"""Photonic-crystal TM band-gap maximizer: symmetry-preserving evolutionary search."""
import time
import math
import random
import numpy as np

# ---- deadline guard: hard wall-clock budget with safety margin ----
_DEADLINE = 210.0          # seconds; evaluator kills at ~230s
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)


# ---------------- symmetry (C4v about cell centre) ----------------
# For an even NxN cell, a pixel's orbit under the square group about the
# centre (0.5, 0.5) is: (i,j),(j,i),(N-1-i,j),(N-1-j,i),(i,N-1-j),(j,N-1-i),
# (N-1-i,N-1-j),(N-1-j,N-1-i).  The 8 orbit representatives (i<j, i<j+N-1)
# tile the cell.

def _orbit_reps(N):
    reps = []
    seen = set()
    for i in range(N):
        for j in range(N):
            if (i, j) in seen:
                continue
            orbit = [(i, j), (j, i), (N - 1 - i, j), (N - 1 - j, i),
                     (i, N - 1 - j), (j, N - 1 - i), (N - 1 - i, N - 1 - j),
                     (N - 1 - j, N - 1 - i)]
            for p in orbit:
                seen.add(p)
            reps.append(orbit)
    return reps


def _orbit_fill(arr, orbit, val):
    for (i, j) in orbit:
        arr[i, j] = val


def _rep_count(N):
    return len(_orbit_reps(N))


# ---------------- fast approximate PWE band solver ----------------
# Build the inverse-permittivity matrix from a coarse Fourier basis and
# solve the 2x2 (electric-field) generalized eigenvalue problem at a set of
# k-points.  This is cheap (matrix ~ (2*NB+1)^2) and good enough to rank
# candidates during the search.

def _build_eps_inv(N, cell, NB):
    """Return (eps_inv, gvecs) for the coarse basis.

    eps_inv[k1, k2] = 1/eps at (k1-k2) sampled on the coarse grid.
    gvecs: list of (gx, gy) in units of 2*pi/L.
    """
    # Fourier coefficients of 1/eps on the coarse grid
    # c[m, n] = sum_{i,j} (1/eps(i,j)) exp(-2*pi i (m i + n j)/N)
    eps = np.where(cell == 1, 12.0, 1.0)
    einv = 1.0 / eps
    # 2D FFT gives c[m,n] = sum einv[i,j] exp(-2*pi i (m i + n j)/N) up to a constant
    F = np.fft.fft2(einv) / (N * N)
    # Build the coarse index set: all (m, n) with |m|<=NB, |n|<=NB
    idx = []
    for m in range(-NB, NB + 1):
        for n in range(-NB, NB + 1):
            idx.append((m, n))
    gvecs = idx
    M = len(idx)
    pos = {(m, n): q for q, (m, n) in enumerate(idx)}
    eps_inv = np.empty((M, M), dtype=float)
    for a, (ma, na) in enumerate(idx):
        for b, (mb, nb) in enumerate(idx):
            dm = ma - mb
            dn = na - nb
            # wrap into [0, N)
            dm = dm % N
            dn = dn % N
            eps_inv[a, b] = F[dm, dn].real
    return eps_inv, gvecs


def _solve_bands(eps_inv, gvecs, kx, ky, NB, n_bands):
    """Solve for n_bands lowest TM band frequencies at (kx, ky).

    Returns list of omega (in units of c/L * ...), only relative values matter.
    """
    M = len(gvecs)
    # (k + G) dot (k + G')
    G = np.array(gvecs, dtype=float)          # (M, 2)
    kvec = np.array([kx, ky], dtype=float)
    KG = kvec[None, :] + G                     # (M, 2)
    K2 = np.einsum('mi,mj->ij', KG, KG)        # (M, M)
    # generalized eigen: (K2 @ eps_inv) E = w E   where w = (k+G)^2
    A = K2 @ eps_inv
    # symmetric-ish? not exactly; use eig
    vals, _ = np.linalg.eig(A)
    vals = np.real(vals)
    vals = vals[vals > 1e-12]
    vals.sort()
    omega = np.sqrt(vals[:n_bands])
    return omega


def _kpoints():
    """Irreducible BZ k-points for the square lattice (2*pi/L units).

    Boundary + interior points of the triangle (0,0)-(1,0.5)-(0,0.5)... using
    the standard irreducible wedge: 0<=kx<=ky, kx+ky<=1  (in 2*pi/L units).
    """
    pts = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, 0.5),
        (0.25, 0.25),
        (0.33, 0.17),
        (0.17, 0.33),
        (0.4, 0.2),
        (0.2, 0.4),
        (0.42, 0.42),
        (0.3, 0.3),
        (0.5, 0.25),
        (0.25, 0.5),
    ]
    return pts


def _gap_score_fast(N, cell, NB, n_bands):
    """Compute the gap-midgap ratio between bands 4 and 5 (0-indexed 3,4)."""
    eps_inv, gvecs = _build_eps_inv(N, cell, NB)
    lower = None
    upper = None
    for (kx, ky) in _kpoints():
        om = _solve_bands(eps_inv, gvecs, kx, ky, NB, n_bands)
        if len(om) < 5:
            continue
        o4 = om[3]
        o5 = om[4]
        if lower is None or o4 > lower:
            lower = o4
        if upper is None or o5 < upper:
            upper = o5
    if lower is None or upper is None:
        return -1.0
    return 2.0 * (upper - lower) / (upper + lower)


# ---------------- initial design: centred dielectric rod ----------------
def _initial_cell(N, radius):
    cell = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            x = (i + 0.5) / N - 0.5
            y = (j + 0.5) / N - 0.5
            if x * x + y * y < radius * radius:
                cell[i, j] = 1
    return cell


# ---------------- evolutionary search ----------------
def design(problem):
    N = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)

    random.seed(12345)
    np.random.seed(12345)

    orbits = _orbit_reps(N)
    n_rep = len(orbits)

    # Coarse solver settings for the search
    NB_coarse = 3          # 7x7 basis
    NB_fine = 8            # 17x17 basis for final refinement
    n_bands = 6

    # Start from the textbook rod, plus a few diverse seeds
    seeds = []
    for r in (0.20, 0.25, 0.30, 0.35):
        seeds.append(_initial_cell(N, r))
    # A few random symmetric seeds
    for _ in range(4):
        c = np.zeros((N, N), dtype=int)
        for orbit in orbits:
            if random.random() < 0.5:
                _orbit_fill(c, orbit, 1)
        seeds.append(c)

    # Evaluate seeds with the coarse solver
    scored = []
    for c in seeds:
        s = _gap_score_fast(N, c, NB_coarse, n_bands)
        scored.append((s, c.copy()))
    scored.sort(key=lambda t: -t[0])
    best_score, best_cell = scored[0]

    # Population
    pop_size = 12
    population = []
    for i in range(pop_size):
        if i < len(scored):
            population.append((scored[i][0], scored[i][1].copy()))
        else:
            c = best_cell.copy()
            # mutate a few orbits
            for _ in range(random.randint(1, 3)):
                orbit = orbits[random.randrange(n_rep)]
                _orbit_fill(c, orbit, random.choice([0, 1]))
            s = _gap_score_fast(N, c, NB_coarse, n_bands)
            population.append((s, c))

    # Main search loop
    generations = 0
    while _time_left() > 30.0:
        generations += 1
        # Elitism: keep the best
        population.sort(key=lambda t: -t[0])
        cur_best = population[0]
        if cur_best[0] > best_score:
            best_score = cur_best[0]
            best_cell = cur_best[1].copy()

        new_pop = [population[0]]  # elite
        for _ in range(pop_size - 1):
            # parent selection (tournament)
            a = population[random.randrange(pop_size)]
            b = population[random.randrange(pop_size)]
            parent = a if a[0] >= b[0] else b
            child = parent[1].copy()
            # mutation: flip a small number of orbits
            n_flip = random.randint(1, 4)
            for _ in range(n_flip):
                orbit = orbits[random.randrange(n_rep)]
                _orbit_fill(child, orbit, random.choice([0, 1]))
            # occasional larger mutation
            if random.random() < 0.15:
                for _ in range(random.randint(4, 8)):
                    orbit = orbits[random.randrange(n_rep)]
                    _orbit_fill(child, orbit, random.choice([0, 1]))
            s = _gap_score_fast(N, child, NB_coarse, n_bands)
            new_pop.append((s, child))
        population = new_pop

        # If time is getting tight, reduce population
        if _time_left() < 60.0 and pop_size > 6:
            pop_size = 6

    # Final refinement: re-evaluate the top few coarse candidates with the
    # finer basis and pick the best.
    population.sort(key=lambda t: -t[0])
    top = population[:5]
    refined_best = best_score
    refined_cell = best_cell
    for (s, c) in top:
        if _time_left() < 15.0:
            break
        s_fine = _gap_score_fast(N, c, NB_fine, n_bands)
        if s_fine > refined_best:
            refined_best = s_fine
            refined_cell = c

    # Return as nested list of ints
    out = []
    for i in range(N):
        row = []
        for j in range(N):
            row.append(int(refined_cell[i, j]))
        out.append(row)
    return out
