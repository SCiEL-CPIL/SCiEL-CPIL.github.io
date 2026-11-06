"""Photonic-crystal TM band-gap optimizer with symmetry-preserving pixel local search."""
import time
import math
import random
import numpy as np

# 32x32 square cell. Symmetry about cell centre: 180-degree rotation + transpose (D4).
# Fundamental orbit of a pixel (i,j) under D4 about the centre (15.5,15.5):
#   (i,j), (31-i,j), (i,31-j), (31-i,31-j), (j,i), (j,31-i), (31-j,i), (31-j,31-i)
# Pixels on the symmetry axes have smaller orbits.

def _orbits(n=32):
    """Return list of D4 orbits (tuples of (i,j)) for an n x n cell."""
    seen = set()
    out = []
    m = n - 1
    for i in range(n):
        for j in range(n):
            if (i, j) in seen:
                continue
            orb = [(i, j), (m - i, j), (i, m - j), (m - i, m - j),
                   (j, i), (j, m - i), (m - j, i), (m - j, m - i)]
            # deduplicate while preserving order
            uniq = []
            for p in orb:
                if p not in seen and p not in uniq:
                    uniq.append(p)
            for p in uniq:
                seen.add(p)
            out.append(tuple(uniq))
    return out

_ORBITS = _orbits(32)

# Fourier coefficients needed for the (2,0) and (2,2) gap: compute epsinv Fourier
# components at G=(2,0),(0,2),(2,2),(-2,0),(0,-2),(-2,-2) and their conjugates.
# For a real symmetric field, we need:
#   F(2,0), F(0,2), F(2,2)
# where F(G) = (1/N^2) sum_{i,j} epsinv(i,j) exp(-i G . r_{ij})
# and r_{ij} = ((i+0.5)/N - 0.5, (j+0.5)/N - 0.5).
# Since epsinv is 1/eps, and eps is either 1 or 12, epsinv is 1 or 1/12.
# The gap depends on |F(2,0)|, |F(0,2)|, |F(2,2)| and their combinations.

def _surrogate_gap(cell):
    """
    Fast surrogate for the 4-5 TM band gap ratio.
    Uses the magnitudes of the key Fourier coefficients of epsinv.
    For TM polarization in a square lattice, the 4-5 gap is driven by
    the (2,0), (0,2), and (2,2) Fourier components of 1/eps.
    A simple surrogate: gap ~ |F(2,0)| + |F(0,2)| + |F(2,2)|
    Normalized by the mean epsinv.
    """
    cell = np.array(cell, dtype=float)
    eps = np.where(cell > 0.5, 12.0, 1.0)
    epsinv = 1.0 / eps
    N = 32
    # Pixel centers
    i_idx, j_idx = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    xi = (i_idx + 0.5) / N - 0.5
    xj = (j_idx + 0.5) / N - 0.5
    # Fourier coefficients at G = (2,0), (0,2), (2,2)
    # F(G) = (1/N^2) sum epsinv * exp(-i 2pi (Gx*xi + Gy*xj))
    def _F(gx, gy):
        phase = np.exp(-2j * np.pi * (gx * xi + gy * xj))
        return np.sum(epsinv * phase) / (N * N)
    F20 = abs(_F(2, 0))
    F02 = abs(_F(0, 2))
    F22 = abs(_F(2, 2))
    # Also include (1,0) and (1,1) which affect lower bands
    F10 = abs(_F(1, 0))
    F11 = abs(_F(1, 1))
    # The gap ratio is roughly proportional to the coupling strength
    # For a better surrogate, use the combination that drives the 4-5 gap
    # In the square lattice, the 4-5 TM gap is primarily from (2,0)/(0,2) coupling
    # with (2,2) providing additional mixing
    surrogate = F20 + F02 + 0.5 * F22
    # Penalize if the mean fill fraction is too extreme (need both materials)
    fill = np.mean(cell)
    if fill < 0.15 or fill > 0.85:
        surrogate *= 0.5
    return surrogate

def _apply_orbit(cell, orbit_idx, val):
    """Set all pixels in orbit to val (0 or 1)."""
    for (i, j) in _ORBITS[orbit_idx]:
        cell[i][j] = val

def _random_cell(rng, fill_target=0.5):
    """Generate a random symmetric cell."""
    cell = [[0] * 32 for _ in range(32)]
    n_orbits = len(_ORBITS)
    # Estimate how many orbits to fill to hit target fill fraction
    # Total pixels = 32*32 = 1024
    # Each orbit has variable size
    orbit_sizes = [len(orb) for orb in _ORBITS]
    total = sum(orbit_sizes)
    # Randomly select orbits to fill
    order = list(range(n_orbits))
    rng.shuffle(order)
    acc = 0
    for idx in order:
        if acc + orbit_sizes[idx] <= fill_target * total:
            _apply_orbit(cell, idx, 1)
            acc += orbit_sizes[idx]
    return cell

def _local_search(cell, rng, max_iters=5000, time_budget=10.0):
    """Local search: flip orbits to improve surrogate gap."""
    cell = [row[:] for row in cell]
    best_score = _surrogate_gap(cell)
    best_cell = [row[:] for row in cell]
    t0 = time.time()
    n_orbits = len(_ORBITS)
    for it in range(max_iters):
        if time.time() - t0 > time_budget:
            break
        # Pick a random orbit to flip
        idx = rng.randint(0, n_orbits - 1)
        # Flip all pixels in orbit
        current_val = cell[_ORBITS[idx][0][0]][_ORBITS[idx][0][1]]
        new_val = 1 - current_val
        _apply_orbit(cell, idx, new_val)
        score = _surrogate_gap(cell)
        if score > best_score:
            best_score = score
            best_cell = [row[:] for row in cell]
        else:
            # Revert
            _apply_orbit(cell, idx, current_val)
    return best_cell, best_score

def _hill_climb(cell, rng, time_budget=30.0):
    """More thorough local search with restarts."""
    cell = [row[:] for row in cell]
    best_score = _surrogate_gap(cell)
    best_cell = [row[:] for row in cell]
    t0 = time.time()
    n_orbits = len(_ORBITS)
    # Multi-pass: each pass tries all orbits, accept improvements
    for _ in range(20):
        if time.time() - t0 > time_budget:
            break
        improved = False
        for idx in range(n_orbits):
            if time.time() - t0 > time_budget:
                break
            current_val = cell[_ORBITS[idx][0][0]][_ORBITS[idx][0][1]]
            new_val = 1 - current_val
            _apply_orbit(cell, idx, new_val)
            score = _surrogate_gap(cell)
            if score > best_score:
                best_score = score
                best_cell = [row[:] for row in cell]
                improved = True
            else:
                _apply_orbit(cell, idx, current_val)
        if not improved:
            break
    return best_cell, best_score

def design(problem):
    """Design a 32x32 symmetric photonic crystal cell maximizing TM 4-5 band gap."""
    size = problem["pixels"]
    t_start = time.time()
    deadline = t_start + 180.0  # Safety margin before 200s limit
    rng = random.Random(42)
    
    # Start with a reasonable initial design: a dielectric disk
    # with radius ~0.35 (larger than the seed's 0.2 for stronger coupling)
    best_cell = [[0] * 32 for _ in range(32)]
    for i in range(32):
        for j in range(32):
            x = (i + 0.5) / 32 - 0.5
            y = (j + 0.5) / 32 - 0.5
            if x * x + y * y < 0.35 * 0.35:
                best_cell[i][j] = 1
    
    best_score = _surrogate_gap(best_cell)
    
    # Phase 1: Hill climb from the disk
    if time.time() < deadline - 60:
        best_cell, best_score = _hill_climb(best_cell, rng, time_budget=30)
    
    # Phase 2: Random restarts with local search
    n_restarts = 0
    while time.time() < deadline - 30:
        # Generate a random initial cell with varied fill fractions
        fill = rng.choice([0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7])
        cell = _random_cell(rng, fill)
        # Quick local search
        budget = min(15.0, deadline - time.time() - 5)
        if budget < 3:
            break
        cell, score = _local_search(cell, rng, max_iters=2000, time_budget=budget)
        if score > best_score:
            best_score = score
            best_cell = cell
        n_restarts += 1
        # If we're running low on time, do a final hill climb on the best
        if time.time() > deadline - 40:
            break
    
    # Phase 3: Final hill climb on the best candidate
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 5
        if remaining > 3:
            best_cell, best_score = _hill_climb(best_cell, rng, time_budget=remaining)
    
    # Verify symmetry
    cell = best_cell
    # Check 180-degree rotation
    for i in range(32):
        for j in range(32):
            assert cell[i][j] == cell[31 - i][31 - j], "180 rot symmetry violated"
    # Check transpose
    for i in range(32):
        for j in range(32):
            assert cell[i][j] == cell[j][i], "transpose symmetry violated"
    
    return cell
