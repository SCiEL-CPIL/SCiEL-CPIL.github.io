"""Hexagonal-lattice packing of 12 unit hexagons inside a minimal regular hexagon.
Structural change: dense lattice cluster + DE search over outer center/side/angle."""
import math
import time
import random
import numpy as np

N = 12
APOTHEM = math.sqrt(3.0) / 2.0   # apothem of a unit hexagon

# Compact 12-cell cluster on the hexagonal lattice (rows of 4,4,4).
# Unit lattice vectors: a1=(1,0), a2=(1/2, sqrt(3)/2).
# 12 cells: (i + j*1/2, j*sqrt(3)/2) for i in 0..3, j in 0..2.
def _lattice_cells():
    cells = []
    for j in range(3):
        for i in range(4):
            cells.append((i + 0.5 * j, (math.sqrt(3.0) / 2.0) * j))
    return cells

_CELLS = _lattice_cells()
assert len(_CELLS) == N

def _outer_halfplanes(cx, cy, s, phi):
    """Return (normals, offsets) of the six half-planes of the regular hexagon.
    The hexagon = intersection of the six half-planes n_k . x <= b_k."""
    angs = [phi + k * math.pi / 3.0 for k in range(6)]
    normals = np.array([[math.cos(a), math.sin(a)] for a in angs])
    # center is at distance s*apothem from each side
    b = s * APOTHEM
    offsets = b - normals @ np.array([cx, cy])
    return normals, offsets

def _min_side_for_orientation(pts, phi):
    """Minimum regular-hexagon side (fixed orientation phi) enclosing pts.
    Returns (side, cx, cy)."""
    angs = [phi + k * math.pi / 3.0 for k in range(6)]
    normals = np.array([[math.cos(a), math.sin(a)] for a in angs])
    proj = pts @ normals.T                 # (N, 6)
    lo = proj.min(axis=0)
    hi = proj.max(axis=0)
    # side must satisfy: for each k, s*apothem - n_k.c >= hi_k  AND  s*apothem + n_k.c >= -lo_k
    s = np.maximum(
        (hi + 0.0) / APOTHEM,
        (-lo + 0.0) / APOTHEM,
    )
    # incorporate center offset: we can shift c to balance; optimal c_k = -(hi_k+lo_k)/2
    # then required s = (hi_k - lo_k)/2 / apothem  ... but also each side distance = s*apothem
    # Re-derive cleanly:
    # Constraint: n_k.c + s*apothem >= hi_k  and  -n_k.c + s*apothem >= -lo_k
    # Add: 2*s*apothem >= hi_k - lo_k  =>  s >= (hi_k - lo_k)/(2*apothem)
    # With c_k = -(hi_k+lo_k)/2, both become s*apothem >= (hi_k-lo_k)/2. Feasible iff s>=max.
    s = np.max((hi - lo) / (2.0 * APOTHEM))
    c = np.array([-(hi[k] + lo[k]) / 2.0 for k in range(6)])
    # solve for center c (2D) from the six equations n_k.c = c_k (least squares)
    # normals (6,2), c (6,)
    center, *_ = np.linalg.lstsq(normals, c, rcond=None)
    return float(s), float(center[0]), float(center[1])

def _cost(params):
    """params = [cx, cy, s, phi]. Return the enclosing side length (>= s if infeasible)."""
    cx, cy, s, phi = params
    pts = np.array(_CELLS)
    s_need, _, _ = _min_side_for_orientation(pts, phi)
    # If the hexagon of side s at (cx,cy) encloses pts, cost = s; else we must grow to s_need.
    # Cheapest enclosing hexagon of this orientation has side s_need; but we also pay for the
    # actual side s only if it already encloses. Use max(s, s_need) as a smooth proxy.
    return max(s, s_need)

def _enclosing_side(pts, phi):
    """True minimal side of a regular hexagon of orientation phi enclosing pts."""
    s, _, _ = _min_side_for_orientation(pts, phi)
    return s

def _construct_from_params(params):
    cx, cy, s, phi = params
    s_need, _, _ = _min_side_for_orientation(np.array(_CELLS), phi)
    s = max(s, s_need)
    inner = [[float(x), float(y), 0.0] for (x, y) in _CELLS]
    return inner, (float(cx), float(cy)), float(s), float(math.degrees(phi))

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 110.0          # safety margin before 120s kill
    pts = np.array(_CELLS)

    # ---- warm start: lattice cluster, minimal enclosing hexagon over phi ----
    best_params = None
    best_score = float('inf')
    for phi in np.linspace(0.0, math.pi / 3.0, 61):
        s, cx, cy = _min_side_for_orientation(pts, phi)
        score = s
        if score < best_score:
            best_score = score
            best_params = [cx, cy, s, phi]

    # ---- differential evolution over [cx, cy, s, phi] ----
    def _init_pop(n):
        pop = []
        for _ in range(n):
            cx = random.uniform(-1.0, 1.0)
            cy = random.uniform(-1.0, 1.0)
            s = random.uniform(best_score * 0.8, best_score * 1.3)
            phi = random.uniform(0.0, math.pi / 3.0)
            pop.append([cx, cy, s, phi])
        return pop

    pop_size = 24
    F = 0.8
    CR = 0.9
    pop = _init_pop(pop_size)
    scores = [_cost(p) for p in pop]
    # ensure warm start is in the population
    pop[0] = list(best_params)
    scores[0] = best_score

    gen = 0
    while time.time() < deadline:
        gen += 1
        order = sorted(range(pop_size), key=lambda i: scores[i])
        for i in range(pop_size):
            if time.time() > deadline:
                break
            a, b, c = random.sample([j for j in range(pop_size) if j != i], 3)
            trial = [
                pop[a][d] + F * (pop[b][d] - pop[c][d]) if random.random() < CR
                else pop[i][d]
                for d in range(4)
            ]
            trial[3] = trial[3] % (math.pi / 3.0)   # keep phi in fundamental domain
            ts = _cost(trial)
            if ts < scores[i]:
                pop[i] = trial
                scores[i] = ts
            # global best
            if ts < best_score:
                best_score = ts
                best_params = list(trial)

    return _construct_from_params(best_params)
# EVOLVE-BLOCK-END
