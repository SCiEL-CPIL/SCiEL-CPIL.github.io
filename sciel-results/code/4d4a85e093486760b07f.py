import math
import time
import numpy as np
from itertools import product

try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

N = 12
SQRT3 = math.sqrt(3.0)
# Normal directions for a flat-top outer hexagon (apex at right)
NORM_ANG = np.array([0.0, math.pi/3, 2*math.pi/3, math.pi, 4*math.pi/3, 5*math.pi/3])
NORM_C = np.cos(NORM_ANG)
NORM_S = np.sin(NORM_ANG)
# Precomputed vertices for unit hexagon with angle 0 (pointy top)
HEX_V0 = np.array([[math.cos(math.radians(30+60*k)), math.sin(math.radians(30+60*k))] for k in range(6)]).T

def _hex_vertices(cx, cy, theta):
    c, s = math.cos(theta), math.sin(theta)
    vx0 = HEX_V0[0]; vy0 = HEX_V0[1]
    return cx + c*vx0 - s*vy0, cy + s*vx0 + c*vy0

def _min_L(cx, cy, theta):
    vx, vy = _hex_vertices(cx, cy, theta)
    projs = vx[:, None]*NORM_C[None, :] + vy[:, None]*NORM_S[None, :]
    return np.max(projs) / (SQRT3/2.0)

def _min_L_all(positions, thetas):
    """positions (N,2), thetas (N,) -> (N,)"""
    out = np.zeros(len(positions))
    for i in range(len(positions)):
        out[i] = _min_L(positions[i,0], positions[i,1], thetas[i])
    return out

def _hex_sep(theta1, theta2):
    """Min center distance for two unit hexes (same orientation frame, angles theta1,theta2) to be disjoint."""
    dth = abs(theta1 - theta2) % (2*math.pi)
    dth = min(dth, 2*math.pi - dth)
    # Due to 6-fold symmetry, dth mod 60 degrees
    dth_mod = dth % (math.pi/3)
    if dth_mod > math.pi/6:
        dth_mod = math.pi/3 - dth_mod
    # Sample 181 directions in [0, pi/6]
    best = float('inf')
    for phi_d in range(0, 31):
        phi = math.radians(phi_d)
        h1 = max(math.cos(phi - math.radians(60*k)) for k in range(6))
        h2 = max(math.cos(phi - dth - math.radians(60*k)) for k in range(6))
        d = h1 + h2
        if d < best: best = d
    return best

# Precompute separation table for relative rotations 0..60 deg, 0.5 deg steps
_SEP_N = 121
_SEP_T = np.array([_hex_sep(0.0, math.radians(i*60.0/_SEP_N)) for i in range(_SEP_N)])

def _sep_lookup(dth):
    dth = abs(dth) % (2*math.pi)
    dth = min(dth, 2*math.pi - dth)
    dth_deg = math.degrees(dth) % 60.0
    idx = dth_deg / 60.0 * (_SEP_N - 1)
    i0 = int(idx); i1 = min(i0+1, _SEP_N-1)
    f = idx - i0
    return _SEP_T[i0]*(1-f) + _SEP_T[i1]*f

def construct():
    t0 = time.time()
    DEADLINE = t0 + 140.0

    # --- Build initial candidate configurations ---
    # Candidate 1: 7 center + 5 of the next ring (side-3 minus 3 corners)
    # Candidate 2: 12 ring of side-3 outer ring
    # Candidate 3: 2x6 grid-like
    # Candidate 4: Random near hex grid

    def hex_center(q, r):
        return 1.5*q, SQRT3*(r + q/2.0)

    def build_7plus5():
        centers = []
        # 7 central: q,r with |q|<=1,|r|<=1,|q+r|<=1
        for q in range(-1, 2):
            for r in range(-1, 2):
                if abs(q+r) <= 1:
                    centers.append(hex_center(q, r))
        # 5 outer: pick 5 of the 6 ring-2 centers (exclude 3 corners for symmetry)
        # Ring 2 centers: distance sqrt(3) from nearest ring-1
        # Pick the 6 non-corner ring-2: (2,0),(0,2),(-2,0),(0,-2),(2,-2),(-2,2) are corners
        # Non-corner ring-2: (1,1),(-1,1),(-1,-1),(1,-1)... wait
        # Actually let's just add 5 specific ones
        outer = [(1,1),(-1,1),(0,2),(1,-2),(-2,1)]
        for q,r in outer:
            centers.append(hex_center(q,r))
        return np.array(centers)

    def build_12ring():
        centers = []
        for k in range(12):
            # 12 positions on a ring of radius ~3*sqrt(3)/2
            ang = math.radians(30 + k*30)
            r = 1.5*SQRT3
            centers.append((r*math.cos(ang), r*math.sin(ang)))
        return np.array(centers)

    def build_2x6():
        centers = []
        for i in range(6):
            for j in range(2):
                x = i*SQRT3
                y = j*3.0 - 1.5
                centers.append((x - 2.5*SQRT3, y))
        return np.array(centers)

    def build_random(seed):
        rng = np.random.RandomState(seed)
        centers = []
        for i in range(12):
            # Roughly hex-grid positions
            q = rng.randint(-2, 3)
            r = rng.randint(-2, 3)
            if abs(q+r) <= 2:
                centers.append(hex_center(q, r))
            else:
                centers.append((rng.uniform(-3,3), rng.uniform(-3,3)))
        # Ensure no duplicates
        seen = set()
        final = []
        for c in centers:
            key = (round(c[0],2), round(c[1],2))
            if key not in seen:
                seen.add(key)
                final.append(c)
            else:
                final.append((rng.uniform(-3,3), rng.uniform(-3,3)))
        return np.array(final)

    def build_hex_grid_partial():
        # 12 positions from hex grid: center + 6 + 5 of next ring
        centers = [hex_center(0,0)]
        for k in range(6):
            ang = math.radians(60*k)
            centers.append((SQRT3*math.cos(ang), SQRT3*math.sin(ang)))
        # 5 more from ring 2 (non-corner)
        for (q,r) in [(1,1),(-1,1),(1,-1),(-1,-1),(0,2)]:
            centers.append(hex_center(q,r))
        return np.array(centers)

    initial_configs = [
        build_7plus5(),
        build_12ring(),
        build_2x6(),
        build_hex_grid_partial(),
        build_random(42),
        build_random(123),
    ]

    # --- Objective function ---
    def objective(params):
        L = params[-1]
        pos = params[:2*N].reshape(N, 2)
        thetas = params[2*N:3*N]

        # Containment: soft max of required L for each hex
        L_req = _min_L_all(pos, thetas)
        max_lr = np.max(L_req)
        if max_lr > 1e-10:
            soft_L = max_lr + math.log(np.sum(np.exp(20.0*(L_req - max_lr))))/20.0
        else:
            soft_L = 0.0

        # Overlap penalty
        pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i,0]-pos[j,0]
                dy = pos[i,1]-pos[j,1]
                d = math.sqrt(dx*dx + dy*dy)
                md = _sep_lookup(thetas[i]-thetas[j])
                if d < md:
                    pen += (md-d)**2 * 80.0

        return soft_L + pen

    def objective_hard(params):
        """Return actual max L, or inf if invalid."""
        L = params[-1]
        pos = params[:2*N].reshape(N, 2)
        thetas = params[2*N:3*N]
        L_req = _min_L_all(pos, thetas)
        max_L = np.max(L_req)
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i,0]-pos[j,0]
                dy = pos[i,1]-pos[j,1]
                d = math.sqrt(dx*dx + dy*dy)
                md = _sep_lookup(thetas[i]-thetas[j])
                if d < md - 1e-9:
                    return float('inf')
        return max_L

    best_L = float('inf')
    best_params = None

    for cfg in initial_configs:
        if time.time() > DEADLINE:
            break
        pos0 = cfg.copy()
        # Initial rotation: 0 for all
        theta0 = np.zeros(N)
        # Initial L estimate
        L_req = _min_L_all(pos0, theta0)
        L0 = np.max(L_req) * 1.02

        params0 = np.concatenate([pos0.flatten(), theta0, [L0]])

        # Quick check
        hard_val = objective_hard(params0)
        if hard_val < best_L:
            best_L = hard_val
            best_params = params0.copy()

        # Optimize
        for method in ['Nelder-Mead', 'Powell']:
            if time.time() > DEADLINE - 5:
                break
            try:
                if method == 'Nelder-Mead':
                    res = minimize(objective, params0, method='Nelder-Mead',
                                   options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-10})
                else:
                    res = minimize(objective, params0, method='Powell',
                                   options={'maxiter': 600, 'xtol': 1e-8, 'ftol': 1e-10})
                if hasattr(res, 'x'):
                    hard_val = objective_hard(res.x)
                    if hard_val < best_L:
                        best_L = hard_val
                        best_params = res.x.copy()
                        params0 = res.x.copy()
            except Exception:
                pass

    if best_params is None:
        # Fallback: just use first config
        pos0 = initial_configs[0]
        theta0 = np.zeros(N)
        L_req = _min_L_all(pos0, theta0)
        best_L = float(np.max(L_req))
        best_params = np.concatenate([pos0.flatten(), theta0, [best_L]])

    pos = best_params[:2*N].reshape(N, 2)
    thetas = best_params[2*N:3*N]

    inner = [[pos[i,0], pos[i,1], math.degrees(thetas[i])] for i in range(N)]
    return inner, (0.0, 0.0), best_L, 0.0
