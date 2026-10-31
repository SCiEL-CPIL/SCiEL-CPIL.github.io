"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import linprog

N = 26
DEADLINE = 112.0  # safety margin below 120s kill

# ---- initial configuration: grid of equal circles ----
def _initial_config():
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side) for i in range(side) for j in range(side)]
    x0 = np.array(cells[:N], dtype=float)
    r0 = 0.5 / side - 1e-4
    r = np.full(N, r0)
    return x0, r

def _feasible(x, r, eps=1e-9):
    # boundary
    if np.any(x + r < -eps) or np.any(x - r < -eps) or np.any(1 - x + r < -eps) or np.any(1 - x - r < -eps):
        return False
    # pairwise
    for i in range(N):
        for j in range(i + 1, N):
            d2 = (x[i,0]-x[j,0])**2 + (x[i,1]-x[j,1])**2
            if d2 < (r[i]+r[j])**2 - eps:
                return False
    return True

def _sum_r(x, r):
    return float(np.sum(r))

# ---- interior-point-like gradient step ----
def _ipm_step(x, r, mu, iters, dt):
    n = len(r)
    # barrier parameters
    alpha = 50.0  # boundary barrier strength
    beta = 50.0   # overlap barrier strength
    # We maximize sum r - mu * (barriers). Use gradient ascent on F = sum r - mu*B.
    # F = sum r - mu*( alpha * sum_i b_i + beta * sum_{i<j} o_ij )
    # b_i = max(0, -x_i + r_i)^2 + max(0, x_i - (1-r_i))^2 + (same for y)
    # o_ij = max(0, (r_i+r_j) - dist_ij)^2
    best_x, best_r = x.copy(), r.copy()
    best_val = -np.inf
    lr = 1e-3
    for it in range(iters):
        if time.time() > dt:
            break
        # compute gradient of B
        gx = np.zeros((n, 2))
        gr = np.zeros(n)
        # boundary
        for i in range(n):
            r_i = r[i]
            # left: x - r < 0
            v = x[i,0] - r_i
            if v < 0:
                gx[i,0] += 2*alpha*(-v) * (-1)   # d/dx of (x-r)^2 = 2(x-r); but barrier is (x-r)^2 when x-r<0
                # actually b = (x-r)^2 for x-r<0 ; db/dx = 2(x-r), db/dr = -2(x-r)
                gr[i] += 2*alpha*(-v) * (-1)
            # right: 1 - x - r < 0  -> x + r > 1
            v2 = 1 - x[i,0] - r_i
            if v2 < 0:
                gx[i,0] += 2*alpha*(-v2) * (-1)
                gr[i] += 2*alpha*(-v2) * (-1)
            # bottom
            v3 = x[i,1] - r_i
            if v3 < 0:
                gx[i,1] += 2*alpha*(-v3) * (-1)
                gr[i] += 2*alpha*(-v3) * (-1)
            # top
            v4 = 1 - x[i,1] - r_i
            if v4 < 0:
                gx[i,1] += 2*alpha*(-v4) * (-1)
                gr[i] += 2*alpha*(-v4) * (-1)
        # pairwise
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i,0]-x[j,0]; dy = x[i,1]-x[j,1]
                d = np.hypot(dx, dy)
                s = r[i]+r[j]
                if d < s:
                    if d < 1e-12:
                        # degenerate, push apart arbitrarily
                        gx[i,0] += 1.0; gx[j,0] -= 1.0
                        gr[i] += 1.0; gr[j] += 1.0
                        continue
                    # o = (s-d)^2 ; do/dx_i = 2(s-d)*(-dx/d) ; do/dr_i = 2(s-d)
                    f = 2.0*(s-d)
                    gx[i,0] += beta*f*(-dx/d)
                    gx[j,0] += beta*f*( dx/d)
                    gx[i,1] += beta*f*(-dy/d)
                    gx[j,1] += beta*f*( dy/d)
                    gr[i] += beta*f
                    gr[j] += beta*f
        # gradient of F = sum r - mu*B  =>  grad_F_x = -mu*gx ; grad_F_r = 1 - mu*gr
        grad_x = -mu*gx
        grad_r = 1.0 - mu*gr
        # line search: try step
        cand_x = x + lr*grad_x
        cand_r = r + lr*grad_r
        # clip radii to positive
        cand_r = np.maximum(cand_r, 1e-6)
        # quick feasibility check of objective improvement using a surrogate:
        # accept if sum r increased (we track sum r, not F, to keep feasibility focus)
        val = np.sum(cand_r)
        if val > best_val:
            best_val = val
            best_x, best_r = cand_x.copy(), cand_r.copy()
        # adaptive lr
        if val <= _sum_r(x, r):
            lr *= 0.7
        else:
            lr = min(lr*1.05, 5e-3)
        x, r = cand_x, cand_r
        # decay mu slowly
        mu *= 0.995
    return best_x, best_r

# ---- final LP polish on contact graph ----
def _lp_polish(x, r, dt):
    n = len(r)
    # build contact graph: pairs close to touching
    tol = 1e-3
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            d = np.hypot(x[i,0]-x[j,0], x[i,1]-x[j,1])
            if d <= r[i]+r[j] + tol:
                adj[i].append(j); adj[j].append(i)
    # boundary "contacts"
    for i in range(n):
        if x[i,0] - r[i] <= tol: adj[i].append(-1)   # left
        if 1 - x[i,0] - r[i] <= tol: adj[i].append(-2) # right
        if x[i,1] - r[i] <= tol: adj[i].append(-3)   # bottom
        if 1 - x[i,1] - r[i] <= tol: adj[i].append(-4) # top
    # LP: max sum r  s.t.  r_i + r_j <= d_ij  for each edge (i,j)
    #      r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    #      r_i >= 0
    m = 0
    A = []
    b = []
    # pair constraints
    for i in range(n):
        for j in adj[i]:
            if j > i:
                d = np.hypot(x[i,0]-x[j,0], x[i,1]-x[j,1])
                row = np.zeros(n)
                row[i] = 1.0; row[j] = 1.0
                A.append(row); b.append(d)
    # boundary constraints
    bounds = []
    for i in range(n):
        # r_i <= x_i
        row = np.zeros(n); row[i] = 1.0
        A.append(row); b.append(x[i,0])
        row = np.zeros(n); row[i] = 1.0
        A.append(row); b.append(1 - x[i,0])
        row = np.zeros(n); row[i] = 1.0
        A.append(row); b.append(x[i,1])
        row = np.zeros(n); row[i] = 1.0
        A.append(row); b.append(1 - x[i,1])
        bounds.append((0, None))
    if not A:
        return r
    A = np.array(A)
    b = np.array(b)
    try:
        res = linprog(c=-np.ones(n), A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.success:
            r_new = res.x
            # verify feasibility of the new radii with fixed positions
            if _feasible(x, r_new, eps=-1e-6):
                return np.maximum(r_new, 1e-6)
    except Exception:
        pass
    return r

def construct():
    t0 = time.time()
    x, r = _initial_config()
    best = (x.copy(), r.copy(), _sum_r(x, r))
    mu = 0.1
    # phase 1: grow radii while keeping feasibility via IPM
    # We'll do a sequence of IPM steps, each with a few iterations.
    step_budget = 200  # total IPM sub-iterations across all phases
    it = 0
    while it < step_budget and time.time() < DEADLINE - 8:
        x, r = _ipm_step(x, r, mu, iters=5, dt=DEADLINE - 8)
        it += 5
        val = _sum_r(x, r)
        if val > best[2]:
            best = (x.copy(), r.copy(), val)
        # anneal mu
        mu *= 0.97
        # occasionally re-verify and repair feasibility
        if not _feasible(x, r, eps=-1e-6):
            # shrink radii slightly to restore feasibility
            scale = 0.95
            r = r * scale
            # push centers toward feasible region
            for i in range(N):
                r_i = r[i]
                x[i,0] = min(max(x[i,0], r_i), 1 - r_i)
                x[i,1] = min(max(x[i,1], r_i), 1 - r_i)
        if time.time() > DEADLINE - 8:
            break
    # final polish: LP on best configuration
    if time.time() < DEADLINE - 3:
        bx, br, _ = best
        br = _lp_polish(bx, br, DEADLINE - 3)
        if _feasible(bx, br, eps=-1e-6):
            best = (bx.copy(), br.copy(), _sum_r(bx, br))
    # assemble output
    bx, br, _ = best
    out = np.zeros((N, 3))
    out[:, 0] = bx
    out[:, 1] = br
    out[:, 2] = br
    return out
