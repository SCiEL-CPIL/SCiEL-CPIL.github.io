# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

# Precompute problem structure
N = 10
DEMANDS = [(0,5,3),(1,7,2),(2,9,4),(3,6,1),(0,4,5),(2,8,2),(5,9,3),(1,4,2),
           (6,0,4),(3,8,1),(7,2,3),(4,9,2),(8,1,5),(0,6,1),(5,2,2)]
D = len(DEMANDS)

# A[e, d] = 1 if demand d routed clockwise uses edge e
A = np.zeros((N, D), dtype=np.float64)
W = np.zeros(D, dtype=np.float64)
for d, (s, t, w) in enumerate(DEMANDS):
    W[d] = w
    e = s
    while e != t:
        A[e, d] = 1.0
        e = (e + 1) % N

# Base load if all clockwise
base_load = A @ W          # (N,)
# Load when demand d is flipped to counter-clockwise:
# new load on edge e = base[e] - w_d * A[e,d] + w_d * (1 - A[e,d])
# = base[e] + w_d * (1 - 2*A[e,d])
flip_delta = (1.0 - 2.0 * A) * W[None, :]   # (N, D)


def load_from_binary(x):
    """x: binary array (D,). Returns edge loads (N,)."""
    return base_load + flip_delta @ x


def soft_max(logs, alpha):
    """Smooth max via log-sum-exp of exp(alpha * L)."""
    m = np.max(logs)
    return m + np.log(np.sum(np.exp(alpha * (logs - m)))) / alpha


def smooth_objective(z, alpha):
    """z continuous in [0,1]^D; load = base + flip_delta @ z."""
    loads = base_load + flip_delta @ z
    return soft_max(loads, alpha)


def smooth_grad(z, alpha):
    loads = base_load + flip_delta @ z
    m = np.max(loads)
    w = np.exp(alpha * (loads - m))
    w = w / w.sum()
    return flip_delta.T @ w


def max_load(x):
    return np.max(load_from_binary(x))


def solve():
    t0 = time.time()
    deadline = t0 + 600.0
    safety = 30.0

    # --- Phase 1: continuous optimization with log-sum-exp surrogate ---
    best_x = None
    best_ml = np.inf

    # Multiple alpha values and restarts
    for alpha in [20.0, 50.0, 100.0, 200.0, 400.0]:
        if time.time() > deadline - safety:
            break
        for attempt in range(8):
            if time.time() > deadline - safety:
                break
            # Random init in [0,1]^D
            z0 = np.random.rand(D)
            res = minimize(smooth_objective, z0, jac=smooth_grad,
                           method='L-BFGS-B',
                           bounds=[(0.0, 1.0)] * D,
                           options={'maxiter': 200, 'ftol': 1e-10})
            z = res.x
            # Round to binary
            x = (z >= 0.5).astype(np.float64)
            ml = max_load(x)
            if ml < best_ml:
                best_ml = ml
                best_x = x.copy()

    # Also try all-clockwise and all-counter-clockwise
    for x0 in [np.zeros(D), np.ones(D)]:
        ml = max_load(x0)
        if ml < best_ml:
            best_ml = ml
            best_x = x0.copy()

    # --- Phase 2: local binary search (flip single demands) ---
    if time.time() < deadline - safety:
        improved = True
        while improved and time.time() < deadline - safety:
            improved = False
            for d in range(D):
                if best_x[d] == 0:
                    cand = best_x.copy()
                    cand[d] = 1
                else:
                    cand = best_x.copy()
                    cand[d] = 0
                ml = max_load(cand)
                if ml < best_ml:
                    best_ml = ml
                    best_x = cand
                    improved = True

    # --- Phase 3: simulated annealing polish ---
    if time.time() < deadline - safety:
        x = best_x.copy()
        cur_ml = best_ml
        T = 2.0
        T_min = 0.01
        cooling = 0.9995
        rng = np.random.default_rng(42)
        it = 0
        max_it = 200000
        while it < max_it and time.time() < deadline - safety:
            it += 1
            d = rng.integers(D)
            cand = x.copy()
            cand[d] = 1.0 - cand[d]
            cand_ml = max_load(cand)
            delta = cand_ml - cur_ml
            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-9)):
                x = cand
                cur_ml = cand_ml
                if cur_ml < best_ml:
                    best_ml = cur_ml
                    best_x = x.copy()
            T *= cooling
            if T < T_min:
                T = T_min

    # Final local search from best
    if time.time() < deadline - safety:
        improved = True
        while improved and time.time() < deadline - safety:
            improved = False
            for d in range(D):
                cand = best_x.copy()
                cand[d] = 1.0 - cand[d]
                ml = max_load(cand)
                if ml < best_ml:
                    best_ml = ml
                    best_x = cand
                    improved = True

    # Encode: >= 0.5 means counter-clockwise, else clockwise
    # We want returned entry >= 0.5 -> counter-clockwise
    # Our x: 1 = counter-clockwise, 0 = clockwise
    return best_x


def ring_routing() -> np.ndarray:
    routing = solve()
    return routing
# EVOLVE-BLOCK-END
