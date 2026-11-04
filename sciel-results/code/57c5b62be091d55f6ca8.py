import numpy as np
import time
import math

N = 11
S3 = np.sqrt(3.0)

# 6 unit-hexagon edge normals (angles 0,60,...300)
NORM = np.array([
    [1.0, 0.0],
    [0.5, S3 / 2.0],
    [-0.5, S3 / 2.0],
    [-1.0, 0.0],
    [-0.5, -S3 / 2.0],
    [0.5, -S3 / 2.0],
])

def outer_side(C):
    """Smallest outer hexagon (centered at origin, rot 0) side length enclosing all hexagons."""
    m = 0.0
    for k in range(6):
        n = NORM[k]
        proj = C @ n + 1.0
        m = max(m, float(proj.max()))
    return 2.0 * m / S3

def separation_violation(C):
    """Sum of squared overlaps for all pairs."""
    viol = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = C[i] - C[j]
            sep = np.abs(d @ NORM.T).max()
            if sep < 2.0:
                viol += (2.0 - sep) ** 2
    return viol

def energy(C, lambda_pen=100.0):
    """Total energy: side length + penalty for overlaps."""
    side = outer_side(C)
    viol = separation_violation(C)
    return side + lambda_pen * viol

def valid(C, S, eps=1e-9):
    """Check if configuration is valid."""
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            d = C[i] - C[j]
            sep = np.abs(d @ NORM.T).max()
            if sep < 2.0 - eps:
                return False
    # Check inside outer
    apothem = S * S3 / 2.0
    for k in range(6):
        n = NORM[k]
        proj = (C @ n) + 1.0
        if proj.max() > apothem + eps:
            return False
    return True

def init_flower_plus():
    """Initialize with a flower pattern (7 hexagons) and 4 more in the gaps."""
    C = np.zeros((N, 2))
    # Center
    C[0] = [0.0, 0.0]
    # Ring 1
    for k in range(6):
        a = k * np.pi / 3
        C[k+1] = [S3 * np.cos(a), S3 * np.sin(a)]
    # Ring 2: Place 4 hexagons in 4 of the 6 gaps
    # Gap centers are at angles 30, 90, 150, ... 
    # Distance to gap center is 1.5, but we need them further out to not overlap.
    # Let's place them at distance 2.5 * S3? No, that's too far.
    # Let's try distance 2.0 * S3 at angles 30, 90, 150, 210
    gap_angles = [np.pi/6, 3*np.pi/6, 5*np.pi/6, 7*np.pi/6]
    for i, a in enumerate(gap_angles):
        # Start a bit further out to avoid initial overlap
        dist = 2.2 * S3
        C[7+i] = [dist * np.cos(a), dist * np.sin(a)]
    return C

def init_random():
    """Random initialization within a reasonable bound."""
    rng = np.random.RandomState(np.random.randint(10000))
    C = rng.uniform(-3.0 * S3, 3.0 * S3, (N, 2))
    return C

def simulated_annealing(C0, T0, T_min, alpha, steps, deadline):
    """Perform Simulated Annealing."""
    C = C0.copy()
    E = energy(C)
    best_C = C.copy()
    best_E = E
    best_S = outer_side(C)
    
    rng = np.random.RandomState(int(time.time() * 1000))
    
    for step in range(steps):
        if time.time() > deadline:
            break
            
        # Move a random hexagon
        i = rng.randint(N)
        # Step size decreases with temperature
        step_size = 0.1 * (1.0 + T0 / (T0 - T + 1e-9)) # Just a heuristic, let's use fixed small step
        step_size = 0.2
        dx = rng.uniform(-step_size, step_size)
        dy = rng.uniform(-step_size, step_size)
        
        C_new = C.copy()
        C_new[i, 0] += dx
        C_new[i, 1] += dy
        
        E_new = energy(C_new)
        
        if E_new < E or rng.random() < math.exp(-(E_new - E) / T):
            C = C_new
            E = E_new
            S_new = outer_side(C)
            if S_new < best_S:
                best_S = S_new
                best_C = C.copy()
        
        T *= alpha
        
        if T < T_min:
            T = T_min

    return best_C, best_S

def construct():
    start = time.time()
    deadline = start + 140.0 # Safety margin
    
    best_S = float('inf')
    best_C = None
    
    # 1. Try the heuristic initialization
    C0 = init_flower_plus()
    C_opt, S_opt = simulated_annealing(C0, T0=10.0, T_min=0.01, alpha=0.999, steps=50000, deadline=deadline - 30)
    if S_opt < best_S:
        best_S = S_opt
        best_C = C_opt
    
    # 2. Try a few random restarts
    for _ in range(5):
        if time.time() > deadline - 20:
            break
        C_rand = init_random()
        C_opt, S_opt = simulated_annealing(C_rand, T0=10.0, T_min=0.01, alpha=0.999, steps=20000, deadline=deadline - 10)
        if S_opt < best_S:
            best_S = S_opt
            best_C = C_opt
            
    # 3. Final polish with Nelder-Mead if time permits
    if best_C is not None and time.time() < deadline - 5:
        from scipy.optimize import minimize
        def objective_flat(flat):
            C = flat.reshape(N, 2)
            return energy(C)
        
        try:
            res = minimize(objective_flat, best_C.flatten(), method="Nelder-Mead",
                           options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6})
            C_polished = res.x.reshape(N, 2)
            S_polished = outer_side(C_polished)
            if valid(C_polished, S_polished) and S_polished < best_S:
                best_S = S_polished
                best_C = C_polished
        except Exception:
            pass

    # Ensure we have a valid result
    if best_C is None:
        best_C = init_flower_plus()
        best_S = outer_side(best_C)
        # If invalid, spread them out
        if not valid(best_C, best_S):
             # Fallback: linear arrangement
             best_C = np.zeros((N, 2))
             for i in range(N):
                 best_C[i] = [i * 2.0 * S3, 0.0]
             best_S = outer_side(best_C)

    inner = [[float(best_C[i, 0]), float(best_C[i, 1]), 0.0] for i in range(N)]
    return inner, (0.0, 0.0), float(best_S), 0.0
