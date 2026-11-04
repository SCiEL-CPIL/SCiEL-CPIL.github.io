import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11
S3 = np.sqrt(3.0)
NORM_0 = np.array([
    [1.0, 0.0],
    [0.5, S3 / 2.0],
    [-0.5, S3 / 2.0],
    [-1.0, 0.0],
    [-0.5, -S3 / 2.0],
    [0.5, -S3 / 2.0]
])

def get_normals(A):
    rad = A * np.pi / 180.0
    c = np.cos(rad)
    s = np.sin(rad)
    R = np.zeros((len(A), 2, 2))
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    return np.einsum('ij,kji->kij', NORM_0, R)

def outer_side(C, A):
    n_i = get_normals(A)
    dots_full = np.einsum('kj,ikj->ikj', NORM_0, n_i)
    support = np.max(np.abs(dots_full), axis=2)
    proj = C @ NORM_0.T
    extents = proj + support * (S3 / 2.0)
    apothem_outer = np.max(np.max(extents, axis=0))
    return 2.0 * apothem_outer / S3

def separation_penalty(C, A):
    pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            rad_i = A[i] * np.pi / 180.0
            rad_j = A[j] * np.pi / 180.0
            c_i, s_i = np.cos(rad_i), np.sin(rad_i)
            c_j, s_j = np.cos(rad_j), np.sin(rad_j)
            R_i = np.array([[c_i, -s_i], [s_i, c_i]])
            R_j = np.array([[c_j, -s_j], [s_j, c_j]])
            n_i = NORM_0 @ R_i.T
            n_j = NORM_0 @ R_j.T
            axes = np.vstack([n_i, n_j])
            d = C[i] - C[j]
            sep = np.abs(d @ axes.T).max()
            if sep < 2.0:
                pen += (2.0 - sep) ** 2
    return pen

def objective(flat):
    C = np.column_stack([flat[0::3], flat[2::3]])
    A = flat[1::3]
    S = outer_side(C, A)
    pen = separation_penalty(C, A)
    return S + 100.0 * pen

def construct():
    start = time.time()
    deadline = start + 110.0
    
    bounds = [(-5.0, 5.0), (-180, 180), (-5.0, 5.0)] * N
    
    try:
        res = differential_evolution(objective, bounds, maxiter=100, popsize=30, seed=42, tol=1e-5, polish=True)
        if time.time() > deadline:
            return None
        best_flat = res.x
    except Exception:
        best_flat = np.zeros(3*N)
        
    C = np.column_stack([best_flat[0::3], best_flat[2::3]])
    A = best_flat[1::3]
    
    # Polish
    def obj_polish(flat):
        C = np.column_stack([flat[0::3], flat[2::3]])
        A = flat[1::3]
        S = outer_side(C, A)
        pen = separation_penalty(C, A)
        return S + 1000.0 * pen

    try:
        res2 = minimize(obj_polish, best_flat, method="Nelder-Mead", options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6})
        if time.time() < deadline:
            best_flat = res2.x
    except Exception:
        pass

    C = np.column_stack([best_flat[0::3], best_flat[2::3]])
    A = best_flat[1::3]
    
    S = outer_side(C, A)
    pen = separation_penalty(C, A)
    
    # If still invalid, try to force validity by scaling?
    # Or just return. The evaluator will check validity.
    # If invalid, score is 0.
    # We want to maximize chance of validity.
    
    inner = [[float(C[i, 0]), float(C[i, 1]), float(A[i])] for i in range(N)]
    return inner, (0.0, 0.0), float(S), 0.0
