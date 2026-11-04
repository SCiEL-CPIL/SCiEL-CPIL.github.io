"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    
    # 1. Symmetrize A and B
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    # 2. MDS Embedding of locations
    D = b_sym ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    M = -0.5 * J @ D @ J
    M = (M + M.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = np.maximum(eigenvalues, 0)
    Y = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    
    # Normalize Y to improve numerical stability
    Y_mean = Y.mean(axis=0)
    Y_std = Y.std(axis=0)
    Y_std[Y_std == 0] = 1.0
    Y = (Y - Y_mean) / Y_std
    
    # Gram matrix of locations
    G = Y.T @ Y
    
    # Normalize A and G to have similar scales
    norm_A = np.linalg.norm(a_sym)
    norm_G = np.linalg.norm(G)
    if norm_A > 0 and norm_G > 0:
        scale = norm_G / norm_A
        a_scaled = a_sym * scale
    else:
        a_scaled = a_sym
    
    # 3. Dynamics Simulation
    X = Y.copy()
    V = np.zeros_like(X)
    
    dt = 0.05
    num_steps = 2000
    sample_freq = 10
    
    candidates = []
    
    for step in range(num_steps):
        F = 2 * (a_scaled @ X) @ G
        
        X_half = X + 0.5 * dt * V
        U, s, Vt = np.linalg.svd(X_half, full_matrices=False)
        X_half_proj = U @ Vt
        
        F_half = 2 * (a_scaled @ X_half_proj) @ G
        
        V = V + 0.5 * dt * (F + F_half)
        
        X = X_half_proj + 0.5 * dt * V
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        X = U @ Vt
        
        if step % sample_freq == 0:
            C = np.sum(X**2, axis=1)[:, None] + np.sum(Y**2, axis=1)[None, :] - 2 * (X @ Y.T)
            row_ind, col_ind = linear_sum_assignment(C)
            p = col_ind.copy()
            D_p = b_sym[p[:, None], p[None, :]]
            cost = np.sum(a * D_p)
            candidates.append((cost, p))
    
    # 4. Local Search on best candidates
    candidates.sort(key=lambda x: x[0])
    
    best_p = None
    best_cost = float('inf')
    seen = set()
    
    start_time = time.time()
    time_limit = 110
    
    # Process top 20 candidates
    for cost, p in candidates[:20]:
        if (time.time() - start_time) > time_limit - 10:
            break
        p_tuple = tuple(p)
        if p_tuple in seen:
            continue
        seen.add(p_tuple)
        
        p = p.copy()
        current_cost = cost
        
        # Optimized 2-opt local search
        improved = True
        while improved and (time.time() - start_time) < time_limit - 5:
            improved = False
            for i in range(n):
                if (time.time() - start_time) >= time_limit - 5:
                    break
                x = p[i]
                best_j = -1
                best_delta = 0
                for j in range(i + 1, n):
                    y = p[j]
                    
                    mask = np.ones(n, dtype=bool)
                    mask[i] = False
                    mask[j] = False
                    pk = p[mask]
                    
                    S1 = np.dot(a[i, mask], b[y, pk] - b[x, pk])
                    S2 = np.dot(a[mask, i], b[pk, y] - b[pk, x])
                    S3 = np.dot(a[j, mask], b[x, pk] - b[y, pk])
                    S4 = np.dot(a[mask, j], b[pk, x] - b[pk, y])
                    
                    S5 = a[i, j] * (b[y, x] - b[x, y])
                    S6 = a[j, i] * (b[x, y] - b[y, x])
                    S_self = a[i, i] * (b[y, y] - b[x, x]) + a[j, j] * (b[x, x] - b[y, y])
                    
                    delta = S1 + S2 + S3 + S4 + S5 + S6 + S_self
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_j = j
                
                if best_j != -1 and best_delta < 0:
                    p[i], p[best_j] = p[best_j], p[i]
                    current_cost += best_delta
                    improved = True
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p
    
    if best_p is None:
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        best_p = np.empty(n, dtype=np.int64)
        best_p[heavy] = central
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
