"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    
    # 1. Symmetrize A for dynamics (Cost is invariant if B is symmetric)
    a_sym = (a + a.T) / 2.0
    
    # 2. MDS Embedding of locations
    # b is distance matrix. We need squared distances for MDS.
    D = b ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    M = -0.5 * J @ D @ J
    # Symmetrize M to handle numerical errors
    M = (M + M.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    # Sort eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Ensure eigenvalues are non-negative
    eigenvalues = np.maximum(eigenvalues, 0)
    Y = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    
    # Gram matrix of locations
    G = Y.T @ Y
    
    # 3. Dynamics Simulation
    # Initialize X (facility coordinates)
    X = Y.copy()
    V = np.zeros_like(X)
    
    # Symplectic Integrator: Velocity Verlet
    # Potential E(X) = -tr(A X G X^T)
    # Force F = -grad E = 2 A X G
    
    dt = 0.01
    num_steps = 1000
    sample_freq = 20
    
    candidates = []
    
    for step in range(num_steps):
        # Compute force at current X
        F = 2 * (a_sym @ X) @ G
        
        # Half step position
        X_half = X + 0.5 * dt * V
        # Project X_half to Stiefel manifold
        U, s, Vt = np.linalg.svd(X_half, full_matrices=False)
        X_half_proj = U @ Vt
        
        # Compute force at X_half
        F_half = 2 * (a_sym @ X_half_proj) @ G
        
        # Update velocity
        V = V + 0.5 * dt * (F + F_half)
        
        # Full step position
        X = X_half_proj + 0.5 * dt * V
        # Project X to Stiefel manifold
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        X = U @ Vt
        
        # Sample state
        if step % sample_freq == 0:
            # Discretize X to permutation
            C = np.sum(X**2, axis=1)[:, None] + np.sum(Y**2, axis=1)[None, :] - 2 * (X @ Y.T)
            row_ind, col_ind = linear_sum_assignment(C)
            p = col_ind.copy()
            D_p = b[p[:, None], p[None, :]]
            cost = np.sum(a * D_p)
            candidates.append((cost, p))
    
    # 4. Local Search on best candidates
    candidates.sort(key=lambda x: x[0])
    
    best_p = None
    best_cost = float('inf')
    seen = set()
    
    start_time = time.time()
    time_limit = 110
    
    # Process top candidates
    for cost, p in candidates:
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
                    # Calculate delta cost for swapping i and j
                    # Precompute differences to avoid repeated indexing
                    # delta = sum_{k != i,j} a[i,k](b[y,p[k]] - b[x,p[k]]) + a[k,i](b[p[k],y] - b[p[k],x])
                    #       + a[j,k](b[x,p[k]] - b[y,p[k]]) + a[k,j](b[p[k],x] - b[p[k],y])
                    #       + a[i,j](b[y,x] - b[x,y]) + a[j,i](b[x,y] - b[y,x])
                    #       + a[i,i](b[y,y] - b[x,x]) + a[j,j](b[x,x] - b[y,y])
                    
                    # Vectorized calculation for k != i, j
                    # We can't easily vectorize the "k != i, j" part without masking
                    # But we can compute the full sums and subtract i, j terms
                    # Let's stick to the mask approach but optimize it slightly
                    
                    mask = np.ones(n, dtype=bool)
                    mask[i] = False
                    mask[j] = False
                    pk = p[mask]
                    
                    # S1: a[i, mask] * (b[y, pk] - b[x, pk])
                    S1 = np.dot(a[i, mask], b[y, pk] - b[x, pk])
                    # S2: a[mask, i] * (b[pk, y] - b[pk, x])
                    S2 = np.dot(a[mask, i], b[pk, y] - b[pk, x])
                    # S3: a[j, mask] * (b[x, pk] - b[y, pk])
                    S3 = np.dot(a[j, mask], b[x, pk] - b[y, pk])
                    # S4: a[mask, j] * (b[pk, x] - b[pk, y])
                    S4 = np.dot(a[mask, j], b[pk, x] - b[pk, y])
                    
                    # Direct terms
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
        # Fallback
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        best_p = np.empty(n, dtype=np.int64)
        best_p[heavy] = central
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
