"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 115.0  # Safety margin

    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)

    # 1. Initial Solution: Spectral Heuristic + Hungarian
    # Compute top eigenvectors of A and B
    try:
        # Use top 10 eigenvectors
        k = min(10, n)
        # A is symmetric? QAP flow is usually symmetric. B is symmetric distance.
        # Use eigh for symmetric matrices
        wA, vA = np.linalg.eigh(a)
        wB, vB = np.linalg.eigh(b)
        
        # Take top k eigenvectors (largest eigenvalues)
        # eigh returns in ascending order
        idxA = np.argsort(wA)[::-1][:k]
        idxB = np.argsort(wB)[::-1][:k]
        
        U = vA[:, idxA]
        V = vB[:, idxB]
        
        # Project facilities and locations
        # Distance in spectral space
        # We want to minimize sum a_ij b_p(i)p(j)
        # Approximation: match based on spectral coordinates
        # Create cost matrix for Hungarian
        # C[i, j] = - dot(U[i], V[j]) ? 
        # Actually, a common heuristic is to align the principal components.
        # Let's use the correlation between the projected vectors.
        # Or simpler: use the row sums of the top eigenvectors?
        
        # Let's try a simpler spectral init:
        # Map the sign/pattern of the top eigenvectors.
        # For robustness, let's use a rank-based matching on the top eigenvector values.
        # This is often effective.
        
        # Get the top eigenvector
        u_top = vA[:, -1]
        v_top = vB[:, -1]
        
        # Sort facilities by u_top, locations by v_top
        # But we need to handle the sign.
        # If we flip the sign of one, the correlation changes.
        # Let's compute the cost for both signs and pick the better one?
        # Or just use the absolute values?
        
        # Let's use the full top-k projection for a better initial guess.
        # Cost matrix for Hungarian:
        # We want to minimize the QAP cost.
        # A good proxy is the distance in the spectral subspace.
        # Let's compute the "spectral distance" matrix D where D[i, j] = ||U[i] - V[j]||^2 ?
        # No, that's for matching points.
        
        # Let's stick to the Rank Matching on top eigenvector as a fast, decent init,
        # then refine with local search.
        
        # Determine sign alignment
        # We want high a_ij to correspond to low b_p(i)p(j).
        # This is complex.
        
        # Fallback to Rank Matching on sums if spectral fails or is too slow
        a_sums = a.sum(axis=1)
        b_sums = b.sum(axis=1)
        
        # Try spectral init
        # Project A and B onto top 5 eigenvectors
        k_spec = 5
        idxA = np.argsort(wA)[::-1][:k_spec]
        idxB = np.argsort(wB)[::-1][:k_spec]
        U = vA[:, idxA]
        V = vB[:, idxB]
        
        # Compute a proxy cost matrix
        # C[i, j] = sum_k U[i, k] * V[j, k] * wA[idxA[k]] * wB[idxB[k]]
        # This is a low-rank approximation of the interaction.
        # We want to minimize this? Or maximize?
        # If A and B have similar structure, we want to align them.
        # Let's just use the row sums of the projected matrices.
        
        # Simpler: Just use the top eigenvector rank matching
        # Sort facilities by u_top
        # Sort locations by v_top
        # Check if reversing one helps?
        
        # Let's just use the Rank Matching on sums as it's robust and fast.
        # The local search is the main driver.
        heavy_idx = np.argsort(-a_sums, kind="stable")
        central_idx = np.argsort(b_sums, kind="stable")
        
        p = np.empty(n, dtype=np.int64)
        p[heavy_idx] = central_idx
        
    except Exception:
        # Fallback
        a_sums = a.sum(axis=1)
        b_sums = b.sum(axis=1)
        heavy_idx = np.argsort(-a_sums, kind="stable")
        central_idx = np.argsort(b_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy_idx] = central_idx

    # Compute initial cost
    cost = int(np.sum(a * b[p]))
    best_cost = cost
    best_p = p.copy()

    # Precompute K = A @ B[p].T
    # K[i, j] = a[i] . b[p[j]]
    B_perm = b[p]
    K = a @ B_perm.T

    # Local Search: Best Improvement 2-Opt
    while True:
        if time.time() - start_time > time_limit:
            break

        improved = False
        best_delta = 0
        best_i, best_j = -1, -1

        # Scan all pairs to find best improvement
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            
            # Vectorized delta calculation for all j != i
            # Delta(i, j) = K[i, j] - K[i, i] - K[j, j] + K[j, i] + 2 * a[i, j] * b[p[i], p[j]]
            
            # Term 1: K[i, :]
            # Term 2: - K[i, i] (scalar)
            # Term 3: - K.diagonal() (vector)
            # Term 4: K[:, i] (vector)
            # Term 5: 2 * a[i, :] * b[p[i], p] (vector)
            
            b_row_i = b[p[i], p]
            deltas = K[i, :] - K[i, i] - np.diag(K) + K[:, i] + 2 * a[i, :] * b_row_i
            
            # Exclude i
            deltas[i] = 0 # Set to 0 so it's not picked (we only care if < 0)
            
            # Find min delta
            min_idx = np.argmin(deltas)
            min_val = deltas[min_idx]
            
            if min_val < best_delta:
                best_delta = min_val
                best_i = i
                best_j = min_idx

        if best_delta < 0:
            i, j = best_i, best_j
            u = p[i]
            v = p[j]
            
            # Swap p[i] and p[j]
            p[i] = v
            p[j] = u
            
            # Update K
            # K[:, i] was A @ b[u].T, now should be A @ b[v].T
            # K[:, j] was A @ b[v].T, now should be A @ b[u].T
            K[:, i] = a @ b[v]
            K[:, j] = a @ b[u]
            
            cost += int(best_delta)
            improved = True
            
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
        else:
            # Local optimum reached
            # Perturbation
            if time.time() - start_time > time_limit - 5:
                break
                
            # Random swap
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j:
                u = p[i]
                v = p[j]
                p[i] = v
                p[j] = u
                
                # Update K
                K[:, i] = a @ b[v]
                K[:, j] = a @ b[u]
                
                # Recompute cost
                cost = int(np.sum(a * b[p]))
                if cost < best_cost:
                    best_cost = cost
                    best_p = p.copy()
            else:
                break

    return best_p.tolist()

# EVOLVE-BLOCK-END
