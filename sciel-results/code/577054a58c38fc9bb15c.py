import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = a.shape[0]
    deadline = time.time() + 117.0  # Safety margin before 120s limit
    
    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
    
    # --- Initialization: Spectral Embedding ---
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    # Spectral embedding for both matrices
    try:
        a_centered = a - a.mean()
        b_centered = b - b.mean()
        
        # Get top eigenvectors
        eigvals_a, eigvecs_a = np.linalg.eigh(a_centered)
        eigvals_b, eigvecs_b = np.linalg.eigh(b_centered)
        
        # Sort by absolute eigenvalue
        idx_a = np.argsort(-np.abs(eigvals_a))
        idx_b = np.argsort(-np.abs(eigvals_b))
        
        # Use top 2 eigenvectors for 2D embedding
        coords_a = eigvecs_a[:, idx_a[:2]]
        coords_b = eigvecs_b[:, idx_b[:2]]
        
        # Project to 1D for ordering
        order_a = np.argsort(coords_a[:, 0], kind='stable')
        order_b = np.argsort(coords_b[:, 0], kind='stable')
        
    except:
        order_a = np.argsort(-a_sums, kind='stable')
        order_b = np.argsort(b_sums, kind='stable')
    
    # Initial assignment: match sorted facilities to sorted locations
    p = np.empty(n, dtype=np.int64)
    p[order_a] = order_b
    
    current_cost = compute_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # --- Local Search with Simulated Annealing ---
    # Precompute for fast delta calculation
    def recompute_precomp(p):
        b_col_p = b[:, p]  # (n, n): [l, k] = b[l, p[k]]
        b_row_p = b[p, :]  # (n, n): [k, l] = b[p[k], l]
        dot_row = a @ b_col_p.T  # dot_row[i, l] = sum_k a[i,k] * b[l, p[k]]
        dot_col = a.T @ b_row_p  # dot_col[i, l] = sum_k a[k,i] * b[p[k], l]
        return dot_row, dot_col
    
    dot_row, dot_col = recompute_precomp(p)
    b_diag = np.diag(b)
    
    # Simulated Annealing parameters
    T = 1000.0  # Initial temperature
    cooling_rate = 0.99995
    min_T = 1.0
    
    # Neighborhood: random swaps
    while time.time() < deadline:
        if T < min_T:
            T = min_T
        
        # Try a batch of swaps
        batch_size = 100
        accepted = 0
        
        for _ in range(batch_size):
            if time.time() > deadline:
                break
            
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            
            u = p[i]
            v = p[j]
            if u == v:
                continue
            
            # Compute delta using precomputed values
            buu = b_diag[u]
            bvv = b_diag[v]
            buv = b[u, v]
            bvu = b[v, u]
            
            delta = 0.0
            delta += a[i, i] * (bvv - buu)
            delta += a[i, j] * (bvu - buv)
            delta += a[j, i] * (buv - bvu)
            delta += a[j, j] * (buu - bvv)
            
            term_row_i = (dot_row[i, v] - dot_row[i, u]) - a[i, i]*(bvv - buu) - a[i, j]*(bvu - buv)
            delta += term_row_i
            
            term_row_j = (dot_row[j, u] - dot_row[j, v]) - a[j, i]*(buv - bvu) - a[j, j]*(buu - bvv)
            delta += term_row_j
            
            term_col_i = (dot_col[i, v] - dot_col[i, u]) - a[i, i]*(bvv - buu) - a[j, i]*(bvu - buv)
            delta += term_col_i
            
            term_col_j = (dot_col[j, u] - dot_col[j, v]) - a[i, j]*(buv - bvu) - a[j, j]*(buu - bvv)
            delta += term_col_j
            
            # Acceptance criterion
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                # Apply swap
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                accepted += 1
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                
                # Incremental update of dot_row and dot_col
                diff_a_row = a[:, i] - a[:, j]
                diff_a_col = a[i, :] - a[j, :]
                
                diff_b_row = b[:, v] - b[:, u]
                diff_b_col = b[v, :] - b[u, :]
                
                dot_row += np.outer(diff_a_row, diff_b_row)
                dot_col += np.outer(diff_a_col, diff_b_col)
        
        # Cool down
        T *= cooling_rate
        
        # If no moves accepted, try a larger perturbation
        if accepted == 0 and T > min_T * 2:
            # Perform several random swaps to escape local minimum
            for _ in range(10):
                ri = np.random.randint(n)
                rj = np.random.randint(n)
                if ri != rj:
                    p[ri], p[rj] = p[rj], p[ri]
            
            current_cost = compute_cost(p)
            dot_row, dot_col = recompute_precomp(p)
    
    return best_p.tolist()
