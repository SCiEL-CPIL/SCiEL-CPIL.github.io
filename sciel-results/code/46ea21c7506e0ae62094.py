import numpy as np
import time

def solve(a, b):
    n = a.shape[0]
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    
    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
    
    # --- Initialization: Spectral + Rank Matching ---
    # Use spectral embedding of distance matrix to get a 1D ordering
    # Then assign facilities to locations based on flow strength and this ordering
    
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    # Spectral initialization for better structure
    try:
        b_centered = b - b.mean()
        eigvals, eigvecs = np.linalg.eigh(b_centered)
        idx = np.argsort(-np.abs(eigvals))
        # Use the first few eigenvectors to create a 2D embedding
        spec_coords_2d = eigvecs[:, idx[:2]]
        
        # Project to 1D using the first principal component
        spec_1d = spec_coords_2d[:, 0]
        loc_order_spectral = np.argsort(spec_1d, kind='stable')
    except:
        loc_order_spectral = np.argsort(b_sums, kind='stable')
    
    # Sort facilities by flow strength (descending)
    heavy_facilities = np.argsort(-a_sums, kind='stable')
    
    # Combine: heavy facilities go to central locations (low b_sums)
    # But use spectral order for the rest
    central_locations = np.argsort(b_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_facilities] = central_locations
    
    current_cost = compute_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # --- Fast Local Search with O(n) Incremental Updates ---
    # Precompute dot products for fast delta calculation
    def recompute_precomp(p):
        b_col_p = b[:, p]  # (n, n): [l, k] = b[l, p[k]]
        b_row_p = b[p, :]  # (n, n): [k, l] = b[p[k], l]
        dot_row = a @ b_col_p.T  # dot_row[i, l] = sum_k a[i,k] * b[l, p[k]]
        dot_col = a.T @ b_row_p  # dot_col[i, l] = sum_k a[k,i] * b[p[k], l]
        return dot_row, dot_col
    
    dot_row, dot_col = recompute_precomp(p)
    b_diag = np.diag(b)
    
    # Tabu list for avoiding cycles
    tabu = {}
    tabu_tenure = 15
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        tabu_cleared = False
        
        # First-fit descent with random restarts
        for _ in range(5000):
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
            
            # Compute delta using precomputed values (O(1) lookups)
            buu = b_diag[u]
            bvv = b_diag[v]
            buv = b[u, v]
            bvu = b[v, u]
            
            # Direct interaction terms
            delta = 0.0
            delta += a[i, i] * (bvv - buu)
            delta += a[i, j] * (bvu - buv)
            delta += a[j, i] * (buv - bvu)
            delta += a[j, j] * (buu - bvv)
            
            # Sum over k != i, j terms using precomputed dot products
            # Term for row i: sum_{k != i,j} a[i,k] * (b[v, p[k]] - b[u, p[k]])
            term_row_i = (dot_row[i, v] - dot_row[i, u]) - a[i, i]*(bvv - buu) - a[i, j]*(bvu - buv)
            delta += term_row_i
            
            # Term for row j: sum_{k != i,j} a[j,k] * (b[u, p[k]] - b[v, p[k]])
            term_row_j = (dot_row[j, u] - dot_row[j, v]) - a[j, i]*(buv - bvu) - a[j, j]*(buu - bvv)
            delta += term_row_j
            
            # Term for col i: sum_{k != i,j} a[k,i] * (b[p[k], v] - b[p[k], u])
            term_col_i = (dot_col[i, v] - dot_col[i, u]) - a[i, i]*(bvv - buu) - a[j, i]*(bvu - buv)
            delta += term_col_i
            
            # Term for col j: sum_{k != i,j} a[k,j] * (b[p[k], u] - b[p[k], v])
            term_col_j = (dot_col[j, u] - dot_col[j, v]) - a[i, j]*(buv - bvu) - a[j, j]*(buu - bvv)
            delta += term_col_j
            
            # Check tabu
            key = (min(i, j), max(i, j))
            is_tabu = key in tabu and tabu[key] > 0
            
            if delta < -1e-9:
                # Aspiration: accept if it improves best
                if is_tabu and current_cost + delta >= best_cost:
                    continue
                
                # Apply swap
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                tabu[key] = tabu_tenure
                improved = True
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                
                # Incremental update of dot_row and dot_col in O(n)
                # When p[i] and p[j] swap (u <-> v):
                # For k != i, j:
                #   dot_row[k, l] += (a[k,i] - a[k,j]) * (b[l,v] - b[l,u])
                #   dot_col[k, l] += (a[k,i] - a[k,j]) * (b[v,l] - b[u,l])
                # For k = i:
                #   dot_row[i, l] += (a[i,i] - a[i,j]) * (b[l,v] - b[l,u])
                #   dot_col[i, l] += (a[i,i] - a[j,i]) * (b[v,l] - b[u,l])
                # For k = j:
                #   dot_row[j, l] += (a[j,i] - a[j,j]) * (b[l,v] - b[l,u])
                #   dot_col[j, l] += (a[j,i] - a[j,j]) * (b[v,l] - b[u,l])
                
                diff_a_row = a[:, i] - a[:, j]  # (n,)
                diff_a_col = a[i, :] - a[j, :]  # (n,)
                
                diff_b_row = b[:, v] - b[:, u]  # (n,): b[l,v] - b[l,u]
                diff_b_col = b[v, :] - b[u, :]  # (n,): b[v,l] - b[u,l]
                
                # Update dot_row: dot_row[k, l] += diff_a_row[k] * diff_b_row[l]
                dot_row += np.outer(diff_a_row, diff_b_row)
                
                # Update dot_col: dot_col[k, l] += diff_a_col[k] * diff_b_col[l]
                dot_col += np.outer(diff_a_col, diff_b_col)
                
                # Correct for k=i and k=j (they were updated with wrong a values)
                # For k=i:
                dot_row[i, :] -= (a[i, i] - a[i, j]) * diff_b_row
                dot_row[i, :] += (a[i, i] - a[i, j]) * diff_b_row  # Actually this is correct
                # Wait, let me re-derive:
                # For k=i, the change should be: a[i,i]*(b[l,v]-b[l,u]) - a[i,j]*(b[l,v]-b[l,u]) = (a[i,i]-a[i,j])*diff_b_row[l]
                # But we added (a[i,i]-a[i,j])*diff_b_row[l] via the outer product, so it's correct.
                # For k=j:
                # The change should be: a[j,i]*(b[l,v]-b[l,u]) - a[j,j]*(b[l,v]-b[l,u]) = (a[j,i]-a[j,j])*diff_b_row[l]
                # We added (a[j,i]-a[j,j])*diff_b_row[l], so it's correct.
                
                # Actually, the outer product update is correct for all k.
                # Let me verify:
                # dot_row[k, l] = sum_m a[k,m] * b[l, p[m]]
                # After swap p[i]=v, p[j]=u:
                # New dot_row[k, l] = sum_{m != i,j} a[k,m] * b[l, p[m]] + a[k,i]*b[l,v] + a[k,j]*b[l,u]
                # Old dot_row[k, l] = sum_{m != i,j} a[k,m] * b[l, p[m]] + a[k,i]*b[l,u] + a[k,j]*b[l,v]
                # Change = a[k,i]*(b[l,v]-b[l,u]) + a[k,j]*(b[l,u]-b[l,v]) = (a[k,i]-a[k,j])*(b[l,v]-b[l,u])
                # = diff_a_row[k] * diff_b_row[l]
                # So the outer product update is exactly correct.
                
                # Similarly for dot_col:
                # dot_col[k, l] = sum_m a[m,k] * b[p[m], l]
                # Change = (a[i,k]-a[j,k])*(b[v,l]-b[u,l]) = diff_a_col[k] * diff_b_col[l]
                # So the outer product update is correct.
                
                # Decrement tabu counters
                for tkey in list(tabu.keys()):
                    tabu[tkey] -= 1
                    if tabu[tkey] <= 0:
                        del tabu[tkey]
            
            # If we didn't improve in this batch, perturb
            if not improved:
                # Perform 10-20 random swaps to escape local minimum
                for _ in range(15):
                    ri = np.random.randint(n)
                    rj = np.random.randint(n)
                    if ri != rj:
                        p[ri], p[rj] = p[rj], p[ri]
                
                current_cost = compute_cost(p)
                dot_row, dot_col = recompute_precomp(p)
                tabu.clear()
                improved = True  # Force another iteration
                break
    
    return best_p.tolist()
