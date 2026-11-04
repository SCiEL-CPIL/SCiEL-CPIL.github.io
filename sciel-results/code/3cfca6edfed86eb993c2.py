import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # Initialization: Rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Compute initial B_perm and M
    # B_perm[i, j] = b[p[i], p[j]]
    b_perm = b[p[:, None], p[None, :]]
    # M[i, j] = sum_k a[i, k] * b_perm[j, k]
    # M = a @ b_perm
    m = a @ b_perm
    d = np.diag(m).copy()
    
    best_cost = np.sum(a * b_perm)
    
    # Local Search
    while time.time() - start_time < time_limit:
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Find best swap
        # Score_j for fixed i: M[i, j] - D_j + M[j, i]
        # We want to minimize this.
        
        for i in range(n):
            # Compute scores for all j
            # scores[j] = M[i, j] - D[j] + M[j, i]
            scores = m[i, :] - d + m[:, i]
            scores[i] = np.inf  # Exclude self
            
            # Find best j
            j = np.argmin(scores)
            delta = scores[j]
            
            if delta < best_delta:
                best_delta = delta
                best_i = i
                best_j = j
        
        if best_i == -1:
            break  # No improvement found
        
        i, j = best_i, best_j
        u, v = p[i], p[j]
        
        # Perform swap
        p[i], p[j] = p[j], p[i]
        
        # Update B_perm
        # Rows i and j are swapped in B_perm
        # Also columns i and j are swapped?
        # B_perm[x, y] = b[p[x], p[y]]
        # If we swap p[i] and p[j]:
        # B_perm[i, :] becomes old B_perm[j, :]
        # B_perm[j, :] becomes old B_perm[i, :]
        # B_perm[:, i] becomes old B_perm[:, j]
        # B_perm[:, j] becomes old B_perm[:, i]
        
        b_perm[i, :], b_perm[j, :] = b_perm[j, :], b_perm[i, :].copy()
        b_perm[:, i], b_perm[:, j] = b_perm[:, j], b_perm[:, i].copy()
        
        # Update M
        # M[i, k] += (a[i, j] - a[i, i]) * (B_perm[j, k] - B_perm[i, k])  -- Using OLD B_perm values for diff?
        # The formula derived was:
        # M'[i, k] = M[i, k] + (a[i, j] - a[i, i]) * (B_old[j, k] - B_old[i, k])
        # M'[j, k] = M[j, k] + (a[j, i] - a[j, j]) * (B_old[i, k] - B_old[j, k])
        # M'[m, k] = M[m, k] + (a[m, j] - a[m, i]) * (B_old[i, k] - B_old[j, k])
        
        # We need to use OLD B_perm values for the differences.
        # Since we just swapped the rows in b_perm, b_perm[i, :] is now OLD b_perm[j, :]
        # and b_perm[j, :] is now OLD b_perm[i, :].
        # So diff_ij = b_perm[j, :] - b_perm[i, :] is OLD[i] - OLD[j].
        # We need OLD[j] - OLD[i] for the first term.
        diff_ji = b_perm[i, :] - b_perm[j, :]  # This is OLD[i] - OLD[j]
        # So (a[i, j] - a[i, i]) * (OLD[j] - OLD[i]) = (a[i, j] - a[i, i]) * (-diff_ji)
        
        # Let's re-verify the update formula signs.
        # M'[i, k] = M[i, k] + (a[i, j] - a[i, i]) * (B_old[j, k] - B_old[i, k])
        # B_old[j, k] is current b_perm[j, k] (which was i) ? No.
        # Before swap: b_perm[i, :] was row i, b_perm[j, :] was row j.
        # After swap: b_perm[i, :] is row j, b_perm[j, :] is row i.
        # So current b_perm[i, :] == B_old[j, :]
        # Current b_perm[j, :] == B_old[i, :]
        # So B_old[j, k] - B_old[i, k] = current_b_perm[i, k] - current_b_perm[j, k].
        
        diff_ji_vec = b_perm[i, :] - b_perm[j, :]
        
        m[i, :] += (a[i, j] - a[i, i]) * diff_ji_vec
        m[j, :] += (a[j, i] - a[j, j]) * (-diff_ji_vec)  # (a[j,i]-a[j,j]) * (B_old[i]-B_old[j])
        
        # Update other rows
        # M'[m, k] = M[m, k] + (a[m, j] - a[m, i]) * (B_old[i, k] - B_old[j, k])
        # B_old[i, k] - B_old[j, k] = current_b_perm[j, k] - current_b_perm[i, k] = -diff_ji_vec
        m[:, :] += np.outer(a[:, j] - a[:, i], -diff_ji_vec)
        
        # Update diagonal d
        d = np.diag(m).copy()
        
        # Update best_cost
        best_cost += best_delta
    
    return p.tolist()
