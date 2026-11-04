"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit

    a = a.astype(np.int64)
    b = b.astype(np.int64)

    # Check if b is symmetric (it should be for QAPLIB)
    # We'll assume it is for the delta formula

    # Initialization: rank matching
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

    # Precompute for fast delta:
    # We need b[u, p] and b[p, u] for various u.
    # b[p] is the permutation of b's rows: b_perm[k, :] = b[p[k], :]
    # b[:, p] is the permutation of b's columns: b_col[:, k] = b[:, p[k]]
    #
    # For delta(i, j) with u=p[i], v=p[j]:
    # Using the formula derived:
    # delta = sum_k [a[i,k]*(b[v,p'[k]]-b[u,p[k]]) + a[k,i]*(b[p'[k],v]-b[p[k],u])
    #         + a[j,k]*(b[u,p'[k]]-b[v,p[k]]) + a[k,j]*(b[p'[k],u]-b[p[k],v])]
    # where p' is p with i,j swapped.
    #
    # For k not in {i,j}: p'[k]=p[k], so:
    #   term = (a[i,k]-a[j,k])*(b[v,p[k]]-b[u,p[k]]) + (a[k,i]-a[k,j])*(b[p[k],v]-b[p[k],u])
    # For k=i: p'[i]=v
    #   term = a[i,i]*(b[v,v]-b[u,u]) + a[i,i]*(b[v,v]-b[u,u]) + a[j,i]*(b[u,v]-b[v,u]) + a[j,i]*(b[u,v]-b[v,u])
    #         = 2*a[i,i]*(b[v,v]-b[u,u]) + 2*a[j,i]*(b[u,v]-b[v,u])
    # For k=j: p'[j]=u
    #   term = a[i,j]*(b[v,u]-b[u,v]) + a[i,j]*(b[v,u]-b[u,v]) + a[j,j]*(b[u,u]-b[v,v]) + a[j,j]*(b[u,u]-b[v,v])
    #         = 2*a[i,j]*(b[v,u]-b[u,v]) + 2*a[j,j]*(b[u,u]-b[v,v])
    #
    # For symmetric b: b[u,v]=b[v,u]
    # k=i: 2*a[i,i]*(b[v,v]-b[u,u])
    # k=j: 2*a[j,j]*(b[u,u]-b[v,v])
    #
    # So: delta = sum_{k!=i,j} [(a[i,k]-a[j,k])*(b[v,p[k]]-b[u,p[k]]) + (a[k,i]-a[k,j])*(b[p[k],v]-b[p[k],u])]
    #         + 2*a[i,i]*(b[v,v]-b[u,u]) + 2*a[j,j]*(b[u,u]-b[v,v])
    #
    # Let me define:
    #   a_row_diff[k] = a[i,k] - a[j,k]
    #   a_col_diff[k] = a[k,i] - a[k,j]
    #   b_row_diff[k] = b[v,p[k]] - b[u,p[k]]
    #   b_col_diff[k] = b[p[k],v] - b[p[k],u]
    #
    # Then: delta = sum_k a_row_diff[k]*b_row_diff[k] + sum_k a_col_diff[k]*b_col_diff[k]
    #           - a_row_diff[i]*b_row_diff[i] - a_row_diff[j]*b_row_diff[j]
    #           - a_col_diff[i]*b_col_diff[i] - a_col_diff[j]*b_col_diff[j]
    #           + 2*a[i,i]*(b[v,v]-b[u,u]) + 2*a[j,j]*(b[u,u]-b[v,v])
    #
    # But a_row_diff[i] = a[i,i]-a[j,i], b_row_diff[i] = b[v,u]-b[u,u]
    # a_row_diff[j] = a[i,j]-a[j,j], b_row_diff[j] = b[v,v]-b[u,v]
    # a_col_diff[i] = a[i,i]-a[j,i], b_col_diff[i] = b[u,v]-b[u,u]
    # a_col_diff[j] = a[i,j]-a[j,j], b_col_diff[j] = b[v,v]-b[u,v]
    #
    # For symmetric b: b_row_diff[i]=b_col_diff[i]=b[v,u]-b[u,u], b_row_diff[j]=b_col_diff[j]=b[v,v]-b[u,v]
    #
    # This is still complex. Let me just use the O(n) vectorized computation without the p_prime copy:
    #
    # b[v, p] is a vector of length n: b[v, p[k]] for k=0..n-1
    # b[u, p] is a vector of length n: b[u, p[k]] for k=0..n-1
    # b[p, v] is a vector of length n: b[p[k], v] for k=0..n-1
    # b[p, u] is a vector of length n: b[p[k], u] for k=0..n-1
    #
    # For the "k not i,j" part, I can compute the full sum and subtract the i and j terms.
    #
    # Let me just do it cleanly:

    # Precompute b as a view for fast row/column access
    # b is already a 2D array, so b[v, p] is fast fancy indexing.

    def fast_delta(i, j, p, a, b):
        """Compute delta for swapping i and j. O(n) with NumPy."""
        u = p[i]
        v = p[j]
        
        # Get the needed vectors
        b_v_p = b[v, p]   # b[v, p[k]] for all k
        b_u_p = b[u, p]   # b[u, p[k]] for all k
        b_p_v = b[p, v]   # b[p[k], v] for all k
        b_p_u = b[p, u]   # b[p[k], u] for all k
        
        diff_bp = b_v_p - b_u_p       # b[v,p[k]] - b[u,p[k]]
        diff_bp_col = b_p_v - b_p_u   # b[p[k],v] - b[p[k],u]
        
        a_i_row = a[i, :]
        a_j_row = a[j, :]
        a_i_col = a[:, i]
        a_j_col = a[:, j]
        
        diff_a_row = a_i_row - a_j_row
        diff_a_col = a_i_col - a_j_col
        
        # Full sum over all k
        s1 = np.dot(diff_a_row, diff_bp)
        s2 = np.dot(diff_a_col, diff_bp_col)
        
        # Subtract k=i and k=j contributions from the "k not i,j" sum
        # k=i: diff_a_row[i]*diff_bp[i] + diff_a_col[i]*diff_bp_col[i]
        # k=j: diff_a_row[j]*diff_bp[j] + diff_a_col[j]*diff_bp_col[j]
        corr_i = diff_a_row[i] * diff_bp[i] + diff_a_col[i] * diff_bp_col[i]
        corr_j = diff_a_row[j] * diff_bp[j] + diff_a_col[j] * diff_bp_col[j]
        
        # Add back the correct k=i and k=j contributions
        # k=i: 2*a[i,i]*(b[v,v]-b[u,u])
        # k=j: 2*a[j,j]*(b[u,u]-b[v,v])
        # For symmetric b, these are the diagonal corrections
        diag_i = 2 * a[i, i] * (b[v, v] - b[u, u])
        diag_j = 2 * a[j, j] * (b[u, u] - b[v, v])
        
        delta = s1 + s2 - corr_i - corr_j + diag_i + diag_j
        return int(delta)

    # SA with adaptive cooling
    T = 5000.0
    alpha = 0.9997
    T_min = 0.001
    step = 0
    
    while time.time() < deadline:
        step += 1
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        
        delta = fast_delta(i, j, p, a, b)
        
        if delta < 0 or (T > T_min and random.random() < np.exp(-delta / T)):
            p[i], p[j] = p[j], p[i]
            cost += delta
            
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
        
        T *= alpha
        if T < T_min:
            T = T_min
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
