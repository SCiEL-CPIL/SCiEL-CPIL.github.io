<<<<<<< SEARCH
def _compute_swap_delta_fast(a, b, p, i, j):
    """
    Fast swap delta for positions i and j in permutation p.
    Uses the fact that we can express delta as:
    delta = sum_{k} (a[k,i] - a[k,j])*(b[p[k],p[j]] - b[p[k],p[i]])
          + sum_{k} (a[i,k] - a[j,k])*(b[p[j],p[k]] - b[p[i],p[k]])
          + (a[i,j]-a[j,i])*(b[p[j],p[i]] - b[p[i],p[j]])
          - corrections for k=i and k=j double counting
    """
    pi = p[i]
    pj = p[j]
    
    # T1: sum_k (a[k,i] - a[k,j]) * (b[p[k], pj] - b[p[k], pi])
    # = a[:,i] @ (b[p,pj] - b[p,pi]) 
    #   minus corrections for k=i and k=j
    diff_b_col = b[p, pj] - b[p, pi]
    T1 = np.dot(a[:, i], diff_b_col)
    T1 -= (a[i,i] - a[i,j]) * diff_b_col[i]
    T1 -= (a[j,i] - a[j,j]) * diff_b_col[j]
    
    # T2: sum_k (a[i,k] - a[j,k]) * (b[pj, p[k]] - b[pi, p[k]])
    diff_b_row = b[pj, p] - b[pi, p]
    T2 = np.dot(a[i, :], diff_b_row)
    T2 -= (a[i,i] - a[j,i]) * diff_b_row[i]
    T2 -= (a[i,j] - a[j,j]) * diff_b_row[j]
    
    # T3
    T3 = (a[i,j] - a[j,i]) * (b[pj, pi] - b[pi, pj])
    
    return T1 + T2 + T3
=======
def _compute_swap_delta_fast(a, b, p, i, j):
    """
    Fast swap delta for positions i and j in permutation p.
    Uses the fact that we can express delta as:
    delta = sum_{k} (a[k,i] - a[k,j])*(b[p[k],p[j]] - b[p[k],p[i]])
          + sum_{k} (a[i,k] - a[j,k])*(b[p[j],p[k]] - b[p[i],p[k]])
          + (a[i,j]-a[j,i])*(b[p[j],p[i]] - b[p[i],p[j]])
          - corrections for k=i and k=j double counting
    """
    pi = p[i]
    pj = p[j]
    
    # T1: sum_k (a[k,i] - a[k,j]) * (b[p[k], pj] - b[p[k], pi])
    # = a[:,i] @ (b[p,pj] - b[p,pi]) 
    #   minus corrections for k=i and k=j
    diff_b_col = b[p, pj] - b[p, pi]
    T1 = np.dot(a[:, i], diff_b_col)
    T1 -= (a[i,i] - a[i,j]) * diff_b_col[i]
    T1 -= (a[j,i] - a[j,j]) * diff_b_col[j]
    
    # T2: sum_k (a[i,k] - a[j,k]) * (b[pj, p[k]] - b[pi, p[k]])
    diff_b_row = b[pj, p] - b[pi, p]
    T2 = np.dot(a[i, :], diff_b_row)
    T2 -= (a[i,i] - a[j,i]) * diff_b_row[i]
    T2 -= (a[i,j] - a[j,j]) * diff_b_row[j]
    
    # T3
    T3 = (a[i,j] - a[j,i]) * (b[pj, pi] - b[pi, pj])
    
    return T1 + T2 + T3

def _compute_all_swap_deltas_vectorized(a, b, p, i):
    """
    Compute swap deltas for position i with all other positions j > i.
    Returns array of deltas for j in range(i+1, n).
    """
    n = len(p)
    pi = p[i]
    
    # For each j, delta(i,j) = T1(i,j) + T2(i,j) + T3(i,j)
    # T1(i,j) = a[:,i] @ (b[p,p[j]] - b[p,pi]) - corrections
    # T2(i,j) = a[i,:] @ (b[p[j],p] - b[pi,p]) - corrections  
    # T3(i,j) = (a[i,j]-a[j,i]) * (b[p[j],pi] - b[pi,p[j]])
    
    # Precompute common terms
    a_col_i = a[:, i]  # (n,)
    a_row_i = a[i, :]  # (n,)
    
    # b[p, pi] is column pi of b indexed by p: shape (n,)
    b_p_pi = b[p, pi]  # (n,)
    b_pi_p = b[pi, p]  # (n,)
    
    # For all j, b[p, p[j]] is b[p, p[i+1:]] if we vectorize
    # Let's compute for all j at once
    
    # b[p, p] is the full matrix indexed by p: shape (n, n)
    # b[p, p[j]] for all j is columns of b[p,p]
    b_pp = b[np.ix_(p, p)]  # (n, n): b_pp[k, j] = b[p[k], p[j]]
    
    # T1 for all j: a_col_i @ (b_pp[:, j] - b_p_pi) - corrections
    # = a_col_i @ b_pp[:, j] - a_col_i @ b_p_pi - corrections
    # a_col_i @ b_pp is (n,): result[j] = sum_k a[k,i] * b[p[k], p[j]]
    T1_all = a_col_i @ b_pp  # (n,)
    T1_const = a_col_i @ b_p_pi  # scalar
    T1_all = T1_all - T1_const
    
    # Corrections for k=i and k=j
    # For k=i: subtract (a[i,i] - a[i,j]) * (b[p[i], p[j]] - b[p[i], pi])
    # For k=j: subtract (a[j,i] - a[j,j]) * (b[p[j], p[j]] - b[p[j], pi])
    # This is complex to vectorize, so we'll handle it in the main loop
    # Or we can precompute:
    # b_pi_pj = b[pi, p]  # already have as b_pi_p
    # b_pj_pj = np.diag(b_pp)  # b[p[j], p[j]]
    # b_pj_pi = b_pp[:, i]  # b[p[j], pi]
    
    # Let's just compute the main part and add corrections in the loop
    # Actually, for efficiency, let's compute the full delta in a semi-vectorized way
    
    # T2 for all j: a_row_i @ (b[p[j], p] - b_pi_p) - corrections
    # b[p[j], p] for all j is rows of b_pp
    T2_all = a_row_i @ b_pp.T  # (n,): result[j] = sum_k a[i,k] * b[p[j], p[k]]
    T2_const = a_row_i @ b_pi_p  # scalar
    T2_all = T2_all - T2_const
    
    # T3 for all j: (a[i,j] - a[j,i]) * (b[p[j], pi] - b[pi, p[j]])
    a_i_j = a[i, :]  # a[i,j] for all j
    a_j_i = a[:, i]  # a[j,i] for all j
    b_pj_pi = b_pp[:, i]  # b[p[j], pi] for all j
    b_pi_pj = b_pi_p  # b[pi, p[j]] for all j
    
    T3_all = (a_i_j - a_j_i) * (b_pj_pi - b_pi_pj)
    
    # Combine main parts
    deltas_all = T1_all + T2_all + T3_all
    
    # Now add corrections for k=i and k=j
    # For k=i correction in T1: -(a[i,i] - a[i,j]) * (b[p[i], p[j]] - b[p[i], pi])
    # b[p[i], p[j]] = b_pp[i, j], b[p[i], pi] = b_pp[i, i]
    corr_T1_i = (a[i,i] - a_i_j) * (b_pp[i, :] - b_pp[i, i])
    
    # For k=j correction in T1: -(a[j,i] - a[j,j]) * (b[p[j], p[j]] - b[p[j], pi])
    # b[p[j], p[j]] = np.diag(b_pp), b[p[j], pi] = b_pp[:, i]
    diag_bpp = np.diag(b_pp)
    corr_T1_j = (a_j_i - np.diag(a)) * (diag_bpp - b_pp[:, i])
    
    # For k=i correction in T2: -(a[i,i] - a[j,i]) * (b[p[j], p[i]] - b[pi, p[i]])
    # b[p[j], p[i]] = b_pp[j, i], b[pi, p[i]] = b_pi_p[i]
    corr_T2_i = (a[i,i] - a_j_i) * (b_pp[:, i] - b_pi_p[i])
    
    # For k=j correction in T2: -(a[i,j] - a[j,j]) * (b[p[j], p[j]] - b[pi, p[j]])
    # b[p[j], p[j]] = diag_bpp, b[pi, p[j]] = b_pi_p
    corr_T2_j = (a_i_j - np.diag(a)) * (diag_bpp - b_pi_p)
    
    deltas_all = deltas_all - corr_T1_i - corr_T1_j - corr_T2_i - corr_T2_j
    
    # Only return deltas for j > i
    return deltas_all[i+1:]
>>>>>>> REPLACE

