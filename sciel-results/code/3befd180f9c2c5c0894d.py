"""QAP solver for tai256c with fast vectorized 2-opt and diverse initializations."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))

def _rank_matching_init(a, b):
    """Rank-matching: heavy flow rows to central location rows."""
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(len(a), dtype=np.int64)
    p[heavy] = central
    return p

def _spectral_init(a, b, n):
    """Spectral embedding initial permutation."""
    try:
        a_sym = (a + a.T) / 2.0
        vals, vecs = np.linalg.eigh(a_sym)
        k = min(8, n)
        embed = vecs[:, -k:]
        scores = embed[:, -1]
        p = np.argsort(scores, kind="stable")
        return p.astype(np.int64)
    except:
        return _rank_matching_init(a, b)

def _random_init(n, rng):
    """Random permutation."""
    p = np.arange(n, dtype=np.int64)
    rng.shuffle(p)
    return p

def _local_search_2opt_fast(a, b, p, deadline):
    """
    Fast 2-opt local search using precomputed interaction vectors.
    
    Delta for swapping positions i and j (facilities f_i=p[i], f_j=p[j]):
    delta = sum_{k} (a[i,k] - a[j,k]) * (b[f_j, p[k]] - b[f_i, p[k]])
          + sum_{k} (a[k,i] - a[k,j]) * (b[p[k], f_j] - b[p[k], f_i])
          - (a[i,i] - a[j,i]) * (b[f_j, f_i] - b[f_i, f_j])  # Correction for k=i term if needed?
          - (a[i,j] - a[j,j]) * (b[f_j, f_j] - b[f_i, f_i])  # Correction for k=j term?
          
    Actually, let's derive it carefully.
    Cost = sum_{u,v} a[u,v] b[p[u], p[v]]
    
    Let S be the set of positions. We swap p[i] and p[j].
    Let f_i = p[i], f_j = p[j].
    
    Terms involving i or j change.
    1. a[i,i] b[f_i, f_i] -> a[i,i] b[f_j, f_j]
    2. a[j,j] b[f_j, f_j] -> a[j,j] b[f_i, f_i]
    3. a[i,j] b[f_i, f_j] -> a[i,j] b[f_j, f_i]
    4. a[j,i] b[f_j, f_i] -> a[j,i] b[f_i, f_j]
    
    Terms involving i and k (k not i,j):
    a[i,k] b[f_i, p[k]] -> a[i,k] b[f_j, p[k]]
    a[k,i] b[p[k], f_i] -> a[k,i] b[p[k], f_j]
    
    Terms involving j and k (k not i,j):
    a[j,k] b[f_j, p[k]] -> a[j,k] b[f_i, p[k]]
    a[k,j] b[p[k], f_j] -> a[k,j] b[p[k], f_i]
    
    Delta = [a[i,i](b[f_j,f_j]-b[f_i,f_i]) + a[j,j](b[f_i,f_i]-b[f_j,f_j])
           + a[i,j](b[f_j,f_i]-b[f_i,f_j]) + a[j,i](b[f_i,f_j]-b[f_j,f_i])
           + sum_{k!=i,j} a[i,k](b[f_j,p[k]]-b[f_i,p[k]]) + a[k,i](b[p[k],f_j]-b[p[k],f_i])
           + a[j,k](b[f_i,p[k]]-b[f_j,p[k]]) + a[k,j](b[p[k],f_i]-b[p[k],f_j])]
           
    We can precompute for each facility f, the vector V_f = b[f, :] and W_f = b[:, f].
    Actually, b is fixed. p changes.
    
    Let's define for current p:
    For each position k, we have facility f_k = p[k].
    
    We want to compute delta(i, j) quickly.
    Let R_i = sum_{k} a[i,k] b[f_i, p[k]]
    Let C_i = sum_{k} a[k,i] b[p[k], f_i]
    
    This is not quite right because b[f_i, p[k]] depends on f_i.
    
    Let's use the matrix formulation.
    Let B_p be the matrix B indexed by p. i.e., (B_p)[u,v] = b[p[u], p[v]].
    Cost = sum a[u,v] (B_p)[u,v].
    
    Swapping i and j in p means swapping rows i and j of B_p? No, p is the assignment.
    If we swap p[i] and p[j], the new matrix B'_p has:
    B'_p[u,v] = B_p[u,v] for u,v not in {i,j}
    B'_p[i,i] = B_p[j,j]
    B'_p[j,j] = B_p[i,i]
    B'_p[i,j] = B_p[j,i]
    B'_p[j,i] = B_p[i,j]
    B'_p[i,k] = B_p[j,k] for k not in {i,j}
    B'_p[k,i] = B_p[k,j] for k not in {i,j}
    B'_p[j,k] = B_p[i,k] for k not in {i,j}
    B'_p[k,j] = B_p[k,i] for k not in {i,j}
    
    Delta = sum_{u,v} a[u,v] (B'_p[u,v] - B_p[u,v])
    = a[i,i](B_p[j,j]-B_p[i,i]) + a[j,j](B_p[i,i]-B_p[j,j])
    + a[i,j](B_p[j,i]-B_p[i,j]) + a[j,i](B_p[i,j]-B_p[j,i])
    + sum_{k!=i,j} [ a[i,k](B_p[j,k]-B_p[i,k]) + a[k,i](B_p[k,j]-B_p[k,i])
                   + a[j,k](B_p[i,k]-B_p[j,k]) + a[k,j](B_p[k,i]-B_p[k,j]) ]
    
    Note B_p[u,v] = b[p[u], p[v]].
    
    For fixed i, we can compute the delta for all j > i in O(n^2) total for the whole pass if we are clever, or O(n) per j if we precompute some vectors.
    
    Let's precompute for each position i:
    RowVec_i = b[p[i], :]  (length n)
    ColVec_i = b[:, p[i]]  (length n)
    
    Then B_p[i,k] = RowVec_i[k]
    B_p[k,i] = ColVec_i[k]
    
    Delta(i,j) = a[i,i](RowVec_j[j]-RowVec_i[i]) + a[j,j](RowVec_i[i]-RowVec_j[j])
               + a[i,j](ColVec_j[i]-RowVec_i[j]) + a[j,i](RowVec_i[j]-ColVec_j[i])
               + sum_{k!=i,j} [ a[i,k](RowVec_j[k]-RowVec_i[k]) + a[k,i](ColVec_j[k]-ColVec_i[k])
                              + a[j,k](RowVec_i[k]-RowVec_j[k]) + a[k,j](ColVec_i[k]-ColVec_j[k]) ]
    
    Let's group the sum over k:
    Sum_k [ (a[i,k] - a[j,k]) (RowVec_j[k] - RowVec_i[k])
          + (a[k,i] - a[k,j]) (ColVec_j[k] - ColVec_i[k]) ]
    
    This sum includes k=i and k=j. We must subtract the k=i and k=j terms from this sum and add the explicit terms calculated above.
    
    Let S1 = sum_k (a[i,k] - a[j,k]) (RowVec_j[k] - RowVec_i[k])
    Let S2 = sum_k (a[k,i] - a[k,j]) (ColVec_j[k] - ColVec_i[k])
    
    Total Delta = S1 + S2 
                 - [Term for k=i in S1] - [Term for k=j in S1]
                 - [Term for k=i in S2] - [Term for k=j in S2]
                 + [Explicit terms for i,i; j,j; i,j; j,i]
    
    Term for k=i in S1: (a[i,i] - a[j,i]) (RowVec_j[i] - RowVec_i[i])
    Term for k=j in S1: (a[i,j] - a[j,j]) (RowVec_j[j] - RowVec_i[j])
    
    Term for k=i in S2: (a[i,i] - a[i,j]) (ColVec_j[i] - ColVec_i[i])
    Term for k=j in S2: (a[j,i] - a[j,j]) (ColVec_j[j] - ColVec_i[j])
    
    Explicit terms:
    E1 = a[i,i](RowVec_j[j]-RowVec_i[i])
    E2 = a[j,j](RowVec_i[i]-RowVec_j[j])
    E3 = a[i,j](ColVec_j[i]-RowVec_i[j])
    E4 = a[j,i](RowVec_i[j]-ColVec_j[i])
    
    This is getting complicated to keep track of signs and indices.
    A simpler, robust O(n^2) per full pass (which is O(n^3) total but with small constant) is often faster in Python/NumPy for n=256 than complex O(n) per swap with high overhead if not vectorized perfectly.
    
    However, we can vectorize the calculation of Delta for all j given i.
    
    For a fixed i:
    DiffRow = RowVec_i - RowVec  # Shape (n, n), DiffRow[j, k] = RowVec_i[k] - RowVec_j[k]
    DiffCol = ColVec_i - ColVec  # Shape (n, n), DiffCol[j, k] = ColVec_i[k] - ColVec_j[k]
    
    Wait, RowVec is an (n, n) matrix where RowVec[i] = b[p[i], :].
    
    Let's define:
    B_rows = b[p, :]  # B_rows[i] = b[p[i], :]
    B_cols = b[:, p]  # B_cols[i] = b[:, p[i]]
    
    For fixed i, we want Delta(i, j) for all j.
    
    S1(j) = sum_k (a[i,k] - a[j,k]) (B_rows[j,k] - B_rows[i,k])
           = sum_k (a[i,k] - a[j,k]) B_rows[j,k] - sum_k (a[i,k] - a[j,k]) B_rows[i,k]
    
    Let A_i = a[i, :]
    Let A_j = a[j, :]
    
    This is still O(n) per j.
    
    Let's try a different vectorization.
    Delta(i, j) = sum_{k} (a[i,k] - a[j,k]) (B_rows[j,k] - B_rows[i,k]) 
                + sum_{k} (a[k,i] - a[k,j]) (B_cols[j,k] - B_cols[i,k])
                - Corrections
    
    Let's just implement the O(n^2) full matrix delta calculation for each i, but do it efficiently.
    Actually, for n=256, O(n^3) is 16 million operations. In NumPy, this is very fast (< 0.1s).
    The bottleneck in the previous code was likely Python loops or inefficient memory access.
    
    Strategy:
    1. Precompute B_rows and B_cols.
    2. In each iteration, for each i, compute the delta for all j > i using vectorized operations.
    3. Find the best j.
    4. Swap.
    5. Update B_rows and B_cols.
    
    To compute Delta(i, j) for all j > i:
    We need:
    T1[j] = sum_k (a[i,k] - a[j,k]) (B_rows[j,k] - B_rows[i,k])
    T2[j] = sum_k (a[k,i] - a[k,j]) (B_cols[j,k] - B_cols[i,k])
    
    Note: B_rows[j,k] - B_rows[i,k] is a matrix D_row[j,k].
    (a[i,k] - a[j,k]) is a matrix D_a[i,j,k]? No.
    
    Let's expand T1[j]:
    T1[j] = sum_k a[i,k]B_rows[j,k] - sum_k a[i,k]B_rows[i,k] - sum_k a[j,k]B_rows[j,k] + sum_k a[j,k]B_rows[i,k]
    
    Let X[j] = sum_k a[i,k] B_rows[j,k]  -> This is (a[i,:] @ B_rows[j,:]).T? No.
    a[i,:] is (n,). B_rows[j,:] is (n,).
    sum_k a[i,k] B_rows[j,k] is the dot product of a[i,:] and B_rows[j,:].
    Let M1 = a @ B_rows.T  # M1[i,j] = sum_k a[i,k] B_rows[j,k]
    
    Then sum_k a[i,k] B_rows[j,k] = M1[i,j].
    sum_k a[j,k] B_rows[i,k] = M1[j,i].
    sum_k a[i,k] B_rows[i,k] = M1[i,i].
    sum_k a[j,k] B_rows[j,k] = M1[j,j].
    
    So T1[j] = M1[i,j] - M1[i,i] - M1[j,j] + M1[j,i].
    
    Similarly for T2[j]:
    T2[j] = sum_k (a[k,i] - a[k,j]) (B_cols[j,k] - B_cols[i,k])
    = sum_k a[k,i]B_cols[j,k] - sum_k a[k,i]B_cols[i,k] - sum_k a[k,j]B_cols[j,k] + sum_k a[k,j]B_cols[i,k]
    
    Let M2 = a.T @ B_cols  # M2[i,j] = sum_k a[k,i] B_cols[j,k]
    Note B_cols[j,k] = b[p[k], p[j]].
    
    Then:
    sum_k a[k,i]B_cols[j,k] = M2[i,j]
    sum_k a[k,i]B_cols[i,k] = M2[i,i]
    sum_k a[k,j]B_cols[j,k] = M2[j,j]
    sum_k a[k,j]B_cols[i,k] = M2[j,i]
    
    So T2[j] = M2[i,j] - M2[i,i] - M2[j,j] + M2[j,i].
    
    Now, T1 and T2 include k=i and k=j. We need to subtract the k=i and k=j contributions and add the explicit "diagonal/cross" terms.
    
    k=i contribution in T1:
    (a[i,i] - a[j,i]) (B_rows[j,i] - B_rows[i,i])
    k=j contribution in T1:
    (a[i,j] - a[j,j]) (B_rows[j,j] - B_rows[i,j])
    
    k=i contribution in T2:
    (a[i,i] - a[i,j]) (B_cols[j,i] - B_cols[i,i])
    k=j contribution in T2:
    (a[j,i] - a[j,j]) (B_cols[j,j] - B_cols[i,j])
    
    Explicit terms (from the swap of i,j positions in the cost function):
    E = a[i,i](B_rows[j,j]-B_rows[i,i]) + a[j,j](B_rows[i,i]-B_rows[j,j])
      + a[i,j](B_cols[j,i]-B_rows[i,j]) + a[j,i](B_rows[i,j]-B_cols[j,i])
      
    Wait, B_cols[j,i] = b[p[i], p[j]] = B_rows[i,j].
    So a[i,j](B_cols[j,i]-B_rows[i,j]) = 0?
    Let's check:
    Old term: a[i,j] b[p[i], p[j]]
    New term: a[i,j] b[p[j], p[i]]
    Delta = a[i,j] (b[p[j], p[i]] - b[p[i], p[j]])
    b[p[j], p[i]] = B_cols[i, j] (since B_cols[i, j] = b[:, p[i]][p[j]] = b[p[j], p[i]])
    b[p[i], p[j]] = B_rows[i, j] (since B_rows[i, j] = b[p[i], :][p[j]] = b[p[i], p[j]])
    
    So E3 = a[i,j] (B_cols[i, j] - B_rows[i, j])
    E4 = a[j,i] (B_cols[j, i] - B_rows[j, i])
    
    Let's re-verify M1 and M2 definitions.
    M1[i,j] = sum_k a[i,k] B_rows[j,k] = sum_k a[i,k] b[p[j], p[k]]
    M2[i,j] = sum_k a[k,i] B_cols[j,k] = sum_k a[k,i] b[p[k], p[j]]
    
    This looks correct.
    
    Total Delta(i, j) = (M1[i,j] - M1[i,i] - M1[j,j] + M1[j,i]) 
                     + (M2[i,j] - M2[i,i] - M2[j,j] + M2[j,i])
                     - SubT1_i - SubT1_j - SubT2_i - SubT2_j
                     + E1 + E2 + E3 + E4
    
    Where:
    SubT1_i = (a[i,i] - a[j,i]) (B_rows[j,i] - B_rows[i,i])
    SubT1_j = (a[i,j] - a[j,j]) (B_rows[j,j] - B_rows[i,j])
    SubT2_i = (a[i,i] - a[i,j]) (B_cols[j,i] - B_cols[i,i])
    SubT2_j = (a[j,i] - a[j,j]) (B_cols[j,j] - B_cols[i,j])
    
    E1 = a[i,i](B_rows[j,j]-B_rows[i,i])
    E2 = a[j,j](B_rows[i,i]-B_rows[j,j])
    E3 = a[i,j](B_cols[i,j]-B_rows[i,j])
    E4 = a[j,i](B_cols[j,i]-B_rows[j,i])
    
    Note: B_rows[j,i] = b[p[j], p[i]] = B_cols[i,j].
    B_rows[i,j] = b[p[i], p[j]] = B_cols[j,i].
    
    We can precompute M1 and M2 at the start of each "round" (or update them incrementally, but O(n^2) recompute is fine for n=256).
    M1 = a @ B_rows.T  (n x n)
    M2 = a.T @ B_cols  (n x n)
    
    Then for each i, we can compute the vector of deltas for all j > i.
    """
    n = len(p)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    
    # Precompute B_rows and B_cols
    B_rows = b[p, :]  # B_rows[i] = b[p[i], :]
    B_cols = b[:, p]  # B_cols[i] = b[:, p[i]]
    
    # Precompute M1 and M2
    M1 = a @ B_rows.T
    M2 = a.T @ B_cols
    
    converged = False
    max_rounds = 500
    
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        best_delta = -1e9
        best_i, best_j = -1, -1
        
        # Recompute M1, M2 if we want to be safe, or update them.
        # Updating M1, M2 after a swap is complex. Recomputing is O(n^3) which is ~16M ops.
        # In NumPy, 16M ops is very fast (~10-20ms).
        # So we can recompute M1, M2 at the start of each round.
        M1 = a @ B_rows.T
        M2 = a.T @ B_cols
        
        for i in range(n):
            if time.time() > deadline:
                return p, current_cost, False
            
            # Compute deltas for all j > i
            # We need M1[i, j], M1[i, i], M1[j, j], M1[j, i] for all j
            # This is vectorized if we slice.
            
            # For j > i:
            # Deltas_M1 = M1[i, i+1:] - M1[i, i] - M1[i+1:, i+1:] + M1[i+1:, i]
            # Deltas_M2 = M2[i, i+1:] - M2[i, i] - M2[i+1:, i+1:] + M2[i+1:, i]
            
            j_indices = np.arange(i + 1, n)
            if len(j_indices) == 0:
                continue
                
            M1_ii = M1[i, i]
            M1_jj = M1[j_indices, j_indices]
            M1_ji = M1[j_indices, i]
            M1_ij = M1[i, j_indices]
            
            D_M1 = M1_ij - M1_ii - M1_jj + M1_ji
            
            M2_ii = M2[i, i]
            M2_jj = M2[j_indices, j_indices]
            M2_ji = M2[j_indices, i]
            M2_ij = M2[i, j_indices]
            
            D_M2 = M2_ij - M2_ii - M2_jj + M2_ji
            
            # Corrections
            # SubT1_i = (a[i,i] - a[j,i]) (B_rows[j,i] - B_rows[i,i])
            # a_ji = a[j_indices, i]
            # B_rows_ji = B_rows[j_indices, i]
            # B_rows_ii = B_rows[i, i]
            
            a_ii = a[i, i]
            a_ij = a[i, j_indices]
            a_ji = a[j_indices, i]
            a_jj = a[j_indices, j_indices]
            
            B_rows_ii = B_rows[i, i]
            B_rows_ij = B_rows[i, j_indices]
            B_rows_ji = B_rows[j_indices, i]
            B_rows_jj = B_rows[j_indices, j_indices]
            
            B_cols_ii = B_cols[i, i]
            B_cols_ij = B_cols[i, j_indices]
            B_cols_ji = B_cols[j_indices, i]
            B_cols_jj = B_cols[j_indices, j_indices]
            
            SubT1_i = (a_ii - a_ji) * (B_rows_ji - B_rows_ii)
            SubT1_j = (a_ij - a_jj) * (B_rows_jj - B_rows_ij)
            
            SubT2_i = (a_ii - a_ij) * (B_cols_ji - B_cols_ii)
            SubT2_j = (a_ji - a_jj) * (B_cols_jj - B_cols_ij)
            
            E1 = a_ii * (B_rows_jj - B_rows_ii)
            E2 = a_jj * (B_rows_ii - B_rows_jj)
            E3 = a_ij * (B_cols_ij - B_rows_ij)
            E4 = a_ji * (B_cols_ji - B_rows_ji)
            
            deltas = D_M1 + D_M2 - SubT1_i - SubT1_j - SubT2_i - SubT2_j + E1 + E2 + E3 + E4
            
            # Find the best j
            min_idx = np.argmin(deltas)
            min_val = deltas[min_idx]
            
            if min_val < best_delta:
                best_delta = min_val
                best_j = j_indices[min_idx]
                best_i = i
        
        if best_i != -1 and best_delta < -1e-9:
            # Perform swap
            i, j = best_i, best_j
            p[i], p[j] = p[j], p[i]
            current_cost += int(round(best_delta))
            
            # Update B_rows and B_cols
            B_rows[i], B_rows[j] = B_rows[j], B_rows[i]
            B_cols[:, i], B_cols[:, j] = B_cols[:, j].copy(), B_cols[:, i].copy()
            # Correct the diagonal elements of B_cols if necessary?
            # B_cols[k, i] = b[p[k], p[i]].
            # After swap, p[i] is old p[j].
            # So B_cols[k, i] should be b[p[k], old_p[j]].
            # B_cols[k, j] should be b[p[k], old_p[i]].
            # The swap of columns i and j in B_cols handles this for all k.
            # B_cols[:, [i, j]] = B_cols[:, [j, i]]
            # But wait, B_cols is b[:, p].
            # B_cols[k, pos] = b[k, p[pos]]? No.
            # B_cols = b[:, p]. So B_cols[k, pos] = b[k, p[pos]].
            # If we swap p[i] and p[j], then for any k, the value at B_cols[k, i] should become b[k, p_new[i]] = b[k, p_old[j]] = B_cols_old[k, j].
            # So yes, swapping columns i and j of B_cols is correct.
            
            improved = True
        else:
            converged = True
    
    return p, current_cost, converged

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    starts = []
    
    # 1. Rank matching
    starts.append(_rank_matching_init(a, b))
    
    # 2. Spectral initialization
    starts.append(_spectral_init(a, b, n))
    
    # 3. Random permutations
    num_random = 5
    for _ in range(num_random):
        starts.append(_random_init(n, rng))
    
    # Evaluate all starts and find the best
    best_p = None
    best_cost = np.inf
    
    for p in starts:
        cost = _qap_cost(a, b, p)
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
    
    # Run local search on the best start
    best_p, best_cost, _ = _local_search_2opt_fast(a, b, best_p, deadline)
    
    # If time remains, try local search on other starts
    for p in starts[1:]:
        if time.time() > deadline:
            break
        cost = _qap_cost(a, b, p)
        if cost >= best_cost:
            continue
        p_ls, cost_ls, _ = _local_search_2opt_fast(a, b, p, deadline)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls
    
    # Final polish: one more round of local search if time permits
    if time.time() < deadline - 5.0:
        best_p, best_cost, _ = _local_search_2opt_fast(a, b, best_p, deadline)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
