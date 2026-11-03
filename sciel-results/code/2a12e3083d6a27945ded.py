"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    """
    Solves the Quadratic Assignment Problem (QAP) using a 2-opt local search
    with correct swap-delta calculations.
    
    Args:
        a: Flow matrix (n, n) int64 array.
        b: Distance matrix (n, n) int64 array.
        
    Returns:
        p: List of integers representing the permutation of facilities to locations.
    """
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    # Initial solution: Rank matching
    # Heavy flows (high row sum in A) go to central locations (low row sum in B)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Deadline: 115 seconds (safety margin for 120s limit)
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Precompute nothing complex, just ensure arrays are C-contiguous for speed
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    
    # Local search
    while time.time() < deadline:
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Find the best improving swap
        # We iterate over all pairs (i, j) with i < j
        for i in range(n):
            if time.time() >= deadline:
                break
                
            ai = a[i, :]
            pi_loc = p[i]
            bi_pi = b[pi_loc, :]
            
            for j in range(i + 1, n):
                aj = a[j, :]
                pj_loc = p[j]
                bi_pj = b[pj_loc, :]
                
                # Calculate Delta
                # S = sum_{k} (ai[k] - aj[k]) * (bi_pj[k] - bi_pi[k])
                # This is a dot product of (ai - aj) and (bi_pj - bi_pi)
                diff_a = ai - aj
                diff_b = bi_pj - bi_pi
                S = np.dot(diff_a, diff_b)
                
                # Subtract terms for k=i and k=j from S
                # S_ii = (ai[i] - aj[i]) * (bi_pj[i] - bi_pi[i])
                S_ii = (ai[i] - aj[i]) * (bi_pj[i] - bi_pi[i])
                
                # S_jj = (ai[j] - aj[j]) * (bi_pj[j] - bi_pi[j])
                S_jj = (ai[j] - aj[j]) * (bi_pj[j] - bi_pi[j])
                
                # Add correct manual changes for k=i and k=j
                # Correct_ii = ai[i]*(bi_pj[i] - bi_pi[i]) + aj[i]*(bi_pi[j] - bi_pj[i])
                # Note: bi_pi[j] is b[pi, p[j]] = b[pi, pj_loc]
                # bi_pj[i] is b[pj, p[i]] = b[pj_loc, pi_loc]
                # Since b is symmetric, b[pi, pj] == b[pj, pi].
                # Let's stick to indices to be safe.
                # b[pi, pj] is b[pi_loc, pj_loc]
                # b[pj, pi] is b[pj_loc, pi_loc]
                
                b_pi_pj = b[pi_loc, pj_loc]
                b_pj_pi = b[pj_loc, pi_loc]
                b_pi_pi = b[pi_loc, pi_loc]
                b_pj_pj = b[pj_loc, pj_loc]
                
                # Correct_ii:
                # Original terms involving i at k=i: A[i,i]B[pi,pi] + A[j,i]B[pj,pi]
                # New terms: A[i,i]B[pj,pi] + A[j,i]B[pi,pj]
                # Change: A[i,i](B[pj,pi] - B[pi,pi]) + A[j,i](B[pi,pj] - B[pj,pi])
                Correct_ii = ai[i] * (b_pj_pi - b_pi_pi) + aj[i] * (b_pi_pj - b_pj_pi)
                
                # Correct_jj:
                # Original terms involving j at k=j: A[i,j]B[pi,pj] + A[j,j]B[pj,pj]
                # New terms: A[i,j]B[pj,pi] + A[j,j]B[pi,pj]
                # Change: A[i,j](B[pj,pi] - B[pi,pj]) + A[j,j](B[pi,pj] - B[pj,pj])
                Correct_jj = ai[j] * (b_pj_pi - b_pi_pj) + aj[j] * (b_pi_pj - b_pj_pj)
                
                delta = S - S_ii - S_jj + Correct_ii + Correct_jj
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        # If no improvement found, stop
        if best_i == -1:
            break
        
        # Apply the best swap
        p[best_i], p[best_j] = p[best_j], p[best_i]
        
        # Optional: Check time again after swap
        if time.time() >= deadline:
            break

    return p.tolist()
# EVOLVE-BLOCK-END
