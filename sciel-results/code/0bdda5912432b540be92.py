"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Calculate exact QAP cost for permutation p."""
    # p is a permutation of 0..n-1
    # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Efficiently: trace(A @ B[p, p])
    B_perm = b[np.ix_(p, p)]
    return np.sum(a * B_perm)

def _local_search(a, b, p, max_iters=1000, deadline=None):
    """
    Perform 2-opt local search on permutation p.
    Returns improved permutation and cost.
    """
    n = len(p)
    p = np.array(p, dtype=np.int64)
    
    # Precompute matrices for fast delta calculation
    # M[i, j] = b[p[i], p[j]]
    M = b[np.ix_(p, p)]
    
    # Precompute A @ M^T and A^T @ M
    # These are needed for the delta calculation
    # S1[i,j] = sum_k A[i,k] M[j,k] = (A @ M.T)[i,j]
    # S2[i,j] = sum_k A[k,i] M[k,j] = (A.T @ M)[i,j]
    
    AMT = a @ M.T
    ATM = a.T @ M
    
    # V_i = sum_k A[i,k] M[i,k]
    V = np.sum(a * M, axis=1)
    # W_i = sum_k A[k,i] M[k,i]
    W = np.sum(a.T * M, axis=1)
    
    current_cost = _qap_cost(a, b, p)
    
    start_time = time.time()
    improved = True
    
    while improved and max_iters > 0:
        if deadline and time.time() > deadline:
            break
        
        improved = False
        max_iters -= 1
        
        # Calculate Delta matrix
        # S1[i,j] = AMT[i,j] - V[i]
        # S2[i,j] = ATM[i,j] - W[i]
        # S3[i,j] = AMT[j,i] - V[j]  (Note: S3 uses A[j,k] M[i,k], which is AMT[j,i])
        # S4[i,j] = ATM[j,i] - W[j]  (Note: S4 uses A[k,j] M[k,i], which is ATM[j,i])
        
        S1 = AMT - V[:, np.newaxis]
        S2 = ATM - W[:, np.newaxis]
        S3 = AMT.T - V[np.newaxis, :]
        S4 = ATM.T - W[np.newaxis, :]
        
        # Base Delta without corrections for i,j indices
        Delta = S1 + S2 + S3 + S4
        
        # Corrections for k=i and k=j terms in the sums above
        # We need to subtract the terms where k=i or k=j from S1..S4
        # and add the direct swap term.
        
        # Term from S1 where k=j: A[i,j] * (M[j,j] - M[i,j])
        # Term from S1 where k=i: A[i,i] * (M[j,i] - M[i,i])
        # Term from S2 where k=j: A[j,i] * (M[j,j] - M[j,i])
        # Term from S2 where k=i: A[i,i] * (M[i,j] - M[i,i])
        # Term from S3 where k=j: A[j,i] * (M[i,j] - M[j,j])
        # Term from S3 where k=i: A[j,j] * (M[i,i] - M[j,i])
        # Term from S4 where k=j: A[j,j] * (M[j,i] - M[j,j])
        # Term from S4 where k=i: A[j,i] * (M[i,i] - M[i,j])
        
        # Direct swap term:
        # A[i,j] * (M[j,j] - M[i,j]) + A[j,i] * (M[i,i] - M[j,i])
        
        # Let's combine corrections:
        # C_ij = - (A[i,j]*(M[j,j]-M[i,j]) + A[i,i]*(M[j,i]-M[i,i]) +
        #           A[j,i]*(M[j,j]-M[j,i]) + A[i,i]*(M[i,j]-M[i,i]) +
        #           A[j,i]*(M[i,j]-M[j,j]) + A[j,j]*(M[i,i]-M[j,i]) +
        #           A[j,j]*(M[j,i]-M[j,j]) + A[j,i]*(M[i,i]-M[i,j]))
        #       + (A[i,j]*(M[j,j]-M[i,j]) + A[j,i]*(M[i,i]-M[j,i]))
        
        # Assuming A is symmetric and zero diagonal (standard QAP):
        # A[i,i] = 0, A[j,j] = 0.
        # A[i,j] = A[j,i].
        
        # Terms with A[i,i] or A[j,j] vanish.
        # Remaining from S1: -A[i,j]*(M[j,j]-M[i,j])
        # Remaining from S2: -A[j,i]*(M[j,j]-M[j,i])
        # Remaining from S3: -A[j,i]*(M[i,j]-M[j,j]) = A[j,i]*(M[j,j]-M[i,j])
        # Remaining from S4: -A[j,i]*(M[i,i]-M[i,j]) = A[j,i]*(M[i,j]-M[i,i])
        
        # Sum of subtractions:
        # -A[i,j]*(M[j,j]-M[i,j]) - A[j,i]*(M[j,j]-M[j,i]) + A[j,i]*(M[j,j]-M[i,j]) + A[j,i]*(M[i,j]-M[i,i])
        # Since A[i,j]=A[j,i]=A_ij:
        # -A_ij*(M[j,j]-M[i,j]) - A_ij*(M[j,j]-M[j,i]) + A_ij*(M[j,j]-M[i,j]) + A_ij*(M[i,j]-M[i,i])
        # = -A_ij*(M[j,j]-M[j,i]) + A_ij*(M[i,j]-M[i,i])
        # = A_ij * (M[j,i] - M[j,j] + M[i,j] - M[i,i])
        
        # Direct term to add:
        # A_ij*(M[j,j]-M[i,j]) + A_ij*(M[i,i]-M[j,i])
        # = A_ij * (M[j,j] - M[i,j] + M[i,i] - M[j,i])
        
        # Total Correction = Subtractions + Direct
        # = A_ij * (M[j,i] - M[j,j] + M[i,j] - M[i,i] + M[j,j] - M[i,j] + M[i,i] - M[j,i])
        # = 0
        
        # Wait, if A is symmetric and zero diagonal, the corrections cancel out exactly?
        # Let's re-verify.
        # The sums S1..S4 included k=i and k=j.
        # S1 included k=j: A[i,j] M[j,j] - A[i,j] M[i,j].
        # S2 included k=j: A[j,i] M[j,j] - A[j,i] M[j,i].
        # S3 included k=j: A[j,i] M[i,j] - A[j,i] M[j,j].
        # S4 included k=j: A[j,j] M[j,i] - A[j,j] M[j,j] = 0.
        
        # Sum of k=j terms in S1..S4:
        # A_ij(M[j,j]-M[i,j]) + A_ij(M[j,j]-M[j,i]) + A_ij(M[i,j]-M[j,j])
        # = A_ij(M[j,j] - M[i,j] + M[j,j] - M[j,i] + M[i,j] - M[j,j])
        # = A_ij(M[j,j] - M[j,i])
        
        # S1 included k=i: A[i,i](M[j,i]-M[i,i]) = 0.
        # S2 included k=i: A[i,i](M[i,j]-M[i,i]) = 0.
        # S3 included k=i: A[j,j](M[i,i]-M[j,i]) = 0.
        # S4 included k=i: A[j,i](M[i,i]-M[i,j]) = A_ij(M[i,i]-M[i,j]).
        
        # Sum of k=i terms in S1..S4:
        # A_ij(M[i,i]-M[i,j])
        
        # Total terms to subtract from S1..S4:
        # A_ij(M[j,j]-M[j,i] + M[i,i]-M[i,j])
        
        # Direct term to add:
        # A_ij(M[j,j]-M[i,j]) + A_ij(M[i,i]-M[j,i])
        # = A_ij(M[j,j] - M[i,j] + M[i,i] - M[j,i])
        
        # Net Correction = Direct - Subtracted
        # = A_ij(M[j,j] - M[i,j] + M[i,i] - M[j,i] - M[j,j] + M[j,i] - M[i,i] + M[i,j])
        # = 0
        
        # So for symmetric A with zero diagonal, Delta = S1 + S2 + S3 + S4 is EXACT.
        
        # Set diagonal to inf to prevent self-swap
        np.fill_diagonal(Delta, np.inf)
        
        # Find best swap
        min_idx = np.unravel_index(np.argmin(Delta), Delta.shape)
        min_val = Delta[min_idx]
        
        if min_val < 0:
            i, j = min_idx
            # Perform swap
            p[i], p[j] = p[j], p[i]
            current_cost += min_val
            improved = True
            
            # Update M
            # M[i, :] and M[j, :] need swapping
            # M[:, i] and M[:, j] need swapping
            M[i, :], M[j, :] = M[j, :], M[i, :].copy()
            M[:, i], M[:, j] = M[:, j].copy(), M[:, i]
            
            # Update AMT, ATM, V, W?
            # It's cheaper to recompute AMT, ATM, V, W from scratch every iteration
            # for n=256, it's ~16M ops, which is fast.
            # Or update them incrementally.
            # Recomputing is safer and likely fast enough.
            AMT = a @ M.T
            ATM = a.T @ M
            V = np.sum(a * M, axis=1)
            W = np.sum(a.T * M, axis=1)
            
    return p.tolist(), current_cost

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    # Initial solution: Rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.tolist()
    
    best_p, best_cost = _local_search(a, b, p, max_iters=200, deadline=deadline)
    
    # Iterated Local Search
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Perturb best_p
        p_current = np.array(best_p, dtype=np.int64)
        
        # Random perturbation: 5-15 random swaps
        num_swaps = rng.integers(5, 15)
        for _ in range(num_swaps):
            i, j = rng.choice(n, size=2, replace=False)
            p_current[i], p_current[j] = p_current[j], p_current[i]
        
        # Local search
        p_improved, cost_improved = _local_search(a, b, p_current, max_iters=200, deadline=deadline)
        
        if cost_improved < best_cost:
            best_cost = cost_improved
            best_p = p_improved
    
    return best_p
# EVOLVE-BLOCK-END
