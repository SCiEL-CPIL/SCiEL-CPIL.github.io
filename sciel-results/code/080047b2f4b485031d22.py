"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition_fast(U, V, W, n, m, p):
    """Fast verification using tensor structure."""
    U = np.asarray(U)
    V = np.asarray(V)
    W = np.asarray(W)
    
    # Check all required entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a, b, c = i*m+j, j*p+k, k*n+i
                val = np.dot(U[a], V[b] * W[c])
                if abs(val - 1.0) > 1e-6:
                    return False
    
    # Check zeros for invalid combinations
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i, j = a//m, a%m
                jj, k = b//p, b%p
                kk, ii = c//n, c%n
                if j != jj or k != kk or i != ii:
                    val = np.dot(U[a], V[b] * W[c])
                    if abs(val) > 1e-6:
                        return False
    return True

def _strassen_2x2x2():
    """Strassen rank-7 decomposition for 2x2 * 2x2."""
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    products = [
        ([((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)]),
        ([((1,0),1),((1,1),1)], [((0,0),1)]),
        ([((0,0),1)], [((0,1),1),((1,1),-1)]),
        ([((1,1),1)], [((1,0),1),((0,0),-1)]),
        ([((0,0),1),((0,1),1)], [((1,1),1)]),
        ([((1,0),1),((0,0),-1)], [((0,0),1),((0,1),1)]),
        ([((0,1),1),((1,1),-1)], [((1,0),1),((1,1),1)]),
    ]
    output_contribs = [
        [((0,0),1),((1,1),1)],
        [((1,0),1),((1,1),-1)],
        [((0,1),1),((1,1),1)],
        [((0,0),1),((1,0),1)],
        [((0,0),-1),((0,1),1)],
        [((1,1),1)],
        [((0,0),1)],
    ]
    
    for s in range(7):
        for (ai,aj), coeff in products[s][0]:
            U[ai*m+aj, s] = coeff
        for (bj,bk), coeff in products[s][1]:
            V[bj*p+bk, s] = coeff
        for (ci,ck), coeff in output_contribs[s]:
            W[ck*n+ci, s] = coeff
    
    return U, V, W

def _construct_2x2x12_strassen():
    """Construct rank-42 decomposition via 6 Strassen 2x2x2 blocks."""
    n, m, p = 2, 2, 12
    rank = 42
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    U2, V2, W2 = _strassen_2x2x2()
    
    for t in range(6):
        base_col = 2*t
        for r in range(7):
            U[:, t*7+r] = U2[:, r]
            for j in range(2):
                for k in range(2):
                    V[j*p + base_col + k, t*7+r] = V2[j*2+k, r]
            for k in range(2):
                for i in range(2):
                    W[(base_col+k)*n+i, t*7+r] = W2[k*2+i, r]
    
    return U, V, W

def _construct_row_aggregation():
    """Row-aggregation: factor 2x2 input into rows, giving two 1x2x12 problems.
    
    For row 0: output row 0 = A[0,:] * B, i.e., c_k = a_00*b_0k + a_01*b_1k for k=0..11
    For row 1: output row 1 = A[1,:] * B, i.e., c_k = a_10*b_0k + a_11*b_1k for k=0..11
    
    Each 1x2x12 can be done with rank 12 (one term per output element).
    Total rank = 24.
    
    But we can share terms between the two rows! The key insight:
    c_0k = a_00*b_0k + a_01*b_1k
    c_1k = a_10*b_0k + a_11*b_1k
    
    We need 4p = 48 terms naively. With Strassen on pairs: 6*7 = 42.
    With row aggregation + sharing: we can potentially do better.
    
    Actually, let's use a different approach: for each column pair (2k, 2k+1),
    use Strassen. That gives rank 42.
    
    Alternative: use the fact that we only need 2x2x12, not full 2x2x2 per pair.
    """
    n, m, p = 2, 2, 12
    rank = 24
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Row 0: c_0k = a_00*b_0k + a_01*b_1k for k=0..11
    # This needs 2*12 = 24 terms if done directly
    # But we can share: for each k, we need a_00*b_0k and a_01*b_1k
    # Row 1: c_1k = a_10*b_0k + a_11*b_1k for k=0..11
    # This needs another 2*12 = 24 terms
    # Total: 48 terms (schoolbook)
    
    # With Strassen on pairs: 6 pairs * 7 = 42
    # Can we do better?
    
    # Let's try: for each k, use 3 terms instead of 4:
    # (a_00+a_11)(b_0k+b_1k) - a_11*b_0k - a_00*b_1k gives a_00*b_0k + a_11*b_1k
    # But we still need a_01*b_1k and a_10*b_0k separately
    # So per k: 3 + 2 = 5 terms, total 12*5 = 60. Worse!
    
    # The Strassen approach is better. Let's just use rank 42.
    # But wait - can we do 2x2x12 with less than 42?
    
    # Known lower bound: rank >= 2p - 2 = 22 for M_{2,2,p}
    # Let's try to construct rank 22 or 24.
    
    # For rank 24: use 12 terms for row 0 and 12 for row 1, but share some.
    # Actually, the simplest: just use the Strassen construction (rank 42).
    
    # Let me try a direct construction for rank 24:
    # For each k in 0..11:
    #   Term 2k: U has a_00, V has b_0k, W has c_0k
    #   Term 2k+1: U has a_01, V has b_1k, W has c_0k  -- NO, W can only contribute to one output
    # 
    # Actually W[c] is a vector of length rank. Each term s contributes W[c]*U[a]*V[b] to output (a,b,c).
    # So for output (0,k) [i.e., row 0, col k of result]:
    #   sum_s U[0,s]*V[2k,s]*W[k*2+0,s] = 1 (for a_00*b_0k)
    #   sum_s U[1,s]*V[2k,s]*W[k*2+0,s] = 0 (no a_01 in row 0)
    #   etc.
    
    # This is getting complex. Let's just return the Strassen rank-42 for now
    # and add a numerical search for lower rank.
    return None  # Fall through to Strassen

def _find_decomposition_als(n, m, p, rank, max_iter=5000, seed=42, tol=1e-8):
    """ALS-based search for rank-r decomposition."""
    np.random.seed(seed)
    U = np.random.randn(n*m, rank) * 0.5
    V = np.random.randn(m*p, rank) * 0.5
    W = np.random.randn(p*n, rank) * 0.5
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(max_iter):
        # Update U
        for a in range(n*m):
            i, j = a // m, a % m
            M = np.array([V[j*p+k] * W[k*n+i] for k in range(p)])
            if M.shape[0] > 0:
                U[a] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
        
        # Update V
        for b in range(m*p):
            j, k = b // p, b % p
            M = np.array([U[i*m+j] * W[k*n+i] for i in range(n)])
            if M.shape[0] > 0:
                V[b] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
        
        # Update W
        for c in range(p*n):
            k, i = c // n, c % n
            M = np.array([U[i*m+j] * V[j*p+k] for j in range(m)])
            if M.shape[0] > 0:
                W[c] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
        
        # Check error
        err_sum = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a, b, c = i*m+j, j*p+k, k*n+i
                    val = np.dot(U[a], V[b] * W[c])
                    err_sum += (val - 1.0)**2
        err = np.sqrt(err_sum)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if err < tol:
            break
    
    return best_U, best_V, best_W, best_err

def construct():
    """Rank-minimal decomposition for 2x2 * 2x12 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 145.0
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    # Strategy 1: Try rank-24 via direct construction
    # For 2x2x12, try to find a rank-24 decomposition
    # Row 0: c_0k = a_00*b_0k + a_01*b_1k (12 outputs, needs at least 12 terms if no sharing)
    # Row 1: c_1k = a_10*b_0k + a_11*b_1k (12 outputs, needs at least 12 terms)
    # With careful sharing, rank 24 might work.
    
    # Actually, let's try a structured rank-24 construction:
    # For each k in 0..11, use 2 terms:
    #   Term 2k: U=[a_00, 0, 0, a_11], V=[b_0k, 0, 0, b_1k], W=[1, 0, 1, 0] for outputs (0,k) and (1,k)
    #   Wait, W needs to be per-output. Let me think differently.
    
    # For output (i, k) where i in {0,1} is row of result, k in 0..11 is col:
    #   (0,k): sum_s U[0,s]*V[2k,s]*W[k*2,s] = a_00*b_0k + a_01*b_1k
    #   (1,k): sum_s U[1,s]*V[2k,s]*W[k*2+1,s] = a_10*b_0k + a_11*b_1k
    #   (0,k+12): should be 0 (but there's no such output since p=12)
    #   Actually outputs are indexed by (k*n+i) = (k*2+i) for k in 0..11, i in 0..1
    
    # So we need:
    #   For each k: sum_s U[0,s]*V[2k,s]*W[2k,s] = coefficient of a_00 in row 0 col k = 1
    #              sum_s U[1,s]*V[2k,s]*W[2k,s] = coefficient of a_01 in row 0 col k = 1
    #              sum_s U[2,s]*V[2k,s]*W[2k,s] = coefficient of a_10 in row 0 col k = 0
    #              sum_s U[3,s]*V[2k,s]*W[2k,s] = coefficient of a_11 in row 0 col k = 0
    #   Similarly for W[2k+1,s] with row 1 coefficients.
    
    # This is a system of equations. Let's set up rank-24 with 2 terms per column pair... 
    # Actually, let's just try numerical search for rank 24, 26, 28, etc.
    
    best_rank = 42
    best_result = None
    
    # Try structured rank-24 first
    # Construction: for each k in 0..11, allocate 2 terms
    # Term 2k: handles a_00*b_0k -> c_0k and a_10*b_0k -> c_1k  
    # Term 2k+1: handles a_01*b_1k -> c_0k and a_11*b_1k -> c_1k
    
    rank_try = 24
    U = np.zeros((n*m, rank_try))
    V = np.zeros((m*p, rank_try))
    W = np.zeros((p*n, rank_try))
    
    for k in range(12):
        # Term 2k: a_00*b_0k and a_10*b_0k
        s0 = 2*k
        U[0, s0] = 1  # a_00
        U[2, s0] = 1  # a_10
        V[0*2+k, s0] = 1  # b_0k
        W[k*2+0, s0] = 1  # contributes to c_0k
        W[k*2+1, s0] = 1  # contributes to c_1k
        
        # Term 2k+1: a_01*b_1k and a_11*b_1k
        s1 = 2*k+1
        U[1, s1] = 1  # a_01
        U[3, s1] = 1  # a_11
        V[1*2+k, s1] = 1  # b_1k
        W[k*2+0, s1] = 1  # contributes to c_0k
        W[k*2+1, s1] = 1  # contributes to c_1k
    
    if _verify_decomposition_fast(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # That didn't work because the cross terms are wrong.
    # Let me check: for output (0, k) [index k*2+0]:
    #   sum_s U[0,s]*V[2k,s]*W[2k,s]
    #   = U[0,2k]*V[2k,2k]*W[2k,2k] + U[0,2k+1]*V[2k,2k+1]*W[2k,2k+1]
    #   = 1*0*1 + 0*0*1 = 0  (WRONG, should be 1 for a_00*b_0k)
    
    # The issue: V[2k, s] is the b-index for row j=0, col k. So V[0*2+k, s] = V[2k, s].
    # For term s0=2k: V[2k, 2k] = 1 (b_0k), but U[0, 2k] = 1 (a_00). 
    # W[2k, 2k] should give the contribution to output (0,k).
    # But W index is (k*n+i) = (k*2+i). For output row 0, col k: index = k*2+0 = 2k.
    # So W[2k, 2k] = 1 means term 2k contributes 1 to output (0,k). Good.
    # But we also set W[2k+1, 2k] = 1, meaning term 2k contributes 1 to output (1,k).
    # For output (1,k): sum_s U[1,s]*V[2k,s]*W[2k+1,s]
    #   = U[1,2k]*V[2k,2k]*W[2k+1,2k] + ... = 0*1*1 = 0. Good (no a_01 in row 1).
    # But we need a_10*b_0k in output (1,k): 
    #   sum_s U[2,s]*V[2k,s]*W[2k+1,s]
    #   = U[2,2k]*V[2k,2k]*W[2k+1,2k] = 1*1*1 = 1. 
    # Wait, U index 2 is a_10 (i=1, j=0, so i*m+j = 1*2+0 = 2). Yes!
    # So this might actually work. Let me re-verify.
    
    # Output (0,k) = index k*2+0 = 2k:
    #   sum_s U[0,s]*V[2k,s]*W[2k,s] where U[0] = a_00
    #   s=2k: U[0,2k]*V[2k,2k]*W[2k,2k] = 1*1*1 = 1 (a_00*b_0k) ✓
    #   s=2k+1: U[0,2k+1]*V[2k,2k+1]*W[2k,2k+1] = 0*0*1 = 0 ✓
    #   Total: 1 ✓ (need a_00*b_0k coefficient = 1)
    
    #   Also need a_01*b_1k in output (0,k):
    #   sum_s U[1,s]*V[2k,s]*W[2k,s] where U[1] = a_01
    #   s=2k: U[1,2k]*V[2k,2k]*W[2k,2k] = 0*1*1 = 0
    #   s=2k+1: U[1,2k+1]*V[2k,2k+1]*W[2k,2k+1] = 1*0*1 = 0  (V[2k,2k+1] = V[0*2+k, 2k+1] = 0 since we set V[1*2+k, 2k+1]=1)
    #   Total: 0 ✗ (need 1 for a_01*b_1k)
    
    # Problem: V[2k, 2k+1] = V[j=0, k, s=2k+1] but we set V[j=1, k, s=2k+1] = V[2k+2, 2k+1] = 1.
    # So V[2k, 2k+1] = 0. The a_01*b_1k term requires V[j=1,k] which is V[2k+2, s], not V[2k, s].
    
    # I see the issue: the V index is j*p+k where j is the row of B. For b_1k, j=1, so index = 1*12+k = 12+k.
    # Not 2k+1! I was confusing the 2x2x2 Strassen indexing with the 2x2x12 indexing.
    
    # Let me redo: V has m*p = 2*12 = 24 rows. V[j*p+k, s] is the coefficient for b_{j,k} in term s.
    # For b_0k: j=0, index = 0*12+k = k
    # For b_1k: j=1, index = 1*12+k = 12+k
    
    # So the correct construction:
    U = np.zeros((n*m, rank_try))
    V = np.zeros((m*p, rank_try))
    W = np.zeros((p*n, rank_try))
    
    for k in range(12):
        s0 = 2*k
        s1 = 2*k+1
        # Term s0: a_00*b_0k -> c_0k, and a_10*b_0k -> c_1k
        U[0, s0] = 1   # a_00 (i=0,j=0)
        U[2, s0] = 1   # a_10 (i=1,j=0)
        V[k, s0] = 1   # b_0k (j=0, col k)
        W[k*2+0, s0] = 1  # c_0k (row 0, col k)
        W[k*2+1, s0] = 1  # c_1k (row 1, col k)
        
        # Term s1: a_01*b_1k -> c_0k, and a_11*b_1k -> c_1k
        U[1, s1] = 1   # a_01 (i=0,j=1)
        U[3, s1] = 1   # a_11 (i=1,j=1)
        V[12+k, s1] = 1  # b_1k (j=1, col k)
        W[k*2+0, s1] = 1  # c_0k
        W[k*2+1, s1] = 1  # c_1k
    
    if _verify_decomposition_fast(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify manually:
    # Output (0,k) = index k*2+0:
    #   sum_s U[0,s]*V[?k,s]*W[k*2,s]
    #   For a_00*b_0k: need U[0,s]*V[k,s]*W[2k,s] = 1
    #     s=s0: U[0,s0]*V[k,s0]*W[2k,s0] = 1*1*1 = 1 ✓
    #   For a_01*b_1k: need U[1,s]*V[12+k,s]*W[2k,s] = 1
    #     s=s1: U[1,s1]*V[12+k,s1]*W[2k,s1] = 1*1*1 = 1 ✓
    #   But wait, the output formula is:
    #   c_{i,k} = sum_j a_{i,j} * b_{j,k}
    #   So c_{0,k} = a_{0,0}*b_{0,k} + a_{0,1}*b_{1,k}
    #   In the tensor: T[i*m+j, j*p+k, k*n+i] = 1
    #   For (i,j,k) = (0,0,k): T[0, k, 2k] = 1
    #   For (i,j,k) = (0,1,k): T[1, 12+k, 2k] = 1
    #   So output index for c_{0,k} is 2k, and we need:
    #   sum_s U[0,s]*V[k,s]*W[2k,s] = 1 (from a_00*b_0k)
    #   sum_s U[1,s]*V[12+k,s]*W[2k,s] = 1 (from a_01*b_1k)
    #   These are SEPARATE sums! The total contribution to W[2k] from all terms should give:
    #   For each (a_idx, b_idx) pair, sum_s U[a_idx,s]*V[b_idx,s]*W[out_idx,s] = T[a_idx,b_idx,out_idx]
    
    # So for out_idx = 2k (c_{0,k}):
    #   a_idx=0 (a_00), b_idx=k (b_0k): sum_s U[0,s]*V[k,s]*W[2k,s] should = 1
    #     s=s0: 1*1*1 = 1, others 0. Total = 1 ✓
    #   a_idx=1 (a_01), b_idx=12+k (b_1k): sum_s U[1,s]*V[12+k,s]*W[2k,s] should = 1
    #     s=s1: 1*1*1 = 1, others 0. Total = 1 ✓
    #   a_idx=2 (a_10), b_idx=k (b_0k): sum_s U[2,s]*V[k,s]*W[2k,s] should = 0
    #     s=s0: 1*1*1 = 1. Total = 1 ✗ (should be 0!)
    
    # Problem! U[2,s0]=1 and V[k,s0]=1 and W[2k,s0]=1 gives a_10*b_0k contributing to c_0k,
    # but a_10*b_0k should contribute to c_1k (row 1), not c_0k (row 0).
    
    # The fix: W[2k, s0] should be 0, and W[2k+1, s0] should be 1 for a_10*b_0k.
    # But W is shared between a_00*b_0k (which needs W[2k,s0]=1) and a_10*b_0k (which needs W[2k+1,s0]=1).
    # We can't have both in the same term!
    
    # So we need separate terms for each (a,b) pair. That's 4*12 = 48 terms (schoolbook).
    # Strassen reduces this to 42 by combining within 2x2 blocks.
    
    # For rank < 42, we need a smarter construction. Let's try numerical search.
    
    # Try ranks from 24 up to 41
    for target_rank in [24, 26, 28, 30, 32, 34, 36, 38, 40]:
        if time.time() > deadline - 20:
            break
        found = False
        for seed in range(20):
            if time.time() > deadline - 15:
                break
            U_t, V_t, W_t, err = _find_decomposition_als(n, m, p, target_rank, max_iter=3000, seed=seed)
            if err < 1e-6:
                if _verify_decomposition_fast(U_t, V_t, W_t, n, m, p):
                    return {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}
                found = True
        if found:
            break
    
    # Fallback: rank-42 Strassen
    U, V, W = _construct_2x2x12_strassen()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
