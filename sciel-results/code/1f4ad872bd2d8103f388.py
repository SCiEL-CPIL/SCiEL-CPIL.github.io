"""Matrix multiplication 2x5x6 rank-minimal exact tensor decomposition."""
import time
import random

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """
    Return the 7 rank-1 tensors for 2x2x2 matrix multiplication (Strassen).
    Each tensor is (U_row, V_row, W_row) where:
      U has 4 entries (rows of U for a 2x2 block),
      V has 4 entries (rows of V for a 2x2 block),
      W has 4 entries (rows of W for a 2x2 block, in k*n+i convention).
    
    Standard Strassen formulas for C = A*B where A,B are 2x2:
      A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
      C = [[c11, c12], [c21, c22]]
      
    Strassen's 7 products:
      p1 = (a11+a22)(b11+b22)
      p2 = (a21+a22)b11
      p3 = a11(b12-b22)
      p4 = a22(b21-b11)
      p5 = (a11+a12)b22
      p6 = (a21-a11)(b11+b12)
      p7 = (a12-a22)(b21+b22)
      
      c11 = p1 + p4 - p5 + p7
      c12 = p3 + p5
      c21 = p2 + p4
      c22 = p1 - p2 + p3 + p6
    
    In tensor decomposition terms, each product p_t gives a rank-1 term:
      U contribution: which (i,j) pair of A
      V contribution: which (j,k) pair of B
      W contribution: which (k,i) pair of C
    
    Let me index: A row i in {0,1}, col j in {0,1}; B row j in {0,1}, col k in {0,1}; C row k in {0,1}, col i in {0,1}.
    
    For product p_t, the coefficient on A[i,j] * B[j,k] contributes to C[k,i].
    So U[i,j] = coeff_A, V[j,k] = coeff_B, W[k,i] = 1 (or we can absorb into U/V).
    
    Actually the standard formulation: each rank-1 term is U_vec * V_vec * W_vec where
    sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] = C[k,i]
    
    Let me enumerate:
    p1: (a00+a11)(b00+b11)
      A terms: a00*1, a11*1 -> U[0,0]=1, U[1,1]=1
      B terms: b00*1, b11*1 -> V[0,0]=1, V[1,1]=1
      C: contributes to c00 (+1), c11 (+1) -> W[0,0]=1, W[1,1]=1
    
    p2: (a10+a11)*b00
      A: a10*1, a11*1 -> U[1,0]=1, U[1,1]=1
      B: b00*1 -> V[0,0]=1
      C: c10 (+1), c11 (-1) -> W[0,1]=1, W[1,1]=-1
    
    p3: a00*(b01-b11)
      A: a00*1 -> U[0,0]=1
      B: b01*1, b11*(-1) -> V[0,1]=1, V[1,1]=-1
      C: c01 (+1), c11 (+1) -> W[1,0]=1, W[1,1]=1
    
    p4: a11*(b10-b00)
      A: a11*1 -> U[1,1]=1
      B: b10*1, b00*(-1) -> V[1,0]=1, V[0,0]=-1
      C: c00 (+1), c10 (+1) -> W[0,0]=1, W[0,1]=1
    
    p5: (a00+a01)*b11
      A: a00*1, a01*1 -> U[0,0]=1, U[0,1]=1
      B: b11*1 -> V[1,1]=1
      C: c00 (-1), c11 (-1) -> W[0,0]=-1, W[1,1]=-1
    
    p6: (a10-a00)*(b00+b01)
      A: a10*1, a00*(-1) -> U[1,0]=1, U[0,0]=-1
      B: b00*1, b01*1 -> V[0,0]=1, V[0,1]=1
      C: c11 (+1) -> W[1,1]=1
    
    p7: (a01-a11)*(b10+b11)
      A: a01*1, a11*(-1) -> U[0,1]=1, U[1,1]=-1
      B: b10*1, b11*1 -> V[1,0]=1, V[1,1]=1
      C: c00 (+1) -> W[0,0]=1
    
    Let me verify p1: U[0,0]*V[0,0]*W[0,0] + U[0,0]*V[1,1]*W[1,0]... 
    Wait, I need to be more careful. The tensor is T[i,j,k] = C[k,i] (with the W convention W[k*n+i]).
    
    For 2x2x2: T[i,j,k] where i,j in {0,1} for A, j,k for B, and C index is (k,i).
    T[i,j,k] = delta (C[k,i] gets A[i,j]*B[j,k]).
    
    So T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[k,i], which is always the case.
    Actually T[i,j,k] = 1 for all valid (i,j,k) since C[k,i] = sum_j A[i,j]*B[j,k].
    
    Wait no. The tensor for matrix multiplication C = A*B is:
    T[i,j,k] = 1 means A[i,j] * B[j,k] contributes to C[k,i].
    But C[k,i] = sum_j A[i,j] * B[j,k], so yes T[i,j,k] = 1 for all i,j,k.
    
    Hmm, that's not right either. Let me re-examine.
    
    For matrix multiplication (n x m) * (m x p) -> (n x p):
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T has indices (i, j, k) and T[i,j,k] = 1.
    Then C[i,k] = sum_j sum_t U[i,j,t] * V[j,k,t] * W[t] ... no.
    
    The rank-r decomposition: T[i,j,k] = sum_{t=1}^{r} U[i,j,t] * V[j,k,t] * W[k,i,t]
    
    Wait, the problem says W has shape (p*n, rank) and the convention is W[k*n+i, r].
    So W index is (k, i) where k is the column of B / row of C, and i is the row of A / column of C.
    
    So the decomposition is:
    For each (i, j, k): sum_{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    And for invalid combinations (where the product doesn't contribute), it should be 0.
    Actually for matrix multiplication, ALL combinations (i,j,k) with i in [0,n), j in [0,m), k in [0,p) contribute,
    and T[i,j,k] = 1 for all of them.
    
    So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    
    OK so for 2x2x2, T[i,j,k] = 1 for all i,j,k in {0,1}.
    
    Let me re-verify Strassen with this understanding.
    Each rank-1 term r has vectors U_r (length 4), V_r (length 4), W_r (length 4).
    U_r[i,j] * V_r[j,k] * W_r[k,i] summed over r should give 1 for all (i,j,k).
    
    p1: U = [1,0,0,1] (i=0,j=0 -> 1; i=1,j=1 -> 1), V = [1,0,0,1] (j=0,k=0 -> 1; j=1,k=1 -> 1), W = [1,0,0,1] (k=0,i=0 -> 1; k=1,i=1 -> 1)
    Contribution to (i,j,k): U[i,j]*V[j,k]*W[k,i]
    (0,0,0): 1*1*1 = 1
    (0,0,1): 1*0*0 = 0
    (0,1,0): 0*0*0 = 0
    (0,1,1): 0*1*0 = 0
    (1,0,0): 0*1*0 = 0
    (1,0,1): 0*0*1 = 0
    (1,1,0): 1*0*0 = 0
    (1,1,1): 1*1*1 = 1
    
    p2: U = [0,0,1,1] (i=1,j=0 -> 1; i=1,j=1 -> 1), V = [1,0,0,0] (j=0,k=0 -> 1), W = [0,1,-1,0] (k=0,i=1 -> 1; k=1,i=1 -> -1)
    (0,0,0): 0
    (0,0,1): 0
    (0,1,0): 0
    (0,1,1): 0
    (1,0,0): 1*1*0 = 0
    (1,0,1): 1*0*1 = 0
    (1,1,0): 1*0*(-1) = 0
    (1,1,1): 1*1*0 = 0
    Hmm that gives 0 everywhere. That's wrong.
    
    I think I'm confusing the indexing. Let me be very explicit.
    
    U has indices (i, j) for i in 0..n-1, j in 0..m-1. Stored as U[i*m+j].
    V has indices (j, k) for j in 0..m-1, k in 0..p-1. Stored as V[j*p+k].
    W has indices (k, i) for k in 0..p-1, i in 0..n-1. Stored as W[k*n+i].
    
    For 2x2x2: n=m=p=2.
    U: 4 entries, indexed by (i,j) -> position i*2+j
    V: 4 entries, indexed by (j,k) -> position j*2+k
    W: 4 entries, indexed by (k,i) -> position k*2+i
    
    The condition is: for ALL (i,j,k) in {0,1}^3:
      sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1
    
    Now Strassen:
    p1 = (a00+a11)(b00+b11)
    This product involves A[0,0], A[1,1] and B[0,0], B[1,1].
    The result contributes to C[0,0] and C[1,1].
    
    So in this rank-1 term:
    U[0,0] = 1, U[1,1] = 1, others 0.
    V[0,0] = 1, V[1,1] = 1, others 0.
    W[0,0] = 1, W[1,1] = 1, others 0.
    
    Check (i=0,j=0,k=0): U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1 ✓
    Check (i=0,j=0,k=1): U[0,0]*V[0,1]*W[1,0] = 1*0*0 = 0
    Check (i=1,j=1,k=0): U[1,1]*V[1,0]*W[0,1] = 1*0*0 = 0
    Check (i=1,j=1,k=1): U[1,1]*V[1,1]*W[1,1] = 1*1*1 = 1 ✓
    All other (i,j,k) give 0. Good.
    
    p2 = (a10+a11)*b00
    A[1,0]=1, A[1,1]=1; B[0,0]=1.
    Result: C[0,1] += 1, C[1,1] -= 1.
    
    U[1,0]=1, U[1,1]=1.
    V[0,0]=1.
    W[0,1]=1, W[1,1]=-1.
    
    Check (i=1,j=0,k=0): U[1,0]*V[0,0]*W[0,1] = 1*1*1 = 1
    Check (i=1,j=1,k=0): U[1,1]*V[0,0]... wait V[1,0] not V[0,0].
    
    I see my error. V is indexed by (j,k). So V[j,k].
    For p2: B[0,0] means j=0,k=0. So V[0,0]=1.
    But then for (i=1,j=1,k=0): U[1,1]*V[1,0]*W[0,1]. V[1,0]=0. So 0. Good.
    For (i=1,j=0,k=0): U[1,0]*V[0,0]*W[0,1] = 1*1*1 = 1. This should contribute to... 
    
    Wait, the sum over r of U[i,j,r]*V[j,k,r]*W[k,i,r] should equal 1 for ALL (i,j,k).
    So (i=1,j=0,k=0) should get total 1 from all rank-1 terms.
    
    Let me just trust the standard Strassen construction and verify the final sum.
    
    Actually, let me just use a known correct Strassen tensor decomposition and verify it computationally.
    """
    # Strassen 7 products, each as (U_vec, V_vec, W_vec) of length 4 each
    # U indexed by (i,j) -> i*2+j, V by (j,k) -> j*2+k, W by (k,i) -> k*2+i
    terms = [
        # p1 = (a00+a11)(b00+b11)
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        # p2 = (a10+a11)*b00
        ([0,0,1,1], [1,0,0,0], [0,1,-1,0]),
        # p3 = a00*(b01-b11)
        ([1,0,0,0], [0,1,0,-1], [0,0,1,1]),
        # p4 = a11*(b10-b00)
        ([0,0,0,1], [-1,0,1,0], [1,1,0,0]),
        # p5 = (a00+a01)*b11
        ([1,1,0,0], [0,0,0,1], [-1,0,0,-1]),
        # p6 = (a10-a00)*(b00+b01)
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        # p7 = (a01-a11)*(b10+b11)
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]
    return terms


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    import numpy as np
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should be 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _build_strassen_2x5x6():
    """
    Build a decomposition for 2x5x6 using Strassen on 2x2x2 sub-blocks.
    
    Strategy: 
    - n=2, m=5, p=6
    - We can pair up the m dimension: (0,1), (2,3), and leave j=4 alone
    - We can pair up the p dimension: (0,1), (2,3), (4,5)
    
    For each combination of m-pair and p-pair, if we have a 2x2x2 block we can use Strassen (rank 7 instead of 8).
    
    m-pairs: (0,1), (2,3), and single j=4
    p-pairs: (0,1), (2,3), (4,5)
    
    For 2x2x2 blocks (when both m and p are paired): use Strassen rank 7
    For 2x1x2 or 1x2x2 or 1x1x2 etc: use schoolbook
    
    Actually, let me think about this differently.
    
    The full tensor is 2x5x6 = 60 entries, all equal to 1.
    Schoolbook: rank 60.
    
    We can decompose this into blocks:
    - For each pair of m-indices (j1, j2) and pair of p-indices (k1, k2), 
      the sub-tensor on those indices (with all n=2) is a 2x2x2 tensor (all ones).
    
    m has 5 indices: we can form pairs (0,1), (2,3) and single 4.
    p has 6 indices: we can form pairs (0,1), (2,3), (4,5).
    
    Number of 2x2x2 blocks: 2 (m-pairs) * 3 (p-pairs) = 6 blocks, each rank 7 = 42
    Remaining: j=4 with all p: 2x1x6 = 12 entries, rank 12 (schoolbook)
    Total: 42 + 12 = 54.
    
    That matches the knowledge: "reduces rank from 58 to 54". Wait, 58? 
    Let me recompute schoolbook: 2*5*6 = 60. So 54 is a saving of 6.
    The knowledge says "from 58 to 54" which is a saving of 4. Maybe they started from a different base.
    
    Anyway, let me construct the rank-54 decomposition.
    
    For each 2x2x2 block (j1,j2) x (k1,k2) with n=2:
      The sub-tensor has entries T[i, j, k] = 1 for i in {0,1}, j in {j1,j2}, k in {k1,k2}.
      We use Strassen's 7 rank-1 terms.
    
    For the remaining j=4:
      T[i, 4, k] = 1 for i in {0,1}, k in {0,...,5}.
      Schoolbook: for each (i,k), one rank-1 term. That's 2*6=12 terms.
    
    Total rank: 6*7 + 12 = 42 + 12 = 54.
    """
    n, m, p = 2, 5, 6
    strassen_terms = _strassen_2x2x2()
    
    U = []
    V = []
    W = []
    
    def add_term(u_vec, v_vec, w_vec, i_offset, j_offset, k_offset):
        """Add a rank-1 term to U, V, W.
        u_vec: length n*m, but only entries at positions corresponding to the block are nonzero.
        Actually, let me build full-length vectors.
        """
        U_full = [0] * (n * m)
        V_full = [0] * (m * p)
        W_full = [0] * (p * n)
        
        # u_vec is for a 2x2 block in (i,j) space, where i in 0..1, j in 0..1 (local)
        # Local j maps to global j = j_offset + local_j
        for local_i in range(2):
            for local_j in range(2):
                U_full[i_offset + local_i, j_offset + local_j] = u_vec[local_i * 2 + local_j]
        
        # v_vec is for a 2x2 block in (j,k) space
        for local_j in range(2):
            for local_k in range(2):
                V_full[j_offset + local_j, k_offset + local_k] = v_vec[local_j * 2 + local_k]
        
        # w_vec is for a 2x2 block in (k,i) space
        for local_k in range(2):
            for local_i in range(2):
                W_full[k_offset + local_i, local_i] = w_vec[local_k * 2 + local_i]
        
        U.append(U_full)
        V.append(V_full)
        W.append(W_full)
    
    # 2x2x2 blocks
    m_pairs = [(0, 1), (2, 3)]
    p_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for (j1, j2) in m_pairs:
        for (k1, k2) in p_pairs:
            for (u_vec, v_vec, w_vec) in strassen_terms:
                add_term(u_vec, v_vec, w_vec, 0, j1, k1)
    
    # Remaining: j=4, all k
    for i in range(n):
        for k in range(p):
            U_full = [0] * (n * m)
            V_full = [0] * (m * p)
            W_full = [0] * (p * n)
            U_full[i * m + 4] = 1
            V_full[4 * p + k] = 1
            W_full[k * n + i] = 1
            U.append(U_full)
            V.append(V_full)
            W.append(W_full)
    
    rank = len(U)
    return {"U": U, "V": V, "W": W}


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x5x6 matrix multiplication.
    Uses Strassen's algorithm on 2x2x2 sub-blocks to reduce rank.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Start with the Strassen-based decomposition (rank 54)
    result = _build_strassen_2x5x6()
    best_rank = len(result["U"])
    
    # Verify
    n, m, p = DIMENSIONS
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        result = {"U": U, "V": V, "W": W}
        best_rank = rank
        return result
    
    # Try to further reduce rank using random search / local optimization
    # This is a hard problem; for now, return the Strassen-based result.
    # In a full implementation, we could try:
    # 1. Random rank-r initializations and gradient descent
    # 2. Exploiting symmetries
    # 3. Trying different block decompositions
    
    elapsed = time.time() - start_time
    if elapsed < deadline:
        # Try some random perturbations to see if we can find a lower rank
        # (This is very unlikely to work for exact integer/half-integer solutions)
        pass
    
    return result
# EVOLVE-BLOCK-END
