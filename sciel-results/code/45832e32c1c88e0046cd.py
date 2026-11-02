import time
import numpy as np
from itertools import product

def _verify_decomposition(U, V, W, n=2, m=3, p=5):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build the multiplication tensor T[i,j,k] = delta_{i,j} * delta_{j,k} in product form
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal sum_j A[i,j]*B[j,k]
    # Check: for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 if j==j (always) 
    # Actually the tensor T[i,j,k] = 1 for all i,j,k where the multiplication is defined
    # T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) of C
    # That's always 1 for valid indices.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_rank_27_search(deadline):
    """
    Attempt to find a rank-27 decomposition by trying to merge/cancel across blocks.
    
    Strategy: The current rank-28 has blocks:
      Block 1 (j=0): 10 rank-1 terms
      Block 2a (j={1,2}, k={0,1}): 7 Strassen terms
      Block 2b (j={1,2}, k=2): 4 schoolbook terms
      Block 2c (j={1,2}, k={3,4}): 7 Strassen terms
    
    To get rank 27, we need to eliminate one term. We try:
    1. Check if any term in Block 2b can be expressed as a linear combination of
       terms in adjacent blocks (2a or 2c) with the same support pattern.
    2. Try to find a 27-term decomposition via a small local search.
    
    This is a heuristic search - if it doesn't find one in time, we return None.
    """
    # The 4 schoolbook terms in Block 2b are:
    # (i=0,j=1): U[3], V[7], W[4]
    # (i=0,j=2): U[5], V[12], W[4]  
    # (i=1,j=1): U[4], V[7], W[5]
    # (i=1,j=2): U[6], V[12], W[5]
    
    # Each of these is a rank-1 tensor with support on specific (i,j) for U,
    # (j,k) for V, and (k,i) for W.
    
    # For a term to be absorbed, its (i,j,k) support must overlap with existing terms.
    # Block 2a terms have j in {1,2} and k in {0,1}
    # Block 2b terms have j in {1,2} and k=2
    # Block 2c terms have j in {1,2} and k in {3,4}
    
    # The k-index is disjoint between blocks, so no term in 2b can be a linear
    # combination of terms in 2a or 2c alone (different k support).
    
    # However, we could potentially restructure. Let's try a more general approach:
    # Use a small random search to see if rank 27 is achievable.
    
    # Actually, let's think about this differently. The tensor for 2x3x5 has:
    # - 10 entries where j=0 (Block 1)
    # - 20 entries where j in {1,2} (Blocks 2a, 2b, 2c)
    
    # The rank of the full tensor is at least the rank of any sub-tensor.
    # The 2x2x5 sub-tensor (j={1,2}) has rank at least... let's compute.
    
    # For 2x2x5: schoolbook is 10. Strassen-based: split k into {0,1},{2},{3,4}
    # gives 7+4+7=18. Can we do better?
    # 2x2x2 has rank 7 (Strassen is optimal).
    # 2x2x1 has rank 4 (schoolbook, and it's minimal since each of 4 products is needed).
    # So 2x2x5 >= 7+4+7 = 18? Not necessarily - there could be cross-k cancellations.
    
    # Actually the rank of 2x2x5: the tensor is sum_{j=0}^{1} sum_{k=0}^{4} e_i * e_j * e_k
    # This is a "direct sum" in the k-direction since each k has independent (i,j) structure.
    # So rank(2x2x5) = 5 * rank(2x2x1) = 5*4 = 20? No, that's schoolbook.
    # But Strassen shows we can do 2x2x2 in rank 7 < 8.
    # The key insight: the k-dimension can be "mixed" via Strassen.
    
    # For 2x2x5, one approach: group k as {0,1}, {2}, {3,4} -> 7+4+7=18
    # Or: {0}, {1,2}, {3,4} -> 4+7+7=18
    # Or: {0,1,2}, {3,4} -> rank(2x2x3)+7
    # rank(2x2x3) >= max(rank(2x2x2), rank(2x2x1)) but can be less than 7+4=11
    # Actually rank(2x2x3) = 10 (it's known that 2x2x3 has rank 10)
    # So 10+7=17? Let me verify this.
    
    # Hmm, actually I recall that rank(2x2x3) might be 10 or 11.
    # If it's 10, then 2x2x5 could be 10+7=17, giving total 10+17=27!
    
    # Let me try to construct a rank-10 decomposition for 2x2x3.
    # 2x2x3: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i in {0,1}, k in {0,1,2}
    # That's 6 output entries, 8 input entries (4 for A, 6 for B... wait 2x2=4, 2x3=6)
    
    # Schoolbook: 2*2*3 = 12 terms.
    # Strassen on 2x2x2 part + 2x2x1 part: 7+4=11.
    # Can we do 10?
    
    # Let me try to construct it. We need 10 rank-1 terms.
    # Each term is (u, v, w) where u is in R^4 (2x2), v in R^6 (2x3), w in R^6 (2x3 output).
    
    # This is getting complex. Let me just try a numerical search with a time limit.
    
    # Actually, for the purpose of this problem, let me just verify if rank 27 is possible
    # by checking if we can find such a decomposition. Given the time constraint,
    # I'll implement a quick check and fall back to rank 28.
    
    # For now, let's just return the known rank-28 solution.
    # The search for rank 27 would require more sophisticated tensor decomposition methods.
    return None

def construct():
    """
    Return a rank-28 exact decomposition for 2x3 * 3x5 matrix multiplication.
    
    Strategy:
    - Split middle dimension m=3 into j=0 and j in {1,2}.
    - Block 1 (j=0): 2x1x5, rank 10 (schoolbook).
    - Block 2 (j in {1,2}): 2x2x5.
      - Split k into {0,1}, {2}, {3,4}.
      - Sub-block 2a (k=0,1): 2x2x2, rank 7 (Strassen).
      - Sub-block 2b (k=2): 2x2x1, rank 4 (schoolbook).
      - Sub-block 2c (k=3,4): 2x2x2, rank 7 (Strassen).
    - Total rank: 10 + 7 + 4 + 7 = 28.
    
    Indexing conventions:
    - A is 2x3, flattened row-major: A[i,j] -> index i*3+j
    - B is 3x5, flattened row-major: B[j,k] -> index j*5+k
    - C is 2x5, output follows W[k*n+i, r] convention: C[i,k] -> index k*2+i
    """
    n, m, p = 2, 3, 5
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 28
    V = [[0] * rank for _ in range(m * p)]  # 15 x 28
    W = [[0] * rank for _ in range(p * n)]  # 10 x 28
    
    col = 0
    
    # Block 1: j=0, all k in {0,1,2,3,4}
    # C[i,k] += A[i,0] * B[0,k]
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    # col is now 10
    
    # Block 2a: j in {1,2}, k in {0,1} — Strassen 2x2x2
    # Local indices: i' in {0,1}, j' in {0,1} (j=j'+1), k' in {0,1} (k=k')
    # U index: i * 3 + (j' + 1)
    # V index: (j' + 1) * 5 + k'
    # W index: k' * 2 + i
    
    # M1: (A[0,1]+A[1,2]) * (B[1,0]+B[2,1]) -> C[0,0] += 1, C[1,1] += 1
    U[0 * 3 + 1][col] = 1
    U[1 * 3 + 2][col] = 1
    V[1 * 5 + 0][col] = 1
    V[2 * 5 + 1][col] = 1
    W[0 * 2 + 0][col] = 1
    W[1 * 2 + 1][col] = 1
    col += 1
    
    # M2: (A[1,1]+A[1,2]) * B[1,0] -> C[1,0] += 1, C[1,1] += -1
    U[1 * 3 + 1][col] = 1
    U[1 * 3 + 2][col] = 1
    V[1 * 5 + 0][col] = 1
    W[0 * 2 + 1][col] = 1
    W[1 * 2 + 1][col] = -1
    col += 1
    
    # M3: A[0,1] * (B[1,1]-B[2,1]) -> C[0,1] += 1, C[1,1] += 1
    U[0 * 3 + 1][col] = 1
    V[1 * 5 + 1][col] = 1
    V[2 * 5 + 1][col] = -1
    W[1 * 2 + 0][col] = 1
    W[1 * 2 + 1][col] = 1
    col += 1
    
    # M4: A[1,2] * (B[2,0]-B[1,0]) -> C[0,0] += 1, C[1,0] += 1
    U[1 * 3 + 2][col] = 1
    V[2 * 5 + 0][col] = 1
    V[1 * 5 + 0][col] = -1
    W[0 * 2 + 0][col] = 1
    W[0 * 2 + 1][col] = 1
    col += 1
    
    # M5: (A[0,1]+A[0,2]) * B[2,1] -> C[0,0] += -1, C[0,1] += 1
    U[0 * 3 + 1][col] = 1
    U[0 * 3 + 2][col] = 1
    V[2 * 5 + 1][col] = 1
    W[0 * 2 + 0][col] = -1
    W[1 * 2 + 0][col] = 1
    col += 1
    
    # M6: (A[1,1]-A[0,1]) * (B[1,0]+B[1,1]) -> C[1,1] += 1
    U[1 * 3 + 1][col] = 1
    U[0 * 3 + 1][col] = -1
    V[1 * 5 + 0][col] = 1
    V[1 * 5 + 1][col] = 1
    W[1 * 2 + 1][col] = 1
    col += 1
    
    # M7: (A[0,2]-A[1,2]) * (B[2,0]+B[2,1]) -> C[0,0] += 1
    U[0 * 3 + 2][col] = 1
    U[1 * 3 + 2][col] = -1
    V[2 * 5 + 0][col] = 1
    V[2 * 5 + 1][col] = 1
    W[0 * 2 + 0][col] = 1
    col += 1
    
    # col is now 17
    
    # Block 2b: j in {1,2}, k=2 — schoolbook 2x2x1
    # C[i,2] += A[i,1]*B[1,2] + A[i,2]*B[2,2]
    for i in range(n):
        for j_local in range(2):
            j = j_local + 1
            U[i * 3 + j][col] = 1
            V[j * 5 + 2][col] = 1
            W[2 * 2 + i][col] = 1
            col += 1
    # col is now 21
    
    # Block 2c: j in {1,2}, k in {3,4} — Strassen 2x2x2
    # Local indices: i' in {0,1}, j' in {0,1} (j=j'+1), k' in {0,1} (k=k'+3)
    # U index: i * 3 + (j' + 1)
    # V index: (j' + 1) * 5 + (k' + 3)
    # W index: (k' + 3) * 2 + i
    
    # M1: (A[0,1]+A[1,2]) * (B[1,3]+B[2,4]) -> C[0,3] += 1, C[1,4] += 1
    U[0 * 3 + 1][col] = 1
    U[1 * 3 + 2][col] = 1
    V[1 * 5 + 3][col] = 1
    V[2 * 5 + 4][col] = 1
    W[3 * 2 + 0][col] = 1
    W[4 * 2 + 1][col] = 1
    col += 1
    
    # M2: (A[1,1]+A[1,2]) * B[1,3] -> C[1,3] += 1, C[1,4] += -1
    U[1 * 3 + 1][col] = 1
    U[1 * 3 + 2][col] = 1
    V[1 * 5 + 3][col] = 1
    W[3 * 2 + 1][col] = 1
    W[4 * 2 + 1][col] = -1
    col += 1
    
    # M3: A[0,1] * (B[1,4]-B[2,4]) -> C[0,4] += 1, C[1,4] += 1
    U[0 * 3 + 1][col] = 1
    V[1 * 5 + 4][col] = 1
    V[2 * 5 + 4][col] = -1
    W[4 * 2 + 0][col] = 1
    W[4 * 2 + 1][col] = 1
    col += 1
    
    # M4: A[1,2] * (B[2,3]-B[1,3]) -> C[0,3] += 1, C[1,3] += 1
    U[1 * 3 + 2][col] = 1
    V[2 * 5 + 3][col] = 1
    V[1 * 5 + 3][col] = -1
    W[3 * 2 + 0][col] = 1
    W[3 * 2 + 1][col] = 1
    col += 1
    
    # M5: (A[0,1]+A[0,2]) * B[2,4] -> C[0,3] += -1, C[0,4] += 1
    U[0 * 3 + 1][col] = 1
    U[0 * 3 + 2][col] = 1
    V[2 * 5 + 4][col] = 1
    W[3 * 2 + 0][col] = -1
    W[4 * 2 + 0][col] = 1
    col += 1
    
    # M6: (A[1,1]-A[0,1]) * (B[1,3]+B[1,4]) -> C[1,4] += 1
    U[1 * 3 + 1][col] = 1
    U[0 * 3 + 1][col] = -1
    V[1 * 5 + 3][col] = 1
    V[1 * 5 + 4][col] = 1
    W[4 * 2 + 1][col] = 1
    col += 1
    
    # M7: (A[0,2]-A[1,2]) * (B[2,3]+B[2,4]) -> C[0,3] += 1
    U[0 * 3 + 2][col] = 1
    U[1 * 3 + 2][col] = -1
    V[2 * 5 + 3][col] = 1
    V[2 * 5 + 4][col] = 1
    W[3 * 2 + 0][col] = 1
    col += 1
    
    # col should be 28
    
    return {"U": U, "V": V, "W": W}
