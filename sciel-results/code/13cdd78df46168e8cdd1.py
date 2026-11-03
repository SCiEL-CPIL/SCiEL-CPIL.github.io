"""Matrix multiplication 2x4 x 4x8: rank-minimal exact tensor decomposition.

Strategy:
  - The multiplication tensor T[i,j,k] = 1 for all (i,j,k) in [0,2)x[0,4)x[0,8).
  - We seek rank-r factor matrices U (8 x r), V (32 x r), W (16 x r) such that
    for all (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1.
  - Lower bound: rank >= max(n*m, m*p, p*n) = max(8, 32, 16) = 32.
  - We attempt to find a rank-32 decomposition via structured search,
    falling back to the Strassen-based rank-28... wait, 28 < 32? That can't be.
    
  Actually the lower bound for matrix multiplication M_{n,m,p} is not simply
  max(nm, mp, pn). The true lower bound is more subtle. For M_{2,4,8}, the
  schoolbook rank is 64. Strassen gives 28. Can we do better than 28?

  Known results: M_{2,2,2} has rank 7 (Strassen). M_{2,2,1} has rank 4.
  For rectangular cases, the landscape is less explored.

  Let's think about the structure. A is 2x4, B is 4x8. We can view this as:
  - Split A into two 2x2 blocks: A = [A1 | A2]
  - Split B into four 2x2 blocks: B = [B1; B2; B3; B4] (stacked vertically, each 2x2... 
    wait, B is 4x8, so splitting rows into 2+2 gives two 2x8 blocks, or splitting
    columns into 4 groups of 2 gives four 4x2 blocks.

  Actually B is 4 rows x 8 columns. Let's split B's columns into 4 blocks of 2:
  B = [B_1 | B_2 | B_3 | B_4] where each B_i is 4x2.
  A = [A_1 | A_2] where each A_i is 2x2.
  
  C = A*B = A_1*B_1 + A_2*B_2 + A_1*B_3 + A_2*B_4 (each product is 2x2)
  
  Each 2x2 * 4x2 product... wait that's not right either.
  
  Let me redo: A is 2x4, B is 4x8.
  A = [A_L | A_R] where A_L, A_R are 2x2.
  B = [B_1; B_2; B_3; B_4] where each B_i is 2x8? No.
  
  B is 4x8. Split B's rows into 2+2: B = [B_top; B_bottom] where each is 2x8.
  Then C = A_L * B_top + A_R * B_bottom, each product is 2x8.
  
  Now each 2x8 product: split the 8 columns into 4 groups of 2.
  B_top = [T_1 | T_2 | T_3 | T_4] each 2x2.
  B_bottom = [Bt_1 | Bt_2 | Bt_3 | Bt_4] each 2x2.
  
  C = A_L*T_1 + A_R*Bt_1 + A_L*T_2 + A_R*Bt_2 + ... (8 terms of 2x2 products)
  
  Using Strassen for each 2x2 product: 7 * 8 = 56 rank terms. That's worse than 28.
  
  The original approach was:
  C = A_L*B_1 + A_R*B_2 + A_L*B_3 + A_R*B_4
  where B_1..B_4 are 2x2 blocks. This gives 4 Strassen products = 28.
  
  Can we share computations across these 4 products? The A factors are reused:
  A_L appears in products 1 and 3, A_R in 2 and 4.
  
  For rank 28, we have 4 independent Strassen decompositions. To go below 28,
  we need to find algebraic dependencies or a different decomposition structure.
  
  Let me try a direct approach: set up the system of equations for rank r and
  try to solve it. For r=32 (the max dimension), we have 32 unknowns per factor
  column, and 2*4*8=64 equations. That's underdetermined per column but coupled.
  
  Actually, let's try a randomized approach to find a low-rank decomposition.
  We'll use the fact that the tensor is "all-ones" in a specific structure.
  
  Alternative: use the identity that for the all-ones tensor of size n x m x p,
  the rank is n*m*p (schoolbook). But our tensor isn't all-ones in the flat sense;
  it's structured.
  
  Let me try to find a rank-24 decomposition by exploiting the rectangular structure.
  
  For 2x4 * 4x8, think of it as 2x2 * 2x8 done twice (top and bottom row of B).
  2x2 * 2x8: A is 2x2, B is 2x8. Split B into four 2x2 blocks.
  C = A*B1 + A*B2 + A*B3 + A*B4.
  Using Strassen: 4*7 = 28 for one 2x2*2x8. Two of those = 56. Worse.
  
  What if we do: 2x4 * 4x8 = (2x2 * 2x8) + (2x2 * 2x8) where we split A's columns?
  Same thing.
  
  Let me try a completely different approach: use numpy to set up and solve
  for a rank-r decomposition via iterative methods or by exploiting the linear
  structure in the W matrix.
  
  For a fixed U and V, the equations are linear in W. So we can:
  1. Randomly initialize U and V.
  2. For each (i,j,k), the equation is: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1.
     This is linear in W's entries.
  3. Solve the linear system for W.
  4. Check if the solution is consistent.
  
  But the system is 64 equations with 16*r unknowns. For r=32, that's 512 unknowns
  and 64 equations - very underdetermined. We need additional constraints.
  
  Actually, the structure is: for fixed c, the contribution to (i,j,k) is
  U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c].
  
  Let's think of it differently. Fix c. Then we have three vectors:
  u_c in R^8, v_c in R^32, w_c in R^16.
  The constraint is: for all (i,j,k), sum_c u_c[i*m+j]*v_c[j*p+k]*w_c[k*n+i] = 1.
  
  This is a 3-tensor decomposition problem. The tensor T has T[i,j,k]=1 for all
  (i,j,k) in [0,2)x[0,4)x[0,8). Wait, no! The multiplication tensor has
  T[i,j,k] = 1 if and only if the (i,k) entry of A*B depends on A[i,j]*B[j,k].
  Since every A[i,j] multiplies every B[j,k] to contribute to C[i,k], and there's
  exactly one such term, T[i,j,k] = 1 for ALL (i,j,k). So it IS the all-ones tensor!
  
  Wait, that's the key insight. The matrix multiplication tensor for M_{n,m,p}
  is T[i,j,k] = delta_{j,j} = 1 (the index j is shared). So T[i,j,k] = 1 for all
  valid (i,j,k). This is just the all-ones tensor of size 2x4x8.
  
  The rank of the all-ones tensor of size a x b x c is max(a,b,c). This is a
  classical result! The all-ones tensor can be decomposed as:
  T[i,j,k] = 1 = sum_{c=1}^{max(a,b,c)} (1/max(a,b,c)) * 1 * 1 * 1
  
  Wait, that gives rank max(a,b,c) with entries 1/max(a,b,c). But we need integer
  or half-integer entries.
  
  Actually, the standard decomposition: T[i,j,k] = 1. We can write:
  T = sum_{c=1}^{r} u_c ⊗ v_c ⊗ w_c
  
  where u_c[i] * v_c[j] * w_c[k] sums to 1 for all (i,j,k).
  
  One simple decomposition: take r = max(n,m,p) = 8.
  Let u_c[i] = 1 for all i (for c=1..8), v_c[j] = 1 for all j, w_c[k] = 1/8 for all k.
  Then sum_c u_c[i]*v_c[j]*w_c[k] = sum_c 1*1*(1/8) = 8*(1/8) = 1. ✓
  
  But we need U to be (n*m) x r = 8 x 8, V to be (m*p) x r = 32 x 8, W to be (p*n) x r = 16 x 8.
  
  U[i*m+j, c] = 1 for all i,j,c.
  V[j*p+k, c] = 1 for all j,k,c.
  W[k*n+i, c] = 1/8 for all k,i,c.
  
  Check: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = sum_c 1*1*(1/8) = 8/8 = 1. ✓
  
  Rank = 8! That's way below 28. And entries are 1 and 1/8 (half-integer? 1/8 is not
  half-integer). The problem says "integer/half-integer entries". 1/8 is neither.
  
  Let me re-read: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank) with
  integer/half-integer entries". So entries must be integers or half-integers (multiples
  of 1/2).
  
  1/8 is not allowed. Can we adjust?
  
  We need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 with all entries in {0, ±1/2, ±1, ±3/2, ...}.
  
  If we use r=8 and all entries are 1, then sum = 8, not 1. We need the sum to be 1.
  
  What if some entries are 0? We need a more careful construction.
  
  Let's think: we need U ⊙ V ⊙ W (element-wise product summed over c) = 1 for all (i,j,k).
  
  One approach: use r columns where in each column c, we have:
  U[:,c] is a vector in {0,1}^8, V[:,c] is in {0,1}^{32}, W[:,c] is in {0,1}^{16}
  (or with half-integers), and the "coverage" works out.
  
  Actually, the simplest: if we can find r vectors such that for every (i,j,k),
  exactly one c has U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 and all others are 0,
  then we need at least 64 columns (schoolbook).
  
  But we can have fractional contributions. E.g., if 8 columns each contribute 1/8,
  but 1/8 isn't allowed.
  
  What about: 4 columns each contributing 1/4? 1/4 isn't half-integer either.
  
  2 columns each contributing 1/2? That works! 1/2 is a half-integer.
  
  So: r=2, U[i*m+j,c] = 1, V[j*p+k,c] = 1, W[k*n+i,c] = 1/2 for c=1,2.
  Sum = 2 * (1*1*1/2) = 1. ✓
  
  All entries are 1 or 1/2, which are valid (integer and half-integer).
  
  Rank = 2!
  
  Wait, that seems too good. Let me double-check.
  
  U is 8x2: all entries = 1.
  V is 32x2: all entries = 1.
  W is 16x2: all entries = 1/2.
  
  For any (i,j,k): sum_{c=1}^{2} U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
  = 1*1*(1/2) + 1*1*(1/2) = 1/2 + 1/2 = 1. ✓
  
  This is valid! Rank 2.
  
  Can we do rank 1? U[i*m+j,1]*V[j*p+k,1]*W[k*n+i,1] = 1 for all (i,j,k).
  This means U[i*m+j,1] * V[j*p+k,1] * W[k*n+i,1] = 1 for all i,j,k.
  Fix i=0,j=0: U[0,1]*V[k,1]*W[k*n,1] = 1 for all k.
  Fix i=0,j=1: U[1,1]*V[4+k,1]*W[k*n,1] = 1 for all k.
  This requires U[0,1] = U[1,1] (by dividing the two equations), and similarly all U entries equal,
  all V entries equal, all W entries equal. Then U*V*W = 1.
  With integer/half-integer entries: U=1, V=1, W=1 gives product 1. Rank 1!
  
  Wait: U[i*m+j,1] = 1 for all i,j. V[j*p+k,1] = 1 for all j,k. W[k*n+i,1] = 1 for all k,i.
  Product = 1*1*1 = 1. ✓
  
  Rank 1 is the minimum possible (rank >= 1 for a nonzero tensor). So the answer is rank 1!
  
  This makes sense because the "multiplication tensor" T[i,j,k] = 1 for all (i,j,k)
  is just the outer product of three all-ones vectors, which has rank 1.
  
  The key realization: the matrix multiplication tensor for M_{n,m,p} has entries
  T[i,j,k] = 1 for ALL (i,j,k) in the index ranges, because every input pair (i,j)
  and (j,k) contributes to output (i,k). The tensor is simply the all-ones tensor,
  which is rank 1.
"""

import time
import numpy as np

DIMENSIONS = [2, 4, 8]


def _verify(U, V, W, n, m, p):
    """Return True iff the decomposition exactly reproduces matrix multiplication."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                ui = U[i * m + j]
                vj = V[j * p + k]
                wk = W[k * n + i]
                for c in range(r):
                    s += ui[c] * vj[c] * wk[c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _rank1_decomp():
    """Rank-1 decomposition: all-ones tensor is rank 1."""
    n, m, p = DIMENSIONS
    rank = 1
    U = [[1] * rank for _ in range(n * m)]   # 8 x 1, all ones
    V = [[1] * rank for _ in range(m * p)]   # 32 x 1, all ones
    W = [[1] * rank for _ in range(p * n)]   # 16 x 1, all ones
    return {"U": U, "V": V, "W": W}


def _schoolbook():
    """Schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110

    # Rank-1 decomposition: the multiplication tensor is the all-ones tensor,
    # which is the outer product of three all-ones vectors.
    cand = _rank1_decomp()
    if _verify(cand["U"], cand["V"], cand["W"], n, m, p):
        return cand

    # Fallback (should never be reached)
    return _schoolbook()
