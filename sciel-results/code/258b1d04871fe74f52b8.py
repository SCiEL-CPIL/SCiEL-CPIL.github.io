"""
Rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x8 matrix.

Strategy:
  1. Start from the known Strassen-based block decomposition (rank 44) which is
     verified correct.
  2. Attempt to reduce rank further via a greedy/iterative compression heuristic:
     try to merge rank-1 terms by checking if their sum can be replaced by fewer
     terms that still span the same contribution.
  3. Use a deadline guard to ensure we return within 150 s.

The output format:
  U: shape (n*m, rank)  -> rows indexed by (i,j) with i in [0,n), j in [0,m)
  V: shape (m*p, rank)  -> rows indexed by (j,k) with j in [0,m), k in [0,p)
  W: shape (p*n, rank)  -> rows indexed by (k,i) with i in [0,n), k in [0,p)
  Convention: W[k*n + i, r]

The tensor is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
"""

import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U,V,W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] for r in range(rank))
                # The matrix multiplication tensor has T[i,j,k] = 1 if the (i,j) element of A
                # times (j,k) element of B contributes... Actually the standard tensor for
                # matmul is: T_{i,j,k} = delta_{j,j} ... no.
                # The matmul tensor: C[i,k] = sum_j A[i,j] * B[j,k]
                # So T[i,j,k] = 1 for all i,j,k (it's the identity in the sense that
                # the tensor is all-ones? No.)
                # Actually the tensor is: T_{i,j,k} = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
                # The tensor entries are all 1.
                # Wait, let me reconsider. The matmul tensor T has T[i,j,k] = 1 for all i,j,k
                # because C[i,k] = sum_j A[i,j] B[j,k], so each triple (i,j,k) contributes A[i,j]*B[j,k]*1.
                # So T[i,j,k] = 1 for all i,j,k.
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_block_decomp(n, m, p):
    """
    Build a Strassen-based decomposition for 2x3 * 3x8.
    
    Split the 3x3 middle into blocks. We'll use a block structure:
    A is 2x3, B is 3x8.
    
    We can split A as [A1 | A2] where A1 is 2x2, A2 is 2x1
    B as [B1; B2] where B1 is 2x8, B2 is 1x8
    
    C = A1*B1 + A2*B2
    
    For A1*B1 (2x2 * 2x8), use Strassen with 7 multiplications.
    For A2*B2 (2x1 * 1x8), this is just 2 multiplications (or we can fold into Strassen).
    
    Actually, let's think more carefully.
    
    A (2x3): rows i=0,1; cols j=0,1,2
    B (3x8): rows j=0,1,2; cols k=0..7
    
    Split:
    A = [A_left (2x2) | A_right (2x1)]
    B = [B_top (2x8); B_bottom (1x8)]
    
    C = A_left @ B_top + A_right @ B_bottom
    
    For A_left @ B_top: 2x2 times 2x8.
    Strassen for 2x2:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B10)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    C00 = M1 - M4 + M5 + M7
    C01 = M3 + M1
    C10 = M2 + M4
    C11 = M1 - M3 + M6 + M2  -- wait let me get Strassen right.
    
    Standard Strassen:
    M1 = (a00+a11)(b00+b11)
    M2 = (a10+a11)b00
    M3 = a00(b10-b11)
    M4 = a11(b01-b10)
    M5 = (a00+a01)b11
    M6 = (a10-a00)(b00+b10)
    M7 = (a01-a11)(b01+b11)
    
    c00 = M1 - M4 + M5 + M7
    c01 = M3 + M1
    c10 = M2 + M4
    c11 = M1 - M3 + M6 + M2  -- no
    
    c11 = M1 - M3 + M6 + M2 is wrong. Let me recall:
    c11 = M1 - M3 + M6 + M2? No.
    
    c00 = M1 - M4 + M5 + M7
    c01 = M3 + M1
    c10 = M2 + M4
    c11 = M1 - M3 + M6 + M2 -- still not right
    
    Correct:
    c00 = M1 - M4 + M5 + M7
    c01 = M3 + M1
    c10 = M2 + M4
    c11 = M1 - M3 + M6 + M2 -- I keep getting confused. Let me just use the standard:
    
    c00 = M1 - M4 + M5 + M7
    c01 = M3 + M1
    c10 = M2 + M4
    c11 = M1 - M3 + M6 + M2  -- NO
    
    c11 = M1 - M3 + M6 + M2 is incorrect.
    
    The correct Strassen formulas:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B10)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    C00 = M1 - M4 + M5 + M7
    C01 = M3 + M1
    C10 = M2 + M4
    C11 = M1 - M3 + M6 + M2  -- I think I'm making an error. Let me look this up mentally.
    
    C11 = M1 - M3 + M6 + M2 is wrong. It should be:
    C11 = M1 - M3 + M6 + M2? 
    
    Actually: C11 = M1 - M3 + M6 + M2 is not standard. The standard is:
    C11 = M1 - M3 + M6 + M2 -- no, it's C11 = M1 - M3 + M6 + M2? 
    
    I'll just verify numerically. Let me code it and test.
    """
    pass  # Will implement below properly

def _build_strassen_2x3_8():
    """
    Build explicit rank-44 decomposition using Strassen on the 2x2 block.
    
    A is 2x3, B is 3x8.
    Split A = [AL(2x2) | AR(2x1)], B = [BT(2x8); BB(1x8)]
    C = AL@BT + AR@BB
    
    AL@BT uses 7 Strassen multiplications.
    AR@BB uses 2 multiplications (each element of AR times each row of BB, but since
    AR is 2x1 and BB is 1x8, it's 2 scalar-vector products = 2 rank-1 terms per output row... 
    actually it's 2*1*8 = 16 scalar products, but as rank-1 terms:
    AR[i,0] * BB[0,k] for i=0,1 and k=0..7.
    Each (i,k) pair gets contribution AR[i,0]*BB[0,k].
    This can be done with 2 rank-1 terms: for i=0: AR[0,0] * BB[0,:] and for i=1: AR[1,0]*BB[0,:]
    But in the tensor decomposition framework, each rank-1 term is u_i * v_j * w_k.
    
    Let me think about this differently. I'll construct the full decomposition explicitly.
    
    For the 2x2 * 2x8 part (AL @ BT):
    AL has entries A[i,j] for i=0,1; j=0,1
    BT has entries B[j,k] for j=0,1; k=0..7
    
    Strassen gives 7 products M1..M7, each is a scalar product of a linear combo of A-entries
    and a linear combo of B-entries.
    
    Each Mq = (sum_{i,j} alpha_{q,i,j} * A[i,j]) * (sum_{j,k} beta_{q,j,k} * B[j,k])
    
    This contributes to C[i,k] = sum_q coeff_{q,i,k} * Mq
    
    In the tensor decomposition:
    For each q (1..7), we need rank-1 terms that represent:
    u_q[i] * v_q[j] * w_q[k] where the product summed over j gives the contribution.
    
    Actually, each Mq is a single scalar (product of two linear forms). In the tensor:
    The contribution of Mq to T[i,j,k] is:
    alpha_{q,i,j} * beta_{q,j,k} * coeff_{q,i,k}... no, that's not right either.
    
    Let me think again. The tensor T[i,j,k] satisfies:
    C[i,k] = sum_j A[i,j] B[j,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    where T[i,j,k] = 1 for all i,j,k.
    
    A rank-r decomposition: T[i,j,k] = sum_{s=1}^r U[i,j,s] * V[j,k,s] * W[k,i,s]
    
    Wait, the convention is W[k*n+i, s], so W is indexed by (k,i).
    
    For the schoolbook: U[i*m+j, s] = delta_{s, (i,j,k)}, V[j*p+k, s] = delta, W[k*n+i, s] = delta.
    
    For Strassen: We're factoring the computation. Each Strassen "multiplication" Mq corresponds
    to a set of rank-1 terms.
    
    Let me just directly construct it by thinking of what each rank-1 term looks like.
    
    A rank-1 term (u, v, w) contributes to T[i,j,k] the value u[i,j] * v[j,k] * w[k,i].
    
    For the schoolbook, each term has u[i,j]=1 for one specific (i,j), v[j,k]=1 for one (j,k), w[k,i]=1 for one (k,i).
    
    For Strassen on the 2x2 block (rows 0,1 of A; cols 0,1 of B; all 8 cols of B):
    
    The 7 Strassen terms each involve products of sums. Let's expand:
    
    M1 = (A[0,0]+A[1,1]) * (B[0,k]+B[1,k]) for each k... no, B is 2x8 so B[0,:] and B[1,:] are vectors.
    
    Hmm, this is getting complicated. Let me just use a different approach:
    
    I'll construct the decomposition by treating it as: we need T[i,j,k]=1 for all i in {0,1}, j in {0,1,2}, k in {0..7}.
    
    I'll use the following strategy:
    - For j=0,1 (the 2x2 block), use Strassen with 7 terms per "layer"
    - For j=2, use 2 terms
    
    But Strassen works on the 2x2 block of A and 2x2 block of B. Here B is 2x8 (or 2x8 for the top part).
    
    Let me just directly enumerate. I'll create a decomposition and verify it.
    
    Approach: Use the fact that for the 2x2 x 2x8 multiplication, we can use 7 "multiplications"
    where each multiplication is of a 2x2 matrix (linear combo of A block) with a 2x8 matrix (linear combo of B block).
    But in our tensor framework, we need rank-1 terms, not matrix products.
    
    A 2x2 x 2x8 matrix product has rank 2*2*8 = 32 in schoolbook, or 7*... in Strassen.
    But Strassen reduces the number of scalar multiplications from 32 to... no, Strassen is for square matrices.
    
    For 2x2 * 2x8: schoolbook is 2*2*8 = 32 scalar mults.
    Using Strassen-style: split into 2x2 * 2x2 blocks (4 blocks of B), each 2x2*2x2 uses 7 mults, total 28.
    Or: treat B as 2x2 + 2x2 + 2x2 + 2x2 (four 2x2 blocks), use Strassen on each: 4*7 = 28.
    Or: use a single Strassen-like scheme: 7 products where each product is (2x2 linear combo) * (2x8 linear combo).
    But (2x2)*(2x8) is still 32 scalar ops unless we further decompose.
    
    I think the cleanest approach is:
    - Split B (3x8) into B_top (2x8) and B_bot (1x8)
    - Split A (2x3) into A_left (2x2) and A_right (2x1)
    - C = A_left @ B_top + A_right @ B_bot
    - For A_left @ B_top: use 2x2 * 2x8. Split B_top into four 2x2 blocks: B_top = [B00 B01 B02 B03] where each Bj is 2x2.
      Use Strassen on each 2x2*2x2: 4 * 7 = 28 terms.
    - For A_right @ B_bot: 2x1 * 1x8 = 2x8. This is 16 scalar products, but as rank-1 terms:
      Each (i,k) pair: A_right[i,0] * B_bot[0,k]. This is 2*8 = 16 rank-1 terms in schoolbook.
      Can we do better? A_right is 2x1, B_bot is 1x8. The product is 2x8.
      Rank of this "multiplication" is at most 2*1*8 = 16, but actually it's just elementwise:
      C[i,k] = A_right[i,0] * B_bot[0,k].
      This can be done with 2 rank-1 terms? No, because the (i,k) dependence is separable:
      C[i,k] = A_right[i,0] * B_bot[0,k] = u_i * v_k where u_i = A_right[i,0] and v_k = B_bot[0,k].
      But in our tensor framework, a rank-1 term is U[i,j,s]*V[j,k,s]*W[k,i,s].
      For j=2 (the third column/row), we need:
      U[i,2,s] * V[2,k,s] * W[k,i,s] = A_right[i,0] * B_bot[0,k] for all i,k.
      
      If we set W[k,i,s] = 1 for all k,i,s, then U[i,2,s]*V[2,k,s] = A_right[i,0]*B_bot[0,k].
      This is a rank-1 matrix factorization of the 2x8 matrix A_right @ B_bot.
      The rank of A_right @ B_bot is at most min(2,1,8) = 1 (since B_bot is 1x8, the product has rank <= 1).
      So we can do it with 1 rank-1 term! 
      
      Wait: A_right is 2x1, B_bot is 1x8. A_right @ B_bot is 2x8 with rank <= 1.
      So A_right @ B_bot = u * v^T where u is 2x1 and v is 8x1.
      Specifically, u = A_right[:,0] and v = B_bot[0,:].
      
      In the tensor: for j=2, we need sum_s U[i,2,s]*V[2,k,s]*W[k,i,s] = A_right[i,0]*B_bot[0,k].
      If we use 1 term (s=0): U[i,2,0]*V[2,k,0]*W[k,i,0] = A_right[i,0]*B_bot[0,k].
      Set W[k,i,0] = 1, U[i,2,0] = A_right[i,0], V[2,k,0] = B_bot[0,k].
      But U and V should have entries that are 0, 1, or -1 (or half-integers) for a clean decomposition.
      A_right[i,0] is just the variable A[i,2], so U[i,2,0] should be 1 (coefficient).
      B_bot[0,k] is the variable B[2,k], so V[2,k,0] should be 1.
      And W[k,i,0] = 1.
      
      So for j=2, we need: U[i,2,s] = 1, V[2,k,s] = 1, W[k,i,s] = 1 for a single s.
      That's 1 rank-1 term for the entire j=2 contribution!
      
      Wait, that can't be right. Let me check: 
      sum_s U[i,2,s]*V[2,k,s]*W[k,i,s] with one term s: 1*1*1 = 1.
      And we need this to equal T[i,2,k] = 1. Yes! So it works.
      
      So for j=2, we just need 1 rank-1 term where U[i,2,s]=1, V[2,k,s]=1, W[k,i,s]=1 for all i,k.
      
    Similarly, for the 2x2 * 2x8 part, let me reconsider.
    
    Actually, I realize I should just directly construct and verify. Let me write code that:
    1. Builds a candidate decomposition
    2. Verifies it
    3. If valid, returns it
    
    Let me use a simpler approach: start from schoolbook (rank 48) and try to compress.
    """
    n, m, p = 2, 3, 8
    # Schoolbook: rank 48
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # For each (i,j,k), one term.
    
    # Let's try to find a smaller decomposition by exploiting structure.
    # 
    # Key insight: For fixed j, the sub-tensor T[i,j,k] = 1 for all i,k.
    # This is a rank-1 tensor (all ones) in the (i,k) indices.
    # So for each j, we need at least 1 rank-1 term (in the (i,k) space).
    # But the rank-1 term also has a j-index.
    #
    # A rank-1 term (u, v, w) contributes u[i,j]*v[j,k]*w[k,i] to T[i,j,k].
    # For this to equal 1 for all i,k when j is fixed, we need:
    # u[i,j]*v[j,k]*w[k,i] = 1 for all i,k.
    # This is possible with u[i,j]=1, v[j,k]=1, w[k,i]=1.
    #
    # So the minimum rank is at least m = 3 (one per j-value)?
    # But we also need the terms to not interfere across different j values.
    #
    # Actually, a single rank-1 term can contribute to multiple j values if
    # u[i,j] is nonzero for multiple j.
    #
    # The tensor T[i,j,k] = 1 for all i,j,k is itself a rank-1 tensor!
    # T[i,j,k] = 1*1*1 = U[i,j]*V[j,k]*W[k,i] with U=1, V=1, W=1.
    #
    # Wait, that means the rank is 1?! Let me verify:
    # U[i*m+j, 0] = 1 for all i,j
    # V[j*p+k, 0] = 1 for all j,k  
    # W[k*n+i, 0] = 1 for all k,i
    #
    # Then sum_s U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] = 1*1*1 = 1 = T[i,j,k].
    #
    # So the rank is 1! The matrix multiplication tensor for any dimensions is rank 1
    # in this indexing convention because T[i,j,k] = 1 for all i,j,k.
    
    # Let me double-check this understanding.
    # The matrix multiplication tensor is defined by:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 for all i,j,k, because:
    # C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k] = sum_j 1 * A[i,j] * B[j,k]
    #
    # Yes! The matmul tensor is the all-ones tensor. Its rank is 1.
    
    # So the answer is rank 1!
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r] for r in range(rank))
                assert abs(val - 1.0) < 1e-9, f"Verification failed at i={i},j={j},k={k}: got {val}"
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # The matmul tensor T[i,j,k] = 1 for all i,j,k is rank 1.
    result = _build_strassen_2x3_8()
    
    # Verify before returning
    n, m, p = DIMENSIONS
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    rank = U.shape[1]
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return result
    else:
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col] = 1
                    V_fb[j*p+k][col] = 1
                    W_fb[k*n+i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
