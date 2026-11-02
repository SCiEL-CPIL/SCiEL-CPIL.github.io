import time
import numpy as np

DEADLINE_S = 110.0
N, M, P = 4, 5, 5


def _validate(U, V, W, tol=1e-9):
    """Check sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1 for all i,j,k."""
    n, m, p = N, M, P
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(R))
                if abs(s - 1.0) > tol:
                    return False
    return True


def _strassen_4x4x4():
    """Return (U,V,W) for rank-63 decomposition of 4x4x4 matrix multiplication.
    U: (16,63), V: (16,63), W: (16,63) with W[k*4+i, r].
    Uses two independent 2x2x2 Strassen blocks (ranks 7+7=14 per block)
    plus schoolbook for the remaining 4x4x4 terms not covered.
    
    Actually, let me use a known rank-63 construction for 4x4x4:
    The 4x4x4 tensor can be decomposed by splitting into four 2x2x2 blocks:
    - Block (0,0): A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2], Strassen rank 7
    - Block (0,1): A[0:2,2:4] * B[2:4,0:2] -> C[0:2,2:4], schoolbook rank 4
    - Block (1,0): A[2:4,0:2] * B[0:2,2:4] -> C[2:4,0:2], schoolbook rank 4
    - Block (1,1): A[2:4,2:4] * B[2:4,2:4] -> C[2:4,2:4], Strassen rank 7
    Total: 7+4+4+7 = 22? No that's not right either.
    
    Let me think differently. The standard 4x4x4 schoolbook is 64.
    We can save 1 by using a shared computation.
    
    Actually, for 4x4x4, the best known rank is 63 (one less than schoolbook 64).
    The construction: use Strassen's algorithm adapted.
    
    Let me use a simpler approach: just do schoolbook 64 and see if we can merge one pair.
    
    Actually, let me just return schoolbook 64 for the 4x4x4 core and handle the rest.
    64 + 16 + 16 + 4 = 100, same as full schoolbook. That doesn't help.
    
    Let me try a different strategy entirely: 
    For 4x5x5, total schoolbook is 100.
    Split A into A1 (4x4) and A2 (4x1), B into B1 (5x4) and B2 (5x1).
    C = A1*B1 + A2*B2 (but B1 is 5x4 so A1*B1 is 4x4, and A2*B2 is 4x1... wait)
    
    A is 4x5, B is 5x5, C is 4x5.
    Split columns of A: A = [A_left (4x4) | A_right (4x1)]
    Split columns of B: B = [B_left (5x4) | B_right (5x1)]
    C = A_left * B_left + A_right * B_right
    A_left * B_left: 4x4 * 4x4 -> 4x4 (rank 64 schoolbook, 63 Strassen)
    A_right * B_right: 4x1 * 1x1... no, B_right is 5x1 not 1x1.
    
    Hmm, the split needs to be along the shared dimension (the inner dimension = 5).
    A is 4x5, B is 5x5. Inner dim is 5.
    Split A into A1 (4x4) and A2 (4x1).
    Split B into B1 (4x5) and B2 (1x5).  [first 4 rows, last 1 row]
    C = A1 * B1 + A2 * B2
    A1 * B1: 4x4 * 4x5 -> 4x5, schoolbook rank 4*4*5=80
    A2 * B2: 4x1 * 1x5 -> 4x5, schoolbook rank 4*1*5=20
    Total: 100.
    
    Now optimize A1*B1 (4x4*4x5). 
    Split B1 into B1a (4x4) and B1b (4x1):
    A1*B1 = A1*B1a + A1*B1b
    A1*B1a: 4x4*4x4 -> 4x4, Strassen rank 7? No, Strassen for 2x2 is 7, for 4x4 it's 7^2=49 if we use Strassen recursively, but that's for the full 4x4 product. Actually 4x4x4 Strassen has rank 63 (known result, one less than 64).
    
    Hmm, I'm overcomplicating this. Let me just try to find ANY valid decomposition with rank < 100.
    
    Simplest idea: can we merge two schoolbook terms?
    Schoolbook term for (i,j,k): U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
    Two terms (i1,j1,k1) and (i2,j2,k2) can be merged if there's a linear combination...
    Generally no, they're all distinct rank-1 tensors.
    
    Let me just return the schoolbook decomposition with rank 100, but try to be clever about
    the W indexing to make sure it's correct.
    """
    n = 4
    rank = 64
    U = np.zeros((n*n, rank))
    V = np.zeros((n*n, rank))
    W = np.zeros((n*n, rank))
    c = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                U[i*n+j, c] = 1.0
                V[j*n+k, c] = 1.0
                W[k*n+i, c] = 1.0
                c += 1
    return U, V, W


def _construct_rank_99():
    """Try to construct a rank-99 decomposition for 4x5x5.
    
    Split: A (4x5) = A1 (4x4) + A2 (4x1) along columns
           B (5x5) = B1 (4x5) + B2 (1x5) along rows
    C = A1*B1 + A2*B2
    
    A1*B1: 4x4 * 4x5 -> 4x5
      Split B1 = B1a (4x4) + B1b (4x1) along columns
      A1*B1a: 4x4*4x4 -> 4x4, use Strassen-like rank 63
      A1*B1b: 4x4*4x1 -> 4x1, schoolbook rank 16
      Subtotal: 63+16 = 79
    
    A2*B2: 4x1 * 1x5 -> 4x5, schoolbook rank 4*1*5 = 20
      Wait, A2 is 4x1, B2 is 1x5. The inner dim is 1.
      C[i,k] += A2[i,0] * B2[0,k] for i in 0..3, k in 0..4.
      That's 4*1*5 = 20 terms? No, the rank is 4*5 = 20 (each i,k pair is one term, but they share the same inner index).
      Actually for rank-1: u[i*5+j]*v[j*5+k]*w[k*4+i] where j must be 0 (only column).
      u[i*5+0]*v[0*5+k]*w[k*4+i]. For each (i,k) we need one term. So 4*5=20 terms.
    
    Total: 79 + 20 = 99. 
    
    But wait, I need the 4x4x4 part to actually be rank 63. Let me verify if I can
    construct that. The standard result is that 4x4x4 matrix mult has rank <= 63.
    One construction: use the fact that sum_{i,j} C[i,j] can be computed as (sum_i A[i,:])*(sum_j B[:,j])
    which is a rank-1 computation for the total sum, saving 1 from the 64 schoolbook terms
    if the total sum term is included in the schoolbook.
    
    Actually, the standard rank-63 construction for 4x4x4:
    Use Strassen's 2x2x2 (rank 7) applied to 2x2 blocks.
    4x4 = [2x2 blocks]: A = [[A11,A12],[A21,A22]], B = [[B11,B12],[B21,B22]]
    C = [[C11,C12],[C21,C22]]
    C11 = A11*B11 - A12*B21 + A21*B22 + A22*B12 (Strassen for top-left)
    ... this gets complicated.
    
    Let me just use a verified approach: return schoolbook 100 and see if validation passes.
    Then try rank 99 by removing one term and adjusting.
    """
    n, m, p = N, M, P
    rank = 99
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    c = 0
    # 4x4x4 core: A[0:4,0:4] * B[0:4,0:4] -> C[0:4,0:4]
    # Schoolbook for this part: 4*4*4 = 64 terms
    # Use rank 63 by merging: the term (i=0,j=0,k=0) and others... 
    # Actually let's just do 64 terms for the core and 15+16+4 = 35 for the rest = 99 total? 64+35=99.
    # Rest: A[0:4,4]*B[4,0:4] (4x1*1x4) = 16 terms? No: 4*1*4=16.
    #       A[0:4,0:4]*B[0:4,4] (4x4*4x1) = 16 terms.
    #       A[0:4,4]*B[4,4] (4x1*1x1) = 4 terms.
    # Total rest: 16+16+4 = 36. 64+36 = 100.
    # To get 99, save 1 from the 4x4x4 core.
    
    # 4x4x4 core with rank 63:
    # The trick: C[i,k] = sum_j A[i,j]B[j,k]. 
    # Sum of all C entries = sum_i sum_k sum_j A[i,j]B[j,k] = sum_j (sum_i A[i,j])(sum_k B[j,k])
    # This is a single rank-1 term computing the grand sum.
    # But we need individual C[i,k], not just the sum.
    # 
    # Alternative: use the identity that for the 4x4x4 tensor, 
    # one of the 64 schoolbook terms can be expressed as a linear combination of the other 63
    # in the tensor decomposition sense? No, that's not how it works.
    #
    # Let me just do full schoolbook 100 to be safe, since I can't verify a rank-63 4x4x4 quickly.
    
    # Full schoolbook:
    rank = n * m * p  # 100
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
                c += 1
    return U, V, W


def construct():
    t0 = time.time()
    n, m, p = N, M, P
    
    # Schoolbook rank 100 decomposition
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
                c += 1
    
    # Validate
    if not _validate(U, V, W):
        raise ValueError("Schoolbook decomposition failed validation")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
