"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

import time
import numpy as np

# EVOLVE-BLOCK-START
def _build_strassen_block(n, m, p):
    """
    Build a decomposition using Strassen's algorithm for the 2x2 block
    multiplied by each column of the 2x13 matrix.
    
    For each column k, we compute:
    C[:,k] = A * B[:,k]
    
    Using Strassen:
    M1 = (A00 + A11) * (B00 + B11)
    M2 = (A10 + A11) * B00
    M3 = A00 * (B01 - B11)
    M4 = A11 * (B10 - B00)
    M5 = (A00 + A01) * B11
    M6 = (A10 - A00) * (B00 + B01)
    M7 = (A01 - A11) * (B10 + B11)
    
    C00 = M1 + M4 - M5 + M7
    C10 = M2 + M4
    C01 = M3 + M5
    C11 = M1 - M3 + M2 + M6
    
    Each Mj is a product of two linear combinations.
    We can decompose each product into rank-1 terms using the fact that
    we're working over a commutative ring (the polynomials Q[x]/(x^13)).
    
    Actually, let me think more carefully. The tensor for 2x2 * 2x13 has
    rank at most 7 * 13 = 91 if we do Strassen per column, but that's worse
    than schoolbook (52).
    
    Wait, schoolbook is 2*2*13 = 52. Strassen per column would be 7*13 = 91.
    That's worse.
    
    Let me reconsider. The key insight is that we can use the algebraic structure
    of 2x2 matrices. The multiplication tensor for 2x2 matrices has rank 7
    (Strassen). But when one matrix is 2x13, we can think of it as 13 columns,
    each of length 2.
    
    Actually, a better approach: represent the 2x13 matrix B as a 2x2 matrix
    with entries in Q[x]/(x^13). Then the multiplication is:
    C = A * B where A is 2x2 over Q and B is 2x2 over Q[x]/(x^13).
    
    The product C is 2x2 over Q[x]/(x^13). Using Strassen over the ring Q[x]/(x^13):
    Each "scalar" multiplication in Strassen becomes a multiplication in Q[x]/(x^13),
    which has rank 13 (since it's just polynomial multiplication truncated to degree 12).
    
    So total rank = 7 * 13 = 91. Still worse than 52.
    
    Hmm, let me think differently. Maybe we can do better by exploiting the
    specific structure.
    
    Actually, wait. The schoolbook rank is n*m*p = 2*2*13 = 52. Can we beat this?
    
    For matrix multiplication M_{n,m,p}, the rank is at most n*m*p (schoolbook).
    For n=2, m=2, p=13, that's 52.
    
    There are known results that for M_{2,2,p}, the rank is at most 2p + 1 for p >= 1?
    Let me think...
    
    Actually, I recall that for M_{2,2,p}, one can achieve rank 2p + 1 using
    a clever decomposition. Let me try to construct it.
    
    Consider A = [[a, b], [c, d]] and B = [[b0, b1, ..., b12], [c0, c1, ..., c12]]
    where B is 2x13.
    
    C = A * B = [[a*b0 + b*c0, a*b1 + b*c1, ..., a*b12 + b*c12],
                 [c*b0 + d*c0, c*b1 + d*c1, ..., c*b12 + d*c12]]
    
    So C[i,k] = sum_j A[i,j] * B[j,k] for i=0,1 and k=0,...,12.
    
    We need to decompose this as a sum of rank-1 tensors.
    
    One approach: use the fact that we can write:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    This is already the schoolbook decomposition with rank 2*2*13 = 52.
    
    To beat this, we need to find a decomposition with rank < 52.
    
    Let me try a different approach. Consider the following:
    For each k, we have a 2x2 matrix multiplication:
    C[:,k] = A * B[:,k]
    
    The tensor for a single 2x2 multiplication has rank 7 (Strassen).
    But the entries of B[:,k] are independent for different k.
    
    However, we can share the "A" part across all k. This is where the
    algebraic structure comes in.
    
    Actually, I think the key is to use the fact that the 2x2 matrix
    multiplication tensor can be decomposed into 7 rank-1 terms, and then
    each "scalar" multiplication in that decomposition can be expanded
    using the polynomial structure.
    
    But as I computed, that gives 7*13 = 91, which is worse.
    
    Let me try yet another approach. What if we use a different decomposition
    of the 2x2 multiplication that has a special structure?
    
    Actually, I just realized: the rank of M_{2,2,p} is known to be at most
    2p + 1 for p >= 1. Let me verify this.
    
    For p=1, M_{2,2,1} is just 2x2 * 2x1, which is a 2x2 matrix times a 2-vector.
    The rank should be... let me compute. C[i] = sum_j A[i,j]*B[j].
    Schoolbook: 2*2*1 = 4. Can we do better?
    
    C[0] = a*B[0] + b*B[1]
    C[1] = c*B[0] + d*B[1]
    
    Using Strassen-like:
    M1 = (a+d)*(B[0]+B[1])
    M2 = (c+d)*B[0]
    M3 = a*(B[1]-B[0])  -- wait, this doesn't work directly since B is a vector.
    
    Actually for 2x2 * 2x1, the rank is 4 (same as schoolbook) since there's
    no way to reduce it below n*m*p when p=1 and n=m=2? No, that's not right either.
    
    Let me just try to construct a rank-5 decomposition for M_{2,2,1}:
    C[0] = a*B[0] + b*B[1]
    C[1] = c*B[0] + d*B[1]
    
    I don't see how to do this in rank 4 or less easily. Schoolbook gives 4.
    
    OK let me just go with the schoolbook decomposition of rank 52, which is
    what the current program does, and see if I can find any improvement.
    
    Actually, I just remembered: for M_{n,m,p}, the rank is at most min(n*m*p, ...).
    There's no general formula, but for specific cases, improvements exist.
    
    Let me just return the schoolbook decomposition.
    """
    n, m, p = 2, 2, 13
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def construct():
    start = time.time()
    deadline = start + 110
    n, m, p = DIMENSIONS
    
    # Return schoolbook decomposition
    result = _build_strassen_block(n, m, p)
    return result
# EVOLVE-BLOCK-END
