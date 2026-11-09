"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """
    Schoolbook rank-n*m*p decomposition for n×m times m×p matrix multiplication.
    Uses the trace convention: output index is k*n + i for W.
    Input: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    For each term c = (i, j, k):
        U[i*m + j, c] = 1
        V[j*p + k, c] = 1
        W[k*n + i, c] = 1
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _gks_split(n, m, p, m1, m2):
    """
    Apply GKS splitting to an n×m×p matrix multiplication problem.
    Split the middle dimension m into m1 + m2 = m.
    Returns a decomposition with rank = rank(n,m1,p) + rank(n,m2,p) + rank(n,m1,m2) + rank(n,m2,m2)
    where each sub-problem is solved by schoolbook.
    
    The idea: C = A*B where A is n×m, B is m×p.
    Split A = [A1 | A2] where A1 is n×m1, A2 is n×m2
    Split B = [B1; B2] where B1 is m1×p, B2 is m2×p
    
    C = A1*B1 + A2*B2 + A1*B2 + A2*B1
    
    Wait, that's not right. Let me reconsider.
    
    Actually, the standard GKS/Strassen approach for rectangular matrices:
    For A (n×m) * B (m×p), we can write:
    
    Let's split the middle index j into two parts: j ∈ [0, m1) and j ∈ [m1, m).
    
    C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
           = sum_{j=0}^{m1-1} A[i,j] * B[j,k] + sum_{j=m1}^{m-1} A[i,j] * B[j,k]
    
    This is just two independent multiplications: (n×m1) * (m1×p) and (n×m2) * (m2×p)
    where m2 = m - m1.
    
    So the rank would be n*m1*p + n*m2*p = n*m*p, which is the same as schoolbook.
    That's not useful.
    
    The actual GKS method uses a different decomposition. Let me think about this differently.
    
    For square matrices, Strassen's trick works by computing:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B21 - B22)
    M4 = A22(B12 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    Then C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    For rectangular matrices, we need to adapt this. The key insight is that we can still
    use the Strassen-like decomposition if we can split both the row and column indices.
    
    For n×m × m×p, let's try to split n into n1+n2, m into m1+m2, and p into p1+p2.
    Then we have 8 sub-problems, but Strassen's method reduces the number of multiplications.
    
    However, for the specific case of 2×5×6, let's try a simpler approach.
    
    Actually, let me reconsider the GKS splitting more carefully. The standard result is:
    For matrix multiplication [n,m,p], the GKS bound gives:
    R(n,m,p) ≤ R(n, m1, p) + R(n, m2, p) + R(n, m1, m2) + R(m1, m2, p)
    where m = m1 + m2.
    
    Wait, I think I was confusing myself. Let me look up the correct formula.
    
    The correct GKS/Strassen rectangular formula is:
    R(n, m, p) ≤ R(n, m1, p) + R(n, m2, p) + R(n, m1, m2) + R(m1, m2, p)
    
    No, that doesn't seem right either. Let me think from first principles.
    
    For A (n×m) * B (m×p) = C (n×p):
    
    Split A into [A1 | A2] where A1 is n×m1, A2 is n×m2
    Split B into [B1; B2] where B1 is m1×p, B2 is m2×p
    
    C = A1*B1 + A2*B2
    
    This gives rank ≤ R(n,m1,p) + R(n,m2,p) = n*m1*p + n*m2*p = n*m*p
    
    That's just the schoolbook rank. So this simple splitting doesn't help.
    
    The Strassen trick requires splitting BOTH dimensions. Let me try:
    
    Split A into blocks:
    A = [[A11, A12],
         [A21, A22]]
    where A11 is n1×m1, A12 is n1×m2, A21 is n2×m1, A22 is n2×m2
    
    Split B into blocks:
    B = [[B11, B12],
         [B21, B22]]
    where B11 is m1×p1, B12 is m1×p2, B21 is m2×p1, B22 is m2×p2
    
    Then C = A*B has blocks:
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    Now apply Strassen's trick:
    M1 = (A11 + A22)(B11 + B22)  -- but this requires A11+A22 to be n1×m1 and B11+B22 to be m1×p1
    This only works if n1 = n2 and m1 = m2 and p1 = p2, i.e., square blocks.
    
    For rectangular matrices, we need a different approach. One common technique is to pad
    to square matrices, but that increases the rank.
    
    Given the complexity, let me just use the schoolbook decomposition for now, since
    finding a better exact decomposition for 2×5×6 is non-trivial and may require
    advanced techniques.
    """
    pass


def construct():
    """
    Return the schoolbook rank-n*m*p decomposition for 2×5 × 5×6 matrix multiplication.
    This is the trivial upper bound; a rank-minimal exact decomposition would be better.
    """
    n, m, p = DIMENSIONS
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
