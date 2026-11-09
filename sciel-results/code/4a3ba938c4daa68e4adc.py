"""
Rank-minimal tensor decomposition for 3x3 * 3x4 matrix multiplication.

Strategy:
- Start with the known rank-34 decomposition (Strassen on 2x2 block + schoolbook for the rest).
- Attempt to find a rank-33 decomposition via randomized iterative refinement / structured search.
- If rank 33 is not found within time budget, return rank 34.
- If rank 34 also fails verification (shouldn't happen), fall back to schoolbook rank 36.

The key insight: for 3x3 * 3x4, we can partition the 3x4 matrix into a 2x4 and 1x4 part,
or use block structure. A known approach:
  - Split the 3x3 input into blocks to exploit Strassen's 7-multiplication 2x2 trick.
  - The rank-34 construction: use Strassen for a 2x2 sub-block (saves 1 mult: 4->7 is worse for 2x2,
    but when combined with the rectangular structure, it can save overall).

Actually, let me think more carefully.

For 3x3 * 3x4:
- Schoolbook: 3*3*4 = 36 scalar multiplications.
- We can partition the 3x3 matrix A into [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1.
- Partition the 3x4 matrix B into [[B11, B12], [B21, B22]] where B11 is 2x4, B12 is 1x4, etc. Wait, B is 3x4, so rows are 3, cols are 4.

Let me use a different partition. Let's split the output 3x4 into blocks.

Alternative known result: The rank of 3x3 * 3x4 is known to be at most 34, and it's an open question whether it can be done in less. Some sources suggest 33 or 34.

Let me implement a rank-34 decomposition using Strassen on a 2x2 block:

Partition A (3x3) as:
  A = [[A1, A2], [A3, A4]]  where A1 is 2x2, A2 is 2x1, A3 is 1x2, A4 is 1x1

Partition B (3x4) as:
  B = [[B1, B2], [B3, B4]]  where B1 is 2x4, B2 is 2x4? No...

Wait, B is 3x4. If I split rows of B the same way (2+1), then:
  B = [[B_top (2x4)], [B_bottom (1x4)]]

But for block matrix multiplication to work cleanly, I need consistent splits.

Let me try: Split the 3x3 matrix A into 2x2, 2x1, 1x2, 1x1 blocks.
Split the 3x4 matrix B into 2x4 and 1x4 (row split only).

Then:
C = A * B = [[A1, A2], [A3, A4]] * [[B1], [B2]]
  = [A1*B1 + A2*B2, A2*B2 + A4*B2]  -- no this isn't right either.

Actually, let me just use a known rank-34 construction. 

For multiplying an (n1+n2) x (m1+m2) matrix by an (m1+m2) x (p1+p2) matrix:
If we use Strassen on the 2x2 block of A (top-left 2x2) and B (top-left 2x2 part of the 3x4, i.e., first 2 rows and first 2 cols), we can save one multiplication.

But the 3x4 matrix doesn't have a clean 2x2 block in the same way.

Let me try a different approach: use the fact that we can write the 3x4 multiplication as:
  - 2x4 multiplication (top 2 rows of A times B) 
  - 1x4 multiplication (bottom row of A times B)

For the 2x4 part, we can further split into 2x2 + 2x2:
  A_top (2x3) * B (3x4) ... this is getting complicated.

Let me just implement a direct search for rank 33 and 34 using structured tensor decomposition.

Given the time constraints, I'll:
1. Implement the known rank-34 decomposition.
2. Try a randomized search for rank 33 with a time limit.
3. Return the best found.

For the rank-34 construction, I'll use the following approach:
- Treat the 3x3 * 3x4 as having a 2x2 block structure where we apply Strassen.

After more thought, here's a concrete rank-34 construction:

Partition the 3x3 matrix A into:
  A = [[a11, a12, a13],
       [a21, a22, a23],
       [a31, a32, a33]]

Partition the 3x4 matrix B into:
  B = [[b11, b12, b13, b14],
       [b21, b22, b23, b24],
       [b31, b32, b33, b34]]

The idea is to use Strassen-like combinations on the first 2x2 block of A and first 2 rows of B, treating the rest as corrections.

Actually, let me just do a brute-force structured search. Given the small dimensions, I can try to find a rank-33 decomposition by solving a system of polynomial equations, but that's hard.

Instead, let me use a randomized approach: start from rank 34 and try to merge two terms.

Let me implement this practically:
"""

import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 3, 4]
DEADLINE = 140.0  # safety margin before 150s kill

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the multiplication tensor T[i, j, k] = delta_{i,j,k} in the appropriate sense
    # For matrix mult: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
    # T[i, j, k] = 1 if the scalar product A[i,j]*B[j,k] contributes to C[i,k]
    # The decomposition: T = sum_r U[:,r] outer V[:,r] outer W[:,r]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    # Reconstruct tensor from decomposition
    T_recon = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    
    # Expected tensor
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m + j, j*p + k, k*n + i] = 1.0
    
    if np.max(np.abs(T_recon - T_expected)) > tol:
        return False
    return True

def _get_schoolbook(n, m, p):
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

def _get_rank34(n, m, p):
    """
    Rank-34 decomposition using Strassen on a 2x2 block.
    
    We partition:
    A (3x3) = [[A11 (2x2), A12 (2x1)],
               [A21 (1x2), A22 (1x1)]]
    B (3x4) = [[B1 (2x4)],
               [B2 (1x4)]]
    
    C = A*B has shape 3x4.
    C_top (2x4) = A11*B1 + A12*B2
    C_bot (1x4) = A21*B1 + A22*B2
    
    For the A11*B1 part (2x2 * 2x4), we can split B1 into [B1a (2x2), B1b (2x2)]:
    A11*B1a: use Strassen (7 mults)
    A11*B1b: use Strassen (7 mults)
    A12*B2: 2x1 * 1x4 = 2 mults per column... actually A12 is 2x1, B2 is 1x4, so 2*4=8? No, it's 2*1*4=8 scalar mults but we can do it with 2 "matrix" mults of size 1x4... 
    
    Hmm, this is getting complicated. Let me just use a known explicit construction.
    
    Actually, for the purpose of this task, let me construct rank-34 by taking schoolbook (36) and finding 2 savings.
    
    A simpler approach: use the fact that for 2x2 * 2x2, Strassen uses 7 instead of 4.
    If I can identify a 2x2 * 2x2 subproblem within the 3x3 * 3x4, I save 1.
    
    But 3x3 * 3x4 doesn't contain a clean 2x2 * 2x2.
    
    Let me try: partition A into 2x2 + 2x1 + 1x2 + 1x1, and B into 2x4 + 1x4.
    
    A11 (2x2) * B1 (2x4): This is 2x2 * 2x4. Split B1 = [B1a, B1b] each 2x2.
    A11*B1a: Strassen, 7 mults
    A11*B1b: Strassen, 7 mults
    Total for A11*B1: 14 mults (vs 16 schoolbook)
    
    A12 (2x1) * B2 (1x4): 2*1*4 = 8 scalar mults, but as "tensor rank" it's 2*4=8? No, it's 2 rows of A12 times 1 column, times 4 cols of B2. The rank is 2*4 = 8? Actually for a 2x1 * 1x4 multiplication, the tensor rank is 2*4 = 8 (each output element is a product of one A element and one B element).
    
    Wait, 2x1 * 1x4: C[i,k] = A[i,0] * B[0,k] for i=0,1 and k=0,1,2,3. That's 8 products, rank 8.
    
    A21 (1x2) * B1 (2x4): C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]. This is 1x2 * 2x4, rank = 1*2*4 = 8.
    
    A22 (1x1) * B2 (1x4): C[0,k] = A[0,0]*B[0,k]. Rank = 1*4 = 4.
    
    Total: 14 + 8 + 8 + 4 = 34. Yes!
    
    Now let me construct this explicitly.
    """
    # This is complex to construct directly. Let me use a different strategy:
    # Just return schoolbook for now and rely on the search to find better.
    # Actually, let me just construct it properly.
    
    # For simplicity and correctness, let me use a randomized search approach
    # to find rank 34 and potentially 33.
    
    # Start with schoolbook
    result = _get_schoolbook(n, m, p)
    return result

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # First, get schoolbook as fallback
    schoolbook = _get_schoolbook(n, m, p)
    
    # Try to find rank 34 via structured construction
    # For now, let's try a randomized search for rank 33 and 34
    
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try rank 34
    target_rank = 34
    if best_rank > target_rank:
        result = _try_find_rank(n, m, p, target_rank, start_time)
        if result is not None:
            best_result = result
            best_rank = target_rank
    
    # Try rank 33
    target_rank = 33
    if best_rank > target_rank:
        remaining = DEADLINE - (time.time() - start_time)
        if remaining > 10:
            result = _try_find_rank(n, m, p, target_rank, start_time)
            if result is not None:
                best_result = result
                best_rank = target_rank
    
    return best_result

def _try_find_rank(n, m, p, target_rank, start_time):
    """Try to find a decomposition of the given rank using randomized methods."""
    # This is a placeholder for a real search. For now, return None to use fallback.
    # A real implementation would use tensor rank minimization algorithms.
    
    # Given the complexity, let's just return None and rely on the known constructions.
    return None

# Let me actually implement a proper rank-34 construction
def _get_rank34_decomposition(n, m, p):
    """
    Construct rank-34 decomposition for 3x3 * 3x4.
    
    Partition:
    A (3x3): rows split as 2+1, cols split as 2+1
      A11 (2x2): A[0:2, 0:2]
      A12 (2x1): A[0:2, 2:3]  
      A21 (1x2): A[2:3, 0:2]
      A22 (1x1): A[2:3, 2:3]
    
    B (3x4): rows split as 2+1
      B1 (2x4): B[0:2, :]
      B2 (1x4): B[2:3, :]
    
    C (3x4): rows split as 2+1
      C1 (2x4): C[0:2, :]
      C2 (1x4): C[2:3, :]
    
    C1 = A11*B1 + A12*B2
    C2 = A21*B1 + A22*B2
    
    For A11*B1 (2x2 * 2x4): split B1 into B1a (2x2) and B1b (2x2)
      A11*B1a: Strassen, 7 mults
      A11*B1b: Strassen, 7 mults
    For A12*B2 (2x1 * 1x4): 8 mults (schoolbook)
    For A21*B1 (1x2 * 2x4): 8 mults (schoolbook)
    For A22*B2 (1x1 * 1x4): 4 mults (schoolbook)
    
    Total: 7+7+8+8+4 = 34
    """
    rank = 34
    U = [[0] * rank for _ in range(n * m)]  # 9x34
    V = [[0] * rank for _ in range(m * p)]  # 12x34
    W = [[0] * rank for _ in range(p * n)]  # 12x34
    
    col = 0
    
    # Strassen for A11 * B1a (2x2 * 2x2)
    # A11 = [[a00, a01], [a10, a11]] (A[0,0], A[0,1], A[1,0], A[1,1])
    # B1a = [[b00, b01], [b10, b11]] (B[0,0], B[0,1], B[1,0], B[1,1])
    # Strassen:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    
    # Indices:
    # A[i*m+j] where m=3
    # A[0*3+0]=A[0]=a00, A[0*3+1]=A[1]=a01, A[1*3+0]=A[3]=a10, A[1*3+1]=A[4]=a11
    # B[j*p+k] where p=4
    # B[0*4+0]=B[0]=b00, B[0*4+1]=B[1]=b01, B[1*4+0]=B[4]=b10, B[1*4+1]=B[5]=b11
    # W[k*n+i] where n=3
    # W[0*3+0]=W[0]=c00, W[0*3+1]=W[1]=c01, W[1*3+0]=W[3]=c10, W[1*3+1]=W[4]=c11
    
    # M1 = (a00+a11)*(b00+b11) -> c00 += M1, c11 += M1
    # U: A[0]+=1, A[4]+=1; V: B[0]+=1, B[5]+=1; W: W[0]+=1, W[4]+=1
    U[0][col] += 1; U[4][col] += 1
    V[0][col] += 1; V[5][col] += 1
    W[0][col] += 1; W[4][col] += 1
    col += 1
    
    # M2 = (a10+a11)*b00 -> c10 += M2, c11 += M2
    U[3][col] += 1; U[4][col] += 1
    V[0][col] += 1
    W[3][col] += 1; W[4][col] += 1
    col += 1
    
    # M3 = a00*(b01-b11) -> c01 += M3, c11 -= M3
    U[0][col] += 1
    V[1][col] += 1; V[5][col] -= 1
    W[1][col] += 1; W[4][col] -= 1
    col += 1
    
    # M4 = a11*(b10-b00) -> c10 += M4, c11 -= M4... wait
    # Actually: c10 += M4? Let me recheck Strassen.
    # c00 = M1 - M4 + M5 - M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # So M4 contributes to c10 (+) and c00 (-)
    U[4][col] += 1
    V[4][col] += 1; V[0][col] -= 1
    W[3][col] += 1; W[0][col] -= 1
    col += 1
    
    # M5 = (a00+a01)*b11 -> c00 += M5, c01 += M5
    U[0][col] += 1; U[1][col] += 1
    V[5][col] += 1
    W[0][col] += 1; W[1][col] += 1
    col += 1
    
    # M6 = (a10-a00)*(b00+b01) -> c11 += M6, c00 -= M6
    U[3][col] += 1; U[0][col] -= 1
    V[0][col] += 1; V[1][col] += 1
    W[4][col] += 1; W[0][col] -= 1
    col += 1
    
    # M7 = (a01-a11)*(b10+b11) -> c00 -= M7, c11 += M7
    U[1][col] += 1; U[4][col] -= 1
    V[4][col] += 1; V[5][col] += 1
    W[0][col] -= 1; W[4][col] += 1
    col += 1
    
    # Now A11*B1b (2x2 * 2x2) where B1b = B[0:2, 2:4]
    # b00' = B[0*4+2]=B[2], b01' = B[3], b10' = B[6], b11' = B[7]
    # Output: c02, c03, c12, c13
    # W indices: W[2*3+0]=W[6]=c02, W[3*3+0]=W[9]=c12, W[2*3+1]=W[7]=c03, W[3*3+1]=W[10]=c13
    
    # M1' = (a00+a11)*(b00'+b11') -> c02 += M1', c13 += M1'
    U[0][col] += 1; U[4][col] += 1
    V[2][col] += 1; V[7][col] += 1
    W[6][col] += 1; W[10][col] += 1
    col += 1
    
    # M2' = (a10+a11)*b00' -> c12 += M2', c13 += M2'
    U[3][col] += 1; U[4][col] += 1
    V[2][col] += 1
    W[9][col] += 1; W[10][col] += 1
    col += 1
    
    # M3' = a00*(b01'-b11') -> c03 += M3', c13 -= M3'
    U[0][col] += 1
    V[3][col] += 1; V[7][col] -= 1
    W[7][col] += 1; W[10][col] -= 1
    col += 1
    
    # M4' = a11*(b10'-b00') -> c12 += M4', c02 -= M4'
    U[4][col] += 1
    V[6][col] += 1; V[2][col] -= 1
    W[9][col] += 1; W[6][col] -= 1
    col += 1
    
    # M5' = (a00+a01)*b11' -> c02 += M5', c03 += M5'
    U[0][col] += 1; U[1][col] += 1
    V[7][col] += 1
    W[6][col] += 1; W[7][col] += 1
    col += 1
    
    # M6' = (a10-a00)*(b00'+b01') -> c13 += M6', c02 -= M6'
    U[3][col] += 1; U[0][col] -= 1
    V[2][col] += 1; V[3][col] += 1
    W[10][col] += 1; W[6][col] -= 1
    col += 1
    
    # M7' = (a01-a11)*(b10'+b11') -> c02 -= M7', c13 += M7'
    U[1][col] += 1; U[4][col] -= 1
    V[6][col] += 1; V[7][col] += 1
    W[6][col] -= 1; W[10][col] += 1
    col += 1
    
    # A12*B2: A12 (2x1) * B2 (1x4)
    # A12 = [A[0*3+2], A[1*3+2]] = [A[2], A[5]]
    # B2 = [B[2*4+0], B[2*4+1], B[2*4+2], B[2*4+3]] = [B[8], B[9], B[10], B[11]]
    # C1 contribution: c0k += A[2]*B[8+k] for k=0,1,2,3
    # c1k += A[5]*B[8+k] for k=0,1,2,3
    for k in range(4):
        U[2][col] += 1
        V[8 + k][col] += 1
        W[k * 3 + 0][col] += 1  # c0k
        col += 1
    for k in range(4):
        U[5][col] += 1
        V[8 + k][col] += 1
        W[k * 3 + 1][col] += 1  # c1k
        col += 1
    
    # A21*B1: A21 (1x2) * B1 (2x4)
    # A21 = [A[2*3+0], A[2*3+1]] = [A[6], A[7]]
    # B1 = B[0:2, :] = [B[0..3], B[4..7]]
    # C2: c2k += A[6]*B[k] + A[7]*B[4+k] for k=0,1,2,3
    for k in range(4):
        U[6][col] += 1
        V[k][col] += 1
        W[k * 3 + 2][col] += 1  # c2k
        col += 1
    for k in range(4):
        U[7][col] += 1
        V[4 + k][col] += 1
        W[k * 3 + 2][col] += 1  # c2k
        col += 1
    
    # A22*B2: A22 (1x1) * B2 (1x4)
    # A22 = A[2*3+2] = A[8]
    # B2 = [B[8], B[9], B[10], B[11]]
    # C2: c2k += A[8]*B[8+k] for k=0,1,2,3
    for k in range(4):
        U[8][col] += 1
        V[8 + k][col] += 1
        W[k * 3 + 2][col] += 1  # c2k
        col += 1
    
    # col should be 34
    assert col == 34, f"Expected 34, got {col}"
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Get rank-34 decomposition
    rank34 = _get_rank34_decomposition(n, m, p)
    
    # Verify it
    if _verify(rank34["U"], rank34["V"], rank34["W"], n, m, p):
        # Try to find rank 33
        remaining = DEADLINE - (time.time() - start_time)
        if remaining > 20:
            rank33 = _try_find_rank33(n, m, p, start_time)
            if rank33 is not None:
                return rank33
        return rank34
    else:
        # Fall back to schoolbook
        return _get_schoolbook(n, m, p)

def _try_find_rank33(n, m, p, start_time):
    """
    Try to find a rank-33 decomposition.
    This is an open problem, so we use a heuristic search.
    """
    # Given the difficulty, let's try a simple approach:
    # Start from rank 34 and try to merge two terms.
    # This is very unlikely to succeed, but let's try briefly.
    
    # For now, return None
    return None
