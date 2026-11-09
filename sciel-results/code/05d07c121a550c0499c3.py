"""Matrix multiplication 3x3 x 3x4 rank-minimal decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def _remaining(deadline):
    return deadline - time.time()

def _verify(U, V, W, n, m, p):
    """Verify decomposition: C[i,k] = sum_j A[i,j]*B[j,k]
    via C[i,k] = sum_r W[k*n+i,r] * (sum_{i',j'} U[i'*m+j',r]*A[i',j']) * (sum_{j',k'} V[j'*p+k',r]*B[j',k'])
    """
    rng = np.random.RandomState(42)
    for _ in range(5):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C = A @ B
        # Build factor matrices
        Ua = np.array(U, dtype=float).reshape(n*m, -1)
        Va = np.array(V, dtype=float).reshape(m*p, -1)
        Wa = np.array(W, dtype=float).reshape(p*n, -1)
        rank = Ua.shape[1]
        # f_r(A) = sum_{i',j'} U[i'*m+j', r] * A[i',j']
        A_flat = A.reshape(n*m)
        B_flat = B.reshape(m*p)
        f = Ua.T @ A_flat  # shape (rank,)
        g = Va.T @ B_flat  # shape (rank,)
        # C_recon[i,k] = sum_r W[k*n+i, r] * f[r] * g[r]
        C_recon_flat = Wa.T @ (f * g)
        C_recon = C_recon_flat.reshape(p, n).T  # C_recon[i,k]
        if not np.allclose(C_recon, C, atol=1e-9):
            return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2_2x4():
    """Strassen for 2x2 * 2x4, then schoolbook for row 2.
    A is 3x3, B is 3x4, C is 3x4.
    Split: A = [[A11, A12, A13],[A21, A22, A23],[A31, A32, A33]]
    B = [[B11, B12, B13, B14],[B21, B22, B23, B24],[B31, B32, B33, B34]]
    
    Use Strassen on the 2x2 block (rows 0,1 of A; rows 0,1 of B) with 2x4 B block.
    Actually, Strassen works for 2x2 * 2x2. For 2x2 * 2x4, we can split B into two 2x2 blocks.
    
    Let me think differently. We have:
    C[0:2, 0:4] = A[0:2, 0:2] * B[0:2, 0:4] + A[0:2, 2] * B[2, 0:4]
    C[2, 0:4] = A[2, 0:2] * B[0:2, 0:4] + A[2, 2] * B[2, 0:4]
    
    For the 2x2 * 2x4 part, we can use Strassen by splitting B into B_left (2x2) and B_right (2x2).
    C[0:2, 0:2] = A[0:2,0:2] * B[0:2,0:2] + A[0:2,2] * B[2,0:2]
    C[0:2, 2:4] = A[0:2,0:2] * B[0:2,2:4] + A[0:2,2] * B[2,2:4]
    
    Hmm, this is getting complex. Let me use a simpler approach:
    Use Strassen for the 2x2 * 2x2 subproblems and handle the rest with schoolbook.
    
    Actually, let me just directly construct a rank-34 decomposition.
    
    The key insight: for 3x3 * 3x4, we can use:
    - 2x2 * 2x2 Strassen (rank 7) for the top-left 2x2 output from A[0:2,0:2]*B[0:2,0:2]
    - But we need to handle all the cross terms...
    
    Let me try a different decomposition strategy:
    Split into:
    - A[0:2, 0:2] * B[0:2, 0:2] -> 2x2 output (use Strassen, rank 7)
    - A[0:2, 0:2] * B[0:2, 2:4] -> 2x2 output (use Strassen, rank 7)  
    - A[0:2, 2] * B[2, 0:2] -> 2x2 output (rank 4 schoolbook, or can merge)
    - A[0:2, 2] * B[2, 2:4] -> 2x2 output (rank 4 schoolbook)
    - A[2, 0:2] * B[0:2, 0:2] -> 1x2 output (rank 2)
    - A[2, 0:2] * B[0:2, 2:4] -> 1x2 output (rank 2)
    - A[2, 2] * B[2, 0:2] -> 1x2 output (rank 2)
    - A[2, 2] * B[2, 2:4] -> 1x2 output (rank 2)
    
    Total: 7+7+4+4+2+2+2+2 = 30? That seems too good. The issue is that Strassen for 2x2*2x2 gives rank 7, but we're computing C[0:2,0:2] which is a 2x2 matrix. The Strassen decomposition has 7 rank-1 terms for the 2x2*2x2 tensor.
    
    Wait, but the Strassen decomposition for 2x2*2x2 has U of shape (4,7), V of shape (4,7), W of shape (4,7). Each rank term contributes to all 4 output entries. So for C[0:2, 0:2], we use 7 terms.
    
    But the problem is that A[0:2,0:2]*B[0:2,0:2] is just ONE part of C[0:2,0:2]. We also have A[0:2,2]*B[2,0:2] contributing to the same output.
    
    So the total rank is the sum of ranks of all independent tensor components. Since they all write to different or overlapping output positions, we need to be careful.
    
    Actually, the tensor decomposition is linear. If we have:
    C = A*B where A is 3x3 and B is 3x4
    
    We can decompose the multiplication tensor as a sum of tensor components. Each component is a rank-1 tensor (or sum of rank-1 tensors). The total rank is the sum of the ranks of the components IF the components operate on disjoint sets of (i,j,k) indices or can be independently decomposed.
    
    Let me think about this more carefully. The multiplication tensor T for 3x3*3x4 has entries T[i,j,k] = 1 for all i in {0,1,2}, j in {0,1,2}, k in {0,1,2,3}. Wait no, T is not all ones. 
    
    The matrix multiplication tensor is defined as: for A (n x m), B (m x p), C (n x p):
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T has indices (i, j, k) and T[i,j,k] = 1 if this term contributes to C[i,k], which is always true for valid indices. So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    Wait, that's the all-ones tensor? No! The tensor is:
    T[i, j, k] = 1 means that A[i,j] * B[j,k] contributes to C[i,k].
    
    A rank-1 decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    where u_r is in R^{n*m}, v_r is in R^{m*p}, w_r is in R^{n*p} (or R^{p*n} depending on convention).
    
    The condition is: for all i, j, k:
    T[i,j,k] = sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[i*p+k]
    
    Wait, I need to get the convention right. The task says W[k*n+i, r]. So W has shape (p*n, rank) and W[k*n+i, r] is the coefficient for output C[i,k] in rank term r.
    
    So the decomposition is:
    C[i,k] = sum_r W[k*n+i, r] * (sum_{i',j'} U[i'*m+j', r] * A[i',j']) * (sum_{j',k'} V[j'*p+k', r] * B[j',k'])
    
    For this to equal sum_j A[i,j]*B[j,k], we need the tensor identity.
    
    OK let me just try to construct a rank-34 decomposition explicitly and verify it.
    
    Strategy: Use Strassen's 7-term algorithm for the 2x2*2x2 subproblem (top-left), and schoolbook for everything else.
    
    For the 2x2*2x2 Strassen (computing C[0:2, 0:2] from A[0:2,0:2] and B[0:2,0:2]):
    M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    M2 = (A[1,0]+A[1,1]) * B[0,0]
    M3 = A[0,0] * (B[0,1]-B[1,1])
    M4 = A[1,1] * (B[1,0]-B[0,0])
    M5 = (A[0,0]+A[0,1]) * B[1,1]
    M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    
    C[0,0] = M1 + M4 - M5 + M7
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M2 + M3 + M6
    
    Now, for the full 3x3 * 3x4 problem:
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    C[0,3] = A[0,0]*B[0,3] + A[0,1]*B[1,3] + A[0,2]*B[2,3]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    C[1,3] = A[1,0]*B[0,3] + A[1,1]*B[1,3] + A[1,2]*B[2,3]
    C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
    C[2,3] = A[2,0]*B[0,3] + A[2,1]*B[1,3] + A[2,2]*B[2,3]
    
    Using Strassen for the A[0:2,0:2]*B[0:2,0:2] part:
    C[0,0] gets contribution M1 + M4 - M5 + M7 from Strassen, plus A[0,2]*B[2,0]
    C[0,1] gets contribution M3 + M5 from Strassen, plus A[0,2]*B[2,1]
    C[1,0] gets contribution M2 + M4 from Strassen, plus A[1,2]*B[2,0]
    C[1,1] gets contribution M1 - M2 + M3 + M6 from Strassen, plus A[1,2]*B[2,1]
    
    For C[0,2], C[0,3], C[1,2], C[1,3]: these involve A[0:2, 0:2]*B[0:2, 2:4] + A[0:2,2]*B[2,2:4]
    We can use another Strassen for A[0:2,0:2]*B[0:2,2:4] (treating B[0:2,2:4] as a 2x2 matrix with columns 2,3):
    N1 = (A[0,0]+A[1,1]) * (B[0,2]+B[1,3])
    N2 = (A[1,0]+A[1,1]) * B[0,2]
    N3 = A[0,0] * (B[0,3]-B[1,3])
    N4 = A[1,1] * (B[1,2]-B[0,2])
    N5 = (A[0,0]+A[0,1]) * B[1,3]
    N6 = (A[1,0]-A[0,0]) * (B[0,2]+B[0,3])
    N7 = (A[0,1]-A[1,1]) * (B[1,2]+B[1,3])
    
    C[0,2] += N1 + N4 - N5 + N7 + A[0,2]*B[2,2]
    C[0,3] += N3 + N5 + A[0,2]*B[2,3]
    C[1,2] += N2 + N4 + A[1,2]*B[2,2]
    C[1,3] += N1 - N2 + N3 + N6 + A[1,2]*B[2,3]
    
    For C[2,0], C[2,1], C[2,2], C[2,3]: schoolbook
    C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
    C[2,3] = A[2,0]*B[0,3] + A[2,1]*B[1,3] + A[2,2]*B[2,3]
    
    Total rank: 7 (M) + 7 (N) + 4 (A[0:2,2]*B[2,0:2]) + 4 (A[0:2,2]*B[2,2:4]) + 4 (A[2,0:2]*B[0:2,0:2]) + 4 (A[2,0:2]*B[0:2,2:4]) + 4 (A[2,2]*B[2,0:4])
    = 7+7+4+4+4+4+4 = 34
    
    Wait, let me recount. The terms A[0:2,2]*B[2,0:2] contribute to C[0:2,0:2]:
    A[0,2]*B[2,0] -> C[0,0]
    A[0,2]*B[2,1] -> C[0,1]
    A[1,2]*B[2,0] -> C[1,0]
    A[1,2]*B[2,1] -> C[1,1]
    That's 4 rank-1 terms.
    
    A[0:2,2]*B[2,2:4] contribute to C[0:2,2:4]:
    A[0,2]*B[2,2] -> C[0,2]
    A[0,2]*B[2,3] -> C[0,3]
    A[1,2]*B[2,2] -> C[1,2]
    A[1,2]*B[2,3] -> C[1,3]
    That's 4 rank-1 terms.
    
    A[2,0:2]*B[0:2,0:2] contribute to C[2,0:2]:
    A[2,0]*B[0,0] -> C[2,0]
    A[2,1]*B[1,0] -> C[2,0]
    A[2,0]*B[0,1] -> C[2,1]
    A[2,1]*B[1,1] -> C[2,1]
    That's 4 rank-1 terms.
    
    A[2,0:2]*B[0:2,2:4] contribute to C[2,2:4]:
    A[2,0]*B[0,2] -> C[2,2]
    A[2,1]*B[1,2] -> C[2,2]
    A[2,0]*B[0,3] -> C[2,3]
    A[2,1]*B[1,3] -> C[2,3]
    That's 4 rank-1 terms.
    
    A[2,2]*B[2,0:4] contribute to C[2,0:4]:
    A[2,2]*B[2,0] -> C[2,0]
    A[2,2]*B[2,1] -> C[2,1]
    A[2,2]*B[2,2] -> C[2,2]
    A[2,2]*B[2,3] -> C[2,3]
    That's 4 rank-1 terms.
    
    Total: 7+7+4+4+4+4+4 = 34. 
    
    Now let me construct the U, V, W matrices for this decomposition.
    """
    n, m, p = 3, 3, 4
    rank = 34
    U = [[0.0] * rank for _ in range(n * m)]  # 9 x 34
    V = [[0.0] * rank for _ in range(m * p)]  # 12 x 34
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x 34
    
    col = 0
    # Strassen M terms for A[0:2,0:2] * B[0:2,0:2]
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # U: A[0,0] coeff 1, A[1,1] coeff 1 -> U[0*3+0]=1, U[1*3+1]=1
    # V: B[0,0] coeff 1, B[1,1] coeff 1 -> V[0*4+0]=1, V[1*4+1]=1
    # W: C[0,0] coeff 1, C[0,1] coeff 0, C[1,0] coeff 0, C[1,1] coeff 1
    # W[k*n+i]: C[0,0] -> W[0*3+0]=W[0], C[1,1] -> W[1*3+1]=W[4]
    U[0][col] = 1.0; U[4][col] = 1.0
    V[0][col] = 1.0; V[5][col] = 1.0
    W[0][col] = 1.0; W[4][col] = 1.0
    col += 1  # col=1
    
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    U[3][col] = 1.0; U[4][col] = 1.0
    V[0][col] = 1.0
    W[3][col] = 1.0; W[4][col] = -1.0  # C[1,0] += M2, C[1,1] -= M2
    col += 1  # col=2
    
    # M3 = A[0,0] * (B[0,1]-B[1,1])
    U[0][col] = 1.0
    V[1][col] = 1.0; V[5][col] = -1.0
    W[1][col] = 1.0; W[4][col] = 1.0  # C[0,1] += M3, C[1,1] += M3
    col += 1  # col=3
    
    # M4 = A[1,1] * (B[1,0]-B[0,0])
    U[4][col] = 1.0
    V[4][col] = 1.0; V[0][col] = -1.0
    W[0][col] = 1.0; W[3][col] = 1.0  # C[0,0] += M4, C[1,0] += M4
    col += 1  # col=4
    
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    U[0][col] = 1.0; U[1][col] = 1.0
    V[5][col] = 1.0
    W[0][col] = -1.0; W[1][col] = 1.0  # C[0,0] -= M5, C[0,1] += M5
    col += 1  # col=5
    
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    U[3][col] = 1.0; U[0][col] = -1.0
    V[0][col] = 1.0; V[1][col] = 1.0
    W[4][col] = 1.0  # C[1,1] += M6
    col += 1  # col=6
    
    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    U[1][col] = 1.0; U[4][col] = -1.0
    V[4][col] = 1.0; V[5][col] = 1.0
    W[0][col] = 1.0  # C[0,0] += M7
    col += 1  # col=7
    
    # Strassen N terms for A[0:2,0:2] * B[0:2,2:4]
    # N1 = (A[0,0]+A[1,1]) * (B[0,2]+B[1,3])
    U[0][col] = 1.0; U[4][col] = 1.0
    V[2][col] = 1.0; V[7][col] = 1.0
    W[2][col] = 1.0; W[6][col] = 1.0  # C[0,2] += N1, C[1,3] += N1
    col += 1  # col=8
    
    # N2 = (A[1,0]+A[1,1]) * B[0,2]
    U[3][col] = 1.0; U[4][col] = 1.0
    V[2][col] = 1.0
    W[5][col] = 1.0; W[6][col] = -1.0  # C[1,2] += N2, C[1,3] -= N2
    col += 1  # col=9
    
    # N3 = A[0,0] * (B[0,3]-B[1,3])
    U[0][col] = 1.0
    V[3][col] = 1.0; V[7][col] = -1.0
    W[3][col] = 1.0; W[6][col] = 1.0  # C[0,3] += N3, C[1,3] += N3
    col += 1  # col=10
    
    # N4 = A[1,1] * (B[1,2]-B[0,2])
    U[4][col] = 1.0
    V[6][col] = 1.0; V[2][col] = -1.0
    W[2][col] = 1.0; W[5][col] = 1.0  # C[0,2] += N4, C[1,2] += N4
    col += 1  # col=11
    
    # N5 = (A[0,0]+A[0,1]) * B[1,3]
    U[0][col] = 1.0; U[1][col] = 1.0
    V[7][col] = 1.0
    W[2][col] = -1.0; W[3][col] = 1.0  # C[0,2] -= N5, C[0,3] += N5
    col += 1  # col=12
    
    # N6 = (A[1,0]-A[0,0]) * (B[0,2]+B[0,3])
    U[3][col] = 1.0; U[0][col] = -1.0
    V[2][col] = 1.0; V[3][col] = 1.0
    W[6][col] = 1.0  # C[1,3] += N6
    col += 1  # col=13
    
    # N7 = (A[0,1]-A[1,1]) * (B[1,2]+B[1,3])
    U[1][col] = 1.0; U[4][col] = -1.0
    V[6][col] = 1.0; V[7][col] = 1.0
    W[2][col] = 1.0  # C[0,2] += N7
    col += 1  # col=14
    
    # A[0:2,2] * B[2,0:2] (4 terms)
    # A[0,2]*B[2,0] -> C[0,0]
    U[2][col] = 1.0
    V[8][col] = 1.0
    W[0][col] = 1.0
    col += 1  # col=15
    # A[0,2]*B[2,1] -> C[0,1]
    U[2][col] = 1.0
    V[9][col] = 1.0
    W[1][col] = 1.0
    col += 1  # col=16
    # A[1,2]*B[2,0] -> C[1,0]
    U[5][col] = 1.0
    V[8][col] = 1.0
    W[3][col] = 1.0
    col += 1  # col=17
    # A[1,2]*B[2,1] -> C[1,1]
    U[5][col] = 1.0
    V[9][col] = 1.0
    W[4][col] = 1.0
    col += 1  # col=18
    
    # A[0:2,2] * B[2,2:4] (4 terms)
    # A[0,2]*B[2,2] -> C[0,2]
    U[2][col] = 1.0
    V[10][col] = 1.0
    W[2][col] = 1.0
    col += 1  # col=19
    # A[0,2]*B[2,3] -> C[0,3]
    U[2][col] = 1.0
    V[11][col] = 1.0
    W[3][col] = 1.0
    col += 1  # col=20
    # A[1,2]*B[2,2] -> C[1,2]
    U[5][col] = 1.0
    V[10][col] = 1.0
    W[5][col] = 1.0
    col += 1  # col=21
    # A[1,2]*B[2,3] -> C[1,3]
    U[5][col] = 1.0
    V[11][col] = 1.0
    W[6][col] = 1.0
    col += 1  # col=22
    
    # A[2,0:2] * B[0:2,0:2] (4 terms)
    # A[2,0]*B[0,0] -> C[2,0]
    U[6][col] = 1.0
    V[0][col] = 1.0
    W[6][col] = 1.0  # C[2,0]: k=0, i=2, W[0*3+2]=W[2]... wait
    # W[k*n+i] where k is column of C (0-3), i is row of C (0-2)
    # C[2,0]: i=2, k=0 -> W[0*3+2] = W[2]
    # C[2,1]: i=2, k=1 -> W[1*3+2] = W[5]
    # C[2,2]: i=2, k=2 -> W[2*3+2] = W[8]
    # C[2,3]: i=2, k=3 -> W[3*3+2] = W[11]
    # Let me redo the W indexing:
    # W[k*n + i] where k in [0,4), i in [0,3)
    # C[i,k] corresponds to W[k*3+i]
    # C[0,0]: W[0*3+0]=W[0]
    # C[0,1]: W[1*3+0]=W[3]
    # C[0,2]: W[2*3+0]=W[6]
    # C[0,3]: W[3*3+0]=W[9]
    # C[1,0]: W[0*3+1]=W[1]
    # C[1,1]: W[1*3+1]=W[4]
    # C[1,2]: W[2*3+1]=W[7]
    # C[1,3]: W[3*3+1]=W[10]
    # C[2,0]: W[0*3+2]=W[2]
    # C[2,1]: W[1*3+2]=W[5]
    # C[2,2]: W[2*3+2]=W[8]
    # C[2,3]: W[3*3+2]=W[11]
    
    # I made errors in the W indices above. Let me restart with correct indexing.
    # Let me redo this more carefully.
    return None  # placeholder, will fix

def _construct_rank34():
    """Construct rank-34 decomposition using Strassen on 2x2 blocks."""
    n, m, p = 3, 3, 4
    rank = 34
    U = [[0.0] * rank for _ in range(n * m)]  # 9 x 34
    V = [[0.0] * rank for _ in range(m * p)]  # 12 x 34
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x 34
    
    # W index: W[k*n + i] for C[i,k], i in [0,3), k in [0,4)
    # C[i,k] -> W[k*3 + i]
    
    col = 0
    # === Strassen M: A[0:2,0:2] * B[0:2,0:2] ===
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # C[0,0]+=M1, C[1,1]+=M1
    U[0][col]=1; U[4][col]=1
    V[0][col]=1; V[5][col]=1
    W[0][col]=1; W[4][col]=1
    col+=1
    
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # C[1,0]+=M2, C[1,1]-=M2
    U[3][col]=1; U[4][col]=1
    V[0][col]=1
    W[1][col]=1; W[4][col]=-1
    col+=1
    
    # M3 = A[0,0]*(B[0,1]-B[1,1])
    # C[0,1]+=M3, C[1,1]+=M3
    U[0][col]=1
    V[1][col]=1; V[5][col]=-1
    W[3][col]=1; W[4][col]=1
    col+=1
    
    # M4 = A[1,1]*(B[1,0]-B[0,0])
    # C[0,0]+=M4, C[1,0]+=M4
    U[4][col]=1
    V[4][col]=1; V[0][col]=-1
    W[0][col]=1; W[1][col]=1
    col+=1
    
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # C[0,0]-=M5, C[0,1]+=M5
    U[0][col]=1; U[1][col]=1
    V[5][col]=1
    W[0][col]=-1; W[3][col]=1
    col+=1
    
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # C[1,1]+=M6
    U[3][col]=1; U[0][col]=-1
    V[0][col]=1; V[1][col]=1
    W[4][col]=1
    col+=1
    
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    # C[0,0]+=M7
    U[1][col]=1; U[4][col]=-1
    V[4][col]=1; V[5][col]=1
    W[0][col]=1
    col+=1
    
    # === Strassen N: A[0:2,0:2] * B[0:2,2:4] ===
    # N1 = (A[0,0]+A[1,1])*(B[0,2]+B[1,3])
    # C[0,2]+=N1, C[1,3]+=N1
    U[0][col]=1; U[4][col]=1
    V[2][col]=1; V[7][col]=1
    W[6][col]=1; W[10][col]=1
    col+=1
    
    # N2 = (A[1,0]+A[1,1])*B[0,2]
    # C[1,2]+=N2, C[1,3]-=N2
    U[3][col]=1; U[4][col]=1
    V[2][col]=1
    W[7][col]=1; W[10][col]=-1
    col+=1
    
    # N3 = A[0,0]*(B[0,3]-B[1,3])
    # C[0,3]+=N3, C[1,3]+=N3
    U[0][col]=1
    V[3][col]=1; V[7][col]=-1
    W[9][col]=1; W[10][col]=1
    col+=1
    
    # N4 = A[1,1]*(B[1,2]-B[0,2])
    # C[0,2]+=N4, C[1,2]+=N4
    U[4][col]=1
    V[6][col]=1; V[2][col]=-1
    W[6][col]=1; W[7][col]=1
    col+=1
    
    # N5 = (A[0,0]+A[0,1])*B[1,3]
    # C[0,2]-=N5, C[0,3]+=N5
    U[0][col]=1; U[1][col]=1
    V[7][col]=1
    W[6][col]=-1; W[9][col]=1
    col+=1
    
    # N6 = (A[1,0]-A[0,0])*(B[0,2]+B[0,3])
    # C[1,3]+=N6
    U[3][col]=1; U[0][col]=-1
    V[2][col]=1; V[3][col]=1
    W[10][col]=1
    col+=1
    
    # N7 = (A[0,1]-A[1,1])*(B[1,2]+B[1,3])
    # C[0,2]+=N7
    U[1][col]=1; U[4][col]=-1
    V[6][col]=1; V[7][col]=1
    W[6][col]=1
    col+=1
    
    # === Schoolbook: A[0:2,2] * B[2,0:2] ===
    # A[0,2]*B[2,0] -> C[0,0]
    U[2][col]=1; V[8][col]=1; W[0][col]=1; col+=1
    # A[0,2]*B[2,1] -> C[0,1]
    U[2][col]=1; V[9][col]=1; W[3][col]=1; col+=1
    # A[1,2]*B[2,0] -> C[1,0]
    U[5][col]=1; V[8][col]=1; W[1][col]=1; col+=1
    # A[1,2]*B[2,1] -> C[1,1]
    U[5][col]=1; V[9][col]=1; W[4][col]=1; col+=1
    
    # === Schoolbook: A[0:2,2] * B[2,2:4] ===
    # A[0,2]*B[2,2] -> C[0,2]
    U[2][col]=1; V[10][col]=1; W[6][col]=1; col+=1
    # A[0,2]*B[2,3] -> C[0,3]
    U[2][col]=1; V[11][col]=1; W[9][col]=1; col+=1
    # A[1,2]*B[2,2] -> C[1,2]
    U[5][col]=1; V[10][col]=1; W[7][col]=1; col+=1
    # A[1,2]*B[2,3] -> C[1,3]
    U[5][col]=1; V[11][col]=1; W[10][col]=1; col+=1
    
    # === Schoolbook: A[2,0:2] * B[0:2,0:2] ===
    # A[2,0]*B[0,0] -> C[2,0]
    U[6][col]=1; V[0][col]=1; W[2][col]=1; col+=1
    # A[2,1]*B[1,0] -> C[2,0]
    U[7][col]=1; V[4][col]=1; W[2][col]=1; col+=1
    # A[2,0]*B[0,1] -> C[2,1]
    U[6][col]=1; V[1][col]=1; W[5][col]=1; col+=1
    # A[2,1]*B[1,1] -> C[2,1]
    U[7][col]=1; V[5][col]=1; W[5][col]=1; col+=1
    
    # === Schoolbook: A[2,0:2] * B[0:2,2:4] ===
    # A[2,0]*B[0,2] -> C[2,2]
    U[6][col]=1; V[2][col]=1; W[8][col]=1; col+=1
    # A[2,1]*B[1,2] -> C[2,2]
    U[7][col]=1; V[6][col]=1; W[8][col]=1; col+=1
    # A[2,0]*B[0,3] -> C[2,3]
    U[6][col]=1; V[3][col]=1; W[11][col]=1; col+=1
    # A[2,1]*B[1,3] -> C[2,3]
    U[7][col]=1; V[7][col]=1; W[11][col]=1; col+=1
    
    # === Schoolbook: A[2,2] * B[2,0:4] ===
    # A[2,2]*B[2,0] -> C[2,0]
    U[8][col]=1; V[8][col]=1; W[2][col]=1; col+=1
    # A[2,2]*B[2,1] -> C[2,1]
    U[8][col]=1; V[9][col]=1; W[5][col]=1; col+=1
    # A[2,2]*B[2,2] -> C[2,2]
    U[8][col]=1; V[10][col]=1; W[8][col]=1; col+=1
    # A[2,2]*B[2,3] -> C[2,3]
    U[8][col]=1; V[11][col]=1; W[11][col]=1; col+=1
    
    assert col == 34, f"Expected 34 columns, got {col}"
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Try rank-34 construction first
    try:
        U, V, W = _construct_rank34()
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
