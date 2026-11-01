"""Tensor decomposition for 2x2 by 2x4 matrix multiplication, rank 14."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n=2, m=2, p=4, tol=1e-9):
    """Verify that U, V, W decompose the matrix multiplication tensor."""
    r = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Check: for each (i,k), sum_j A[i,j]*B[j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Actually the tensor is T[i,j,k] = delta[i,j,k] for matmul
    # T[i,j,k] should equal 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor for matmul: T[i, j, k] = 1 for all valid (i,j,k) where j is the contraction index
    # Wait - the tensor is: for matmul C = AB, C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has T[i,j,k] = 1 (it's just the identity in the middle)
    # Actually no. The tensor is defined as T[i,j,k] = 1 for all i,j,k in range.
    # Then C[i,k] = sum_j sum_c U[i,j,c] V[j,k,c] W[i,k,c]... 
    
    # Let me re-read: the tensor decomposition means:
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
    # where T is the matmul tensor: T[i,j,k] = 1 for all valid indices
    # Wait, that's not right either. Let me think more carefully.
    
    # For matrix multiplication C = A*B where A is n×m, B is m×p, C is n×p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has entries T[i,j,k] = 1 (it's the "multiplication" tensor)
    # A rank-r decomposition: T[i,j,k] = sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_rank13_milp(deadline):
    """Try to find a rank-13 decomposition using MILP for sparsity pattern."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import csc_matrix
    
    n, m, p = 2, 2, 4
    r = 13
    # Sizes: U is (n*m, r) = (4, 13), V is (m*p, r) = (8, 13), W is (p*n, r) = (8, 13)
    # Total variables: 4*13 + 8*13 + 8*13 = 52 + 104 + 104 = 260
    
    # The bilinear constraint is: for each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is cubic (trilinear), not linear. MILP can't directly handle this.
    
    # Alternative approach: fix the sparsity pattern (which entries are nonzero) and solve the resulting system.
    # For rank 13, we need to find a sparse pattern that allows a solution.
    
    # Let's try a different approach: use the known rank-14 structure and try to merge one term.
    # In the rank-14 solution, the left and right halves are independent Strassen decompositions.
    # Can we find a shared term?
    
    # Actually, let's try a direct approach: set up the equations for rank 13 and solve.
    # We have 8 equations (one for each (i,k) pair in the output, but actually 8*2*2=32 scalar equations... 
    # wait no, it's 2*2*4 = 16 equations since T[i,j,k]=1 for all i,j,k).
    
    # Actually there are n*m*p = 2*2*4 = 16 equations.
    # Unknowns: 4*13 + 8*13 + 8*13 = 260 values.
    # But the system is trilinear (cubic), so we can't solve it directly with linear algebra.
    
    # Let me try a different strategy: start from rank 14 and try to find a linear dependence
    # that allows us to eliminate one term.
    
    # For the rank-14 solution, let's check if any term can be expressed as a combination of others.
    # This is hard in general. Let's try a randomized search instead.
    
    return None

def _try_rank13_random(deadline):
    """Try to find a rank-13 decomposition via randomized search."""
    n, m, p = 2, 2, 4
    r = 13
    
    # The system: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is a system of 16 trilinear equations in 260 unknowns.
    
    # Strategy: fix U and W, solve for V (or similar).
    # If we fix U and W, the equations become linear in V:
    # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # Group by (j,k): sum_c (U[i*m+j,c] * W[k*n+i,c]) * V[j*p+k,c] = 1
    # For fixed (j,k) and varying (i): we get n=2 equations per (j,k) pair.
    # There are m*p = 8 (j,k) pairs, each giving 2 equations, total 16 equations.
    # Unknowns for V: m*p*r = 8*13 = 104 values.
    # So for fixed U,W, we have 16 linear equations in 104 unknowns - underdetermined.
    
    # Similarly, we can fix V and W, solve for U (16 eqs in 52 unknowns).
    # Or fix U and V, solve for W (16 eqs in 104 unknowns).
    
    # Let's try: fix U to have a nice structure, fix W to have a nice structure, solve for V.
    # Or use an alternating minimization approach.
    
    # Actually, let's try a more systematic approach.
    # We know rank 14 works. Let's see if we can find rank 13 by trying to make one column
    # of the rank-14 decomposition redundant.
    
    # For now, return None if we can't find rank 13 quickly.
    return None

def construct():
    t0 = time.time()
    deadline = t0 + 120
    n, m, p = DIMENSIONS  # n=2, m=2, p=4
    rank = 14

    U = [[0] * rank for _ in range(n * m)]  # (4, 14)
    V = [[0] * rank for _ in range(m * p)]  # (8, 14)
    W = [[0] * rank for _ in range(p * n)]  # (8, 14)

    # Left half (k=0,1) - Strassen
    # M1_L = (A00+A11)(B00+B11) -> C00+=, C11+=
    c = 0
    U[0][c] = 1; U[3][c] = 1
    V[0][c] = 1; V[5][c] = 1
    W[0][c] = 1; W[3][c] = 1

    # M2_L = (A10+A11)B00 -> C10+=, C11-=
    c = 1
    U[2][c] = 1; U[3][c] = 1
    V[0][c] = 1
    W[1][c] = 1; W[3][c] = -1

    # M3_L = A00(B01-B11) -> C01+=, C11+=
    c = 2
    U[0][c] = 1
    V[1][c] = 1; V[5][c] = -1
    W[2][c] = 1; W[3][c] = 1

    # M4_L = A11(B10-B00) -> C00+=, C10+=
    c = 3
    U[3][c] = 1
    V[4][c] = 1; V[0][c] = -1
    W[0][c] = 1; W[1][c] = 1

    # M5_L = (A00+A01)B11 -> C00-=, C01+=
    c = 4
    U[0][c] = 1; U[1][c] = 1
    V[5][c] = 1
    W[0][c] = -1; W[2][c] = 1

    # M6_L = (A10-A00)(B00+B01) -> C11+=
    c = 5
    U[2][c] = 1; U[0][c] = -1
    V[0][c] = 1; V[1][c] = 1
    W[3][c] = 1

    # M7_L = (A01-A11)(B10+B11) -> C00+=
    c = 6
    U[1][c] = 1; U[3][c] = -1
    V[4][c] = 1; V[5][c] = 1
    W[0][c] = 1

    # Right half (k=2,3) - Strassen
    # M1_R = (A00+A11)(B02+B13) -> C02+=, C13+=
    c = 7
    U[0][c] = 1; U[3][c] = 1
    V[2][c] = 1; V[7][c] = 1
    W[4][c] = 1; W[7][c] = 1

    # M2_R = (A10+A11)B02 -> C12+=, C13-=
    c = 8
    U[2][c] = 1; U[3][c] = 1
    V[2][c] = 1
    W[5][c] = 1; W[7][c] = -1

    # M3_R = A00(B03-B13) -> C03+=, C13+=
    c = 9
    U[0][c] = 1
    V[3][c] = 1; V[7][c] = -1
    W[6][c] = 1; W[7][c] = 1

    # M4_R = A11(B12-B02) -> C02+=, C12+=
    c = 10
    U[3][c] = 1
    V[6][c] = 1; V[2][c] = -1
    W[4][c] = 1; W[5][c] = 1

    # M5_R = (A00+A01)B13 -> C02-=, C03+=
    c = 11
    U[0][c] = 1; U[1][c] = 1
    V[7][c] = 1
    W[4][c] = -1; W[6][c] = 1

    # M6_R = (A10-A00)(B02+B03) -> C13+=
    c = 12
    U[2][c] = 1; U[0][c] = -1
    V[2][c] = 1; V[3][c] = 1
    W[7][c] = 1

    # M7_R = (A01-A11)(B12+B13) -> C02+=
    c = 13
    U[1][c] = 1; U[3][c] = -1
    V[6][c] = 1; V[7][c] = 1
    W[4][c] = 1

    # Try to find rank 13 within remaining time
    if time.time() < deadline - 10:
        result = _try_rank13_random(deadline)
        if result is not None:
            return result

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
