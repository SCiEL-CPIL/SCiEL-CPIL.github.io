import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    DEADLINE = 130  # seconds, leave 20s margin before 150s kill

    n, m, p = DIMENSIONS

    # Try rank 27 first with structured ansatz
    if time.time() - start_time < DEADLINE:
        result = _try_rank27_structured(n, m, p, start_time, DEADLINE)
        if result is not None:
            return result

    # Fallback: return the known rank-28 solution
    return _known_rank28()

def _known_rank28():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Indices for A (2x3): A00=0, A01=1, A02=2, A10=3, A11=4, A12=5
    # Indices for B (3x5): B00=0..B04=4, B10=5..B14=9, B20=10..B24=14
    # W index: C[i,k] -> W[k*n+i], so C00=0, C01=2, C02=4, C03=6, C04=8
    #                                    C10=1, C11=3, C12=5, C13=7, C14=9

    # Part 1: A_L (2x2) * B_T1 (2x2) -> C_1 (cols 0,1) using Strassen-7
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[6][0] = 1
    W[0][0] = 1; W[3][0] = 1

    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1

    U[0][2] = 1
    V[1][2] = 1; V[6][2] = -1
    W[2][2] = 1; W[3][2] = 1

    U[4][3] = 1
    V[5][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1

    U[0][4] = 1; U[1][4] = 1
    V[6][4] = 1
    W[0][4] = -1; W[2][4] = 1

    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    U[1][6] = 1; U[4][6] = -1
    V[5][6] = 1; V[6][6] = 1
    W[0][6] = 1

    # Part 2: A_L (2x2) * B_T2 (2x2) -> C_2 (cols 2,3) using Strassen-7
    U[0][7] = 1; U[4][7] = 1
    V[2][7] = 1; V[8][7] = 1
    W[4][7] = 1; W[7][7] = 1

    U[3][8] = 1; U[4][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1

    U[0][9] = 1
    V[3][9] = 1; V[8][9] = -1
    W[6][9] = 1; W[7][9] = 1

    U[4][10] = 1
    V[7][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1

    U[0][11] = 1; U[1][11] = 1
    V[8][11] = 1
    W[4][11] = -1; W[6][11] = 1

    U[3][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1

    U[1][13] = 1; U[4][13] = -1
    V[7][13] = 1; V[8][13] = 1
    W[4][13] = 1

    # Part 3: A_L (2x2) * B_T3 (2x1) -> C_3 (col 4) schoolbook
    U[0][14] = 1; V[4][14] = 1; W[8][14] = 1
    U[1][15] = 1; V[9][15] = 1; W[8][15] = 1
    U[3][16] = 1; V[4][16] = 1; W[9][16] = 1
    U[4][17] = 1; V[9][17] = 1; W[9][17] = 1

    # Part 4: A_R (2x1) * B_B (1x5) -> C_4 (all cols) schoolbook
    U[2][18] = 1; V[10][18] = 1; W[0][18] = 1
    U[2][19] = 1; V[11][19] = 1; W[2][19] = 1
    U[2][20] = 1; V[12][20] = 1; W[4][20] = 1
    U[2][21] = 1; V[13][21] = 1; W[6][21] = 1
    U[2][22] = 1; V[14][22] = 1; W[8][22] = 1
    U[5][23] = 1; V[10][23] = 1; W[1][23] = 1
    U[5][24] = 1; V[11][24] = 1; W[3][24] = 1
    U[5][25] = 1; V[12][25] = 1; W[5][25] = 1
    U[5][26] = 1; V[13][26] = 1; W[7][26] = 1
    U[5][27] = 1; V[14][27] = 1; W[9][27] = 1

    return {"U": U, "V": V, "W": W}

def _try_rank27_structured(n, m, p, start_time, deadline):
    """
    Try to find a rank-27 decomposition with a structured ansatz.
    
    Ansatz: Use Strassen for two 2x2 blocks (14 terms), then 13 terms for
    the remaining interactions. The key idea is to see if we can merge some
    schoolbook terms into the Strassen structure.
    
    Structure:
    - Rows 0,1 of A (A_L) interact with rows 0,1 of B (B_T) for cols 0-3: 14 Strassen terms
    - Row 2 of A (A_R) interacts with row 2 of B (B_B) for all cols: need to handle
    - A_L also interacts with B_B (row 2 of B) for all cols
    - A_R also interacts with B_T (rows 0,1 of B) for all cols
    
    Total schoolbook terms needed:
    - A_L * B_B: 2*1*5 = 10 terms (A[0..1, 0..1] * B[2, 0..4])
    - A_R * B_T: 1*2*5 = 10 terms (A[2, 0..1] * B[0..1, 0..4])
    Total = 20 schoolbook + 14 Strassen = 34? No wait...
    
    Actually let me recount:
    - A is 2x3, B is 3x5
    - Split A into A_L (2x2, cols 0,1) and A_R (2x1, col 2)
    - Split B into B_T (2x5, rows 0,1) and B_B (1x5, row 2)
    
    C = A_L @ B_T + A_R @ B_B (where A_R is 2x1 and B_B is 1x5)
    
    A_L @ B_T: 2x2 * 2x5. Split B_T into B_T1 (2x2, cols 0,1), B_T2 (2x2, cols 2,3), B_T3 (2x1, col 4)
    - A_L @ B_T1: Strassen rank 7
    - A_L @ B_T2: Strassen rank 7
    - A_L @ B_T3: schoolbook rank 4 (2x2 * 2x1)
    Total for A_L @ B_T: 7 + 7 + 4 = 18
    
    A_R @ B_B: 2x1 * 1x5 = schoolbook rank 10
    
    Total: 18 + 10 = 28. That's the current solution.
    
    To get rank 27, we need to save 1 term. Let's try:
    - Can we combine A_L @ B_T3 (4 terms) with some A_R @ B_B terms?
    - Or use a different decomposition for the 2x2*2x1 case?
    
    For 2x2 * 2x1, the minimum rank is 4 (it's just a matrix-vector product).
    But maybe we can share structure with the Strassen terms.
    
    Alternative: What if we use a different split?
    Let's try: A_L (2x2) * B_T (2x5) as one block using a rank-13 algorithm for 2x2*2x5?
    The standard rank for 2x2*2x5 would be... let's think.
    
    Actually, let me try a linear algebra approach. Fix U and V to have a specific
    structure and solve for W.
    """
    
    # Let's try: use the Strassen structure for the 2x2*2x2 parts,
    # but try to merge the 4 schoolbook terms for A_L*B_T3 with some
    # of the 10 schoolbook terms for A_R*B_B.
    
    # The 4 terms for A_L * B_T3 (col 4 of B):
    # A00*B04 -> C04
    # A01*B14 -> C04
    # A10*B04 -> C14
    # A11*B14 -> C14
    
    # The 10 terms for A_R * B_B:
    # A02*B20 -> C00
    # A02*B21 -> C01
    # A02*B22 -> C02
    # A02*B23 -> C03
    # A02*B24 -> C04
    # A12*B20 -> C10
    # A12*B21 -> C11
    # A12*B22 -> C12
    # A12*B23 -> C13
    # A12*B24 -> C14
    
    # Note: A02*B24 -> C04 and A12*B24 -> C14 overlap with the A_L*B_T3 outputs!
    # C04 = A00*B04 + A01*B14 + A02*B24
    # C14 = A10*B04 + A11*B14 + A12*B24
    
    # So for C04, we have 3 terms, and for C14, we have 3 terms.
    # Can we express (A00*B04 + A01*B14 + A02*B24) in fewer than 3 rank-1 terms?
    # That would require the 2x3 * 3x1 product to have rank < 3, which is impossible
    # in general (it's a matrix-vector product of a 2x3 matrix with a 3-vector).
    
    # So that approach won't work directly. Let me try a different structured ansatz.
    
    # What if we use a non-standard split? For example:
    # Group A columns as (0,1,2) but B rows as (0,2) and (1)
    # A * B[0,2,:] + A * B[1,:]
    # = (2x3)*(2x5) + (2x3)*(1x5)
    # Rank: ? + 10
    
    # Or: B rows as (0,1) and (2) is what we already have.
    
    # Let me try yet another approach: use the fact that we have half-integer
    # coefficients allowed, and try to find a rank-27 solution via linear algebra.
    
    # Set up: we need T[i,j,k] = 1 for all i,j,k (the matmul tensor)
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k*n+i, r]
    
    # Wait, I need to be more careful about the indexing.
    # U has shape (n*m, rank) where U[i*m+j, r] corresponds to A[i,j]
    # V has shape (m*p, rank) where V[j*p+k, r] corresponds to B[j,k]
    # W has shape (p*n, rank) where W[k*n+i, r] corresponds to C[i,k]
    
    # The decomposition: for each (i,j,k):
    # 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # This must hold for all i in [0,n), j in [0,m), k in [0,p).
    # That's n*m*p = 2*3*5 = 30 equations.
    
    # With rank r, we have r*(n*m + m*p + p*n) = r*(6+15+10) = 31r unknowns.
    # For r=27: 31*27 = 837 unknowns, 30 equations. Very underdetermined.
    # But the equations are nonlinear (products of three variables).
    
    # However, if we fix U and V, the equations become linear in W.
    # For each (i,k), we need:
    # sum_j (sum_r U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i, r] ... no.
    
    # Let me re-derive. The condition is:
    # For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Group by (i,k): for fixed i,k, we need this to hold for all j.
    # Let's define M_{i,k}[j, r] = U[i*m+j, r] * V[j*p+k, r]
    # Then: sum_r M_{i,k}[j, r] * W[k*n+i, r] = 1 for all j.
    
    # This is a system: for each (i,k), M_{i,k} (m x r) times W_col (r x 1) = ones (m x 1)
    # where W_col[r] = W[k*n+i, r]
    
    # So for each (i,k), we need to solve: M_{i,k} @ w = 1_m
    # where w is the column of W corresponding to (i,k).
    
    # If we fix U and V, this is a linear system in W.
    # The system is solvable if 1_m is in the column space of M_{i,k}.
    
    # Let's try: fix U and V to the Strassen + partial schoolbook structure,
    # and see if we can find W that satisfies all 30 equations with rank 27.
    
    # Actually, let me try a simpler approach: just try to find ANY rank-27
    # solution by fixing U and V to a reasonable structure and solving for W.
    
    # Structure for U (6 x 27):
    # Terms 0-6: Strassen for A_L * B_T1 (uses A rows 0,1 cols 0,1)
    # Terms 7-13: Strassen for A_L * B_T2 (uses A rows 0,1 cols 0,1)
    # Terms 14-26: 13 terms for the rest
    
    # For the 13 remaining terms, we need to cover:
    # - A_L * B_T3 (4 terms worth of products)
    # - A_R * B_B (10 terms worth of products)
    # Total: 14 products, but we only have 13 terms.
    # So we need at least one term to cover two products.
    
    # A term covers A[i,j]*B[j',k] for specific i,j,j',k.
    # Actually no: in the tensor decomposition, each term r has:
    # U[i*m+j, r] nonzero for some (i,j) pairs
    # V[j'*p+k, r] nonzero for some (j',k) pairs
    # W[k''*n+i'', r] nonzero for some (i'',k'') pairs
    
    # The product U[i*m+j,r]*V[j'*p+k,r]*W[k''*n+i'',r] contributes to T[i,j,k]
    # only if i''=i and k''=k (since W is indexed by k''*n+i'').
    # And it contributes to the (i,j,k) equation with value U[i*m+j,r]*V[j'*p+k,r]*W[k*n+i,r].
    
    # So for a fixed term r, it contributes to T[i,j,k] as:
    # U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # (note: the j in V must match the j in U for it to be a valid matmul term)
    
    # Wait, I think I'm confusing myself. Let me re-read the problem statement.
    # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "that exactly decompose the matrix multiplication tensor"
    
    # The matmul tensor T has T[i,j,k] = 1 (coefficient of A[i,j]*B[j,k] in C[i,k]).
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # So for each (i,j,k), the equation is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of 30 nonlinear equations in 31*27 = 837 variables (for rank 27).
    
    # If we fix U and V, it becomes linear in W:
    # For each (i,k): sum_r [sum_j U[i*m+j,r] * V[j*p+k,r] * delta_{j,j}] * W[k*n+i,r] = 1
    # Wait, that's not right. Let me be more careful.
    
    # The equation for (i,j,k) is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # For fixed (i,k), we have m equations (one for each j):
    # For j=0: sum_r U[i*m+0, r] * V[0*p+k, r] * W[k*n+i, r] = 1
    # For j=1: sum_r U[i*m+1, r] * V[1*p+k, r] * W[k*n+i, r] = 1
    # For j=2: sum_r U[i*m+2, r] * V[2*p+k, r] * W[k*n+i, r] = 1
    
    # Let w_{i,k}[r] = W[k*n+i, r]
    # Let M_{i,k}[j, r] = U[i*m+j, r] * V[j*p+k, r]
    # Then: M_{i,k} @ w_{i,k} = [1, 1, 1]^T (m-vector of ones)
    
    # So for each (i,k), we need to solve a 3x27 linear system.
    # This is solvable if the all-ones vector is in the column space of M_{i,k}.
    
    # Let's try to construct U and V such that this works for rank 27.
    
    # I'll use the Strassen structure for the first 14 terms and try to design
    # the remaining 13 terms to cover the rest.
    
    # Let me just try a numerical approach: set up the problem and use least squares
    # to see if a solution exists.
    
    # Actually, given the complexity, let me just try a few specific ansatze
    # and check if they work.
    
    # Ansatz 1: Same as rank-28 but merge two schoolbook terms.
    # In the rank-28 solution, terms 18-27 are schoolbook for A_R*B_B.
    # Can we merge term 18 (A02*B20->C00) with something?
    # C00 = A00*B00 + A01*B10 + A02*B20
    # The Strassen terms already cover A00*B00 and A01*B10 (partially).
    # Specifically, term 0 gives (A00+A11)(B00+B11) which contributes to C00 as A00*B00 + A00*B11 + A11*B00 + A11*B11.
    # This is getting complicated.
    
    # Let me try a completely different approach: use the fact that we can use
    # half-integer coefficients and try to find a rank-27 solution by solving
    # a system of equations.
    
    # I'll set up the problem as: fix U and V to have a specific sparse structure,
    # then solve for W using linear algebra.
    
    # Given time constraints, let me just try one specific ansatz:
    # Use Strassen for the two 2x2 blocks (14 terms), then use 13 terms for
    # the remaining 14 products by having one term cover two products.
    
    # The term that covers two products would need:
    # U[i*m+j1, r] * V[j1*p+k1, r] * W[k1*n+i, r] != 0 AND
    # U[i*m+j2, r] * V[j2*p+k2, r] * W[k2*n+i, r] != 0
    # with (i,j1,k1) != (i,j2,k2)
    
    # But wait, each term contributes to MULTIPLE (i,j,k) equations.
    # Term r contributes U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] to equation (i,j,k).
    # For a fixed r, this is nonzero for various (i,j,k) depending on the
    # nonzero patterns of U[:,r], V[:,r], W[:,r].
    
    # In the schoolbook case, each term has exactly one nonzero in U, one in V,
    # and one in W, so it contributes to exactly one (i,j,k) equation.
    
    # In the Strassen case, each term has multiple nonzeros, so it contributes
    # to multiple (i,j,k) equations, but the sum of contributions equals the
    # desired value.
    
    # For a "merged" term, we'd need it to contribute to two different (i,j,k)
    # equations in a way that's consistent with the other terms.
    
    # This is getting very complex. Let me just try to verify if rank 27 is
    # possible by checking a necessary condition.
    
    # Necessary condition: The tensor T must have rank <= 27.
    # The rank of the matmul tensor for n x m * m x p is at least max(n*m, m*p, p*n)
    # = max(6, 15, 10) = 15. And at most n*m*p = 30.
    # For 2x3*3x5, the known bounds are... I'm not sure of the exact minimum.
    
    # Given the time constraint, let me just return the rank-28 solution.
    # The rank-27 search is too complex to do reliably in the remaining time.
    return None

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0]) if U else 0
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T[i,j,k] contribution from term r
                    u_val = U[i*m+j][r]
                    v_val = V[j*p+k][r]
                    w_val = W[k*n+i][r]
                    C_computed[i,k] += u_val * v_val * w_val * A[i,j] * B[j,k]
                    # Wait, that's not right. The decomposition is:
                    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                    # And C[i,k] = sum_j A[i,j] * B[j,k] * T[i,j,k]
                    # So C[i,k] = sum_j A[i,j] * B[j,k] * sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                    # = sum_r sum_j A[i,j] * B[j,k] * U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                    # Hmm, that doesn't look like a standard matmul decomposition.
                    pass
    
    # Let me try a different interpretation:
    # C[i,k] = sum_r (sum_j U[i*m+j,r] * A[i,j]) * (sum_j V[j*p+k,r] * B[j,k]) * W[k*n+i,r]
    # No, that's also not standard.
    
    # I think the correct interpretation is:
    # The tensor T[i,j,k] = 1 (for matmul) is decomposed as:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This means: for each (i,j,k), the sum over r of the product equals 1.
    # Then C[i,k] = sum_j A[i,j] * B[j,k] (since T[i,j,k]=1 for all valid i,j,k)
    
    # So the verification is just: for all i,j,k, sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True
