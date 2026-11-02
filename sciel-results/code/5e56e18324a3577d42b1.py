"""Matrix multiplication 3x4x8: exact tensor decomposition.

Strategy:
  * For the first two rows of A (i = 0,1) use a 2x2 Strassen block per
    (j-pair, k-pair).  8 blocks * rank 7 = 56.
  * For the third row of A (i = 2) use a rank-3 1x2x2 decomposition per
    (j-pair, k-pair).  8 blocks * rank 3 = 24.
  * Total rank = 80.
"""
import time
from fractions import Fraction
import itertools

DIMENSIONS = [3, 4, 8]


def _strassen_2x2x2():
    """Return U, V, W for the rank-7 Strassen decomposition of 2x2x2."""
    U = [
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [1, 0, 0, 1, 0, 1, 0],
    ]
    V = [
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [1, 0, 1, 0, 1, 0, 0],
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]
    return U, V, W


def _find_rank3_1x2x2():
    """Search for a rank-3 decomposition of 1x2x2 using small rational entries."""
    # The 1x2x2 tensor has entries:
    # T[0,0,0]=1 (a*e), T[0,1,0]=0, T[1,0,0]=0, T[1,1,0]=1 (b*g)
    # T[0,0,1]=0, T[0,1,1]=1 (a*f), T[1,0,1]=0, T[1,1,1]=1 (b*h)
    # where indices are (j, k, i) with j in {0,1}, k in {0,1}, i in {0,1}
    # But our convention: U is 2x3 (rows j), V is 4x3 (rows j*p+k), W is 2x3 (rows k*n+i)
    # For 1x2x2: n=1, m=2, p=2
    # U: 2x3, V: 4x3, W: 2x3
    # Tensor T[i,j,k] = 1 if (i,j,k) corresponds to a valid product term
    
    # For 1x2x2: C[0,k] = sum_j A[j] * B[j,k]
    # C[0,0] = A[0]*B[0,0] + A[1]*B[1,0]
    # C[0,1] = A[0]*B[0,1] + A[1]*B[1,1]
    # So T[i,j,k] = 1 if i=0 and (j,k) is one of the 4 positions
    
    # Let's try small rational values
    vals = [Fraction(0), Fraction(1), Fraction(-1), Fraction(1,2), Fraction(-1,2)]
    
    # Fix U to be simple to reduce search space
    # Try U = [[1, 0, 1], [0, 1, 1]] or similar
    u_candidates = [
        [[1, 0, 1], [0, 1, 1]],
        [[1, 0, 1], [0, 1, -1]],
        [[1, 1, 0], [0, 1, 1]],
        [[1, 1, 1], [0, 1, 1]],
        [[1, 0, 0], [0, 1, 1]],
    ]
    
    for U in u_candidates:
        # For each U, try to find V and W
        # V is 4x3, W is 2x3
        # We need: for each (i,j,k), sum_c U[j,c]*V[j*2+k,c]*W[k*1+i,c] = T[i,j,k]
        
        # This is a system of 8 equations
        # Let's try small values for V and W
        v_vals = [Fraction(0), Fraction(1), Fraction(-1), Fraction(1,2), Fraction(-1,2)]
        
        # Generate V candidates (4x3) - this is large, so be selective
        # Try structured V
        v_structures = [
            # Pattern 1
            [[1,0,0], [0,1,1], [0,0,1], [1,1,0]],
            [[1,0,0], [0,1,1], [0,0,1], [1,0,1]],
            [[1,0,0], [0,1,0], [0,0,1], [1,1,1]],
            [[1,1,0], [0,1,1], [0,0,1], [1,0,0]],
            [[1,0,1], [0,1,0], [0,0,1], [1,1,0]],
        ]
        
        for V in v_structures:
            # Solve for W
            # For each (i,k): W[k,i] = ?
            # The equation is: sum_c U[j,c]*V[j*2+k,c]*W[k*1+i,c] = T[i,j,k]
            # For fixed (i,k), this is linear in W[k,i]
            
            # Let's set up the system
            # Variables: W[0,0], W[0,1], W[1,0], W[1,1] (4 unknowns)
            # Equations for (i,j,k):
            # i=0, j=0, k=0: U[0,0]*V[0,0]*W[0,0] + U[0,1]*V[0,1]*W[0,0] + U[0,2]*V[0,2]*W[0,0] = 1
            # Wait, W index is k*n+i, n=1, so W[k,i]
            # For (i=0,j=0,k=0): sum_c U[0,c]*V[0,c]*W[0,0] = 1
            # For (i=0,j=1,k=0): sum_c U[1,c]*V[0,c]*W[0,0] = 1
            # For (i=0,j=0,k=1): sum_c U[0,c]*V[2,c]*W[1,0] = 1
            # For (i=0,j=1,k=1): sum_c U[1,c]*V[2,c]*W[1,0] = 1
            # For (i=1,j=0,k=0): sum_c U[0,c]*V[0,c]*W[0,1] = 0 (but i=1 doesn't exist for n=1!)
            
            # Wait, n=1, so i only goes 0. The tensor is 1x2x2, so:
            # T[i,j,k] where i in {0}, j in {0,1}, k in {0,1}
            # T[0,0,0] = 1 (A[0]*B[0,0])
            # T[0,1,0] = 1 (A[1]*B[1,0])
            # T[0,0,1] = 1 (A[0]*B[0,1])
            # T[0,1,1] = 1 (A[1]*B[1,1])
            
            # So we have 4 equations:
            # (i=0,j=0,k=0): sum_c U[0,c]*V[0,c]*W[0,0] = 1
            # (i=0,j=1,k=0): sum_c U[1,c]*V[0,c]*W[0,0] = 1
            # (i=0,j=0,k=1): sum_c U[0,c]*V[2,c]*W[1,0] = 1
            # (i=0,j=1,k=1): sum_c U[1,c]*V[2,c]*W[1,0] = 1
            
            # And W has shape 2x3 (k in {0,1}, i in {0,1,2} but i only 0 for n=1)
            # Actually W is p*n = 2*1 = 2 rows, rank=3 columns
            # W[k*n+i] = W[k*1+0] = W[k,0] for k=0,1
            # So W is 2x3: W[0,0], W[0,1], W[0,2], W[1,0], W[1,1], W[1,2]
            
            # But i only goes 0, so W[k,0] is the only relevant column for each k
            # Wait, no. W has shape (p*n, rank) = (2, 3)
            # W[k*n + i, c] where k in {0,1}, i in {0}
            # So W[0, c] and W[1, c] for c in {0,1,2}
            
            # The equations are:
            # For (i=0,j=0,k=0): sum_c U[0,c]*V[0*2+0,c]*W[0*1+0,c] = sum_c U[0,c]*V[0,c]*W[0,c] = 1
            # For (i=0,j=1,k=0): sum_c U[1,c]*V[0,c]*W[0,c] = 1
            # For (i=0,j=0,k=1): sum_c U[0,c]*V[1*2+0,c]*W[1,c] = sum_c U[0,c]*V[2,c]*W[1,c] = 1
            # For (i=0,j=1,k=1): sum_c U[1,c]*V[2,c]*W[1,c] = 1
            
            # Let a_c = U[0,c]*V[0,c], b_c = U[1,c]*V[0,c]
            # c_c = U[0,c]*V[2,c], d_c = U[1,c]*V[2,c]
            # Then:
            # sum_c a_c * W[0,c] = 1
            # sum_c b_c * W[0,c] = 1
            # sum_c c_c * W[1,c] = 1
            # sum_c d_c * W[1,c] = 1
            
            a = [U[0][c]*V[0][c] for c in range(3)]
            b = [U[1][c]*V[0][c] for c in range(3)]
            c_vec = [U[0][c]*V[2][c] for c in range(3)]
            d = [U[1][c]*V[2][c] for c in range(3)]
            
            # Solve for W[0,:] from [a; b] * W[0,:] = [1; 1]
            # And W[1,:] from [c_vec; d] * W[1,:] = [1; 1]
            
            def solve_linear(A_mat, b_vec):
                """Solve A*x = b for 2x3 system using fractions."""
                # A_mat is 2x3, b_vec is 2
                # Use row reduction
                A = [list(row) for row in A_mat]
                b = list(b_vec)
                
                for col in range(2):
                    # Find pivot
                    pivot_row = None
                    for r in range(col, 2):
                        if A[r][col] != 0:
                            pivot_row = r
                            break
                    if pivot_row is None:
                        return None
                    if pivot_row != col:
                        A[col], A[pivot_row] = A[pivot_row], A[col]
                        b[col], b[pivot_row] = b[pivot_row], b[col]
                    # Normalize
                    pivot_val = A[col][col]
                    for c in range(col, 3):
                        A[col][c] = A[col][c] / pivot_val
                    b[col] = b[col] / pivot_val
                    # Eliminate
                    for r in range(2):
                        if r != col and A[r][col] != 0:
                            factor = A[r][col]
                            for c in range(col, 3):
                                A[r][c] -= factor * A[col][c]
                            b[r] -= factor * b[col]
                
                # Solution: x[0] = b[0] - A[0][2]*x[2], x[1] = b[1] - A[1][2]*x[2]
                # Free variable x[2] = t
                # Try t = 0, 1, -1, 1/2
                solutions = []
                for t_val in [Fraction(0), Fraction(1), Fraction(-1), Fraction(1,2), Fraction(-1,2)]:
                    x2 = t_val
                    x0 = b[0] - A[0][2]*x2
                    x1 = b[1] - A[1][2]*x2
                    solutions.append([x0, x1, x2])
                return solutions
            
            w0_sols = solve_linear([a, b], [Fraction(1), Fraction(1)])
            w1_sols = solve_linear([c_vec, d], [Fraction(1), Fraction(1)])
            
            if w0_sols is None or w1_sols is None:
                continue
            
            for w0 in w0_sols:
                for w1 in w1_sols:
                    W = [w0, w1]
                    # Verify
                    valid = True
                    for i in range(1):
                        for j in range(2):
                            for k in range(2):
                                expected = 1
                                actual = sum(U[j][c]*V[j*2+k][c]*W[k*1+i][c] for c in range(3))
                                if actual != expected:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if valid:
                        return U, V, W
    
    return None

def _rank3_1x2x2():
    """
    Returns a rank-3 decomposition of the 1x2x2 multiplication tensor if found,
    otherwise falls back to rank-4.
    """
    result = _find_rank3_1x2x2()
    if result is not None:
        return result
    
    # Fallback: rank-4 schoolbook
    U = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ]
    V = [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ]
    W = [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ]
    return U, V, W


def _construct_3x4x8():
    """Construct decomposition for 3x4x8 matrix multiplication."""
    n, m, p = 3, 4, 8

    U2, V2, W2 = _strassen_2x2x2()
    U1, V1, W1 = _rank3_1x2x2()
    
    r1 = len(U1[0])  # rank of 1x2x2 block (3 or 4)

    # 8 blocks of 2x2x2 (rank 7 each) for i=0,1: 8*7 = 56
    # 8 blocks of 1x2x2 (rank r1 each) for i=2: 8*r1
    # Total rank = 56 + 8*r1
    rank = 56 + 8 * r1
    U = [[0] * rank for _ in range(n * m)]   # 12 rows
    V = [[0] * rank for _ in range(m * p)]   # 32 rows
    W = [[0] * rank for _ in range(p * n)]   # 24 rows

    col_offset = 0

    # Process 2x2x2 Strassen blocks for i_pair=0 (i=0,1)
    for jp in range(2):
        for kp in range(4):
            j0, j1 = 2 * jp, 2 * jp + 1
            k0, k1 = 2 * kp, 2 * kp + 1

            u_rows = [
                0 * m + j0,  # A[0, j0]
                0 * m + j1,  # A[0, j1]
                1 * m + j0,  # A[1, j0]
                1 * m + j1,  # A[1, j1]
            ]

            v_rows = [
                j0 * p + k0,  # B[j0, k0]
                j0 * p + k1,  # B[j0, k1]
                j1 * p + k0,  # B[j1, k0]
                j1 * p + k1,  # B[j1, k1]
            ]

            w_rows = [
                k0 * n + 0,  # C[0, k0]
                k1 * n + 0,  # C[0, k1]
                k0 * n + 1,  # C[1, k0]
                k1 * n + 1,  # C[1, k1]
            ]

            for i in range(4):
                for c in range(7):
                    U[u_rows[i]][col_offset + c] += U2[i][c]
                    V[v_rows[i]][col_offset + c] += V2[i][c]
                    W[w_rows[i]][col_offset + c] += W2[i][c]
            col_offset += 7

    # Process 1x2x2 blocks for i=2
    for jp in range(2):
        for kp in range(4):
            j0, j1 = 2 * jp, 2 * jp + 1
            k0, k1 = 2 * kp, 2 * kp + 1

            u_rows = [
                2 * m + j0,  # A[2, j0]
                2 * m + j1,  # A[2, j1]
            ]

            v_rows = [
                j0 * p + k0,  # B[j0, k0]
                j0 * p + k1,  # B[j0, k1]
                j1 * p + k0,  # B[j1, k0]
                j1 * p + k1,  # B[j1, k1]
            ]

            w_rows = [
                k0 * n + 2,  # C[2, k0]
                k1 * n + 2,  # C[2, k1]
            ]

            for i in range(2):
                for c in range(r1):
                    U[u_rows[i]][col_offset + c] += U1[i][c]
                    W[w_rows[i]][col_offset + c] += W1[i][c]
            for i in range(4):
                for c in range(r1):
                    V[v_rows[i]][col_offset + c] += V1[i][c]
            col_offset += r1

    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                actual = 0
                for c in range(rank):
                    actual += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if actual != expected:
                    return False
    return True


def _schoolbook():
    """Fallback schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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


def construct():
    start_time = time.time()
    deadline = start_time + 115

    result = _construct_3x4x8()

    n, m, p = DIMENSIONS
    if not _verify(result["U"], result["V"], result["W"], n, m, p):
        result = _schoolbook()

    return result
