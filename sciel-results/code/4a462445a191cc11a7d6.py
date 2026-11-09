"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

DEADLINE = 145.0

def _construct_block_decomp(n, m, p):
    """
    Construct a block-based decomposition for n×m × m×p matrix multiplication.
    Splits into 2×2 blocks of A and 2×3 blocks of B (for the case n=4,m=4,p=9).
    """
    # For 4x4 x 4x9:
    # A is 4x4, split into 2x2 blocks
    # B is 4x9, split into 2x3 blocks (2 row-blocks of 2, 3 col-blocks of 3)
    # C is 4x9, split into 2x3 blocks (2 row-blocks of 2, 3 col-blocks of 3)
    
    if n != 4 or m != 4 or p != 9:
        # Fallback to schoolbook
        return _schoolbook(n, m, p)
    
    # Block structure:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01, B02], [B10, B11, B12]] where each is 2x3
    # C = [[C00, C01, C02], [C10, C11, C12]] where each is 2x3
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12
    
    # Each term X*Y is 2x2 * 2x3, rank 6 (schoolbook)
    # Total: 12 terms * 6 = 72
    
    rank = 72
    U = np.zeros((n * m, rank))  # 16 x 72
    V = np.zeros((m * p, rank))  # 36 x 72
    W = np.zeros((p * n, rank))  # 36 x 72
    
    col = 0
    
    # Helper: add a 2x2 * 2x3 schoolbook term
    # A_block is at rows (ar0, ar0+1) x cols (ac0, ac0+1) of A
    # B_block is at rows (br0, br0+1) x cols (bc0, bc0+1, bc0+2) of B
    # C_block is at rows (cr0, cr0+1) x cols (cc0, cc0+1, cc0+2) of C
    def add_2x2_x_2x3(ar0, ac0, br0, bc0, cr0, cc0):
        nonlocal col
        # 2x2 * 2x3 schoolbook: 6 multiplications
        # A elements: (ar0,ac0), (ar0,ac0+1), (ar0+1,ac0), (ar0+1,ac0+1)
        # B elements: (br0,bc0), (br0,bc0+1), (br0,bc0+2), (br0+1,bc0), (br0+1,bc0+1), (br0+1,bc0+2)
        # C elements: (cr0,cc0), (cr0,cc0+1), (cr0,cc0+2), (cr0+1,cc0), (cr0+1,cc0+1), (cr0+1,cc0+2)
        
        # For each (i,j) in 2x2 and (k) in 3:
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    # U index: row (ar0+i)*m + (ac0+j) = (ar0+i)*4 + (ac0+j)
                    u_idx = (ar0 + i) * m + (ac0 + j)
                    # V index: row (br0+j)*p + (bc0+k) = (br0+j)*9 + (bc0+k)
                    v_idx = (br0 + j) * p + (bc0 + k)
                    # W index: row (cc0+k)*n + (cr0+i) = (cc0+k)*4 + (cr0+i)
                    w_idx = (cc0 + k) * n + (cr0 + i)
                    
                    U[u_idx, col] = 1.0
                    V[v_idx, col] = 1.0
                    W[w_idx, col] = 1.0
                    col += 1
    
    # C00 = A00*B00 + A01*B10
    add_2x2_x_2x3(0, 0, 0, 0, 0, 0)   # A00*B00 -> C00
    add_2x2_x_2x3(0, 2, 2, 0, 0, 0)   # A01*B10 -> C00
    
    # C01 = A00*B01 + A01*B11
    add_2x2_x_2x3(0, 0, 0, 3, 0, 3)   # A00*B01 -> C01
    add_2x2_x_2x3(0, 2, 2, 3, 0, 3)   # A01*B11 -> C01
    
    # C02 = A00*B02 + A01*B12
    add_2x2_x_2x3(0, 0, 0, 6, 0, 6)   # A00*B02 -> C02
    add_2x2_x_2x3(0, 2, 2, 6, 0, 6)   # A01*B12 -> C02
    
    # C10 = A10*B00 + A11*B10
    add_2x2_x_2x3(2, 0, 0, 0, 2, 0)   # A10*B00 -> C10
    add_2x2_x_2x3(2, 2, 2, 0, 2, 0)   # A11*B10 -> C10
    
    # C11 = A10*B01 + A11*B11
    add_2x2_x_2x3(2, 0, 0, 3, 2, 3)   # A10*B01 -> C11
    add_2x2_x_2x3(2, 2, 2, 3, 2, 3)   # A11*B11 -> C11
    
    # C12 = A10*B02 + A11*B12
    add_2x2_x_2x3(2, 0, 0, 6, 2, 6)   # A10*B02 -> C12
    add_2x2_x_2x3(2, 2, 2, 6, 2, 6)   # A11*B12 -> C12
    
    assert col == rank, f"Column mismatch: {col} != {rank}"
    
    return U, V, W

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check that the decomposition is exact for matrix multiplication."""
    rank = U.shape[1]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _to_rational(x, max_den=8):
    """Snap a float to a half-integer rational if close; else return None."""
    if x is None:
        return None
    for den in (1, 2):
        num = round(x * den)
        if abs(x - num / den) < 1e-6:
            return num / den
    return None

def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    time_budget = DEADLINE - t_start - 5.0
    
    if time_budget < 1.0:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try the block decomposition first
    U, V, W = _construct_block_decomp(n, m, p)
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
    
    # Convert to lists
    U_list = [[_to_rational(x) if _to_rational(x) is not None else float(x) for x in row] for row in U]
    V_list = [[_to_rational(x) if _to_rational(x) is not None else float(x) for x in row] for row in V]
    W_list = [[_to_rational(x) if _to_rational(x) is not None else float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
