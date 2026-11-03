"""Matrix multiplication 3,4,7 - rank-minimal exact tensor decomposition"""
import time
import itertools
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 7]
TIME_LIMIT = 130  # seconds, with safety margin

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a flat array."""
    tensor = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor[a, b, c] = 1.0
    return tensor

def _check_decomposition(U, V, W, tensor, tol=1e-9):
    """Check if U, V, W exactly decompose the tensor."""
    r = U.shape[1]
    for a in range(tensor.shape[0]):
        for b in range(tensor.shape[1]):
            for c in range(tensor.shape[2]):
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - tensor[a, b, c]) > tol:
                    return False
    return True

def _strassen_2x2():
    """Return Strassen decomposition for 2x2x2. Rank 7."""
    # Strassen's 7 products for 2x2 matrix multiplication
    # U: (4,7), V: (4,7), W: (4,7)
    # Indices: U row = i*2+j, V row = j*2+k, W row = k*2+i
    # Products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # A00=0, A01=1, A10=2, A11=3
    # B00=0, B01=1, B10=2, B11=3
    # C00=0, C01=1, C10=2, C11=3
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1; W[1, 0] = -1; W[2, 0] = -1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[1, 2] = 1
    
    # M4 = A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[3, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[2, 5] = 1; W[3, 5] = 1
    
    # M7 = (A01-A11)(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[1, 6] = 1
    
    return U, V, W

def _strassen_2x2_2x3():
    """
    Decompose 2x2 * 2x3 multiplication.
    We can view this as two 2x2*2x2 multiplications plus adjustments,
    or use a direct approach.
    
    Actually, let's use a block approach:
    A is 2x2, B is 2x3, C is 2x3.
    
    We can split B into B_left (2x2) and B_right (2x1).
    C_left = A * B_left (rank 7 via Strassen)
    C_right = A * B_right (rank 4 via schoolbook, or less)
    
    For A (2x2) * B_right (2x1):
    C_right[0] = A[0,0]*B_right[0] + A[0,1]*B_right[1]
    C_right[1] = A[1,0]*B_right[0] + A[1,1]*B_right[1]
    
    This needs 4 products. Total rank = 7 + 4 = 11.
    
    But we can do better. Let me think of a direct decomposition.
    
    Actually, for 2x2 * 2x3, the schoolbook rank is 12.
    Let me try to find a rank-11 or lower decomposition.
    
    For now, let's use the block approach: rank 7 + 4 = 11.
    """
    # Strassen for the 2x2 * 2x2 part (columns 0,1 of B)
    U_str, V_str, W_str = _strassen_2x2()
    
    # Schoolbook for the 2x2 * 2x1 part (column 2 of B)
    U_sb = np.zeros((4, 4))
    V_sb = np.zeros((4, 4))
    W_sb = np.zeros((4, 4))
    
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B_right indices: 0=B_right[0], 1=B_right[1] -> but in the 2x3 matrix, these are at positions 2 and 3
    # Wait, B is 2x3. B[j,k] where j=0,1 (rows of B = cols of A) and k=0,1,2.
    # B index in V: j*3+k
    # So B_right is at B[0,2] (index 2) and B[1,2] (index 5)
    
    # C is 2x3. C[i,k] where i=0,1 and k=0,1,2.
    # C index in W: k*2+i
    # C_right is at C[0,2] (index 4) and C[1,2] (index 5)
    
    # Product 1: A00 * B02
    U_sb[0, 0] = 1
    V_sb[2, 0] = 1  # B[0,2]
    W_sb[4, 0] = 1  # C[0,2]
    
    # Product 2: A01 * B12
    U_sb[1, 1] = 1
    V_sb[5, 1] = 1  # B[1,2]
    W_sb[4, 1] = 1  # C[0,2]
    
    # Product 3: A10 * B02
    U_sb[2, 2] = 1
    V_sb[2, 2] = 1  # B[0,2]
    W_sb[5, 2] = 1  # C[1,2]
    
    # Product 4: A11 * B12
    U_sb[3, 3] = 1
    V_sb[5, 3] = 1  # B[1,2]
    W_sb[5, 3] = 1  # C[1,2]
    
    # Combine: U has 7+4=11 columns
    U = np.hstack([U_str, U_sb])
    V = np.hstack([V_str, V_sb])
    W = np.hstack([W_str, W_sb])
    
    return U, V, W

def _decompose_3x4x3():
    """
    Decompose 3x4 * 4x3 multiplication.
    Split into blocks:
    A is 3x4, B is 4x3, C is 3x3.
    
    We can split A into rows: A_top (2x4), A_bottom (1x4)
    And B into columns: B_left (4x2), B_right (4x1)
    
    C = A_top * B_left + A_top * B_right + A_bottom * B_left + A_bottom * B_right
    
    A_top (2x4) * B_left (4x2): This is 2x4*4x2. 
    We can split A_top into A_top_left (2x2) and A_top_right (2x2).
    Similarly B_left into B_left_top (2x2) and B_left_bottom (2x2).
    
    This is getting complex. Let me use a simpler approach.
    
    For 3x4 * 4x3, schoolbook rank is 36.
    Let's try to use known results or a systematic approach.
    
    Actually, let me just use the schoolbook decomposition for now and see if we can improve.
    """
    n, m, p = 3, 4, 3
    rank = n * m * p  # 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return U, V, W

def _decompose_3x4x4():
    """
    Decompose 3x4 * 4x4 multiplication.
    Schoolbook rank is 48.
    """
    n, m, p = 3, 4, 4
    rank = n * m * p  # 48
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return U, V, W

def _try_strassen_block_decomposition(n, m, p):
    """
    Try to decompose n*m*p multiplication using block structure.
    For 3x4*4x7: split the 7 columns into 3+4.
    
    C = A * B where A is 3x4, B is 4x7.
    Split B = [B1 | B2] where B1 is 4x3, B2 is 4x4.
    C = [C1 | C2] where C1 = A*B1 (3x3), C2 = A*B2 (3x4).
    
    We need decompositions for:
    1. 3x4 * 4x3 (rank r1)
    2. 3x4 * 4x4 (rank r2)
    
    Total rank = r1 + r2.
    """
    start_time = time.time()
    
    # For 3x4 * 4x3:
    # Let's try to find a good decomposition.
    # One approach: split A into 2+1 rows, B into 3+0... no.
    # Split A into top 2 rows and bottom 1 row.
    # A_top (2x4), A_bottom (1x4)
    # B (4x3)
    # C_top = A_top * B (2x3)
    # C_bottom = A_bottom * B (1x3)
    
    # For 2x4 * 4x3:
    # Split A_top into 2x2 blocks: A_top = [A11 A12] where A11, A12 are 2x2
    # Split B into 2x2 blocks vertically: B = [B1; B2] where B1, B2 are 2x3
    # C_top = A11*B1 + A12*B2
    # A11*B1: 2x2*2x3, rank ~11
    # A12*B2: 2x2*2x3, rank ~11
    # Total for 2x4*4x3: ~22
    
    # For 1x4 * 4x3:
    # This is just a dot product for each column of C_bottom.
    # C_bottom[k] = sum_j A_bottom[j] * B[j,k]
    # This needs 4 products per column, 3 columns = 12 products.
    # But we can share: for each j, we compute A_bottom[j] * B[:,j] which gives a 1x3 vector.
    # So we need 4 rank-1 operations, but each produces 3 outputs.
    # In tensor decomposition terms, this is rank 4 (one per j).
    
    # Hmm, this is getting complicated. Let me just use schoolbook for the sub-problems
    # and see if the total is better than 84.
    
    # 3x4*4x3 schoolbook: 36
    # 3x4*4x4 schoolbook: 48
    # Total: 84. Same as full schoolbook.
    
    # We need to find better decompositions for the sub-problems.
    
    # Let me try a different approach: use the known result that 
    # rank(3x4*4x7) might be achievable by clever blocking.
    
    # Actually, let me try to directly construct a low-rank decomposition
    # using the structure of the problem.
    
    # The tensor T[i*4+j, j*7+k, k*3+i] = 1
    
    # Let me try random search with verification, but be smarter about initialization.
    
    tensor = _build_tensor(n, m, p)
    
    # Try ranks from a lower bound upward
    # Lower bound: the rank must be at least max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # But actually, for matrix multiplication, the lower bound is often higher.
    
    # Let's try ranks from 35 to 84
    for rank in range(35, 85):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        
        max_attempts = 30 if rank < 50 else 10
        
        for attempt in range(max_attempts):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            rng = np.random.default_rng(rank * 10000 + attempt * 7 + 42)
            
            # Sparse random initialization
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            for c in range(rank):
                # Each column corresponds to a rank-1 tensor
                # Use sparse patterns
                nz_U = rng.choice(n * m, size=min(3, n * m), replace=False)
                nz_V = rng.choice(m * p, size=min(3, m * p), replace=False)
                nz_W = rng.choice(p * n, size=min(3, p * n), replace=False)
                
                for idx in nz_U:
                    U[idx, c] = rng.choice([-1, 1])
                for idx in nz_V:
                    V[idx, c] = rng.choice([-1, 1])
                for idx in nz_W:
                    W[idx, c] = rng.choice([-1, 1])
            
            # Check decomposition
            valid = True
            for a in range(n * m):
                if not valid:
                    break
                for b in range(m * p):
                    if not valid:
                        break
                    for c_idx in range(p * n):
                        val = np.dot(U[a, :], V[b, :] * W[c_idx, :])
                        if abs(val - tensor[a, b, c_idx]) > 1e-9:
                            valid = False
                            break
            
            if valid:
                U_list = [[int(x) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in U]
                V_list = [[int(x) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in V]
                W_list = [[int(x) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in W]
                return U_list, V_list, W_list, rank
    
    return None

def construct():
    """Find rank-minimal exact tensor decomposition for 3x4 * 4x7."""
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try block decomposition approach
    result = _try_strassen_block_decomposition(n, m, p)
    if result is not None:
        U_found, V_found, W_found, rank_found = result
        if rank_found < best_rank:
            best_rank = rank_found
            best_U, best_V, best_W = U_found, V_found, W_found
    
    return {"U": best_U, "V": best_V, "W": best_W}
