import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    """
    Returns a rank-56 exact tensor decomposition for multiplying a 2x4 matrix 
    by a 4x8 matrix.
    
    The decomposition is based on applying Strassen's algorithm to the 4 
    independent 2x2x2 sub-problems formed by blocking the 4x8 matrix into 
    four 2x2 blocks, while sharing the 2x4 matrix structure appropriately.
    
    Actually, the 2x4 matrix A is partitioned into two 2x2 blocks A_L, A_R.
    The 4x8 matrix B is partitioned into four 2x2 blocks B_LL, B_LR, B_UR, B_UR (top-left, top-right, bottom-left, bottom-right).
    The output C (2x8) is partitioned into four 2x2 blocks C_L, C_R.
    
    Wait, the standard block decomposition for (2x4) * (4x8) -> (2x8) is:
    A = [A11 A12] (2x4) -> A11 is 2x2, A12 is 2x2
    B = [B11 B12; B21 B22] (4x8) -> Each is 2x4? No.
    Let's partition B into 4 blocks of 2x2?
    B is 4x8.
    Rows 0-1, Cols 0-1: B00 (2x2)
    Rows 0-1, Cols 2-3: B01 (2x2)
    Rows 0-1, Cols 4-5: B02 (2x2)
    Rows 0-1, Cols 6-7: B03 (2x2)
    Rows 2-3, Cols 0-1: B10 (2x2)
    ...
    
    The previous program treated it as 2 blocks of A (2x2 each) and 4 blocks of B (2x2 each).
    A is 2x4. It has 2 columns of 2x2? No, A is 2 rows, 4 cols.
    Block 0 of A: A[:, 0:2] (2x2)
    Block 1 of A: A[:, 2:4] (2x2)
    
    B is 4x8.
    The multiplication is C = A * B.
    C is 2x8.
    
    The previous code iterated `b` in 0..1 (A blocks) and `c` in 0..3 (B blocks).
    This implies it computed 8 independent 2x2 * 2x2 products?
    If A_block_b (2x2) multiplies B_block_bc (2x2), we get a 2x2 result.
    But C is 2x8.
    The result of A_block_b * B_block_bc contributes to C[:, 2c:2c+2].
    
    So we have 2 * 4 = 8 independent 2x2 matrix multiplications.
    Standard rank for 2x2x2 is 8.
    Strassen reduces this to 7.
    Total rank = 8 * 7 = 56.
    
    This is the design we must keep.
    """
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    rank = 56
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    # b: block of A (2 blocks: rows 0-1, 2-3 of the 4x4 B? No, A is 2x4)
    # A is 2x4. Blocks are A[:, 0:2] and A[:, 2:4].
    # B is 4x8. Blocks are B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6], B[0:2, 6:8] AND B[2:4, ...]?
    
    # Let's re-verify the indices from the previous program.
    # idx_a = 0 * m + 2 * b       # A[0, 2b]
    # idx_b = 0 * m + 2 * b + 1   # A[0, 2b+1]
    # idx_c_a = 1 * m + 2 * b     # A[1, 2b]
    # idx_d = 1 * m + 2 * b + 1   # A[1, 2b+1]
    # This maps A to a 2x2 block A_b.
    # A is 2x4. m=4.
    # b=0: A[0,0], A[0,1], A[1,0], A[1,1] -> A[:, 0:2]
    # b=1: A[0,2], A[0,3], A[1,2], A[1,3] -> A[:, 2:4]
    
    # idx_e = 2 * b * p + 2 * c       # B[2b, 2c]
    # idx_f = 2 * b * p + 2 * c + 1   # B[2b, 2c+1]
    # idx_g = (2 * b + 1) * p + 2 * c     # B[2b+1, 2c]
    # idx_h = (2 * b + 1) * p + 2 * c + 1 # B[2b+1, 2c+1]
    # This maps B to a 2x2 block B_bc.
    # B is 4x8. p=8.
    # b=0, c=0: B[0,0], B[0,1], B[1,0], B[1,1] -> B[0:2, 0:2]
    # b=0, c=1: B[0,2], B[0,3], B[1,2], B[1,3] -> B[0:2, 2:4]
    # ...
    # b=1, c=0: B[2,0], B[2,1], B[3,0], B[3,1] -> B[2:4, 0:2]
    
    # So we are computing 8 independent 2x2 * 2x2 products:
    # A_b * B_bc -> contributes to C_block_bc.
    # C is 2x8.
    # C_block_bc is C[0:2, 2c:2c+2]?
    # Let's check W indices.
    # idx_c11 = 2 * c * n + 0       # C[0, 2c]
    # idx_c12 = (2 * c + 1) * n + 0 # C[0, 2c+1]
    # idx_c21 = 2 * c * n + 1       # C[1, 2c]
    # idx_c22 = (2 * c + 1) * n + 1 # C[1, 2c+1]
    # n=2.
    # c=0: C[0,0], C[0,1], C[1,0], C[1,1] -> C[:, 0:2]
    # c=1: C[0,2], C[0,3], C[1,2], C[1,3] -> C[:, 2:4]
    
    # Wait, the W indices depend on `c` but not `b`?
    # If b=0 and b=1 both write to the same C columns, they are ADDING to the same output.
    # C = A * B = A[:,0:2] * B[0:2,:] + A[:,2:4] * B[2:4,:]
    # So C[:, 2c:2c+2] = A[:, 0:2] * B[0:2, 2c:2c+2] + A[:, 2:4] * B[2:4, 2c:2c+2]
    # This means the contributions from b=0 and b=1 are summed.
    # The tensor decomposition sums over the rank index.
    # So we can just add the Strassen terms for each (b,c) pair into the same W columns.
    
    for b in range(2):
        for c in range(4):
            # Local indices for A (2x2 block)
            idx_a = 0 * m + 2 * b       # A[0, 2b]
            idx_b = 0 * m + 2 * b + 1   # A[0, 2b+1]
            idx_c_a = 1 * m + 2 * b     # A[1, 2b]
            idx_d = 1 * m + 2 * b + 1   # A[1, 2b+1]
            
            # Local indices for B (2x2 block)
            idx_e = 2 * b * p + 2 * c       # B[2b, 2c]
            idx_f = 2 * b * p + 2 * c + 1   # B[2b, 2c+1]
            idx_g = (2 * b + 1) * p + 2 * c     # B[2b+1, 2c]
            idx_h = (2 * b + 1) * p + 2 * c + 1 # B[2b+1, 2c+1]
            
            # Local indices for C (2x2 output block)
            idx_c11 = 2 * c * n + 0       # C[0, 2c]
            idx_c12 = (2 * c + 1) * n + 0 # C[0, 2c+1]
            idx_c21 = 2 * c * n + 1       # C[1, 2c]
            idx_c22 = (2 * c + 1) * n + 1 # C[1, 2c+1]
            
            # M1 = (a+d)(e+h)
            U[idx_a, col] += 1
            U[idx_d, col] += 1
            V[idx_e, col] += 1
            V[idx_h, col] += 1
            W[idx_c11, col] += 1
            W[idx_c22, col] += 1
            col += 1
            
            # M2 = (c+d)e
            U[idx_c_a, col] += 1
            U[idx_d, col] += 1
            V[idx_e, col] += 1
            W[idx_c21, col] += 1
            W[idx_c22, col] -= 1
            col += 1
            
            # M3 = a(f-h)
            U[idx_a, col] += 1
            V[idx_f, col] += 1
            V[idx_h, col] -= 1
            W[idx_c12, col] += 1
            W[idx_c22, col] += 1
            col += 1
            
            # M4 = d(g-e)
            U[idx_d, col] += 1
            V[idx_g, col] += 1
            V[idx_e, col] -= 1
            W[idx_c11, col] += 1
            W[idx_c21, col] += 1
            col += 1
            
            # M5 = (a+b)h
            U[idx_a, col] += 1
            U[idx_b, col] += 1
            V[idx_h, col] += 1
            W[idx_c11, col] -= 1
            W[idx_c12, col] += 1
            col += 1
            
            # M6 = (c-a)(e+f)
            U[idx_c_a, col] += 1
            U[idx_a, col] -= 1
            V[idx_e, col] += 1
            V[idx_f, col] += 1
            W[idx_c22, col] += 1
            col += 1
            
            # M7 = (b-d)(g+h)
            U[idx_b, col] += 1
            U[idx_d, col] -= 1
            V[idx_g, col] += 1
            V[idx_h, col] += 1
            W[idx_c11, col] += 1
            col += 1
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Convert to lists of lists as per interface requirement if needed, 
    # but numpy arrays are usually accepted if they behave like lists.
    # The prompt says "factor matrices of shapes...".
    # Previous program returned lists of lists.
    # I will return lists of lists to be safe and distinct from a numpy-only return.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    # Simple self-test
    res = construct()
    print("Rank:", len(res["U"][0]))
    print("Shapes:", len(res["U"]), len(res["U"][0]), len(res["V"]), len(res["V"][0]), len(res["W"]), len(res["W"][0]))
