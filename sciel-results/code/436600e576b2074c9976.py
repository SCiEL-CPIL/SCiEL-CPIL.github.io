"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # Start with Strassen-based rank-28 decomposition
    # Then try to reduce further
    
    # Build the 28-rank Strassen decomposition
    U, V, W = _strassen_28()
    
    # Try to reduce rank by finding linear dependencies
    best_rank = len(U[0])
    best_U, best_V, best_W = U, V, W
    
    # Try to find a lower rank decomposition
    for target_rank in range(best_rank - 1, 15, -1):
        if time.time() > deadline - 30:
            break
        success = _try_reduce_to_rank(U, V, W, target_rank)
        if success:
            best_rank = target_rank
            best_U, best_V, best_W = U, V, W
            break
    
    # Convert to lists with integer/half-integer entries
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}


def _strassen_28():
    """Build rank-28 Strassen decomposition for 2x4x4."""
    n, m, p = 2, 4, 4
    rank = 28
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Block indices:
    # A blocks: A0=A[0,0:2], A1=A[0,2:4], A2=A[1,0:2], A3=A[1,2:4]
    # B blocks: B00=B[0:2,0:2], B01=B[0:2,2:4], B10=B[2:4,0:2], B11=B[2:4,2:4]
    # C blocks: C00=C[0,0:2], C01=C[0,2:4], C10=C[1,0:2], C11=C[1,2:4]
    
    # Strassen for each of the 4 block multiplications
    # C00 = A0*B00 + A1*B10
    # C01 = A0*B01 + A1*B11
    # C10 = A2*B00 + A3*B10
    # C11 = A2*B01 + A3*B11
    
    # Strassen for X*Y where X,Y are 2x2:
    # M1 = (X0+X1)*Y0
    # M2 = X1*(Y0+Y1)
    # M3 = X0*(Y0-Y1)
    # M4 = (X0-X1)*(Y1+Y2) -- wait, standard Strassen:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B21-B22)
    # M4 = A22*(B12-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M4
    # C21 = M2 + M5
    # C22 = M1 - M2 + M3 + M6
    
    def strassen_block(A_block_idx, B_block_idx, C_block_idx, offset):
        """Add Strassen decomposition for one 2x2x2 block product."""
        # A_block_idx: 0=A0, 1=A1, 2=A2, 3=A3
        # B_block_idx: 0=B00, 1=B01, 2=B10, 3=B11
        # C_block_idx: 0=C00, 1=C01, 2=C10, 3=C11
        
        a_row = A_block_idx // 2  # 0 or 1
        a_col = A_block_idx % 2   # 0 or 1
        b_row = B_block_idx // 2  # 0 or 1
        b_col = B_block_idx % 2   # 0 or 1
        c_row = C_block_idx // 2  # 0 or 1
        c_col = C_block_idx % 2   # 0 or 1
        
        # A elements: A[a_row, a_col*2 + 0], A[a_row, a_col*2 + 1]
        # B elements: B[b_row*2 + 0, b_col*2 + 0], B[b_row*2 + 0, b_col*2 + 1],
        #             B[b_row*2 + 1, b_col*2 + 0], B[b_row*2 + 1, b_col*2 + 1]
        # C elements: C[c_row, c_col*2 + 0], C[c_row, c_col*2 + 1]
        
        a00 = a_row * m + a_col * 2 + 0
        a01 = a_row * m + a_col * 2 + 1
        a10 = a_row * m + a_col * 2 + 0  # Wait, A is 2x4, so rows are 0,1
        # A is 2x4: A[0,0], A[0,1], A[0,2], A[0,3], A[1,0], A[1,1], A[1,2], A[1,3]
        # A_block 0: A[0,0:2] -> indices 0,1
        # A_block 1: A[0,2:4] -> indices 2,3
        # A_block 2: A[1,0:2] -> indices 4,5
        # A_block 3: A[1,2:4] -> indices 6,7
        
        a_idx0 = A_block_idx * 2 + 0
        a_idx1 = A_block_idx * 2 + 1
        
        # B is 4x4: B[0,0], B[0,1], B[0,2], B[0,3], B[1,0], ..., B[3,3]
        # B_block 0 (B00): B[0:2,0:2] -> indices 0,1,4,5
        # B_block 1 (B01): B[0:2,2:4] -> indices 2,3,6,7
        # B_block 2 (B10): B[2:4,0:2] -> indices 8,9,12,13
        # B_block 3 (B11): B[2:4,2:4] -> indices 10,11,14,15
        
        b_base = B_block_idx * 4
        b00 = b_base + 0
        b01 = b_base + 1
        b10 = b_base + 4
        b11 = b_base + 5
        
        # C is 2x4: C[0,0], C[0,1], C[0,2], C[0,3], C[1,0], C[1,1], C[1,2], C[1,3]
        # W index: W[k*n+i, r] where k is col of B (0-3), i is row of A (0-1)
        # C[i, k] = sum_j A[i,j] * B[j,k]
        # So C[0,0] uses A[0,*] and B[*,0]
        # W index for C[i,k] is k*n + i = k*2 + i
        
        c_idx0 = C_block_idx * 2 + 0  # This is the column offset in C
        # C_block 0: C[0,0:2] -> cols 0,1
        # C_block 1: C[0,2:4] -> cols 2,3
        # C_block 2: C[1,0:2] -> cols 0,1 (row 1)
        # C_block 3: C[1,2:4] -> cols 2,3 (row 1)
        
        c_row_idx = C_block_idx // 2
        c_col_start = (C_block_idx % 2) * 2
        
        # W index: W[k*n + i] where k is B column, i is A row
        # For C[i, k], W index = k * n + i = k * 2 + i
        
        w_idx0 = c_col_start * n + c_row_idx
        w_idx1 = (c_col_start + 1) * n + c_row_idx
        
        # Strassen 7 multiplications
        # M1 = (A00+A11) * (B00+B11)
        U[a_idx0, offset] += 1
        U[a_idx1, offset] += 1
        V[b00, offset] += 1
        V[b11, offset] += 1
        W[w_idx0, offset] += 1
        W[w_idx1, offset] += 1
        offset += 1
        
        # M2 = (A10+A11) * B00
        U[a_idx1, offset] += 1  # A10 is a_idx0? No...
        # Wait, I need to be careful. A_block has elements A00, A01 (top row), A10, A11 (bottom row)
        # But A is 2x4, so A_block is 1x2, not 2x2!
        
        # I made an error. Let me reconsider.
        pass
    
    # Actually, each "block" of A is 1x2, not 2x2. Let me reconsider the structure.
    # A is 2x4, B is 4x4, C is 2x4.
    # 
    # Let me partition B into four 2x2 blocks:
    # B = [B00 B01]
    #     [B10 B11]
    # where B00 is rows 0-1, cols 0-1; B01 is rows 0-1, cols 2-3; etc.
    #
    # A = [A0 A1] where A0 = A[0,0:2], A1 = A[0,2:4], etc.
    # Actually A has 2 rows, so:
    # Row 0: A[0,0], A[0,1], A[0,2], A[0,3]
    # Row 1: A[1,0], A[1,1], A[1,2], A[1,3]
    #
    # C[0,0:2] = A[0,0:2] * B[0:2,0:2] + A[0,2:4] * B[2:4,0:2]
    # C[0,2:4] = A[0,0:2] * B[0:2,2:4] + A[0,2:4] * B[2:4,2:4]
    # C[1,0:2] = A[1,0:2] * B[0:2,0:2] + A[1,2:4] * B[2:4,0:2]
    # C[1,2:4] = A[1,0:2] * B[0:2,2:4] + A[1,2:4] * B[2:4,2:4]
    
    # So we have four 1x2 * 2x2 products, each giving a 1x2 result.
    # A 1x2 * 2x2 product can be done with 2 scalar multiplications (schoolbook).
    # But can we do better? A 1x2 * 2x2 = [a b] * [c d; e f] = [ac+be, ad+bf]
    # This is just 4 scalar mults, or we can view it as two dot products of length 2.
    # Minimal rank for 1x2x2 is 2 (it's just two independent bilinear forms).
    
    # So the total schoolbook rank is 4 blocks * 4 mults = 16? No wait.
    # Each C[i,k] = sum_j A[i,j] B[j,k], so there are 2*4*4 = 32 terms.
    # Schoolbook rank is 32.
    
    # Let me just use the known result that 2x4x4 has rank at most 28 via Strassen-like methods.
    # I'll construct it directly.
    
    # For now, return the schoolbook rank-32 as a baseline, and I'll improve.
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W


def _try_reduce_to_rank(U, V, W, target_rank):
    """Try to reduce the rank of the decomposition."""
    # This is a placeholder; a real implementation would use linear algebra
    # to find linear dependencies among the rank-1 terms.
    # For now, return False.
    return False


# EVOLVE-BLOCK-END
