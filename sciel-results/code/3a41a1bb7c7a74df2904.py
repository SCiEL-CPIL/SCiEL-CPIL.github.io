"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Use Strassen recursion:
    # 4x4x4 = 7 * (2x2x2) where each 2x2x2 has rank 7
    # So 4x4x4 has rank 49
    # 4x4x8 = 2 * (4x4x4) so rank 98
    
    # First, define the 2x2x2 Strassen decomposition
    # For multiplying 2x2 by 2x2:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # The 2x2x2 Strassen terms (u, v, w) for each of the 7 products:
    # u is in R^4 (indices 0,1,2,3 for A11,A12,A21,A22)
    # v is in R^4 (indices 0,1,2,3 for B11,B12,B21,B22)
    # w is in R^4 (indices 0,1,2,3 for C11,C12,C21,C22)
    
    # Standard Strassen coefficients
    strassen_2x2x2 = [
        # (u, v, w) for each of 7 terms
        # u: A11, A12, A21, A22
        # v: B11, B12, B21, B22
        # w: C11, C12, C21, C22
        
        # P1 = (A11+A22)(B11+B22)
        # C11 += P1, C12 -= P1, C21 += P1, C22 -= P1
        ((1, 0, 0, 1), (1, 0, 0, 1), (1, -1, 1, -1)),
        
        # P2 = (A21+A22)B11
        # C21 += P2, C22 += P2
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, 1)),
        
        # P3 = A11(B12-B22)
        # C12 += P3, C22 -= P3
        ((1, 0, 0, 0), (0, 1, 0, -1), (0, 1, 0, -1)),
        
        # P4 = A22(B21-B11)
        # C21 += P4, C11 -= P4
        ((0, 0, 0, 1), (0, 0, 1, -1), (-1, 0, 1, 0)),
        
        # P5 = (A11+A12)B22
        # C11 += P5, C12 += P5
        ((1, 1, 0, 0), (0, 0, 0, 1), (1, 1, 0, 0)),
        
        # P6 = (A21-A11)(B11+B12)
        # C21 += P6, C22 += P6
        ((-1, 0, 1, 0), (1, 1, 0, 0), (0, 0, 1, 1)),
        
        # P7 = (A12-A22)(B21+B22)
        # C12 += P7, C22 += P7
        ((0, 1, 0, -1), (0, 0, 1, 1), (0, 1, 0, 1)),
    ]
    
    # Now for 4x4x4, we use Strassen at the block level
    # A is 4x4, split into four 2x2 blocks: A11, A12, A21, A22
    # B is 4x4, split into four 2x2 blocks: B11, B12, B21, B22
    # C is 4x4, split into four 2x2 blocks: C11, C12, C21, C22
    
    # The 7 Strassen products at the 4x4 level:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # Each Pk is a 2x2x2 multiplication, so we apply the 7-term Strassen to each
    # Total: 7 * 7 = 49 terms for 4x4x4
    
    # For 4x4x8, we split B into two 4x4 blocks: B_left and B_right
    # C = A @ B = [A @ B_left, A @ B_right]
    # Each A @ B_left and A @ B_right is a 4x4x4 multiplication with rank 49
    # Total rank: 49 + 49 = 98
    
    # Let's build the full decomposition
    rank = 98
    U = np.zeros((16, rank))  # 4*4 = 16
    V = np.zeros((32, rank))  # 4*8 = 32
    W = np.zeros((32, rank))  # 8*4 = 32
    
    col = 0
    
    # For each of the two 4x4x4 subproblems (left half and right half of B)
    for half in range(2):
        # half=0: B columns 0-3, C columns 0-3
        # half=1: B columns 4-7, C columns 4-7
        
        # For each of the 7 Strassen steps at the 4x4 level
        strassen_4x4_blocks = [
            # (A_block_comb, B_block_comb, C_block_comb)
            # Each is a 4-element vector indicating which 2x2 blocks are used
            # with coefficients
            
            # P1 = (A11+A22)(B11+B22)
            ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)),
            
            # P2 = (A21+A22)B11
            ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, 0)),
            
            # P3 = A11(B12-B22)
            ((1, 0, 0, 0), (0, 1, 0, -1), (0, 1, 0, 0)),
            
            # P4 = A22(B21-B11)
            ((0, 0, 0, 1), (0, 0, 1, -1), (0, 0, 1, 0)),
            
            # P5 = (A11+A12)B22
            ((1, 1, 0, 0), (0, 0, 0, 1), (1, 1, 0, 0)),
            
            # P6 = (A21-A11)(B11+B12)
            ((-1, 0, 1, 0), (1, 1, 0, 0), (0, 0, 1, 0)),
            
            # P7 = (A12-A22)(B21+B22)
            ((0, 1, 0, -1), (0, 0, 1, 1), (0, 1, 0, 0)),
        ]
        
        for step in range(7):
            a_comb, b_comb, c_comb = strassen_4x4_blocks[step]
            
            # For each of the 7 Strassen terms in the 2x2x2 subproblem
            for sub in range(7):
                u_sub, v_sub, w_sub = strassen_2x2x2[sub]
                
                # Build U, V, W for this term
                # A has 4x4 = 16 elements, indexed as [i,j] -> i*4+j
                # Blocks: A11 = [0:2, 0:2], A12 = [0:2, 2:4], A21 = [2:4, 0:2], A22 = [2:4, 2:4]
                
                # u_sub gives coefficients for the 2x2 blocks within the A_block
                # a_comb gives which 4x4-level blocks are used
                
                # For each 2x2 block position (bi, bj) in the 4x4 A:
                # bi, bj in {0, 1}
                for bi in range(2):
                    for bj in range(2):
                        a_coeff = a_comb[bi * 2 + bj]
                        if a_coeff == 0:
                            continue
                        
                        # Within this 2x2 block, u_sub gives the 4 coefficients
                        # u_sub = (u11, u12, u21, u22) for the local 2x2
                        for li in range(2):
                            for lj in range(2):
                                local_coeff = u_sub[li * 2 + lj]
                                if local_coeff == 0:
                                    continue
                                
                                global_i = bi * 2 + li
                                global_j = bj * 2 + lj
                                U[global_i * 4 + global_j, col] = a_coeff * local_coeff
                
                # V: B has 4x8 = 32 elements
                # For half=0: B columns 0-3, for half=1: B columns 4-7
                # B blocks within the 4x4 half: B11 = [0:2, 0:2], B12 = [0:2, 2:4], B21 = [2:4, 0:2], B22 = [2:4, 2:4]
                # Global column offset: half * 4
                
                for bi in range(2):
                    for bj in range(2):
                        b_coeff = b_comb[bi * 2 + bj]
                        if b_coeff == 0:
                            continue
                        
                        for li in range(2):
                            for lj in range(2):
                                local_coeff = v_sub[li * 2 + lj]
                                if local_coeff == 0:
                                    continue
                                
                                global_i = bi * 2 + li  # row in B (0-3)
                                global_j = half * 4 + bj * 2 + lj  # column in B (0-7)
                                V[global_i * 8 + global_j, col] = b_coeff * local_coeff
                
                # W: C has 8x4 = 32 elements, indexed as [k*n+i] = [k*4+i]
                # C blocks within the 4x4: C11 = [0:2, 0:2], C12 = [0:2, 2:4], C21 = [2:4, 0:2], C22 = [2:4, 2:4]
                # Global row offset for C: half * 4 (since C columns are offset)
                
                for bi in range(2):
                    for bj in range(2):
                        c_coeff = c_comb[bi * 2 + bj]
                        if c_coeff == 0:
                            continue
                        
                        for li in range(2):
                            for lj in range(2):
                                local_coeff = w_sub[li * 2 + lj]
                                if local_coeff == 0:
                                    continue
                                
                                global_i = bi * 2 + li  # this is the C row index (k)
                                global_j = half * 4 + bj * 2 + lj  # this is the C column index (i)
                                # W index: k*n + i = global_i * 4 + global_j
                                W[global_i * 4 + global_j, col] = c_coeff * local_coeff
                
                col += 1
    
    # Verify the decomposition
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 8)
    C_test = A_test @ B_test
    
    C_check = np.zeros((4, 8))
    for c in range(rank):
        u = U[:, c].reshape(4, 4)
        v = V[:, c].reshape(4, 8)
        w = W[:, c].reshape(8, 4)
        
        # Contribution: C[i,k] += sum_j u[i,j] * v[j,k] * w[k,i]
        for i in range(4):
            for k in range(8):
                s = 0
                for j in range(4):
                    s += u[i, j] * v[j, k] * w[k, i]
                C_check[i, k] += s
    
    if not np.allclose(C_test, C_check, atol=1e-10):
        # Fallback to schoolbook
        rank_fb = 128
        U_fb = np.zeros((16, rank_fb))
        V_fb = np.zeros((32, rank_fb))
        W_fb = np.zeros((32, rank_fb))
        col_fb = 0
        for i in range(4):
            for j in range(4):
                for k in range(8):
                    U_fb[i * 4 + j, col_fb] = 1
                    V_fb[j * 8 + k, col_fb] = 1
                    W_fb[k * 4 + i, col_fb] = 1
                    col_fb += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
