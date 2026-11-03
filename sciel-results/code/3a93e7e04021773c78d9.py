"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    start_time = time.time()
    
    # We'll build a decomposition by splitting p=7 into p1=4 and p2=3
    # Part 1: 3x4 * 4x4 (columns 0-3 of result)
    # Part 2: 3x4 * 4x3 (columns 4-6 of result)
    
    # For 3x4 * 4x4: split A into 3 rows of 4, B into 4x4
    # We can use the fact that 3x4x4 can be done with rank <= 24 (schoolbook) 
    # but we can optimize using Strassen on sub-blocks
    
    # Strategy: 
    # - Split the 4x4 B into four 2x2 blocks
    # - For each pair of A rows and B 2x2 block, use Strassen (rank 7 instead of 8)
    # - But we need to handle the 3 rows carefully
    
    # Let's use a direct approach first, then optimize:
    # 3x4x4: We can write this as sum over j of A[:,j] * B[j,:]
    # Each term is outer product of 3-vector and 4-vector = rank 1 tensor
    # Total rank = 4 (for the 4 columns of B) * 3 (rows of A) = 12? No...
    
    # Actually, the tensor for 3x4x4 multiplication has rank at most 3*4*4 = 48
    # But we can do better.
    
    # Let me use a different decomposition:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # For 3x4x4: 
    #   A is 3x4, B is 4x4, C is 3x4
    #   We can use: C = A * B
    #   Decompose B into columns: B = [b0, b1, b2, b3] where bj is 4x1
    #   C[:,k] = sum_j A[:,j] * B[j,k]
    
    # This is just schoolbook with rank 3*4*4 = 48 for this part alone.
    
    # Better: Use the known optimal rank for small matrix multiplication.
    # For 2x2x2: rank 7 (Strassen)
    # For 3x3x3: rank 23 (known lower bound ~23, best known 23)
    
    # Let me try a block approach:
    # Split A (3x4) into A1 (2x4) and A2 (1x4)
    # Split B (4x4) into B11, B12, B21, B22 (each 2x2)
    
    # C1 = A1 * B1 (2x4 * 4x4 = 2x4)
    # C2 = A2 * B2 (1x4 * 4x4 = 1x4)
    
    # For 2x4 * 4x4:
    #   Split B into [B_top; B_bottom] where B_top is 2x4, B_bottom is 2x4
    #   C1_top = A1 * B_top (2x4 * 2x4 -> wait, dimensions don't work)
    
    # Let me restart with a cleaner approach.
    
    # The multiplication tensor M_{n,m,p} has entries T[i,j,k] = 1 if the (i,k) entry 
    # of the product depends on A[i,j]*B[j,k].
    
    # For 3x4x7: T[i,j,k] = 1 for i in 0..2, j in 0..3, k in 0..6
    # Rank is the minimum number of rank-1 tensors that sum to T.
    
    # Known results:
    # - 2x2x2: rank 7
    # - 3x3x3: rank 23 (best known)
    # - 2x3x3: rank 15 (best known)
    # - 3x3x4: rank ? 
    
    # Let me use a constructive approach with Strassen on 2x2 blocks.
    
    # Decompose 3x4x7 as:
    # Rows of A: 3
    # Cols of A / Rows of B: 4  
    # Cols of B: 7 = 2+2+2+1 or 4+3
    
    # Try: Split B's columns into two groups: first 4, last 3
    # Part 1: 3x4 * 4x4
    # Part 2: 3x4 * 4x3
    
    # For 3x4 * 4x4:
    #   Split A into A_top (2x4) and A_bot (1x4)
    #   C_top = A_top * B (2x4)
    #   C_bot = A_bot * B (1x4)
    #   
    #   For 2x4 * 4x4:
    #     Split B into B_left (4x2) and B_right (4x2)
    #     C_top_left = A_top * B_left (2x2) -> use Strassen rank 7
    #     C_top_right = A_top * B_right (2x2) -> use Strassen rank 7
    #     Total for 2x4*4x4: 7+7 = 14
    #   
    #   For 1x4 * 4x4:
    #     This is a row vector times a 4x4 matrix = 1x4 vector
    #     C_bot[k] = sum_j A_bot[j] * B[j,k]
    #     This is just 4 terms, rank 4
    #   
    #   Total for 3x4*4x4: 14 + 4 = 18
    
    # For 3x4 * 4x3:
    #   Split A into A_top (2x4) and A_bot (1x4)
    #   C_top = A_top * B (2x3)
    #   C_bot = A_bot * B (1x3)
    #   
    #   For 2x4 * 4x3:
    #     Split B into B_col0 (4x1), B_col1 (4x1), B_col2 (4x1)
    #     C_top[:,k] = A_top * B[:,k] (2x1)
    #     Each 2x4 * 4x1 = 2x1 can be done with rank 2 (since it's just dot products)
    #     Actually, 2x4 * 4x1: result is 2x1, each entry is dot product of 4-dim vectors
    #     This is rank 2 (two independent dot products)
    #     So 3 such terms: rank 6
    #     But wait, we can do better? 
    #     2x4 * 4x1: M[i,j] where i in 0..1, j in 0..3, result C[i] = sum_j A[i,j]*B[j]
    #     This is two rank-1 tensors: A[0,:] * B and A[1,:] * B
    #     So rank 2 per column of B, total 3*2 = 6 for 2x4*4x3
    #   
    #   For 1x4 * 4x3:
    #     C_bot[k] = sum_j A_bot[j] * B[j,k]
    #     This is 3 dot products, rank 3
    #   
    #   Total for 3x4*4x3: 6 + 3 = 9
    
    # Grand total: 18 + 9 = 27
    
    # Let me verify this construction works and build the U, V, W matrices.
    
    # I'll build the decomposition explicitly.
    
    U_rows = []  # Will become list of lists, each of length rank
    V_rows = []
    W_rows = []
    
    def add_term(u_vec, v_vec, w_vec):
        """Add a rank-1 term. u_vec is length n*m, v_vec is length m*p, w_vec is length p*n"""
        U_rows.append(u_vec)
        V_rows.append(v_vec)
        W_rows.append(w_vec)
    
    # Part 1: 3x4 * 4x4 (columns 0-3 of output)
    # A is 3x4, B is 4x4, C is 3x4 (columns 0-3)
    
    # 2x4 * 4x4 part (top 2 rows of A, all 4 rows of B)
    # Split B into left 2 cols and right 2 cols
    # C_top_left = A_top * B_left (2x2) using Strassen
    # C_top_right = A_top * B_right (2x2) using Strassen
    
    # Strassen for 2x2 * 2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B11-B21)
    # M4 = A22*(B22-B12)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    def strassen_2x2(A_blocks, B_blocks, col_offset, row_offset):
        """
        A_blocks: dict with keys '11','12','21','22' each being a function to get the 2x2 block
        Actually, let me pass the actual index ranges.
        
        A is 2x2 in block form: A[ri, rj] where ri, rj in {0,1}
        B is 2x2 in block form: B[bi, bj] where bi, bj in {0,1}
        
        For the 2x4 * 4x2 case:
        A_top is 2x4, B_left is 4x2
        A_top[i, j] for i in 0..1, j in 0..3
        B_left[j, k] for j in 0..3, k in 0..1
        
        We need to view A_top as 2x2 blocks of size 1x2? No, that's not standard.
        
        Let me reconsider. For 2x4 * 4x2:
        A is 2x4, B is 4x2, C is 2x2
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        This is NOT a 2x2 * 2x2 multiplication. It's 2x4 * 4x2.
        So Strassen doesn't directly apply.
        
        I need to decompose differently.
        """
        pass
    
    # Let me use a simpler, verified approach.
    
    # Direct schoolbook but with some sharing.
    
    # Actually, let me just implement the schoolbook with rank 84 first to ensure correctness,
    # then try to optimize.
    
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
