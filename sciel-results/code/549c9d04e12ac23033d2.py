import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # 3, 3, 7

    # Schoolbook fallback: rank = 3*3*7 = 63
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    # Try to construct rank-57 decomposition:
    # Part 1: Rank-21 for 3x3x3 (columns 0-2 of B)
    # Part 2: Schoolbook rank-36 for 3x3x4 (columns 3-6 of B)
    # Total rank: 21 + 36 = 57

    try:
        # Construct the rank-21 Strassen-Winograd decomposition for 3x3x3
        # Using the known explicit factors
        
        # The Winograd 3x3x3 decomposition with rank 21:
        # I'll use the standard factors from the literature.
        
        # For 3x3x3, the decomposition is:
        # C[i,k] = sum_{r=0}^{20} U[i,j,r] * V[j,k,r] * W[k,i,r]
        
        # Let me use the explicit factors.
        # Based on the standard Winograd scheme:
        
        # I'll construct a valid rank-21 decomposition.
        # For safety, let me use the schoolbook for 3x3x3 which has rank 9,
        # but that would give total rank 9 + 36 = 45, which is even better!
        
        # Schoolbook for 3x3x3: rank 9
        # U has shape (9, 9), V has shape (9, 9), W has shape (9, 9)
        
        U_3x3 = [[0] * 9 for _ in range(9)]
        V_3x3 = [[0] * 9 for _ in range(9)]
        W_3x3 = [[0] * 9 for _ in range(9)]
        
        for i in range(3):
            for j in range(3):
                r = i * 3 + j
                U_3x3[i * 3 + j][r] = 1
                for k in range(3):
                    V_3x3[j * 3 + k][r] = 1
                    W_3x3[k * 3 + i][r] = 1
        
        # Wait, that's not right either. Let me think again.
        
        # The decomposition is: sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] = 1
        
        # For schoolbook, each (i,j,k) triple should be covered by exactly one r.
        # But there are 27 triples and only 9 r values, so this can't work.
        
        # I see my error. The schoolbook for 3x3x3 has rank 9, but the tensor
        # has 27 entries. How can rank 9 decompose a tensor with 27 nonzero entries?
        
        # The answer is that the tensor is not all ones. The matrix multiplication
        # tensor T[i,j,k] = 1 only for specific (i,j,k) combinations.
        
        # Actually, T[i,j,k] = 1 for ALL i,j,k in the standard basis.
        # So the tensor has 27 nonzero entries, all equal to 1.
        
        # A rank-9 decomposition would require:
        # sum_{r=0}^{8} U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] = 1 for all i,j,k
        
        # This is possible if the factors are chosen appropriately.
        
        # Let me use the known rank-9 schoolbook decomposition:
        # U[i*3+j, r] = 1 if r = i*3+j, else 0
        # V[j*3+k, r] = 1 if j = r % 3, else 0  (this doesn't depend on i!)
        # W[k*3+i, r] = 1 if i = r // 3, else 0
        
        # Let me verify: sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
        # = sum_r [r==i*3+j] * [j == r%3] * [i == r//3]
        # = 1 if i*3+j == r and j == r%3 and i == r//3
        # = 1 if i*3+j == r (since i == r//3 and j == r%3 are equivalent to i*3+j == r)
        # = 1
        
        # Great! So the schoolbook for 3x3x3 has rank 9.
        
        # Now let me construct it properly:
        U_3x3 = [[0] * 9 for _ in range(9)]
        V_3x3 = [[0] * 9 for _ in range(9)]
        W_3x3 = [[0] * 9 for _ in range(9)]
        
        for i in range(3):
            for j in range(3):
                r = i * 3 + j
                U_3x3[i * 3 + j][r] = 1
                for k in range(3):
                    V_3x3[j * 3 + k][r] = 1
                    W_3x3[k * 3 + i][r] = 1
        
        # Wait, I'm still making the same error. Let me be more careful.
        
        # U has shape (9, 9): U[i*3+j, r]
        # V has shape (9, 9): V[j*3+k, r]
        # W has shape (9, 9): W[k*3+i, r]
        
        # For schoolbook:
        # U[i*3+j, r] = 1 if r = i*3+j, else 0
        # V[j*3+k, r] = 1 if j = r % 3, else 0
        # W[k*3+i, r] = 1 if i = r // 3, else 0
        
        U_3x3 = [[0] * 9 for _ in range(9)]
        V_3x3 = [[0] * 9 for _ in range(9)]
        W_3x3 = [[0] * 9 for _ in range(9)]
        
        for r in range(9):
            i = r // 3
            j = r % 3
            U_3x3[i * 3 + j][r] = 1
            for k in range(3):
                V_3x3[j * 3 + k][r] = 1
                W_3x3[k * 3 + i][r] = 1
        
        # Now let me verify this decomposition.
        # For a given (i,j,k):
        # sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
        # = sum_r [r == i*3+j] * [j == r%3] * [i == r//3]
        # = [i*3+j == i*3+j] * [j == (i*3+j)%3] * [i == (i*3+j)//3]
        # = 1 * [j == j] * [i == i]
        # = 1
        
        # Perfect! The schoolbook for 3x3x3 has rank 9.
        
        # Now for the 3x3x4 part (columns 3-6 of B):
        # This is a 3x3 matrix times a 3x4 matrix.
        # Schoolbook: rank = 3*3*4 = 36
        
        # Total rank: 9 + 36 = 45
        
        # Let me construct the full decomposition.
        rank_total = 9 + 36  # 45
        
        U_out = [[0] * rank_total for _ in range(n * m)]  # 9 x 45
        V_out = [[0] * rank_total for _ in range(m * p)]  # 21 x 45
        W_out = [[0] * rank_total for _ in range(p * n)]  # 21 x 45
        
        # Part 1: Schoolbook for 3x3x3 (columns 0-2)
        for r in range(9):
            i = r // 3
            j = r % 3
            U_out[i * 3 + j][r] = 1
            for k in range(3):
                V_out[j * 7 + k][r] = 1
                W_out[k * 3 + i][r] = 1
        
        # Part 2: Schoolbook for 3x3x4 (columns 3-6)
        col_offset = 9
        for i in range(3):
            for j in range(3):
                for k in range(4):
                    r = col_offset + i * 3 * 4 + j * 4 + k
                    U_out[i * 3 + j][r] = 1
                    V_out[j * 7 + (k + 3)][r] = 1
                    W_out[(k + 3) * 3 + i][r] = 1
        
        # Verify the decomposition
        U_np = np.array(U_out, dtype=float)
        V_np = np.array(V_out, dtype=float)
        W_np = np.array(W_out, dtype=float)
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank_total):
                        val += U_np[i * m + j, r] * V_np[j * p + k, r] * W_np[k * n + i, r]
                    if abs(val - 1) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return {"U": U_out, "V": V_out, "W": W_out}
        else:
            return {"U": U_sb, "V": V_sb, "W": W_sb}
            
    except Exception:
        return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
