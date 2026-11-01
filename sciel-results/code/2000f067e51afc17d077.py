"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 4, 8]

def _verify_decomposition(U, V, W, n=2, m=4, p=8, trials=50, tol=1e-9):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    rng = np.random.RandomState(42)
    for _ in range(trials):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        
        # C[i,k] = sum_c (sum_j A[i,j]*U[i*m+j,c]) * (sum_l V[l*p+k,c]*B[l,k]) * W[k*n+i,c]
        C_computed = np.zeros((n, p))
        for c in range(rank):
            # factor_A[i] = sum_j A[i,j] * U[i*m+j, c]
            factor_A = np.zeros(n)
            for i in range(n):
                factor_A[i] = sum(A[i, j] * U[i * m + j][c] for j in range(m))
            # factor_B[k] = sum_l V[l*p+k, c] * B[l,k]
            factor_B = np.zeros(p)
            for k in range(p):
                factor_B[k] = sum(V[l * p + k][c] * B[l, k] for l in range(m))
            # Add contribution
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += factor_A[i] * factor_B[k] * W[k * n + i][c]
        
        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False
    return True

def _construct_schoolbook(n=2, m=4, p=8):
    """Construct the standard schoolbook decomposition of rank n*m*p."""
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
    return U, V, W, rank

def _strassen_2x2_decomp():
    """
    Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns U (4x7), V (4x7), W (4x7) for the 2x2 case.
    
    A = [[a, b], [c, d]]
    B = [[e, f], [g, h]]
    
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    C[0,0] = M1 + M4 - M5 + M7
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M2 + M3 + M6
    """
    # U[i*m+j, c] where i in {0,1}, j in {0,1}, c in {0..6}
    # V[j*p+k, c] where j in {0,1}, k in {0,1}, c in {0..6}
    # W[k*n+i, c] where k in {0,1}, i in {0,1}, c in {0..6}
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a+d)(e+h): U[0,0]=1, U[1,1]=1, V[0,0]=1, V[1,1]=1
    # U indices: i*m+j => i*2+j
    U[0][0] = 1  # a
    U[3][0] = 1  # d
    V[0][0] = 1  # e
    V[3][0] = 1  # h
    
    # M2 = (c+d)e: U[1,0]=1, U[1,1]=1, V[0,0]=1
    U[2][1] = 1  # c
    U[3][1] = 1  # d
    V[0][1] = 1  # e
    
    # M3 = a(f-h): U[0,0]=1, V[0,1]=1, V[1,1]=-1
    U[0][2] = 1  # a
    V[1][2] = 1  # f
    V[3][2] = -1  # h
    
    # M4 = d(g-e): U[1,1]=1, V[0,0]=-1, V[1,0]=1
    U[3][3] = 1  # d
    V[0][3] = -1  # e
    V[2][3] = 1  # g
    
    # M5 = (a+b)h: U[0,0]=1, U[0,1]=1, V[1,1]=1
    U[0][4] = 1  # a
    U[1][4] = 1  # b
    V[3][4] = 1  # h
    
    # M6 = (c-a)(e+f): U[0,0]=-1, U[1,0]=1, V[0,0]=1, V[0,1]=1
    U[0][5] = -1  # a
    U[2][5] = 1  # c
    V[0][5] = 1  # e
    V[1][5] = 1  # f
    
    # M7 = (b-d)(g+h): U[0,1]=1, U[1,1]=-1, V[1,0]=1, V[1,1]=1
    U[1][6] = 1  # b
    U[3][6] = -1  # d
    V[2][6] = 1  # g
    V[3][6] = 1  # h
    
    # W[k*n+i, c]: W indices k*2+i
    # C[0,0] = M1 + M4 - M5 + M7
    W[0][0] = 1   # M1
    W[0][3] = 1   # M4
    W[0][4] = -1  # M5
    W[0][6] = 1   # M7
    
    # C[0,1] = M3 + M5
    W[1][2] = 1   # M3
    W[1][4] = 1   # M5
    
    # C[1,0] = M2 + M4
    W[2][1] = 1   # M2
    W[2][3] = 1   # M4
    
    # C[1,1] = M1 - M2 + M3 + M6
    W[3][0] = 1   # M1
    W[3][1] = -1  # M2
    W[3][2] = 1   # M3
    W[3][5] = 1   # M6
    
    return U, V, W

def _build_strassen_2x4x8():
    """
    Build rank-56 decomposition for 2x4 * 4x8 using Strassen.
    
    Strategy: Split the 4x8 matrix into four 2x2 blocks along the output dimension.
    A is 2x4, B is 4x8.
    
    Split A into A1 (2x2) and A2 (2x2) along columns.
    Split B into B1, B2, B3, B4 (each 2x2) along columns.
    
    C = A1*B1 + A1*B2 + A2*B3 + A2*B4 ... no wait.
    
    Actually, let me think more carefully.
    
    A is 2x4: A = [A_left, A_right] where A_left is 2x2, A_right is 2x2
    B is 4x8: B = [[B11, B12], [B21, B22]] where each B* is 2x2
    
    Then C = A * B = [A_left, A_right] * [[B11, B12], [B21, B22]]
                  = A_left * [B11, B12] + A_right * [B21, B22]
                  = [A_left*B11 + A_right*B21, A_left*B12 + A_right*B22]
    
    So C is 2x8, split into C_left (2x4) and C_right (2x4).
    
    C_left = A_left * B11 + A_right * B21
    C_right = A_left * B12 + A_right * B22
    
    Hmm, that's not directly Strassen-applicable in the simple way.
    
    Let me try a different split. 
    
    Alternative: Think of it as computing C = A * B where A is 2x4 and B is 4x8.
    
    We can view this as two independent 2x2 multiplications if we split appropriately.
    
    Let me split A into A1 (2x2) using columns 0,1 and A2 (2x2) using columns 2,3.
    Split B into B1 (2x2) using rows 0,1 and B2 (2x2) using rows 2,3.
    
    Then C = A1 * B1 + A2 * B2, but B is 4x8 so this doesn't quite work directly.
    
    Let me try yet another approach. 
    
    View the 4x8 matrix B as two 4x4 matrices B_a and B_b.
    C = A * B = A * [B_a, B_b] = [A*B_a, A*B_b]
    
    Now A*B_a is 2x4. Split A into A1 (2x2, cols 0,1) and A2 (2x2, cols 2,3).
    Split B_a into B_a1 (2x2, rows 0,1) and B_a2 (2x2, rows 2,3).
    
    A*B_a = A1*B_a1 + A2*B_a2
    
    Similarly A*B_b = A1*B_b1 + A2*B_b2
    
    So C = [A1*B_a1 + A2*B_a2, A1*B_b1 + A2*B_b2]
    
    Now each A1*B_a1, A2*B_a2, etc. is a 2x2 * 2x2 multiplication.
    
    Using Strassen for each:
    A1*B_a1: 7 scalar mults
    A2*B_a2: 7 scalar mults
    A1*B_b1: 7 scalar mults
    A2*B_b2: 7 scalar mults
    
    Total: 4 * 7 = 28? That seems too low. Let me verify.
    
    Wait, I think I'm making an error. Let me re-examine.
    
    A is 2x4. A1 = A[:, 0:2] (2x2), A2 = A[:, 2:4] (2x2).
    B is 4x8. 
    B_a = B[:, 0:4] (4x4), B_b = B[:, 4:8] (4x4).
    
    B_a1 = B_a[0:2, :] = B[0:2, 0:4] (2x4) -- wait, this is 2x4 not 2x2.
    
    I think the issue is that B is 4x8, so when I split rows into 2+2, each block is 2x4.
    
    Let me redo:
    A = [A1, A2] where A1, A2 are 2x2 (columns split)
    B split by rows: B_top = B[0:2, :] (2x8), B_bot = B[2:4, :] (2x8)
    
    C = A1 * B_top + A2 * B_bot
    
    Now A1 * B_top is 2x2 * 2x8 = 2x8.
    A2 * B_bot is 2x2 * 2x8 = 2x8.
    
    To use Strassen, I need to further split B_top into four 2x2 blocks:
    B_top = [B1, B2] where B1, B2 are 2x2 (columns 0,1 and 2,3)
    Wait, B_top is 2x8, so split into four 2x2 blocks: B_top1, B_top2, B_top3, B_top4
    
    A1 * B_top = A1 * [B_top1, B_top2, B_top3, B_top4]
               = [A1*B_top1, A1*B_top2, A1*B_top3, A1*B_top4]
    
    Each A1*B_topi is 2x2 * 2x2, using 7 mults.
    
    Similarly for A2 * B_bot.
    
    Total: 8 * 7 = 56 mults. That's the rank-56 decomposition!
    
    So the structure is:
    - 8 independent 2x2 * 2x2 multiplications
    - Each uses Strassen's 7 scalar multiplications
    - Total rank: 56
    """
    n, m, p = 2, 4, 8
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]  # 8x56
    V = [[0] * rank for _ in range(m * p)]  # 32x56
    W = [[0] * rank for _ in range(p * n)]  # 16x56
    
    # Strassen 2x2 matrices
    U_s, V_s, W_s = _strassen_2x2_decomp()
    
    # Define the 8 blocks
    # Block r: which A part (0 or 1), which B part (top or bot), which 2x2 block (0,1,2,3)
    # A part 0: columns 0,1 -> A indices 0,1
    # A part 1: columns 2,3 -> A indices 2,3
    # B part top: rows 0,1 -> V row indices 0,1 (for each k)
    # B part bot: rows 2,3 -> V row indices 2,3
    # 2x2 block b: columns 2b, 2b+1
    
    blocks = []
    for a_part in range(2):  # 0, 1
        for b_part in range(2):  # 0=top, 1=bot
            for block in range(4):  # 0,1,2,3
                blocks.append((a_part, b_part, block))
    
    for idx, (a_part, b_part, block) in enumerate(blocks):
        # Column offset in the rank-56 decomposition
        col_off = idx * 7
        
        # A row indices: a_part * 2 + {0, 1}
        a_rows = [a_part * 2, a_part * 2 + 1]
        
        # For each Strassen component c (0..6):
        for c in range(7):
            col = col_off + c
            
            # U: A rows
            for a_r in a_rows:
                U[a_r][col] = U_s[a_r][c]
            
            # V: B rows
            # b_part determines which rows of B (0,1 for top; 2,3 for bot)
            # block determines which columns (2*block, 2*block+1)
            b_rows = [b_part * 2, b_part * 2 + 1]
            b_cols = [2 * block, 2 * block + 1]
            
            for b_r in b_rows:
                for b_c in b_cols:
                    # V index: b_r * p + b_c
                    V[b_r * p + b_c][col] = V_s[b_r * 2 + b_c][c]
            
            # W: C indices
            # C row: determined by a_part (0 for rows 0,1; 1 for rows 2,3)? 
            # No, C is 2x8. C row is always 0 or 1 (the 2 rows of A).
            # C col: determined by block (2*block, 2*block+1)
            
            # Wait, I need to think about this more carefully.
            # C = A * B, C is 2x8.
            # C[i, k] where i in {0,1}, k in {0,...,7}
            
            # The block structure:
            # C[:, 2*block : 2*block+2] = A[:, 2*a_part : 2*a_part+2] * B[2*b_part : 2*b_part+2, 2*block : 2*block+2]
            
            # Hmm, that's not right either. Let me reconsider.
            
            # Actually, I think the correct decomposition is:
            # C[i, 2*block + dc] = sum over j of A[i, j] * B[j, 2*block + dc]
            
            # Split j into j = 2*b_part + dj where dj in {0,1}
            # And A[i, j] = A[i, 2*b_part + dj] -- but this mixes up the A partition.
            
            # I think I need to be more careful. Let me reconsider the block structure.
            
            pass  # Will fix below
    
    # Let me redo this more carefully.
    # 
    # C[i, k] = sum_{j=0}^{3} A[i, j] * B[j, k]
    # 
    # Split j into two groups: j in {0,1} and j in {2,3}
    # C[i, k] = sum_{j=0}^{1} A[i, j] * B[j, k] + sum_{j=2}^{3} A[i, j] * B[j, k]
    # 
    # For the first sum: let j' = j (0,1)
    # For the second sum: let j' = j - 2 (0,1)
    # 
    # C[i, k] = sum_{j'=0}^{1} A[i, j'] * B[j', k] + sum_{j'=0}^{1} A[i, 2+j'] * B[2+j', k]
    # 
    # Now split k into 4 groups: k in {0,1}, {2,3}, {4,5}, {6,7}
    # For block b: k = 2*b + k' where k' in {0,1}
    # 
    # C[i, 2*b + k'] = sum_{j'=0}^{1} A[i, j'] * B[j', 2*b+k'] + sum_{j'=0}^{1} A[i, 2+j'] * B[2+j', 2*b+k']
    # 
    # This is: (A1 * B1b)[i, k'] + (A2 * B2b)[i, k']
    # where A1 = A[:, 0:2], A2 = A[:, 2:4], B1b = B[0:2, 2*b:2*b+2], B2b = B[2:4, 2*b:2*b+2]
    # 
    # Each is a 2x2 * 2x2 product.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_s, V_s, W_s = _strassen_2x2_decomp()
    
    # 8 blocks: (a_part, block) where a_part in {0,1}, block in {0,1,2,3}
    for a_part in range(2):
        for block in range(4):
            for c in range(7):
                col = a_part * 4 * 7 + block * 7 + c
                
                # U: A rows for this a_part
                # A[i, j] where j = a_part * 2 + j_local
                for i in range(2):
                    for j_local in range(2):
                        j = a_part * 2 + j_local
                        U[i * m + j][col] = U_s[i * 2 + j_local][c]
                
                # V: B rows
                # B[j, k] where j = j_global, k = block * 2 + k_local
                # But we split j into two groups: j = a_part * 2 + j_local... 
                
                # Wait, I think the issue is that the a_part split is on A columns,
                # but B rows are not split the same way.
                
                # Let me reconsider. 
                # 
                # C[i, k] = sum_j A[i,j] B[j,k]
                # 
                # I want to apply Strassen to sub-problems of the form:
                # C_sub = A_sub * B_sub where A_sub is 2x2 and B_sub is 2x2.
                # 
                # A_sub = A[i1:i1+2, j1:j1+2] for some i1, j1
                # B_sub = B[j1:j1+2, k1:k1+2] for some k1
                # 
                # For the contribution to C[i1:i1+2, k1:k1+2]:
                # C[i1+i, k1+k] += A[i1+i, j1+j] * B[j1+j, k1+k]
                # 
                # So for each pair (j1, k1) where j1 in {0,2} and k1 in {0,2,4,6},
                # we have a 2x2 * 2x2 sub-problem.
                
                # j1 in {0, 2}, k1 in {0, 2, 4, 6}
                # That's 2 * 4 = 8 sub-problems.
                
                # For sub-problem (j1, k1):
                # A_sub[i, j] = A[i1+i, j1+j] where i1=0 (since A has 2 rows)
                # B_sub[j, k] = B[j1+j, k1+k]
                
                # Since A has only 2 rows, i1 = 0 always.
                
                # So for (j1, k1):
                # U contribution: A[0, j1+j] and A[1, j1+j]
                # V contribution: B[j1+j, k1+k]
                # W contribution: C[0, k1+k] and C[1, k1+k]
                
                pass  # Will implement correctly below
    
    # Correct implementation:
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_s, V_s, W_s = _strassen_2x2_decomp()
    
    subproblems = []
    for j1 in [0, 2]:
        for k1 in [0, 2, 4, 6]:
            subproblems.append((j1, k1))
    
    for idx, (j1, k1) in enumerate(subproblems):
        for c in range(7):
            col = idx * 7 + c
            
            # U: A rows
            # A[i, j1 + j_local] for i in {0,1}, j_local in {0,1}
            for i in range(2):
                for j_local in range(2):
                    j = j1 + j_local
                    U[i * m + j][col] = U_s[i * 2 + j_local][c]
            
            # V: B rows
            # B[j1 + j_local, k1 + k_local] for j_local in {0,1}, k_local in {0,1}
            for j_local in range(2):
                for k_local in range(2):
                    j = j1 + j_local
                    k = k1 + k_local
                    V[j * p + k][col] = V_s[j_local * 2 + k_local][c]
            
            # W: C entries
            # C[i, k1 + k_local] for i in {0,1}, k_local in {0,1}
            for i in range(2):
                for k_local in range(2):
                    k = k1 + k_local
                    W[k * n + i][col] = W_s[k_local * 2 + i][c]
    
    return U, V, W, rank

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Build the rank-56 Strassen-based decomposition
    U, V, W, rank = _build_strassen_2x4x8()
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p, trials=50):
        # Fall back to schoolbook
        U, V, W, rank = _construct_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank further using greedy removal
    if time.time() < deadline - 10:
        removed = True
        while removed and time.time() < deadline - 10:
            removed = False
            cur_rank = len(U[0])
            for c in range(cur_rank):
                if time.time() > deadline - 10:
                    break
                U_test = [row[:c] + row[c+1:] for row in U]
                V_test = [row[:c] + row[c+1:] for row in V]
                W_test = [row[:c] + row[c+1:] for row in W]
                if _verify_decomposition(U_test, V_test, W_test, n, m, p, trials=30):
                    U, V, W = U_test, V_test, W_test
                    removed = True
                    break
    
    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p, trials=50):
        # Rebuild rank-56
        U, V, W, rank = _build_strassen_2x4x8()
    
    return {"U": U, "V": V, "W": W}
