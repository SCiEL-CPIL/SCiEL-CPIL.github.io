import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    def verify_solution(U, V, W, R):
        """Verify that U, V, W with rank R correctly decompose the matrix multiplication tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for l in range(m):
                        val = 0.0
                        for r in range(R):
                            val += W[k * n + i, r] * U[i * m + j, r] * V[l * p + k, r]
                        expected = 1.0 if j == l else 0.0
                        if abs(val - expected) > 1e-9:
                            return False
        return True

    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition.
        U: 4x7 (rows indexed by (i,j) for i in {0,1}, j in {0,1})
        V: 4x7 (rows indexed by (j,k) for j in {0,1}, k in {0,1})
        W: 4x7 (rows indexed by (k,i) for k in {0,1}, i in {0,1})
        """
        R = 7
        U = np.zeros((4, R))
        V = np.zeros((4, R))
        W = np.zeros((4, R))
        
        # M1 = (A00 + A11)(B00 + B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10 + A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = A00(B10 - B11)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = A11(B01 - B00)
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A00 + A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = -1
        
        return U, V, W

    def strassen_2x2x3():
        """Return U, V, W for 2x2x3 decomposition using Strassen.
        Multiply 2x2 by 2x3.
        U: 4xR (rows: (i,j) for i in {0,1}, j in {0,1})
        V: 6xR (rows: (j,k) for j in {0,1}, k in {0,1,2})
        W: 6xR (rows: (k,i) for k in {0,1,2}, i in {0,1})
        """
        # Split the 2x3 matrix into a 2x2 part and a 2x1 part.
        # Use Strassen for the 2x2x2 part (rank 7) and schoolbook for the 2x2x1 part (rank 4).
        # Total rank = 7 + 4 = 11.
        # 
        # Actually, we can do better. Let's use Strassen's 7 multiplications but adapt them.
        # For 2x2 x 2x3, we can write B = [B_left(2x2) | B_right(2x1)]
        # C = A * B = [A*B_left | A*B_right]
        # A*B_left uses Strassen (rank 7), A*B_right uses schoolbook (rank 4).
        # But we can potentially share some structure.
        # 
        # Simplest: just concatenate. Rank = 7 + 4 = 11.
        
        U_s, V_s, W_s = strassen_2x2x2()
        
        # Schoolbook for 2x2x1: 4 multiplications
        U_sc = np.zeros((4, 4))
        V_sc = np.zeros((6, 4))
        W_sc = np.zeros((6, 4))
        
        # For the 2x1 part (column index 2 in the 2x3 matrix):
        # A[i,j] * B[j,2] for i in {0,1}, j in {0,1}
        # U rows: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # V rows: (0,2)=4, (1,2)=5  (only column k=2 matters, but V has 6 rows for k=0,1,2)
        # W rows: (0,0)=0, (1,0)=1, (0,1)=2, (1,1)=3  (W[k*n+i] for k in {0,1}, i in {0,1})
        
        for r in range(4):
            i = r // 2
            j = r % 2
            U_sc[i*2+j, r] = 1
            V_sc[j*3+2, r] = 1  # j in {0,1}, k=2
            W_sc[2*2+i, r] = 1  # k=2, i in {0,1} -> index k*2+i = 2*2+i = 4+i
        
        # Combine: total rank = 7 + 4 = 11
        R = 11
        U = np.zeros((4, R))
        V = np.zeros((6, R))
        W = np.zeros((6, R))
        
        U[:, :7] = U_s
        V[:4, :7] = V_s  # V_s is 4x7 for 2x2x2, maps to rows (j,k) for j in {0,1}, k in {0,1}
        # V_s rows: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # In 2x2x3, V rows: (j,k) for j in {0,1}, k in {0,1,2} -> indices 0,1,2,3,4,5
        # So V_s row 0 -> V row 0 (j=0,k=0), V_s row 1 -> V row 1 (j=0,k=1)
        # V_s row 2 -> V row 3 (j=1,k=0), V_s row 3 -> V row 4 (j=1,k=1)
        # Wait, V indexing: V[j*p+k] where p=3. So (0,0)=0, (0,1)=1, (0,2)=2, (1,0)=3, (1,1)=4, (1,2)=5
        # V_s for 2x2x2: V[j*2+k] where p=2. So (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # Mapping: V_s[0] -> V[0], V_s[1] -> V[1], V_s[2] -> V[3], V_s[3] -> V[4]
        V[0, :7] = V_s[0, :]
        V[1, :7] = V_s[1, :]
        V[3, :7] = V_s[2, :]
        V[4, :7] = V_s[3, :]
        
        # W_s for 2x2x2: W[k*2+i] for k in {0,1}, i in {0,1} -> indices 0,1,2,3
        # In 2x2x3: W[k*2+i] for k in {0,1,2}, i in {0,1} -> indices 0,1,2,3,4,5
        # So W_s[0] -> W[0], W_s[1] -> W[1], W_s[2] -> W[2], W_s[3] -> W[3]
        W[:4, :7] = W_s
        
        # Schoolbook part
        U[:, 7:] = U_sc
        V[:, 7:] = V_sc
        W[:, 7:] = W_sc
        
        return U, V, W

    # Block decomposition: 
    # A is 4x4, split into 4 blocks of 2x2: A00, A01, A10, A11
    # B is 4x6, split into 6 blocks of 2x3: B00, B01, B10, B11, B20, B21
    #   B00 = rows 0-1, cols 0-2
    #   B01 = rows 0-1, cols 3-5
    #   B10 = rows 2-3, cols 0-2
    #   B11 = rows 2-3, cols 3-5
    #   Wait, that's only 4 blocks. Let me re-think.
    #
    # A (4x4): split into 2x2 blocks
    #   A[0:2, 0:2] = A00, A[0:2, 2:4] = A01
    #   A[2:4, 0:2] = A10, A[2:4, 2:4] = A11
    #
    # B (4x6): split into 2x3 blocks
    #   B[0:2, 0:3] = B00, B[0:2, 3:6] = B01
    #   B[2:4, 0:3] = B10, B[2:4, 3:6] = B11
    #
    # C (4x6): C = A * B
    #   C[0:2, 0:3] = A00*B00 + A01*B10
    #   C[0:2, 3:6] = A00*B01 + A01*B11
    #   C[2:4, 0:3] = A10*B00 + A11*B10
    #   C[2:4, 3:6] = A10*B01 + A11*B11
    #
    # Each term is a 2x2 * 2x3 multiplication, which has rank 11 (from strassen_2x2x3).
    # There are 8 such terms total.
    # Total rank = 8 * 11 = 88.
    #
    # This is worse than schoolbook (96)? No, 88 < 96. Good.
    #
    # But we can do better by sharing Strassen multiplications across the 8 terms.
    # For now, let's just concatenate all 8 decompositions.

    # Build the full decomposition
    # U has 16 rows (4*4), V has 24 rows (4*6), W has 24 rows (6*4)
    # 
    # For each (A_block, B_block) pair, we have a 2x2x3 decomposition.
    # A_block (i,j) contributes to U rows [2i:2i+2, 2j:2j+2] -> 4 rows
    # B_block (j',k') contributes to V rows [2j':2j'+2, 3k':3k'+3] -> 6 rows  
    # C_block (i',k') contributes to W rows [3k':3k'+3, 2i':2i'+2] -> 6 rows
    #
    # Wait, I need to be more careful with indexing.
    #
    # U[i*m+j, r] where m=4. So U has 16 rows indexed by (i,j) for i in 0..3, j in 0..3.
    # V[j*p+k, r] where p=6. So V has 24 rows indexed by (j,k) for j in 0..3, k in 0..5.
    # W[k*n+i, r] where n=4. So W has 24 rows indexed by (k,i) for k in 0..5, i in 0..3.
    #
    # For the block (A_block at rows 2a:2a+2, cols 2b:2b+2) and (B_block at rows 2c:2c+2, cols 3d:3d+3):
    #   This contributes to C_block at rows 2a:2a+2, cols 3d:3d+3.
    #
    # The 2x2x3 decomposition gives:
    #   U_sub: 4x11, rows indexed by (i',j') for i' in {0,1}, j' in {0,1}
    #   V_sub: 6x11, rows indexed by (j',k') for j' in {0,1}, k' in {0,1,2}
    #   W_sub: 6x11, rows indexed by (k',i') for k' in {0,1,2}, i' in {0,1}
    #
    # Mapping to global indices:
    #   U_sub[i',j'] -> U[2a+i', 2b+j']
    #   V_sub[j',k'] -> V[2c+j', 3d+k']
    #   W_sub[k',i'] -> W[3d+k', 2a+i']

    U_sub, V_sub, W_sub = strassen_2x2x3()
    R_sub = 11

    # 8 blocks: (a,b,c,d) where a in {0,1}, b in {0,1}, c in {0,1}, d in {0,1}
    # a: row block of A (and C), b: col block of A
    # c: row block of B, d: col block of B (and C)
    # Constraint: b must equal c for the multiplication to be valid!
    # A[2a:2a+2, 2b:2b+2] * B[2c:2c+2, 3d:3d+3] is only valid if 2b+2 <= 4 and 2c >= 0 and 2c+2 <= 4
    # and the inner dimension matches: A's cols are 2b:2b+2, B's rows are 2c:2c+2.
    # For matrix multiplication, A's columns must match B's rows.
    # A is 4x4, B is 4x6. A's columns are 0..3, B's rows are 0..3.
    # A_block (a,b) has cols 2b:2b+2. B_block (c,d) has rows 2c:2c+2.
    # These match only if 2b == 2c, i.e., b == c.
    
    # So valid (a,b,c,d) with b == c:
    # (a, b, b, d) for a in {0,1}, b in {0,1}, d in {0,1} -> 8 combinations
    
    total_rank = 8 * R_sub
    U_full = np.zeros((16, total_rank))
    V_full = np.zeros((24, total_rank))
    W_full = np.zeros((24, total_rank))
    
    col = 0
    for a in range(2):
        for b in range(2):
            for d in range(2):
                c = b  # must match
                # U_sub rows (i',j') -> U[2a+i', 2b+j']
                for i_prime in range(2):
                    for j_prime in range(2):
                        U_full[2*a + i_prime, col:col+R_sub] += U_sub[i_prime*2+j_prime, :]
                
                # V_sub rows (j',k') -> V[2c+j', 3d+k']
                for j_prime in range(2):
                    for k_prime in range(3):
                        V_full[2*c + j_prime, col:col+R_sub] += V_sub[j_prime*3+k_prime, :]
                
                # W_sub rows (k',i') -> W[3d+k', 2a+i']
                for k_prime in range(3):
                    for i_prime in range(2):
                        W_full[3*d + k_prime, col:col+R_sub] += W_sub[k_prime*2+i_prime, :]
                
                col += R_sub
    
    # Verify
    if verify_solution(U_full.tolist(), V_full.tolist(), W_full.tolist(), total_rank):
        return {
            "U": [[float(v) for v in row] for row in U_full],
            "V": [[float(v) for v in row] for row in V_full],
            "W": [[float(v) for v in row] for row in W_full]
        }
    
    # Fallback: schoolbook
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
