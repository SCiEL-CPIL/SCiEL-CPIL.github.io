import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = DIMENSIONS
    
    # Try to construct a better decomposition
    # Strategy: use a 2x2x4 decomposition with rank 10, applied 3 times
    # First, find a rank-10 decomposition for 2x2x4
    
    # The tensor for 2x2x4: T[i,j,k] = A[i,j] * B[j,k] for i in {0,1}, j in {0,1}, k in {0,1,2,3}
    # Schoolbook rank: 8
    # Can we do better?
    
    # Let me try to use a combination of Strassen and direct methods
    # Split 4 columns into two pairs: (0,1) and (2,3)
    # Use Strassen for each pair: rank 14
    # Or use a mixed approach
    
    # Actually, let me just use the schoolbook rank-8 for each 2x2x4 block
    # Total rank: 3 * 8 = 24
    
    # Wait, that's better than 42! Let me verify.
    # Schoolbook for 2x2x4: 2*4 = 8 rank-1 terms
    # For 2x2x12: 2*12 = 24 rank-1 terms
    # But schoolbook for 2x2x12 is 2*12 = 24? No, it's (2*2)*(2*12)/... 
    # Actually, schoolbook rank for multiplying (n x m) by (m x p) is n*m*p
    # For 2x2 by 2x12: 2*2*12 = 48
    
    # So schoolbook is 48, Strassen gives 42.
    # A 2x2x4 schoolbook is 2*2*4 = 16, not 8.
    
    # I was confused. Let me reconsider.
    # Schoolbook: for each i, j, k, C[i,k] += A[i,j] * B[j,k]
    # Number of rank-1 terms: n * m * p = 2 * 2 * 12 = 48
    
    # Strassen for 2x2x2: 7 terms
    # For 2x2x12 with 6 blocks of 2: 6 * 7 = 42
    
    # Can I do better?
    # What if I use a 2x2x4 decomposition?
    # Schoolbook for 2x2x4: 2*2*4 = 16
    # Strassen-based: 2 blocks of 2x2x2, each with 7 terms: 14
    # So Strassen-based is better for 2x2x4 too.
    
    # For 2x2x12: 3 blocks of 2x2x4, each with 14 terms: 42 (same as before)
    # Or 6 blocks of 2x2x2, each with 7 terms: 42 (same)
    
    # So I need a different approach.
    
    # Let me try to find a rank-10 decomposition for 2x2x4
    # This would give 3*10 = 30 for 2x2x12
    
    # To find a rank-10 decomposition, I can try to use linear combinations
    # of the 14 Strassen terms to eliminate 4 of them
    
    # This is a hard problem. Let me just return the rank-42 solution.
    
    # Actually, let me try a different idea: use a rank-6 decomposition for 2x2x2
    # (which doesn't exist, since 7 is optimal) - so that's out.
    
    # What about using a non-uniform partition?
    # E.g., 2x2x12 = 2x2x(2+2+2+2+2+2) = 6 blocks of 2x2x2, rank 42
    # Or 2x2x(3+3+3+3) = 4 blocks of 2x2x3
    # What's the rank of 2x2x3?
    # Schoolbook: 2*2*3 = 12
    # Can I do better?
    
    # For 2x2x3, I can try to use a Strassen-like approach
    # Split 3 columns into (2,1): use Strassen for the 2-column part (7 terms)
    # and schoolbook for the 1-column part (2 terms): total 9
    # Or (1,2): same, 9 terms
    # So rank 9 for 2x2x3?
    
    # Then 4 blocks of 2x2x3: 4*9 = 36
    
    # That's better than 42! Let me implement this.
    
    # For 2x2x3: columns 0, 1, 2
    # Use Strassen for columns 0,1 (7 terms) and schoolbook for column 2 (2 terms)
    # Total: 9 terms
    
    # Let me construct this
    
    def construct_2x2x3():
        """Construct a rank-9 decomposition for 2x2x3 matrix multiplication"""
        n_local, m_local, p_local = 2, 2, 3
        rank_local = 9
        
        U_local = [[0] * rank_local for _ in range(n_local * m_local)]  # (4, 9)
        V_local = [[0] * rank_local for _ in range(m_local * p_local)]  # (6, 9)
        W_local = [[0] * rank_local for _ in range(p_local * n_local)]  # (6, 9)
        
        # Strassen terms for columns 0,1 (7 terms)
        strassen_terms = [
            {0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1},       # M1
            {2: 1, 3: 1}, {0: 1},        {2: 1, 3: -1},      # M2
            {0: 1},       {1: 1, 3: -1}, {1: 1, 3: 1},       # M3
            {3: 1},       {2: 1, 0: -1}, {0: 1, 2: 1},       # M4
            {0: 1, 1: 1}, {3: 1},        {0: -1, 1: 1},      # M5
            {0: -1, 2: 1},{0: 1, 1: 1},  {3: 1},             # M6
            {1: 1, 3: -1},{2: 1, 3: 1},  {0: 1},             # M7
        ]
        
        for s in range(7):
            col = s
            u_coeffs = strassen_terms[s * 3]
            v_coeffs = strassen_terms[s * 3 + 1]
            w_coeffs = strassen_terms[s * 3 + 2]
            
            for a_idx, val in u_coeffs.items():
                U_local[a_idx][col] = val
            
            for local_b, val in v_coeffs.items():
                if local_b == 0:
                    global_b = 0  # B[0,0]
                elif local_b == 1:
                    global_b = 1  # B[0,1]
                elif local_b == 2:
                    global_b = 3  # B[1,0] = m_local*p_local//2 + 0 = 3
                else:
                    global_b = 4  # B[1,1] = 3 + 1 = 4
                V_local[global_b][col] = val
            
            for local_c, val in w_coeffs.items():
                if local_c == 0:
                    global_w = 0 * n_local + 0  # C[0,0]
                elif local_c == 1:
                    global_w = 1 * n_local + 0  # C[0,1]
                elif local_c == 2:
                    global_w = 0 * n_local + 1  # C[1,0]
                else:
                    global_w = 1 * n_local + 1  # C[1,1]
                W_local[global_w][col] += val
        
        # Schoolbook terms for column 2 (2 terms)
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
        
        # Term 7: A[0,0] * B[0,2]
        col = 7
        U_local[0][col] = 1  # A[0,0]
        V_local[2][col] = 1  # B[0,2] = index 2 in (m*p) = (2*3) = 6, so B[0,2] is at index 0*3+2=2
        W_local[2 * n_local + 0][col] = 1  # C[0,2] = index 2*2+0 = 4
        
        # Term 8: A[0,1] * B[1,2]
        col = 8
        U_local[1][col] = 1  # A[0,1]
        V_local[5][col] = 1  # B[1,2] = index 1*3+2=5
        W_local[2 * n_local + 0][col] = 1  # C[0,2]
        
        # Wait, I need 4 terms for column 2, not 2:
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] (2 terms)
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] (2 terms)
        # Total: 4 terms for column 2
        
        # So rank_local should be 7 + 4 = 11, not 9!
        
        # Let me fix this
        
        return None  # Placeholder
    
    # I made an error. Let me reconsider.
    # For 2x2x3:
    # Strassen for columns 0,1: 7 terms
    # Schoolbook for column 2: 4 terms (2 rows * 2 columns of A)
    # Total: 11 terms
    
    # That's worse than the 12 from schoolbook for 2x2x3.
    # So this approach doesn't help.
    
    # Let me try a different approach: use a rank-10 decomposition for 2x2x4
    # and apply it 3 times for 2x2x12, giving rank 30.
    
    # To find a rank-10 decomposition for 2x2x4, I can try to use
    # a combination of Strassen and direct methods.
    
    # For 2x2x4:
    # Option 1: Two Strassen blocks: 14 terms
    # Option 2: One Strassen block (2 cols) + schoolbook (2 cols): 7 + 8 = 15
    # Option 3: Find a better decomposition
    
    # I don't know of a rank-10 decomposition for 2x2x4.
    
    # Let me just return the rank-42 solution.
    
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = [
        {0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1},
        {2: 1, 3: 1}, {0: 1},        {2: 1, 3: -1},
        {0: 1},       {1: 1, 3: -1}, {1: 1, 3: 1},
        {3: 1},       {2: 1, 0: -1}, {0: 1, 2: 1},
        {0: 1, 1: 1}, {3: 1},        {0: -1, 1: 1},
        {0: -1, 2: 1},{0: 1, 1: 1},  {3: 1},
        {1: 1, 3: -1},{2: 1, 3: 1},  {0: 1},
    ]
    
    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            u_coeffs = strassen_terms[s * 3]
            v_coeffs = strassen_terms[s * 3 + 1]
            w_coeffs = strassen_terms[s * 3 + 2]
            
            for a_idx, val in u_coeffs.items():
                U[a_idx][col] = val
            
            for local_b, val in v_coeffs.items():
                if local_b == 0:
                    global_b = 2 * t
                elif local_b == 1:
                    global_b = 2 * t + 1
                elif local_b == 2:
                    global_b = m * p // 2 + 2 * t
                else:
                    global_b = m * p // 2 + 2 * t + 1
                V[global_b][col] = val
            
            for local_c, val in w_coeffs.items():
                if local_c == 0:
                    global_w = (2 * t) * n + 0
                elif local_c == 1:
                    global_w = (2 * t + 1) * n + 0
                elif local_c == 2:
                    global_w = (2 * t) * n + 1
                else:
                    global_w = (2 * t + 1) * n + 1
                W[global_w][col] += val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
