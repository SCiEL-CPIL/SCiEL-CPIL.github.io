import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    # Build schoolbook decomposition as fallback
    rank_school = n * m * p  # 40
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": U_school}  # placeholder, will fix
    best_rank = rank_school
    
    # Rebuild schoolbook properly
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Strategy: Non-rectangular block decomposition
    # Partition m=4 into m1=3 and m2=1
    # For 2x3x5: use a rank-12 decomposition
    # For 2x1x5: use a rank-4 decomposition (schoolbook for 2x1x5)
    # Total: 12 + 4 = 16
    
    # Let's try to find a good decomposition for 2x3x5
    # The tensor T[i,j,k] = 1 if (i,j,k) is a valid triple, else 0
    # For 2x3x5, we have indices i in {0,1}, j in {0,1,2}, k in {0,1,2,3,4}
    
    # One approach: use the fact that 2x3x5 can be seen as combining
    # a 2x2x5 part and a 2x1x5 part
    
    # For 2x2x5: we can use Strassen-like for the 2x2 part with 5 columns
    # Actually, let's think differently.
    
    # For multiplying 2x3 by 3x5:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # etc.
    
    # Let's try a direct approach: find a rank-r decomposition of the 2x3x5 tensor
    # using integer or half-integer entries.
    
    # The 2x3x5 tensor has 2*3*5 = 30 entries that are 1.
    # We need U (2x3 x r), V (3x5 x r), W (5x2 x r) such that
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = delta[i,j,k] (1 if valid, 0 otherwise)
    
    # Let's try rank 12 for 2x3x5
    # We can think of it as: for each (i,j), we need to "cover" all k values
    # Using linearity: if we can express each A[i,j] * (B[0,k], B[1,k], B[2,k], B[3,k], B[4,k])
    # with a fixed number of multiplications...
    
    # Actually, let's use a different strategy:
    # Use the known result that matmul(2,3,5) has rank at most 12
    # This can be achieved by:
    # - Using 2x2x5 subproblem solved with rank 8 (using Strassen for 2x2x2 twice + 2x2x1)
    #   Wait, 2x2x5: we can do 2x2x2 (rank 7) + 2x2x2 (rank 7) + 2x2x1 (rank 4) = 18? No.
    #   Actually 2x2x5 = 2x2x2 + 2x2x2 + 2x2x1, ranks 7+7+4=18. That's worse than schoolbook 20.
    #   Hmm, but we can do better.
    
    # Let me try a different approach: use the fact that we can share computations.
    
    # For 2x3x5, let's try to use a rank-12 decomposition directly.
    # The idea: 
    # M1 = A[0,0] * (B[0,0]+B[1,0]+B[2,0])  -> contributes to C[0,0]
    # This is getting complicated. Let me try a systematic search.
    
    # Actually, let's try a known construction:
    # For 2x3x5, we can use the following rank-12 scheme:
    # Define the 12 rank-1 terms.
    
    # Let me try a different structural approach:
    # Use the "bilinear" structure more cleverly.
    
    # For 2x4x5, let's try to use a rank-16 decomposition directly.
    # This would be a significant improvement over 36.
    
    # The key insight: we can use the fact that the matrix multiplication tensor
    # has a lot of symmetry and structure that can be exploited.
    
    # Let's try: partition into 2x2x5 and 2x2x5 blocks (using m=4 = 2+2)
    # For each 2x2x5: use a rank-8 decomposition (if it exists)
    # Total: 8+8 = 16
    
    # Can we decompose 2x2x5 with rank 8?
    # 2x2x5 = 2x2x2 + 2x2x2 + 2x2x1
    # If 2x2x2 has rank 7 (Strassen) and 2x2x1 has rank 4, total = 7+7+4 = 18.
    # But maybe we can do better by sharing between the two 2x2x2 blocks.
    
    # Alternative: 2x2x5 can be seen as computing 2x2 product with 5 right columns.
    # The 2x2x2 part uses 7 multiplications. For the remaining 3 columns,
    # we need 2x2x3. Can we do 2x2x3 with rank 5?
    # 2x2x3: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for k=0,1,2
    # This is 4 independent products for each k, but we can share.
    # Actually, for fixed k, it's a 2x2x1 which needs rank 4 (no savings for single column).
    # For 2x2x3, we can use:
    # M1 = A[0,0]*B[0,0], M2 = A[0,1]*B[1,0], M3 = A[1,0]*B[0,1], M4 = A[1,1]*B[1,1]
    # That's 4 for the first two columns... no, that's not right.
    
    # Let me just try to find a rank-16 decomposition for 2x4x5 by direct construction.
    
    # Approach: Use the "Laderman" style or similar for rectangular cases.
    
    # For now, let's try to improve on the Strassen approach by using a better
    # 2x2x1 decomposition or by sharing between blocks.
    
    # Actually, the current Strassen approach gives rank 36. Let's verify it works
    # and then try to improve.
    
    # Let's try a rank-24 approach:
    # Use 2x2x5 blocks with a rank-12 decomposition each.
    # 2x2x5: We can compute this as:
    # For the 2x2 part, use Strassen (7 mults) for each pair of columns.
    # 5 columns = 2 pairs + 1 single
    # 2x2x2 (7) + 2x2x2 (7) + 2x2x1 (4) = 18 per 2x2x5 block
    # Two blocks: 18+18 = 36. Same as before.
    
    # Let's try to do 2x2x5 with rank 12:
    # Idea: use the fact that we have 5 columns, so we can use more aggressive sharing.
    # 
    # For 2x2x5, the output is a 2x5 matrix C where:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # 
    # We need 10 output values. With rank r, we have r multiplications.
    # Each multiplication gives a "pattern" across the 10 outputs.
    # 
    # Let's try rank 12 for 2x2x5.
    # 
    # Define 12 rank-1 terms:
    # For k=0,1: use Strassen on 2x2x2 (7 terms)
    # For k=2,3: use Strassen on another 2x2x2 (7 terms)  
    # For k=4: need 4 more terms
    # Total: 7+7+4 = 18. Still 18.
    
    # To get rank 12, we need to share between the k blocks.
    # This is the key challenge.
    
    # Let's try a completely different approach: use the "tensor rank" literature.
    # The tensor rank of matmul(2,4,5) is known to be at most 24 (from various bounds).
    # Let's try to achieve rank 24.
    
    # Construction for rank 24:
    # Use the fact that 2x4x5 = 2x4x2 + 2x4x2 + 2x4x1
    # 2x4x2: This is multiplying 2x4 by 4x2. 
    #   We can view this as: for each of the 2 output columns, we compute a 2-vector.
    #   C[:,k] = A * B[:,k]
    #   This is 2 independent 2x4x1 products, each of rank 8 (schoolbook).
    #   But we can do better: use the structure.
    #   
    #   Actually, 2x4x2 can be seen as a 2x4 by 4x2 product.
    #   Let's partition A into [A1; A2] where A1 is 2x2 and A2 is 2x2.
    #   B = [B1, B2] where B1 is 4x1 and B2 is 4x1.
    #   C = [A1*B1_top + A2*B1_bottom, A1*B2_top + A2*B2_bottom]
    #   Wait, B is 4x2, so B1 and B2 are 4x1 vectors.
    #   Let B1 = [b1_top; b1_bottom] where b1_top is 2x1 and b1_bottom is 2x1.
    #   Similarly for B2.
    #   C[:,0] = A1*b1_top + A2*b1_bottom
    #   C[:,1] = A1*b2_top + A2*b2_bottom
    #   
    #   A1*b1_top is a 2x1x1 product (rank 2)
    #   A2*b1_bottom is a 2x1x1 product (rank 2)
    #   So C[:,0] needs rank 4.
    #   Similarly C[:,1] needs rank 4.
    #   Total for 2x4x2: rank 8.
    #   
    #   But wait, we can share between the two output columns!
    #   Let's use:
    #   M1 = A[0,0] * (B[0,0] + B[0,1])
    #   M2 = A[0,1] * (B[1,0] + B[1,1])
    #   M3 = A[0,2] * (B[2,0] + B[2,1])
    #   M4 = A[0,3] * (B[3,0] + B[3,1])
    #   M5 = A[1,0] * (B[0,0] + B[0,1])
    #   M6 = A[1,1] * (B[1,0] + B[1,1])
    #   M7 = A[1,2] * (B[2,0] + B[2,1])
    #   M8 = A[1,3] * (B[3,0] + B[3,1])
    #   That's 8 multiplications, but we only get sums. We need individual values.
    #   
    #   To get individual values, we need more. This approach doesn't directly work.
    
    # Let's try yet another approach: use the "Winograd" style transform.
    
    # For now, let's just try to get a valid rank-36 decomposition (the Strassen one)
    # and verify it works. If it does, we'll try to improve from there.
    
    # Let's rebuild the Strassen decomposition more carefully.
    
    total_rank = 36
    U = np.zeros((nm, total_rank), dtype=np.float64)
    V = np.zeros((mp, total_rank), dtype=np.float64)
    W = np.zeros((pn, total_rank), dtype=np.float64)
    
    def add_2x2x2_block(U, V, W, a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset, col_offset):
        # Standard Strassen for 2x2x2
        # Local indices for A: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # Local indices for B: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # Local indices for C: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        
        strassen_patterns = [
            # M1 = (A00+A11)*(B00+B11)
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            # M2 = (A10+A11)*B00
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            # M3 = A00*(B10-B11)
            ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]),
            # M4 = A11*(B00-B10)
            ([0, 0, 0, 1], [1, 0, -1, 0], [1, 0, 1, 0]),
            # M5 = (A00+A01)*B11
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            # M6 = (A10-A00)*(B00+B01)
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            # M7 = (A01-A11)*(B10+B11)
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
        ]
        
        for u_pat, v_pat, w_pat in strassen_patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    i_global = a_i_offset + i_local
                    j_global = a_j_offset + j_local
                    U[i_global * m + j_global, col_offset] = val
            
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx // 2
                    k_local = idx % 2
                    j_global = b_j_offset + j_local
                    k_global = b_k_offset + k_local
                    V[j_global * p + k_global, col_offset] = val
            
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx // 2
                    k_local = idx % 2
                    i_global = c_i_offset + i_local
                    k_global = c_k_offset + k_local
                    W[k_global * n + i_global, col_offset] = val
            
            col_offset += 1
        
        return col_offset
    
    def add_2x2x1_block(U, V, W, a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset, col_offset):
        # Schoolbook for 2x2x1 (rank 4)
        patterns = [
            ([1, 0, 0, 0], [1], [1]),  # A00*B00 -> C00
            ([0, 1, 0, 0], [1], [0]),  # A01*B10 -> C01... wait
            # Let me redo this.
            # C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
            # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
            # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
            # So we need:
            # M1 = A[0,0]*B[0,0] -> contributes to C[0,0]
            # M2 = A[0,1]*B[1,0] -> contributes to C[0,0]
            # M3 = A[1,0]*B[0,0] -> contributes to C[1,0]
            # M4 = A[1,1]*B[1,0] -> contributes to C[1,0]
        ]
        patterns = [
            ([1, 0, 0, 0], [1], [1, 0]),  # A00*B00 -> C[0,0]
            ([0, 1, 0, 0], [0], [1, 0]),  # A01*B10 -> C[0,0]... 
            # Wait, B is 2x1, so B[0,0] and B[1,0].
            # B index: j in {0,1}, k=0. So B[0,0] is index 0, B[1,0] is index 1.
            # C is 2x1, so C[0,0] is index 0, C[1,0] is index 1.
            # 
            # M1: A[0,0] * B[0,0] -> C[0,0]
            # U pattern: A[0,0]=1, others 0. V pattern: B[0,0]=1, B[1,0]=0. W pattern: C[0,0]=1, C[1,0]=0.
            # M2: A[0,1] * B[1,0] -> C[0,0]
            # U pattern: A[0,1]=1. V pattern: B[1,0]=1. W pattern: C[0,0]=1.
            # M3: A[1,0] * B[0,0] -> C[1,0]
            # U pattern: A[1,0]=1. V pattern: B[0,0]=1. W pattern: C[1,0]=1.
            # M4: A[1,1] * B[1,0] -> C[1,0]
            # U pattern: A[1,1]=1. V pattern: B[1,0]=1. W pattern: C[1,0]=1.
        ]
        patterns = [
            ([1, 0, 0, 0], [1, 0], [1, 0]),  # A00*B00 -> C00
            ([0, 1, 0, 0], [0, 1], [1, 0]),  # A01*B10 -> C00
            ([0, 0, 1, 0], [1, 0], [0, 1]),  # A10*B00 -> C10
            ([0, 0, 0, 1], [0, 1], [0, 1]),  # A11*B10 -> C10
        ]
        
        for u_pat, v_pat, w_pat in patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    i_global = a_i_offset + i_local
                    j_global = a_j_offset + j_local
                    U[i_global * m + j_global, col_offset] = val
            
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx
                    k_local = 0
                    j_global = b_j_offset + j_local
                    k_global = b_k_offset + k_local
                    V[j_global * p + k_global, col_offset] = val
            
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx
                    k_local = 0
                    i_global = c_i_offset + i_local
                    k_global = c_k_offset + k_local
                    W[k_global * n + i_global, col_offset] = val
            
            col_offset += 1
        
        return col_offset
    
    col = 0
    
    # A is 2x4, B is 4x5, C is 2x5
    # Partition B's columns into: cols 0-1, cols 2-3, col 4
    # For each, use 2x2x2 or 2x2x1
    
    # Block 1: A (2x4) * B[:, 0:2] (4x2) -> C[:, 0:2] (2x2)
    # We can use Strassen if we partition A's rows... but A is 2x4, not 2x2.
    # We need to partition A's columns too.
    
    # Let's partition A into A_left (2x2) and A_right (2x2)
    # A_left = A[:, 0:2], A_right = A[:, 2:4]
    # B_top = B[0:2, :], B_bottom = B[2:4, :]
    
    # C = A_left * B_top + A_right * B_bottom
    
    # For A_left * B_top: 2x2 * 2x5 = 2x5
    #   Partition B_top into B_top[:, 0:2], B_top[:, 2:4], B_top[:, 4:5]
    #   A_left * B_top[:, 0:2]: 2x2x2, rank 7
    #   A_left * B_top[:, 2:4]: 2x2x2, rank 7
    #   A_left * B_top[:, 4:5]: 2x2x1, rank 4
    #   Total: 18
    
    # For A_right * B_bottom: 2x2 * 2x5 = 2x5
    #   Similarly: 7+7+4 = 18
    
    # Total: 36. Same as before.
    
    # Let's just implement this and verify.
    
    # A_left * B_top[:, 0:2]
    col = add_2x2x2_block(U, V, W, 0, 0, 0, 0, 0, 0, col)
    # A_left * B_top[:, 2:4]
    col = add_2x2x2_block(U, V, W, 0, 0, 0, 2, 0, 2, col)
    # A_left * B_top[:, 4:5]
    col = add_2x2x1_block(U, V, W, 0, 0, 0, 4, 0, 4, col)
    
    # A_right * B_bottom[:, 0:2]
    col = add_2x2x2_block(U, V, W, 0, 2, 2, 0, 0, 0, col)
    # A_right * B_bottom[:, 2:4]
    col = add_2x2x2_block(U, V, W, 0, 2, 2, 2, 0, 2, col)
    # A_right * B_bottom[:, 4:5]
    col = add_2x2x1_block(U, V, W, 0, 2, 2, 4, 0, 4, col)
    
    assert col == total_rank, f"Expected {total_rank}, got {col}"
    
    # Verify
    def verify_decomposition(U, V, W):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    s = np.sum(u_val * v_val * w_val)
                    if abs(s - 1.0) > 1e-9:
                        return False
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        continue
                    s = np.sum(U[a] * V[b] * W[c])
                    if abs(s) > 1e-9:
                        return False
        return True
    
    if verify_decomposition(U, V, W):
        U_list = [[int(round(U[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(nm)]
        V_list = [[int(round(V[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(mp)]
        W_list = [[int(round(W[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(pn)]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        return best_result
# EVOLVE-BLOCK-END
