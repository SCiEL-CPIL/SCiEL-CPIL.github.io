import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Strategy: Use rank-49 Strassen decomposition for 4x4x4
    # Embed 2x4 into 4x4 (top-left), split 4x8 into two 4x4 blocks
    # Result: 4x4 output, project first 2 rows, both 4-col blocks -> 2x8
    
    # Strassen 7-term for 4x4x4 (each 4x4 is built from 2x2 blocks)
    # Actually, let's use the direct 4x4 Strassen: rank 49 = 7^2
    # The 4x4 multiplication tensor has rank 49 via two levels of Strassen
    
    # For 4x4 x 4x4 -> 4x4:
    # Split into 2x2 blocks: A = [[A11,A12],[A21,A22]], B = [[B11,B12],[B21,B22]]
    # C11 = M1 - M3, C12 = M3 + M4, C21 = M2 - M4, C22 = M1 - M2 + M5 + M6
    # where M1=(A11+A22)(B11+B22), M2=(A21+A22)B11, M3=A11(B12-B22),
    #       M4=A22(B21-B11), M5=(A11+A12)B22, M6=(A21-A11)(B11+B12),
    #       M7=(A12-A22)(B21+B22)
    
    # Each Mi is a 2x2 product, which itself uses rank 7 Strassen
    # So total rank = 7 * 7 = 49
    
    # Now embed our problem:
    # A is 2x4, embed into 4x4: A_full = [[A, 0], [0, 0]]
    # B is 4x8, split into two 4x4: B1 = B[:,0:4], B2 = B[:,4:8]
    # We need C = A @ B = A @ [B1, B2] = [A@B1, A@B2]
    # 
    # Using 4x4 Strassen: C_full = A_full @ B_full
    # But we only need the top-left 2x4 of C_full for B1, and top-right 2x4 for B2
    #
    # Actually, let's think differently. We'll do TWO 4x4 multiplications:
    # C1 = A_full @ B1 (4x4 result, take top 2 rows -> 2x4)
    # C2 = A_full @ B2 (4x4 result, take top 2 rows -> 2x4)
    # Total rank = 49 + 49 = 98? No, that's worse.
    
    # Better approach: Single 4x4x4 with appropriate embedding
    # Let's use the fact that A is 2x4, so embed as 4x4 with zeros
    # B is 4x8, but we can only multiply 4x4. So we need two separate products.
    
    # Alternative: Use a different decomposition.
    # The tensor for (2,4,8) has dimension 2*4*8 = 64.
    # Rank lower bound: max(2*4, 4*8, 2*8) = 32.
    # Upper bound via Strassen embedding: 
    #   2x4 x 4x8: embed A into 4x4 (pad with zeros), B into 4x8 (already)
    #   But 4x4 x 4x8 is not standard.
    
    # Let me try: decompose 4x8 as two 4x4, do two 4x4x4 Strassen (rank 49 each)
    # but share the A factor. Total rank = 49 + 49 = 98. Too high.
    
    # Better: Use the structure. A is 2x4, so effectively only 2 rows matter.
    # We can write: C[i,k] = sum_j A[i,j] * B[j,k]
    # For i=0,1 and k=0..7
    
    # Let's use a rank-49 approach differently:
    # The (2,4,8) tensor can be seen as (4,4,4) with constraints.
    # Actually, let's just do: embed into (4,4,8) and use rank 4*49 = 196? No.
    
    # Let me reconsider. The current rank is 56. Let's try to get 49.
    # 
    # Direct approach: The (2,4,8) multiplication tensor T[i,j,k] = delta
    # We need T = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # 
    # Rank 49 = 7^2. This suggests a 2-level Strassen structure.
    # 
    # Level 1: Split j dimension (size 4) into two halves (size 2)
    #   A[i, j1], A[i, j2] where j1 = 0,1 and j2 = 2,3
    #   B[j1, k], B[j2, k]
    #   C[i, k] = A[i,j1]@B[j1,k] + A[i,j2]@B[j2,k]
    # 
    # This is a (2,2,8) + (2,2,8) problem, each of rank at most 7*... 
    # (2,2,8) has rank at most 7*2 = 14? No, (2,2,p) has rank 7 for p=2, 
    # for p=8 it's more complex.
    
    # Let me just try the straightforward 4x4 embedding with rank 49:
    # We'll construct U, V, W of size (8, 49), (32, 49), (16, 49)
    
    rank = 49
    U = np.zeros((n * m, rank))  # (8, 49)
    V = np.zeros((m * p, rank))  # (32, 49)
    W = np.zeros((p * n, rank))  # (16, 49)
    
    # Strassen 4x4x4: rank 49
    # The 4x4 matrix A has entries A[i,j] for i,j in 0..3
    # The 4x4 matrix B has entries B[j,k] for j,k in 0..3
    # The 4x4 matrix C has entries C[i,k] for i,k in 0..3
    #
    # For our problem:
    # A_full[i,j] = A_orig[i,j] if i<2, else 0  (2x4 embedded in 4x4)
    # B_full[j,k] = B_orig[j,k] if k<4, else 0  (4x8, first half in 4x4)
    # C_full[i,k] = C_orig[i,k] if i<2, k<4  (top-left 2x4 of 4x4)
    #
    # But we also need the second half of B. So we need TWO such products.
    # That gives rank 98, which is worse than 56.
    
    # Alternative: Use a (2,4,8) specific decomposition.
    # Let's try rank 49 with a different structure.
    # 
    # Actually, the theoretical minimum for (2,4,8) is at least 2*4 = 8 (trivial)
    # and at most 2*4*8 = 64 (naive).
    # Strassen gives: 2x4 x 4x8 -> embed in 4x4 x 4x4, rank 49, but we only
    # use part of it. The effective rank might be less.
    
    # Let me try a different approach: use the known (2,4,4) rank and extend.
    # (2,4,4) has rank at most 7*2 = 14? Or 4*7 = 28?
    # Actually (2,4,4): A is 2x4, B is 4x4, C is 2x4.
    # Embed A into 4x4, do 4x4x4 Strassen (rank 49), take top 2 rows.
    # But many of the 49 terms will be zero because A has zeros in bottom 2 rows.
    
    # Let's count: In the 4x4 Strassen, A_full = [[A, 0], [0, 0]]
    # The 7 intermediate matrices M1..M7 involve sums of A blocks.
    # A11 = A (2x2), A12 = 0 (2x2), A21 = 0, A22 = 0
    # M1 = (A11+A22)(B11+B22) = A*(B11+B22) -> nonzero
    # M2 = (A21+A22)B11 = 0
    # M3 = A11(B12-B22) = A*(B12-B22) -> nonzero
    # M4 = A22(B21-B11) = 0
    # M5 = (A11+A12)B22 = A*B22 -> nonzero
    # M6 = (A21-A11)(B11+B12) = -A*(B11+B12) -> nonzero
    # M7 = (A12-A22)(B21+B22) = 0
    # So M2, M4, M7 are zero. That's 3 out of 7 zero.
    # Each Mi is a 2x2 product with rank 7, so 3*7 = 21 terms are zero.
    # Remaining: 4*7 = 28 terms.
    # So rank for (2,4,4) is at most 28.
    
    # For (2,4,8): split B into two 4x4 halves.
    # First half: rank 28 (as above)
    # Second half: same structure, rank 28
    # Total: 56. That's what we already have!
    
    # To get below 56, we need a better decomposition.
    # 
    # Let's try: (2,4,8) directly.
    # Split the 8 columns into two groups of 4.
    # C[:, 0:4] = A @ B[:, 0:4]
    # C[:, 4:8] = A @ B[:, 4:8]
    # Each is a (2,4,4) problem with rank at most 28.
    # But can we share some factors?
    
    # The A factor is the same in both. In the Strassen decomposition,
    # the U matrix (for A) is the same for both halves.
    # The V matrix (for B) is different.
    # The W matrix (for C) is different.
    # So the total rank is 28 + 28 = 56, with no sharing.
    
    # To reduce rank, we need a smarter decomposition.
    # 
    # Let's try: use the (2,4,4) rank-28 decomposition and see if we can
    # combine the two halves to get below 56.
    
    # Actually, let's just implement the rank-56 solution more efficiently
    # and then try to reduce it.
    
    # Rank-56 Strassen-based decomposition:
    # 8 blocks of 7-term Strassen for 2x2x2, arranged as 2x4 grid.
    
    rank = 56
    U = np.zeros((n * m, rank))  # (8, 56)
    V = np.zeros((m * p, rank))  # (32, 56)
    W = np.zeros((p * n, rank))  # (16, 56)
    
    # Strassen 2x2x2: 7 terms
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    # M1 = (a+d)(e+h) -> i,l
    # M2 = (c+d)e -> k
    # M3 = a(f-h) -> j
    # M4 = d(g-e) -> k
    # M5 = (a+b)h -> j,l
    # M6 = (c-a)(e+f) -> i,k
    # M7 = (b-d)(g+h) -> i,l
    
    strassen_U = [
        {(0,0): 1, (1,1): 1},  # M1: a+d
        {(1,0): 1, (1,1): 1},  # M2: c+d
        {(0,0): 1},             # M3: a
        {(1,1): 1},             # M4: d
        {(0,0): 1, (0,1): 1},  # M5: a+b
        {(1,0): 1, (0,0): -1}, # M6: c-a
        {(0,1): 1, (1,1): -1}, # M7: b-d
    ]
    strassen_V = [
        {(0,0): 1, (1,1): 1},  # M1: e+h
        {(0,0): 1},             # M2: e
        {(0,1): 1, (1,1): -1}, # M3: f-h
        {(1,0): 1, (0,0): -1}, # M4: g-e
        {(1,1): 1},             # M5: h
        {(0,0): 1, (0,1): 1},  # M6: e+f
        {(1,0): 1, (1,1): 1},  # M7: g+h
    ]
    strassen_W = [
        {(0,0): 1, (1,1): 1},  # M1: i,l
        {(1,0): 1},             # M2: k
        {(0,1): 1},             # M3: j
        {(1,0): 1},             # M4: k
        {(0,1): 1, (1,1): 1},  # M5: j,l
        {(0,0): -1, (1,0): 1}, # M6: i,k
        {(0,0): 1, (1,1): 1},  # M7: i,l
    ]
    
    def add_block(base_rank, i_base, j_base, k_base):
        """Add a 2x2x2 Strassen block at position (i_base, j_base, k_base)."""
        for r in range(7):
            cr = base_rank + r
            for (li, lj), val in strassen_U[r].items():
                gi = i_base + li
                gj = j_base + lj
                U[gi * 4 + gj, cr] += val
            for (lj, lk), val in strassen_V[r].items():
                gj = j_base + lj
                gk = k_base + lk
                V[gj * 8 + gk, cr] += val
            for (lk, li), val in strassen_W[r].items():
                gk = k_base + lk
                gi = i_base + li
                W[gk * 2 + gi, cr] += val
    
    # 8 blocks: 2 rows (i=0,1) x 4 columns (k=0,1,2,3) x j=0,1
    # For i in 0,1 and k in 0,1,2,3:
    #   Block at (i_base=i, j_base=0, k_base=k*2)
    #   Block at (i_base=i, j_base=2, k_base=k*2)
    # Wait, j has size 4, so j_base can be 0 or 2.
    # k has size 8, so k_base can be 0, 2, 4, 6.
    # i has size 2, so i_base = 0 only (since 2x2 blocks, and i has size 2).
    
    # Actually, A is 2x4, so i has size 2, j has size 4.
    # B is 4x8, so j has size 4, k has size 8.
    # C is 2x8, so i has size 2, k has size 8.
    
    # 2x2x2 blocks:
    # i_base = 0 (only one block in i direction, since i has size 2)
    # j_base = 0 or 2 (two blocks in j direction, since j has size 4)
    # k_base = 0, 2, 4, 6 (four blocks in k direction, since k has size 8)
    # Total: 1 * 2 * 4 = 8 blocks, each of rank 7, total rank 56.
    
    add_block(0, 0, 0, 0)
    add_block(7, 0, 0, 2)
    add_block(14, 0, 0, 4)
    add_block(21, 0, 0, 6)
    add_block(28, 0, 2, 0)
    add_block(35, 0, 2, 2)
    add_block(42, 0, 2, 4)
    add_block(49, 0, 2, 6)
    
    # Verify
    def compute_tensor(U_arr, V_arr, W_arr, rank_val):
        tensor = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_vals = U_arr[i*m+j, :rank_val]
                    v_vals = V_arr[j*p+k, :rank_val]
                    w_vals = W_arr[k*n+i, :rank_val]
                    tensor[i,j,k] = np.dot(u_vals, v_vals * w_vals)
        return tensor
    
    def is_valid(U_arr, V_arr, W_arr, rank_val):
        tensor = compute_tensor(U_arr, V_arr, W_arr, rank_val)
        return np.allclose(tensor, np.ones((n, m, p)), atol=1e-10)
    
    if not is_valid(U, V, W, rank):
        # Fallback: return naive rank-64
        rank = 64
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j, r] = 1
                    V[j * p + k, r] = 1
                    W[k * n + i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = rank
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    # Greedy column removal
    current_rank = rank
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    
    improved = True
    while improved:
        improved = False
        for r in range(current_rank):
            if time.time() > deadline - 5:
                break
            
            test_U = current_U.copy()
            test_V = current_V.copy()
            test_W = current_W.copy()
            test_U[:, r] = 0
            test_V[:, r] = 0
            test_W[:, r] = 0
            
            if is_valid(test_U, test_V, test_W, current_rank):
                mask = [x for x in range(current_rank) if x != r]
                current_U = current_U[:, mask]
                current_V = current_V[:, mask]
                current_W = current_W[:, mask]
                current_rank -= 1
                improved = True
                if current_rank < best_rank:
                    best_rank = current_rank
                    best_U = current_U.copy()
                    best_V = current_V.copy()
                    best_W = current_W.copy()
                break
        
        if time.time() > deadline - 5:
            break
    
    # Local search: try to zero out individual entries
    if time.time() < deadline - 10:
        import random
        random.seed(42)
        
        for trial in range(200):
            if time.time() > deadline - 5:
                break
            
            test_U = best_U.copy()
            test_V = best_V.copy()
            test_W = best_W.copy()
            test_rank = best_rank
            
            for _ in range(30):
                if time.time() > deadline - 5:
                    break
                
                mat_choice = random.randint(0, 2)
                if mat_choice == 0:
                    row = random.randint(0, test_U.shape[0]-1)
                    col = random.randint(0, test_rank-1)
                    if test_U[row, col] == 0:
                        continue
                    old_val = test_U[row, col]
                    test_U[row, col] = 0
                    if not is_valid(test_U, test_V, test_W, test_rank):
                        test_U[row, col] = old_val
                elif mat_choice == 1:
                    row = random.randint(0, test_V.shape[0]-1)
                    col = random.randint(0, test_rank-1)
                    if test_V[row, col] == 0:
                        continue
                    old_val = test_V[row, col]
                    test_V[row, col] = 0
                    if not is_valid(test_U, test_V, test_W, test_rank):
                        test_V[row, col] = old_val
                else:
                    row = random.randint(0, test_W.shape[0]-1)
                    col = random.randint(0, test_rank-1)
                    if test_W[row, col] == 0:
                        continue
                    old_val = test_W[row, col]
                    test_W[row, col] = 0
                    if not is_valid(test_U, test_V, test_W, test_rank):
                        test_W[row, col] = old_val
            
            # Try greedy column removal on the modified matrices
            cur_rank = test_rank
            cur_U = test_U.copy()
            cur_V = test_V.copy()
            cur_W = test_W.copy()
            imp = True
            while imp:
                imp = False
                for r in range(cur_rank):
                    if time.time() > deadline - 5:
                        break
                    tU = cur_U.copy()
                    tV = cur_V.copy()
                    tW = cur_W.copy()
                    tU[:, r] = 0
                    tV[:, r] = 0
                    tW[:, r] = 0
                    if is_valid(tU, tV, tW, cur_rank):
                        mask = [x for x in range(cur_rank) if x != r]
                        cur_U = cur_U[:, mask]
                        cur_V = cur_V[:, mask]
                        cur_W = cur_W[:, mask]
                        cur_rank -= 1
                        imp = True
                        break
                if time.time() > deadline - 5:
                    break
            
            if cur_rank < best_rank:
                best_rank = cur_rank
                best_U = cur_U.copy()
                best_V = cur_V.copy()
                best_W = cur_W.copy()
    
    # Return result
    U_final = best_U.tolist()
    V_final = best_V.tolist()
    W_final = best_W.tolist()
    
    # Round to half-integers if close
    for mat in [U_final, V_final, W_final]:
        for i in range(len(mat)):
            for j in range(len(mat[i])):
                val = mat[i][j]
                rounded = round(val * 2) / 2
                if abs(val - rounded) < 1e-10:
                    mat[i][j] = rounded
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
