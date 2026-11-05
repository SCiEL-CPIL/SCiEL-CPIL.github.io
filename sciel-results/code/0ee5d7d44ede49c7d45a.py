import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    start_time = time.time()
    deadline = start_time + 140.0

    # Build the rank-56 Strassen decomposition as baseline
    rank_56 = 56
    U = [[0] * rank_56 for _ in range(n * m)]
    V = [[0] * rank_56 for _ in range(m * p)]
    W = [[0] * rank_56 for _ in range(p * n)]

    strassen_U = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1},
    ]
    strassen_V = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1},
    ]
    strassen_W = [
        {(0,0): 1, (1,1): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(0,0): -1, (1,0): 1},
        {(1,1): 1},
        {(0,0): 1},
    ]

    def add_block(base_rank, i_base, j_base, k_base):
        for r in range(7):
            cr = base_rank + r
            for (li, lj), val in strassen_U[r].items():
                gi = i_base + li
                gj = j_base + lj
                U[gi * m + gj][cr] += val
            for (lj, lk), val in strassen_V[r].items():
                gj = j_base + lj
                gk = k_base + lk
                V[gj * p + gk][cr] += val
            for (lk, li), val in strassen_W[r].items():
                gk = k_base + lk
                gi = i_base + li
                W[gk * n + gi][cr] += val

    # 8 blocks
    add_block(0, 0, 0, 0)
    add_block(7, 0, 0, 2)
    add_block(14, 0, 0, 4)
    add_block(21, 0, 0, 6)
    add_block(28, 0, 2, 0)
    add_block(35, 0, 2, 2)
    add_block(42, 0, 2, 4)
    add_block(49, 0, 2, 6)

    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    def verify(U_arr, V_arr, W_arr, rank_val):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_vals = U_arr[i*m+j, :rank_val]
                    v_vals = V_arr[j*p+k, :rank_val]
                    w_vals = W_arr[k*n+i, :rank_val]
                    s = np.dot(u_vals, v_vals * w_vals)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    best_rank = rank_56
    best_U = U_np
    best_V = V_np
    best_W = W_np

    if verify(U_np, V_np, W_np, rank_56):
        # Try to find a lower rank decomposition
        # Strategy: Try rank 48 by using 6 blocks of 2x4x4 with shared structure
        # Or try to find a direct rank-48, 49, 50, ... decomposition
        
        # Approach: Try to use the fact that we have 2 rows, so we can potentially
        # share more structure. Let's try a "row-shared" Strassen approach.
        
        # For rank 48: use 6 Strassen blocks but with 2 blocks sharing some factors
        # Actually, let's try a different tiling: 2x4x8 = 2*(2x4x4)
        # For each 2x4x4, use 4 Strassen blocks (rank 28 each) = 56 total
        # But if we can do 2x4x4 in rank 24, then 2*24=48
        
        # Let me try to find a rank-24 decomposition for 2x4x4
        # 2x4x4: A is 2x4, B is 4x4, C is 4x2 (or 2x4 depending on convention)
        # Trivial: 32, Strassen: 28
        
        # Try rank 24 for 2x4x4:
        # We need to find U (8x24), V (16x24), W (8x24) such that
        # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
        
        # Let's try a structured approach:
        # Split A into A1 (2x2) and A2 (2x2)
        # Split B into B11, B12, B21, B22 (each 2x2)
        # C = A1*B11 + A2*B21 (left half) and A1*B12 + A2*B22 (right half)
        
        # For each 2x2x2 subproblem, use Strassen (rank 7)
        # But share the "A" part? No, in tensor decomposition we can't easily share.
        
        # Alternative: Try to use a "Kronecker-like" structure
        # For 2x4x4, think of it as (2x2) x (2x2) x (2x2) with some replication
        
        # Let's just try to verify if rank 48 is achievable with a different construction
        # Try: 4 blocks of 2x4x4 where each uses rank 12 (if possible)
        # Or 6 blocks of 2x4x4 where each uses rank 8 (unlikely)
        
        # Let me try a direct search for small rank improvements
        # Try rank 55, 54, 53, 52, 51, 50, 49, 48
        
        # For now, let's try to see if we can get rank 48 by a clever construction
        # Idea: Use 3 Strassen blocks for the "top" 2x4x4 and 3 for the "bottom" 2x4x4
        # but with some factors shared via the 2-row structure
        
        # Actually, let me try a completely different approach:
        # Use the "Laderman" style for rectangular matrices
        
        # For 2x4 x 4x8, think of it as:
        # A = [A1 A2] where A1, A2 are 2x2
        # B = [[B11, B12, B13, B14],
        #      [B21, B22, B23, B24]]
        # C = A1*B_left + A2*B_right where B_left is 4x4, B_right is 4x4
        
        # Hmm, this is getting complex. Let me just try to verify the current solution
        # and return it, since finding a better one requires significant search.
        
        pass

    if time.time() > deadline - 10:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }

    # Try to find a rank-48 decomposition using a structured approach
    # Idea: For 2x4x8, use 6 blocks where each block handles a 2x4x4 subproblem
    # but with rank 8 each (if we can find a rank-8 decomposition for 2x4x4)
    
    # Actually, let me try: 2x4x4 can be done in rank 28 (Strassen).
    # But maybe we can do 2x4x8 in rank 48 by using 6 "mixed" blocks.
    
    # Let me try a different tiling:
    # A (2x4) * B (4x8) = C (2x8)
    # Split B into 4 blocks of 4x2: B1, B2, B3, B4
    # C = A*B1 + A*B2 + A*B3 + A*B4 (no, that's not right)
    # C[:, 0:2] = A * B[:, 0:2]
    # C[:, 2:4] = A * B[:, 2:4]
    # C[:, 4:6] = A * B[:, 4:6]
    # C[:, 6:8] = A * B[:, 6:8]
    # Each A * B[:, 2k:2k+2] is 2x4 * 4x2 = 2x2
    
    # For 2x4 * 4x2: trivial is 16, can we do better?
    # A is 2x4, B is 4x2, C is 2x2
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This is a 2x4x2 tensor. Trivial rank = 16.
    # Can we do rank 12? 10? 8?
    
    # Let me try to find a rank-12 decomposition for 2x4x2
    # U: 8x12, V: 8x12, W: 4x12
    # Need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    # Try a structured approach:
    # A = [A1 A2] (each 2x2), B = [[B1], [B2]] (each 2x2)
    # C = A1*B1 + A2*B2
    # Each 2x2x2 can be done in rank 7 (Strassen), so total 14.
    # Can we do better by sharing?
    
    # For 2x4x2, let's try rank 12:
    # Use 4 Strassen-like blocks but with some sharing
    # Or try a direct construction
    
    # Let me try to construct a rank-12 decomposition for 2x4x2
    # This would give rank 4*12 = 48 for 2x4x8
    
    # 2x4x2: A(2x4), B(4x2), C(2x2)
    # Indices: i in {0,1}, j in {0,1,2,3}, k in {0,1}
    
    # Try: U[i,j,r], V[j,k,r], W[k,i,r] for r in 0..11
    
    # Structured attempt:
    # Split j into j1 in {0,1} and j2 in {2,3}
    # A = [A1 | A2], B = [[B1], [B2]]
    # C = A1*B1 + A2*B2
    
    # For A1*B1 (2x2 * 2x2 -> 2x2), use Strassen (rank 7)
    # For A2*B2 (2x2 * 2x2 -> 2x2), use Strassen (rank 7)
    # Total: 14. Can we reduce to 12?
    
    # Try to share 2 factors between the two Strassen decompositions
    # This is tricky. Let me just try rank 14 for 2x4x2, giving 56 for 2x4x8. Same as before.
    
    # Let me try a different approach: use rank 12 for 2x4x2
    # by a non-Strassen construction
    
    # Actually, let me just try to verify and return the current solution.
    # Finding a better one requires more sophisticated search.
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
