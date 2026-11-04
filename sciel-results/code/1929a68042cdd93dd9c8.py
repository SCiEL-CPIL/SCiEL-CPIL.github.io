import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    def build_strassen_56():
        """Build the standard 56-rank Strassen decomposition."""
        U_local = np.zeros((4, 7), dtype=int)
        V_local = np.zeros((4, 7), dtype=int)
        W_local = np.zeros((4, 7), dtype=int)
        
        # Strassen coefficients for 2x2 matrix multiplication
        # U (A factors), V (B factors), W (C factors)
        # Indices 0,1,2,3 correspond to A00, A01, A10, A11
        
        # M1 = (A00 + A11)(B00 + B11)
        U_local[0,0] = 1; U_local[3,0] = 1
        V_local[0,0] = 1; V_local[3,0] = 1
        W_local[0,0] = 1; W_local[3,0] = 1
        
        # M2 = (A10 + A11) B00
        U_local[2,1] = 1; U_local[3,1] = 1
        V_local[0,1] = 1
        W_local[2,1] = 1; W_local[3,1] = -1
        
        # M3 = A00 (B01 - B11)
        U_local[0,2] = 1
        V_local[1,2] = 1; V_local[3,2] = -1
        W_local[1,2] = 1; W_local[3,2] = 1
        
        # M4 = A11 B10
        U_local[3,3] = 1
        V_local[2,3] = 1; V_local[0,3] = -1
        W_local[0,3] = 1; W_local[2,3] = 1
        
        # M5 = (A00 + A01) B11
        U_local[0,4] = 1; U_local[1,4] = 1
        V_local[3,4] = 1
        W_local[0,4] = -1; W_local[1,4] = 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U_local[2,5] = 1; U_local[0,5] = -1
        V_local[0,5] = 1; V_local[1,5] = 1
        W_local[3,5] = 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U_local[1,6] = 1; U_local[3,6] = -1
        V_local[2,6] = 1; V_local[3,6] = 1
        W_local[0,6] = 1
        
        total_rank = 56
        U = np.zeros((8, total_rank), dtype=int)
        V = np.zeros((32, total_rank), dtype=int)
        W = np.zeros((16, total_rank), dtype=int)
        
        # Map local 2x2 indices to global indices
        # A is 2x4. Indices i in {0,1}, j in {0,1,2,3}.
        # We split j into two groups: {0,1} and {2,3}.
        # For each group s in {0,1}, we have a 2x2 block of A.
        # A_block_s has rows i=0,1 and cols j=2s, 2s+1.
        # Local index la in {0,1,2,3} maps to i, j_local.
        # la = i * 2 + j_local.
        # Global A index: i * 4 + (2s + j_local).
        
        # B is 4x8. Indices j in {0,1,2,3}, k in {0..7}.
        # We split j into two groups: {0,1} and {2,3}.
        # For group s=0, B rows are 0,1. For s=1, B rows are 2,3.
        # Local index lb in {0,1,2,3} maps to j_local, k.
        # lb = j_local * 2 + k_local? No, B is 2x8 in the block.
        # In the 2x2 block multiplication, B is 2x2.
        # But here B is 2x8. We are doing 4 independent 2x2 multiplications for the k dimension?
        # No, the current code structure implies:
        # For each j_block (2 blocks of size 2), and for each k_block?
        
        # Let's re-read the current code's mapping logic to ensure consistency.
        # The current code iterates j in range(4) and s in range(2).
        # This seems to be iterating over 8 blocks?
        # Wait, the loop is:
        # for j in range(4):
        #   for s in range(2):
        # This is 8 iterations.
        # base_r = (j * 2 + s) * 7.
        # This assigns 7 ranks to each iteration. 8 * 7 = 56.
        
        # Let's trace one iteration: j=0, s=0.
        # a_map: s=0 -> [0, 1, 4, 5]. 
        # la=0 (A00) -> A[0,0]. la=1 (A01) -> A[0,1]. la=2 (A10) -> A[1,0]. la=3 (A11) -> A[1,1].
        # This corresponds to A columns 0,1.
        
        # b_map: s=0, j=0 -> [0*8+0, 0*8+1, 1*8+0, 1*8+1].
        # lb=0 (B00) -> B[0,0]. lb=1 (B01) -> B[0,1]. lb=2 (B10) -> B[1,0]. lb=3 (B11) -> B[1,1].
        # This corresponds to B rows 0,1 and cols 0,1.
        
        # c_map: j=0 -> [0, 1, 2, 3].
        # lc=0 (C00) -> C[0,0]. lc=1 (C01) -> C[0,1]. lc=2 (C10) -> C[1,0]. lc=3 (C11) -> C[1,1].
        # This corresponds to C rows 0,1 and cols 0,1.
        
        # So this iteration computes the top-left 2x2 block of C using A[0:2, 0:2] and B[0:2, 0:2].
        # This is correct for the first 2x2 block.
        
        # Next iteration: j=0, s=1.
        # a_map: s=1 -> [2, 3, 6, 7].
        # la=0 -> A[0,2]. la=1 -> A[0,3]. la=2 -> A[1,2]. la=3 -> A[1,3].
        # A columns 2,3.
        
        # b_map: s=1, j=0 -> [2*8+0, 2*8+1, 3*8+0, 3*8+1].
        # B rows 2,3 and cols 0,1.
        
        # c_map: j=0 -> [0, 1, 2, 3].
        # C rows 0,1 and cols 0,1.
        
        # So this iteration adds to the same C block C[0:2, 0:2] using A[0:2, 2:4] and B[2:4, 0:2].
        # This is correct. C[0:2, 0:2] = A[0:2, 0:2]B[0:2, 0:2] + A[0:2, 2:4]B[2:4, 0:2].
        
        # Next: j=1, s=0.
        # a_map: s=0 -> [0, 1, 4, 5]. A cols 0,1.
        # b_map: s=0, j=1 -> [0*8+2, 0*8+3, 1*8+2, 1*8+3]. B rows 0,1, cols 2,3.
        # c_map: j=1 -> [2, 3, 4, 5]. C cols 2,3.
        
        # This computes C[0:2, 2:4] += A[0:2, 0:2]B[0:2, 2:4].
        
        # It seems the logic is sound. It decomposes the 2x4 x 4x8 multiplication into 8 independent 2x2 x 2x2 multiplications,
        # each handled by Strassen (rank 7), and sums them up.
        
        for j in range(4):
            for s in range(2):
                base_r = (j * 2 + s) * 7
                
                # A mapping
                if s == 0:
                    a_map = [0, 1, 4, 5] # i=0,j=0; i=0,j=1; i=1,j=0; i=1,j=1
                else:
                    a_map = [2, 3, 6, 7] # i=0,j=2; i=0,j=3; i=1,j=2; i=1,j=3
                    
                # B mapping
                if s == 0:
                    # Rows 0,1 of B
                    b_map = [0*8 + 2*j, 0*8 + 2*j+1, 1*8 + 2*j, 1*8 + 2*j+1]
                else:
                    # Rows 2,3 of B
                    b_map = [2*8 + 2*j, 2*8 + 2*j+1, 3*8 + 2*j, 3*8 + 2*j+1]
                    
                # C mapping
                # C is 2x8. j determines the column block (0,1,2,3 -> cols 0-1, 2-3, 4-5, 6-7)
                c_map = [2*j*2+0, (2*j+1)*2+0, 2*j*2+1, (2*j+1)*2+1]
                # c_map[0] = 2j, c_map[1] = 2j+1, c_map[2] = 2j+2, c_map[3] = 2j+3?
                # Wait.
                # If j=0, c_map = [0, 1, 2, 3]. C[0,0], C[0,1], C[1,0], C[1,1].
                # If j=1, c_map = [2, 3, 4, 5]. C[0,2], C[0,3], C[1,2], C[1,3].
                # This is correct.
                
                for l in range(7):
                    r = base_r + l
                    for la in range(4):
                        val = U_local[la, l]
                        if val != 0:
                            U[a_map[la], r] += val
                    for lb in range(4):
                        val = V_local[lb, l]
                        if val != 0:
                            V[b_map[lb], r] += val
                    for lc in range(4):
                        val = W_local[lc, l]
                        if val != 0:
                            W[c_map[lc], r] += val
        
        return U, V, W

    def verify_decomposition(U, V, W):
        """Verify tensor equation: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k."""
        R = U.shape[1]
        # Use NumPy for faster verification
        # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # We can reshape U, V, W and use einsum or loops.
        # Given small sizes, a simple loop is fine, but let's try to be efficient.
        
        # Create index arrays
        i_idx = np.repeat(np.arange(n), m*p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n*m)
        
        u_flat = U[i_idx*m + j_idx, :]
        v_flat = V[j_idx*p + k_idx, :]
        w_flat = W[k_idx*n + i_idx, :]
        
        # Sum over rank dimension
        # u_flat is (nmp, R), v_flat is (nmp, R), w_flat is (nmp, R)
        # We want sum_r u[i,r] v[i,r] w[i,r] for each i in 0..nmp-1
        sums = np.sum(u_flat * v_flat * w_flat, axis=1)
        
        if not np.all(sums == 1):
            return False
        return True

    U56, V56, W56 = build_strassen_56()
    valid = verify_decomposition(U56, V56, W56)
    
    if not valid:
        # Fallback to naive rank 64
        U = np.zeros((8, 64), dtype=int)
        V = np.zeros((32, 64), dtype=int)
        W = np.zeros((16, 64), dtype=int)
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    r = i*32 + j*8 + k
                    U[i*4+j, r] = 1
                    V[j*8+k, r] = 1
                    W[k*2+i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U56.tolist(), "V": V56.tolist(), "W": W56.tolist()}
# EVOLVE-BLOCK-END
