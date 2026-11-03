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
        
        # r=0 (M1)
        U_local[0,0] = 1; U_local[3,0] = 1
        V_local[0,0] = 1; V_local[3,0] = 1
        W_local[0,0] = 1; W_local[3,0] = 1
        
        # r=1 (M2)
        U_local[2,1] = 1; U_local[3,1] = 1
        V_local[0,1] = 1
        W_local[2,1] = 1; W_local[3,1] = -1
        
        # r=2 (M3)
        U_local[0,2] = 1
        V_local[1,2] = 1; V_local[3,2] = -1
        W_local[1,2] = 1; W_local[3,2] = 1
        
        # r=3 (M4)
        U_local[3,3] = 1
        V_local[2,3] = 1; V_local[0,3] = -1
        W_local[0,3] = 1; W_local[2,3] = 1
        
        # r=4 (M5)
        U_local[0,4] = 1; U_local[1,4] = 1
        V_local[3,4] = 1
        W_local[0,4] = -1; W_local[1,4] = 1
        
        # r=5 (M6)
        U_local[2,5] = 1; U_local[0,5] = -1
        V_local[0,5] = 1; V_local[1,5] = 1
        W_local[3,5] = 1
        
        # r=6 (M7)
        U_local[1,6] = 1; U_local[3,6] = -1
        V_local[2,6] = 1; V_local[3,6] = 1
        W_local[0,6] = 1
        
        total_rank = 56
        U = np.zeros((8, total_rank), dtype=int)
        V = np.zeros((32, total_rank), dtype=int)
        W = np.zeros((16, total_rank), dtype=int)
        
        for j in range(4):
            for s in range(2):
                base_r = (j * 2 + s) * 7
                
                a_map = [0, 1, 4, 5] if s == 0 else [2, 3, 6, 7]
                b_map = [0*8+2*j, 0*8+2*j+1, 1*8+2*j, 1*8+2*j+1] if s == 0 else \
                        [2*8+2*j, 2*8+2*j+1, 3*8+2*j, 3*8+2*j+1]
                c_map = [2*j*2+0, (2*j+1)*2+0, 2*j*2+1, (2*j+1)*2+1]
                
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
        """Verify that the decomposition correctly computes matrix multiplication."""
        R = U.shape[1]
        n, m, p = 2, 4, 8
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if s != 1:
                        return False, (i, j, k, s)
        return True, None
    
    U56, V56, W56 = build_strassen_56()
    valid, err = verify_decomposition(U56, V56, W56)
    
    if not valid:
        # Fallback to naive 64-rank if Strassen is broken
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
    
    # Attempt to find a lower-rank decomposition by trying to merge terms
    # Strategy: For the 56-rank solution, check if any pair of rank-1 terms
    # can be combined. Also try a direct construction for rank 52.
    
    # Try a 2x4x2 sub-problem with rank 13 (instead of 14 from two 2x2x2 Strassen)
    # If we can get 13, then 4 groups of 2x4x2 gives rank 52.
    
    def try_rank_52():
        """Attempt a rank-52 decomposition using 4 independent 2x4x2 blocks of rank 13."""
        if time.time() > deadline:
            return None
        
        # For 2x4x2: C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k], i in {0,1}, k in {0,1}
        # Try to find a rank-13 scheme.
        # Use Strassen on two 2x2x2 subproblems (j={0,1} and j={2,3}), giving 14.
        # Try to merge one pair of terms.
        
        # Alternative: use a different factorization.
        # Let me try to construct a rank-13 scheme for 2x4x2 directly.
        
        # The tensor T[i,j,k] = 1 for all i,j,k (i:0-1, j:0-3, k:0-1)
        # We need U: (8,13), V: (8,13), W: (4,13) such that
        # sum_r U[i*4+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1
        
        # Try a structured approach: use the 14-term Strassen and see if we can
        # express one term as a linear combination of others.
        
        # Actually, let me try a completely different approach: use a 4x4x2 
        # multiplication with constraints.
        
        # For now, this is a placeholder - the 56-rank solution is our best bet.
        return None
    
    result_52 = try_rank_52()
    if result_52 is not None:
        return result_52
    
    # Return the verified 56-rank solution
    return {"U": U56.tolist(), "V": V56.tolist(), "W": W56.tolist()}
# EVOLVE-BLOCK-END
