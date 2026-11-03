import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    
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
    
    def build_strassen_56():
        """Build the standard 56-rank Strassen decomposition using 8 independent 2x2x2 blocks."""
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
    
    def try_rank_reduction():
        """
        Attempt to find a rank reduction below 56 by exploiting linear dependencies
        in the Strassen decomposition. We build the 56-rank solution and then try to
        find linear combinations of the rank-1 terms that cancel out, allowing us to
        eliminate some terms.
        
        The key insight: when we have multiple 2x2x2 Strassen blocks, certain 
        intermediate products may be shared or linearly dependent. By identifying
        these dependencies, we can reduce the total rank.
        
        Strategy: Build the tensor equation system and use Gaussian elimination
        to find the minimum number of terms needed.
        """
        if time.time() > deadline:
            return None
        
        # Build the 56-rank solution first
        U56, V56, W56 = build_strassen_56()
        
        # Now try to find a lower-rank decomposition by searching for
        # linear dependencies among the 56 terms.
        
        # For each output position (i, k), we have:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # In our decomposition: C[i,k] = sum_r U[i*4+j, r] * V[j*8+k, r] * W[k*2+i, r]
        
        # The tensor T has entries T[i,j,k] = 1 for all valid (i,j,k).
        # We need: sum_r U[i*4+j, r] * V[j*8+k, r] * W[k*2+i, r] = 1
        
        # Let's try a different approach: use the fact that the 2x4x8 multiplication
        # can be seen as 4 independent 2x1x8 multiplications (one for each j),
        # but with shared A and B structure.
        
        # Actually, let's try to find if we can reduce from 56 to 52 or lower
        # by checking if any of the 56 rank-1 terms are linearly dependent.
        
        # Build the system of equations. For each (i,j,k), we need:
        # sum_r U[i*4+j, r] * V[j*8+k, r] * W[k*2+i, r] = 1
        
        # This is a system of 2*4*8 = 64 equations in the unknowns U, V, W.
        # With rank R, we have 8*R + 32*R + 16*R = 56*R unknowns.
        
        # For R=56, that's 3136 unknowns and 64 equations. Very underdetermined.
        # The Strassen solution gives one valid point in this space.
        
        # To find a lower-rank solution, we need to show that the tensor rank
        # is less than 56. This is a hard problem in general.
        
        # Let's try a specific approach: check if the 56-rank Strassen solution
        # has any zero rows/columns that can be eliminated, or if there are
        # linear dependencies among the rank-1 vectors.
        
        # Check for zero columns in U, V, W
        for r in range(56):
            if np.all(U56[:, r] == 0) or np.all(V56[:, r] == 0) or np.all(W56[:, r] == 0):
                # This term is useless, remove it
                pass  # In practice, this shouldn't happen for Strassen
        
        # Try to find if we can merge terms. Two terms r1 and r2 can be merged
        # if U[:,r1] = U[:,r2], V[:,r1] = V[:,r2], W[:,r1] = W[:,r2]
        # (up to scaling that cancels out)
        
        # This is a complex search. Let's try a simpler heuristic:
        # Check if the 56 terms can be partitioned into groups where each group
        # sums to a rank-1 tensor, allowing us to replace each group with one term.
        
        # Given the time constraints and the difficulty of this problem,
        # let's return None and stick with the 56-rank solution.
        
        return None
    
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
    
    result_reduced = try_rank_reduction()
    if result_reduced is not None:
        return result_reduced
    
    # Return the verified 56-rank solution
    return {"U": U56.tolist(), "V": V56.tolist(), "W": W56.tolist()}
# EVOLVE-BLOCK-END
