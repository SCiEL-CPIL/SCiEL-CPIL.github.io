import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # The tensor T[i,j,k] = 1 for all valid indices
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # U: (8, R), V: (32, R), W: (16, R)
    
    # Strategy: Try to find a rank-48 decomposition using a structured approach
    # Use the fact that we can group the 64 naive terms and find linear combinations
    
    # First, let's try a direct construction using block Strassen with sharing
    # Split into 2 groups of 2 in the j-dimension: j∈{0,1} and j∈{2,3}
    # For each group, we have a 2×2×8 sub-problem
    # But instead of treating each 2×2×8 independently, we try to share rank-1 terms
    
    # Alternative: use a 4×4×8 view where A is 2×4 = (2×2)×2 blocks
    # Let me try a different decomposition:
    
    # For 2×4×8, think of it as:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    # Group terms: (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
    # Each group is a 2×2×8 multiplication
    
    # For 2×2×8, use Strassen on 2×2×2 blocks: 4 blocks × 7 = 28 per group
    # Two groups: 56 total (same as before)
    
    # To get below 56, we need cross-group sharing.
    # Idea: use a 4×4 Strassen-like scheme for the j-dimension.
    
    # Let me try: construct a rank-48 decomposition by finding a valid set of
    # 48 rank-1 terms. I'll use a greedy approach with verification.
    
    # Actually, let me try a known result: the rank of MM_{2,4,8} might be 48.
    # Let me construct it using a tensor decomposition approach.
    
    # For small problems, we can set up the problem as finding U, V, W such that
    # the tensor equation holds. Let me try a specific construction.
    
    # Construction idea: use 6 rank-1 terms per "super-block" of 4 j-values
    # No, that doesn't work directly.
    
    # Let me try a different approach: use the 56-rank decomposition and try to
    # find a linear combination that reduces the rank by 8.
    
    # Actually, let me just try to verify if rank 48 is achievable by constructing
    # a candidate and checking.
    
    # I'll use a heuristic: start with random integer matrices and solve for one
    # given the other two, then check if the solution is integral.
    
    # Better: use the known structure. Let me try a 2×4×8 decomposition based on
    # the following idea:
    
    # View the problem as 8 independent 2×4×1 multiplications (one for each k).
    # Each 2×4×1 has rank 8 naively, but can be done in rank... well, it's just
    # computing 2 dot products of length 4, so rank 8 (can't really reduce).
    # 8 × 8 = 64. Not helpful.
    
    # Let me try yet another approach: use a 4×4 matrix multiplication sub-structure.
    # A is 2×4, B is 4×8. If I could view this as a 4×4 × 4×8 problem with
    # a constraint on A, I could use rank-19 for 4×4×4 (if it exists)... but
    # the constraint makes it harder.
    
    # I think the safest approach is to stick with 56 but try to find a bug or
    # optimization in the current code. Let me verify the current decomposition
    # is correct and see if I can shave off a few terms.
    
    # Let me verify the current 56-rank decomposition and then try to improve.
    
    # Verify: build U, V, W from the Strassen construction and check the tensor equation.
    
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
    
    # Now try to improve: attempt rank 48
    # Idea: find 8 linear dependencies in the 56-rank decomposition
    # or construct a new rank-48 decomposition from scratch
    
    # Let me try a different algebraic construction for rank 48.
    # Use a 4×4 block structure with a custom scheme.
    
    # For 2×4×8, I'll try the following:
    # Split k into 4 groups of 2: k∈{0,1}, {2,3}, {4,5}, {6,7}
    # For each group, we have a 2×4×2 sub-problem
    # Naive rank for 2×4×2: 16
    # Can we do better? 2×4×2 = (2×2×2) + (2×2×2) with shared structure?
    # Each 2×2×2 uses rank 7, so 2×4×2 might use rank 12 or 14.
    
    # Actually, 2×4×2: C[i,k] = sum_j A[i,j]*B[j,k] for i∈{0,1}, k∈{0,1}, j∈{0,1,2,3}
    # That's 4 output values, each a dot product of length 4.
    # Naive: 4×4 = 16 terms.
    # Can we do 12? Let's see...
    
    # I'll try to construct a rank-12 scheme for 2×4×2.
    # If successful, 4 groups × 12 = 48 total.
    
    # For 2×4×2, the tensor has 16 entries all equal to 1.
    # We need 12 rank-1 terms.
    
    # Let me try: use Strassen on 2×2×2 blocks but share some terms.
    # Split j into {0,1} and {2,3}:
    # C[i,k] = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
    # First part: 2×2×2, rank 7
    # Second part: 2×2×2, rank 7
    # Total: 14. Can we share 2 terms to get 12?
    
    # The two 2×2×2 sub-problems are independent (different j indices), so
    # sharing seems unlikely without changing the structure.
    
    # Let me try a different split: interleave j indices.
    # C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0] + A[i,2]*B[2,0] + A[i,3]*B[3,0]
    # C[i,1] = A[i,0]*B[0,1] + A[i,1]*B[1,1] + A[i,2]*B[2,1] + A[i,3]*B[3,1]
    
    # This is essentially computing two 2×4 dot products.
    # I don't see an obvious way to reduce below 14.
    
    # Let me try rank 13 for 2×4×2. If I can get 13, then 4×13 = 52 < 56.
    
    # Actually, let me just try to verify if the 56-rank solution is correct
    # and return it. The improvement from 56 to 48 or 52 is uncertain.
    
    # Let me try one more idea: use a 2×8×4 view (transposed).
    # C^T[k,i] = sum_j B[k,j]*A[j,i]
    # This is 8×2 × 4×2 = 8×2 result.
    # Same problem essentially.
    
    # I'll return the 56-rank solution which is verified to work.
    # The structural change here is the verification step, which ensures correctness.
    
    if time.time() > deadline:
        return {"U": U56.tolist(), "V": V56.tolist(), "W": W56.tolist()}
    
    # Try to find a lower-rank decomposition by attempting to merge Strassen terms
    # Key idea: the 56-rank decomposition has 8 independent 2×2×2 Strassen blocks.
    # Can we find linear dependencies to reduce the rank?
    
    # Let me try a different block structure: instead of splitting j into 4 and k into 2,
    # try splitting j into 2 and k into 4, and see if we can get a better rank.
    
    def build_alternative_structure():
        """Try an alternative block structure that might give lower rank."""
        # Split j into 2 groups of 2: j∈{0,1} and j∈{2,3}
        # Split k into 4 groups of 2: k∈{0,1}, {2,3}, {4,5}, {6,7}
        # For each (j_group, k_group), we have a 2×2×2 sub-problem
        # That's 2×4 = 8 sub-problems, each rank 7, total 56 (same)
        
        # To get below 56, we need to share terms between sub-problems.
        # This is the key challenge.
        
        # Let me try: use a 4×4×8 view where I apply a 4×4 Strassen-like scheme
        # to the j-dimension, which might allow sharing.
        
        # For now, this is a placeholder. Return None if we can't find improvement.
        return None
    
    alt = build_alternative_structure()
    if alt is not None:
        U, V, W = alt
        valid, err = verify_decomposition(U, V, W)
        if valid:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Return the verified 56-rank solution
    return {"U": U56.tolist(), "V": V56.tolist(), "W": W56.tolist()}
# EVOLVE-BLOCK-END
