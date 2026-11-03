import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 2, 4, 8
    
    # Strategy: Construct a rank-48 decomposition using a global algebraic scheme.
    # For each k-pair (k, k+1), we need to compute:
    #   C[i, k]   = sum_j A[i,j] * B[j, k]
    #   C[i, k+1] = sum_j A[i,j] * B[j, k+1]
    # for i=0,1 and j=0,1,2,3.
    #
    # Instead of two independent 2x2x2 Strassen blocks (rank 14 per k-pair),
    # we use a unified 12-term scheme per k-pair that shares structure.
    #
    # The key idea: for a fixed k-pair, view the problem as computing
    #   C[0,k]   = a00*b0 + a01*b1 + a02*b2 + a03*b3
    #   C[1,k]   = a10*b0 + a11*b1 + a12*b2 + a13*b3
    #   C[0,k+1] = a00*c0 + a01*c1 + a02*c2 + a03*c3
    #   C[1,k+1] = a10*c0 + a11*c1 + a12*c2 + a13*c3
    # where b_j = B[j,k], c_j = B[j,k+1].
    #
    # Use 12 rank-1 terms with a structured pattern that exploits n=2.
    
    # For each k-pair, define 12 terms (r=0..11) with the following structure:
    # We group j into {0,1} and {2,3}, and use a modified Strassen that
    # shares 2 terms between the two j-groups.
    
    # Actually, let me try a different approach: use the fact that we can
    # write 12 = 7 + 5, where 7 is Strassen for j∈{0,1} and 5 is a reduced
    # scheme for j∈{2,3} that reuses 2 of the Strassen terms.
    
    # Better: let me try to construct a valid rank-48 solution by solving
    # the tensor equation directly with a structured ansatz.
    
    # Ansatz: for each k-pair (kk, kk+1), use 12 terms with the following
    # U, V, W patterns (indices within the k-pair):
    
    # Let me try a specific construction. For k-pair at position kk (kk=0,2,4,6):
    # Term r within the k-pair (0..11):
    
    # I'll use a pattern based on the "laser" method or direct construction.
    # Let me try: 6 terms for the "sum" part and 6 for the "difference" part.
    
    # Actually, the cleanest approach: try rank 48 by constructing U, V, W
    # with a specific algebraic structure and verifying.
    
    # Let me use a different structural idea: treat the 4x8 B matrix as
    # 4 rows, each of length 8. The multiplication C = A*B where A is 2x4.
    # C[i, k] = sum_{j=0}^{3} A[i,j] * B[j, k]
    #
    # View this as: for each j, the contribution is A[i,j] * B[j,:] (a row of B
    # scaled by A[i,j]). The tensor rank is the minimum number of rank-1
    # decompositions of the 2x4x8 all-ones tensor.
    #
    # Key structural idea: use a 4x4 matrix multiplication sub-structure.
    # Embed the 2x4 * 4x8 problem into a 4x4 * 4x4 problem by padding.
    # A_padded is 4x4 (top-left 2x4 block is A, rest zeros... no, that doesn't work).
    #
    # Alternative: use the "sum of products" structure more cleverly.
    #
    # Let me try a direct construction for rank 48:
    # Split into 4 k-pairs. For each k-pair, use 12 terms.
    # The 12 terms per k-pair use the following scheme:
    #   - 7 terms: Strassen for j∈{0,1} (standard 2x2x2)
    #   - 5 terms: a reduced scheme for j∈{2,3} that reuses 2 Strassen terms
    
    # For the reduced 5-term scheme for j∈{2,3}, k-pair (kk, kk+1):
    # We need: C[i, kk] += A[i,2]*B[2,kk] + A[i,3]*B[3,kk]
    #          C[i, kk+1] += A[i,2]*B[2,kk+1] + A[i,3]*B[3,kk+1]
    # This is a 2x2x2 problem (i∈{0,1}, j∈{2,3}, k∈{kk,kk+1}).
    # Naive rank: 8. Strassen rank: 7. Can we do 5?
    # Lower bound for 2x2x2 is 7, so no. Minimum is 7.
    # So 7+7=14 per k-pair, total 56. Can't get 12 this way.
    
    # Different approach: don't split j into pairs. Use all 4 j's together.
    # For a k-pair, the problem is 2x4x2. What's the rank?
    # Naive: 16. Can we do 12?
    # 
    # Let me try to construct a rank-12 scheme for 2x4x2.
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k] for i∈{0,1}, k∈{0,1}
    # 
    # 12 terms. Let me try:
    # Use 4 "base" terms for the diagonal: A[i,j]*B[j,k] for i=j pattern... no.
    # 
    # Try: 6 terms for "sums" and 6 for "differences" in j.
    # 
    # Let me use the following 12-term scheme for 2x4x2:
    # Terms 0-3: handle j=0,1 for both k values with a 2x2x2 Strassen-like
    # Terms 4-7: handle j=2,3 for both k values
    # Terms 8-11: cross terms that enable sharing
    # 
    # This is getting complex. Let me just try to verify if rank 56 is correct
    # and return it, but with a cleaner implementation.
    
    # Actually, let me try a completely different structural approach:
    # Use the "border rank" idea. The border rank of MM_{2,4,8} might be lower
    # than the tensor rank, but we need exact decomposition.
    
    # Let me try rank 48 with a specific construction:
    # For each k (8 values), use 6 terms. Total: 48.
    # For fixed k, C[i,k] = sum_j A[i,j]*B[j,k]. This is 2 dot products of length 4.
    # 6 terms for 2x4x1? Naive is 8. Can we do 6?
    # Lower bound: max(2*1, 4*1, 2*4) = max(2,4,8) = 8. So no, minimum is 8.
    # 8*8 = 64. Can't do better per-k.
    
    # OK so the per-k approach gives 64. The Strassen approach gives 56.
    # Let me stick with 56 but make sure it's correct.
    
    # Build the 56-rank Strassen decomposition
    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)
    
    # Standard Strassen for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    U_local[0,0] = 1; U_local[3,0] = 1
    V_local[0,0] = 1; V_local[3,0] = 1
    W_local[0,0] = 1; W_local[3,0] = 1
    
    # M2 = (A10+A11)B00
    U_local[2,1] = 1; U_local[3,1] = 1
    V_local[0,1] = 1
    W_local[2,1] = 1; W_local[3,1] = -1
    
    # M3 = A00(B10-B11)
    U_local[0,2] = 1
    V_local[1,2] = 1; V_local[3,2] = -1
    W_local[1,2] = 1; W_local[3,2] = 1
    
    # M4 = A11(B01-B00)
    U_local[3,3] = 1
    V_local[2,3] = 1; V_local[0,3] = -1
    W_local[0,3] = 1; W_local[2,3] = 1
    
    # M5 = (A00+A01)B11
    U_local[0,4] = 1; U_local[1,4] = 1
    V_local[3,4] = 1
    W_local[0,4] = -1; W_local[1,4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U_local[2,5] = 1; U_local[0,5] = -1
    V_local[0,5] = 1; V_local[1,5] = 1
    W_local[3,5] = 1
    
    # M7 = (A01-A11)(B10+B11)
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
            
            # a_map: maps local 2x2 index (la) to global U index
            # s=0: k columns 2j, 2j+1 -> A columns 0,1
            # s=1: k columns 2j, 2j+1 -> A columns 2,3
            if s == 0:
                a_map = [0, 1, 4, 5]  # A[0,0], A[0,1], A[1,0], A[1,1]
            else:
                a_map = [2, 3, 6, 7]  # A[0,2], A[0,3], A[1,2], A[1,3]
            
            # b_map: maps local 2x2 index (lb) to global V index
            # lb=0: B row 0, lb=1: B row 1, lb=2: B row 2, lb=3: B row 3
            # For j-group j, s=0: rows 0,1; s=1: rows 2,3
            if s == 0:
                b_map = [0*8 + 2*j, 0*8 + 2*j+1, 1*8 + 2*j, 1*8 + 2*j+1]
            else:
                b_map = [2*8 + 2*j, 2*8 + 2*j+1, 3*8 + 2*j, 3*8 + 2*j+1]
            
            # c_map: maps local 2x2 index (lc) to global W index
            # lc=0: C[0, 2j], lc=1: C[0, 2j+1], lc=2: C[1, 2j], lc=3: C[1, 2j+1]
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
    
    # Verify
    def verify(U, V, W):
        R = U.shape[1]
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    s = 0
                    for r in range(R):
                        s += U[i*4+j, r] * V[j*8+k, r] * W[k*2+i, r]
                    if s != 1:
                        return False
        return True
    
    if not verify(U, V, W):
        # Fallback to naive 64
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
    
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to improve: attempt to find a rank-48 decomposition
    # by searching for 8 linear dependencies in the 56-rank system.
    # This is a structural change: instead of a fixed block construction,
    # use linear algebra to find a lower-rank equivalent decomposition.
    
    # The 56-rank decomposition gives us 56 rank-1 terms.
    # If there are 8 linear dependencies among the "tensor vectors"
    # t_r = U[:,r] ⊗ V[:,r] ⊗ W[:,r], we can reduce to rank 48.
    
    # To check for dependencies, we can flatten the tensor and check
    # if the 56 vectors span a space of dimension < 56.
    
    # Flatten each rank-1 term into a vector of length 8*32*16 = 4096
    # and check the rank of the 56x4096 matrix.
    
    tensor_size = 8 * 32 * 16  # 4096
    T = np.zeros((total_rank, tensor_size), dtype=float)
    
    for r in range(total_rank):
        # t_r[i*32+j*16+k] = U[i*4+j, r] * V[j*8+k, r] * W[k*2+i, r]
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    idx = i * 32 * 16 + j * 16 + k
                    T[r, idx] = U[i*4+j, r] * V[j*8+k, r] * W[k*2+i, r]
    
    # Compute rank of T
    try:
        rank_T = np.linalg.matrix_rank(T, tol=1e-10)
    except:
        rank_T = total_rank
    
    if rank_T < total_rank:
        # There are dependencies! Try to find a lower-rank decomposition.
        # Use SVD to find the null space and reduce.
        # This is complex; for now, just note it and return 56.
        pass
    
    # Return the verified 56-rank solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
