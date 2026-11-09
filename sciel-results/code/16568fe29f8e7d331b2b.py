import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 4, 5
    
    # First, try to verify if rank 35 is achievable via nuclear norm relaxation
    # The matrix multiplication tensor T[i,j,k] = 1 if (i,j,k) corresponds to C[i,k] = sum_j A[i,j]B[j,k]
    # T has shape (n*m, m*p, p*n) = (8, 20, 10)
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    
    # Build the tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Try nuclear norm relaxation for rank 35
    # Unfold along mode 1: shape (8, 20*10)
    # This is a large problem, let's try a simpler approach
    
    # Alternative: try to construct rank 35 directly
    # Strategy: 4 Strassen blocks but share one rank term
    # Or: use a different block decomposition
    
    # Let's try: 3 Strassen blocks for 2x2x4 part (rank 3*7=21 for 2x2x4? No...)
    # Actually 2x2x4: rank is 11, not 3*7=21
    # 2x4x4: can be seen as 2x(2x2)x(2x2) = two 2x2x2 blocks? No, that's not right either.
    # 2x4x4: A is 2x4, B is 4x4. C is 2x4.
    # Split B into two 4x2 blocks: C[:,0:2] = A*B[:,0:2], C[:,2:4] = A*B[:,2:4]
    # Each 2x4x2: rank? 2x4x2 schoolbook = 16, Strassen-like?
    # 2x4x2: A(2x4)*B(4x2) -> C(2x2). This is 2x(4x2) = 8 schoolbook terms.
    # Can we do better? 2x4x2 tensor rank... 
    
    # Let me just try the nuclear norm approach quickly
    try:
        from scipy.linalg import svd
        # Mode-1 unfolding: (8, 200)
        T_unfold1 = T.reshape(n*m, m*p*p*n)
        # Nuclear norm is sum of singular values
        # For rank-r approximation, we want the (r+1)th singular value to be 0
        # Let's check singular values for different ranks
        U_s, s_vals, Vt = np.linalg.svd(T_unfold1, full_matrices=False)
        # The number of non-zero singular values gives a lower bound on rank
        # But this is just a lower bound, not achievable rank
        nz_count = np.sum(s_vals > 1e-10)
        # This lower bound is likely 8 (since T_unfold1 is 8x200)
        # Not very helpful directly
        
        # Let's try a different approach: structured rank-35 search
        # Use 4 Strassen blocks but try to merge one term
        # The 4 Strassen blocks operate on independent sub-tensors, so they can't be merged
        # unless there's overlap in the factor matrices
        
        # Alternative: use a 2x4x4 decomposition of rank 27 instead of 28
        # Is rank(2x4x4) <= 27? Known: rank(2xn x m) ... 
        # Actually, let's just verify the current rank-36 is correct and try small variations
        
        # Try: can we replace one Strassen block (rank 7) with a rank-6 decomposition?
        # No, Strassen is optimal for 2x2x2 (rank 7 is proven minimal)
        
        # Try: can we do 2x4x5 with rank 35 by sharing between blocks?
        # The blocks are: (i in 0-1, j in 0-1, k in 0-1), (i in 0-1, j in 2-3, k in 0-1),
        # (i in 0-1, j in 0-1, k in 2-3), (i in 0-1, j in 2-3, k in 2-3), and (i in 0-1, j in 0-3, k in 4)
        # These are all disjoint sub-tensors, so no sharing possible in this decomposition.
        
        # What if we use a different block structure?
        # E.g., split differently: use 2x2x5 for the first 2 rows of A?
        # 2x2x5: A(2x2)*B(2x5) -> C(2x5). Rank of 2x2x5?
        # Schoolbook: 2*2*5=20. Strassen for 2x2x2 part: 7 for k=0-1, then 2x2x3 for k=2-4?
        # 2x2x3: schoolbook 12, can we do better?
        
        # This is getting complex. Let me just keep the rank-36 solution but verify it's correct.
    except Exception:
        pass
    
    # Return the known rank-36 decomposition
    strassen_U = [
        [1,0,0,1], [0,0,1,1], [1,0,0,0], [0,0,0,1],
        [1,1,0,0], [-1,0,1,0], [0,1,0,-1]
    ]
    strassen_V = [
        [1,0,0,1], [1,0,0,0], [0,1,0,-1], [-1,0,1,0],
        [0,0,0,1], [1,1,0,0], [0,0,1,1]
    ]
    strassen_W = [
        [1,0,0,1], [0,1,0,-1], [0,0,1,1], [1,1,0,0],
        [-1,0,1,0], [0,0,0,1], [1,0,0,0]
    ]
    
    rank = 4 * 7 + 8
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j0 in [0, 2]:
        for k0 in [0, 2]:
            for r in range(7):
                for li in range(2):
                    for lj in range(2):
                        gi = li
                        gj = j0 + lj
                        U[gi * m + gj][col] = strassen_U[r][li * 2 + lj]
                for lj in range(2):
                    for lk in range(2):
                        gj = j0 + lj
                        gk = k0 + lk
                        V[gj * p + gk][col] = strassen_V[r][lj * 2 + lk]
                for lk in range(2):
                    for li in range(2):
                        gk = k0 + lk
                        gi = li
                        W[gk * n + gi][col] = strassen_W[r][lk * 2 + li]
                col += 1
    
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
