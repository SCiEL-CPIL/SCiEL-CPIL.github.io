import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor T with shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T[a, b, d] = 1.0

    start_time = time.time()
    deadline = start_time + 130  # Safety margin

    best_rank = nm * mp * pn  # schoolbook
    best_U = None
    best_V = None
    best_W = None

    # Strategy:
    # 1. Try to find a rank-5 decomposition for 2x3x1 using discrete search
    # 2. If found, tile it 10 times to get rank 50
    # 3. If not found, try other constructions
    # 4. Fallback to schoolbook rank 60

    # First, let's try the known Strassen-based approach
    # For 2x2x2, rank is 7 (Strassen). For 2x3x1, rank is 5.
    # Let's try to find the rank-5 decomposition for 2x3x1.

    T_231 = np.zeros((6, 3, 2))
    for i in range(2):
        for j in range(3):
            T_231[i * 3 + j, j, i] = 1.0

    found_231 = False
    U_231 = None
    V_231 = None
    W_231 = None

    # Search for rank-5 decomposition of 2x3x1
    # U: 6x5, V: 3x5, W: 2x5, entries in {-1, 0, 1}
    # We can use a smarter search: fix some structure.
    
    # Known rank-5 algorithm for 2x3x1 (from literature):
    # This is essentially the "fast rectangular" algorithm.
    # Let me try to construct it directly.
    
    # Actually, let's try a direct construction.
    # The 2x3x1 tensor can be decomposed as:
    # We need 5 terms of the form u_c * v_c * w_c where:
    # u_c is a 6-vector, v_c is a 3-vector, w_c is a 2-vector
    # and sum_c (u_c * v_c * w_c) = T_231
    
    # Let's try a known construction:
    # Use the fact that 2x3 = 2x2 + 2x1
    # 2x2x1 has rank 3 (schoolbook: 4, but can we do 3?)
    # Actually 2x2x1 has rank 4 (it's just 4 independent multiplications)
    # Wait, no. 2x2x1: A(2x2) * B(2x1) = C(2x1)
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    # This is 4 scalar multiplications. Can we do 3?
    # (A[0,0]+A[1,1])*(B[0]+B[1]) = A[0,0]*B[0] + A[0,0]*B[1] + A[1,1]*B[0] + A[1,1]*B[1]
    # Not directly helpful for 2x2x1.
    
    # Let me just try random search for rank 5 on 2x3x1 with {-1,0,1} entries.
    # The space is large but let's try.
    
    max_attempts = 20000
    np.random.seed(42)
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 20:
            break
            
        U_rand = np.random.randint(-1, 2, (6, 5))
        V_rand = np.random.randint(-1, 2, (3, 5))
        W_rand = np.random.randint(-1, 2, (2, 5))
        
        # Compute tensor product and check
        # T_est[a,b,d] = sum_c U_rand[a,c] * V_rand[b,c] * W_rand[d,c]
        T_est = np.einsum('ac,bc,dc->abd', U_rand, V_rand, W_rand)
        
        if np.allclose(T_est, T_231):
            found_231 = True
            U_231 = U_rand
            V_231 = V_rand
            W_231 = W_rand
            break

    if found_231:
        # Tile the 2x3x1 solution 10 times for 2x3x10
        rank = 50
        U_new = np.zeros((6, rank))
        V_new = np.zeros((30, rank))
        W_new = np.zeros((20, rank))
        
        for k in range(10):
            for c in range(5):
                idx = k * 5 + c
                U_new[:, idx] = U_231[:, c]
                for j in range(3):
                    V_new[j * 10 + k, idx] = V_231[j, c]
                for i in range(2):
                    W_new[k * 2 + i, idx] = W_231[i, c]
        
        best_rank = rank
        best_U = U_new
        best_V = V_new
        best_W = W_new
    else:
        # Fallback: try to use a block-wise approach
        # Use Strassen for 2x2 blocks where possible
        # 2x3x10: split columns into 2x2 and 2x1 blocks
        # 5 blocks of 2x2x10? No, that doesn't make sense.
        
        # Let's try: treat as 2x3x2 * 5 blocks (each 2x3x2)
        # 2x3x2: A(2x3) * B(3x2) = C(2x2)
        # Schoolbook: 12
        # Can we do better?
        # Split B into two 3x1 columns.
        # 2x3x1 for each column. If rank(2x3x1)=5, then 2*5=10 per 2x3x2 block.
        # 5 blocks * 10 = 50. Same as tiling.
        
        # Try a different approach: use the rank-7 Strassen for 2x2x2
        # 2x3x10 = ?
        # Not directly applicable.
        
        # Let's try to find a rank-6 decomposition for 2x3x1 with {-1,0,1}
        # or use a continuous optimization approach.
        
        # For now, fall back to schoolbook
        rank = nm * mp * pn
        best_U = np.zeros((nm, rank))
        best_V = np.zeros((mp, rank))
        best_W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i * m + j, col] = 1.0
                    best_V[j * p + k, col] = 1.0
                    best_W[k * n + i, col] = 1.0
                    col += 1
        best_rank = rank

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
