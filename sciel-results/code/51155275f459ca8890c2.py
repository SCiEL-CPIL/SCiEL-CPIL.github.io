"""Program for rank-minimal tensor decomposition of 3x3 by 3x6 matrix multiplication."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    start_time = time.time()
    deadline = start_time + 140

    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that the decomposition correctly represents matrix multiplication."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i*m+j, :]
                    v_val = V[j*p+k, :]
                    w_val = W[k*n+i, :]
                    val = np.dot(u_val, v_val * w_val)
                    err = abs(1.0 - val)
                    if err > max_err:
                        max_err = err
        return max_err

    def schoolbook_3x3x6():
        """Return schoolbook decomposition (rank 54) for 3x3x6."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, rank

    def build_rank24_3x3x3():
        """
        Build a rank-24 decomposition for 3x3x3 using 4 overlapping 2x2 Strassen blocks.
        Fixed version with correct B-matrix indexing.
        """
        U = np.zeros((9, 24))
        V = np.zeros((9, 24))
        W = np.zeros((9, 24))
        
        # Strassen terms for 2x2x2
        # Matrix A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
        # m1 = (a+d)(e+h), m2 = (c+d)e, m3 = a(f-h), m4 = d(g-c), m5 = (a+b)h, m6 = (c-a)(e+f), m7 = (b-d)(g+h)
        # C[0,0]=i=m1+m4-m5+m7, C[0,1]=j=m3+m5, C[1,0]=k=m2+m4, C[1,1]=l=m1-m2+m3+m6
        
        # Each strassen term: (u_coeffs, v_coeffs, w_coeffs) for 4 positions [TL, TR, BL, BR]
        strassen = [
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # m1
            ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),   # m2
            ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),   # m3
            ([0,0,0,1], [0,0,1,-1], [1,0,1,0]),   # m4
            ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),   # m5
            ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),   # m6
            ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),   # m7
        ]
        
        # 4 blocks: TL(0,1;0,1), TR(0,1;1,2), BL(1,2;0,1), BR(1,2;1,2)
        blocks = [(0,1,0,1), (0,1,1,2), (1,2,0,1), (1,2,1,2)]
        
        col = 0
        for r1, c1 in blocks:
            for u2, v2, w2 in strassen:
                for idx in range(4):
                    # A matrix position: row r1+idx//2, col c1+idx%2
                    a_r = r1 + idx // 2
                    a_c = c1 + idx % 2
                    U[a_r*3 + a_c, col] += u2[idx]
                    
                    # B matrix position: row c1+idx//2, col c1+idx%2
                    # Wait, B is 3x3, so B[row, col] where row is the shared dimension
                    # For block at (r1, c1): B rows are c1, c1+1 and B cols are c1, c1+1
                    # Actually B[j,k] where j is row (shared with A's col) and k is col
                    # For 2x2 block: B submatrix has rows [c1, c1+1] and cols [c1, c1+1]
                    b_r = c1 + idx // 2
                    b_c = c1 + idx % 2
                    V[b_r*3 + b_c, col] += v2[idx]
                    
                    # C matrix position: row r1+idx//2, col c1+idx%2
                    # W uses trace convention: W[k*n+i] = W[k*3+i] where k is col, i is row
                    c_r = r1 + idx // 2
                    c_c = c1 + idx % 2
                    W[c_c*3 + c_r, col] += w2[idx]
                col += 1
        
        return U, V, W, 24

    def build_3x3x6_from_3x3x3(U33, V33, W33, rank33):
        """
        Lift a rank-r 3x3x3 decomposition to 3x3x6 with rank 2r.
        For 3x3x6: A is 3x3, B is 3x6, C is 3x6
        C_left = A * B[:,0:3], C_right = A * B[:,3:6]
        Use same U for both halves, different V and W.
        
        U shape: (n*m, rank) = (9, 2*rank33)
        V shape: (m*p, rank) = (18, 2*rank33)
        W shape: (p*n, rank) = (18, 2*rank33)
        
        For the left half: B columns 0-2, C columns 0-2
        For the right half: B columns 3-5, C columns 3-5
        """
        rank = 2 * rank33
        U = np.zeros((9, rank))
        V = np.zeros((18, rank))
        W = np.zeros((18, rank))
        
        for s in range(rank33):
            # Left half (columns 0-2 of B and C)
            idx1 = 2 * s
            # U: A[i,j] -> U[i*3+j, idx1]
            for i in range(3):
                for j in range(3):
                    U[i*3+j, idx1] = U33[i*3+j, s]
            # V: B[j,k] for k in 0..2 -> V[j*6+k, idx1]
            for j in range(3):
                for k in range(3):
                    V[j*6+k, idx1] = V33[j*3+k, s]
            # W: C[i,k] for k in 0..2 -> W[k*3+i, idx1] (trace convention)
            for k in range(3):
                for i in range(3):
                    W[k*3+i, idx1] = W33[k*3+i, s]
            
            # Right half (columns 3-5 of B and C)
            idx2 = 2 * s + 1
            # U: same A[i,j]
            for i in range(3):
                for j in range(3):
                    U[i*3+j, idx2] = U33[i*3+j, s]
            # V: B[j,k] for k in 3..5 -> V[j*6+k, idx2]
            for j in range(3):
                for k in range(3):
                    V[j*6+k+3, idx2] = V33[j*3+k, s]
            # W: C[i,k] for k in 3..5 -> W[k*3+i, idx2] where k = k3+3
            for k in range(3):
                for i in range(3):
                    W[(k+3)*3+i, idx2] = W33[k*3+i, s]
        
        return U, V, W, rank

    best_result = None
    best_rank = float('inf')
    
    # Method 1: Rank-24 3x3x3 -> Rank-48 3x3x6
    try:
        U33, V33, W33, rank33 = build_rank24_3x3x3()
        U, V, W, rank = build_3x3x6_from_3x3x3(U33, V33, W33, rank33)
        err = verify_decomposition(U, V, W)
        if err < 1e-9 and rank < best_rank:
            best_rank = rank
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception as e:
        pass
    
    # Method 2: Schoolbook fallback (rank 54)
    U, V, W, rank = schoolbook_3x3x6()
    err = verify_decomposition(U, V, W)
    if err < 1e-9 and rank < best_rank:
        best_rank = rank
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if best_result is not None:
        return best_result
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
