"""Program for rank-minimal tensor decomposition of 3x3 by 3x6 matrix multiplication."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    start_time = time.time()
    deadline = start_time + 140

    def strassen_2x2x2_terms():
        """
        Return the 7 Strassen terms for 2x2x2 matrix multiplication.
        Each term is (u, v, w) where u, v, w are 4-dimensional vectors.
        A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
        C = A*B: i=a*e+b*g, j=a*f+b*h, k=c*e+d*g, l=c*f+d*h
        
        Strassen:
        m1 = (a+d)(e+h)
        m2 = (c+d)e
        m3 = a(f-h)
        m4 = d(g-c)
        m5 = (a+b)h
        m6 = (c-a)(e+f)
        m7 = (b-d)(g+h)
        
        i = m1 + m4 - m5 + m7
        j = m3 + m5
        k = m2 + m4
        l = m1 - m2 + m3 + m6
        
        Tensor form: T = sum_s u_s ⊗ v_s ⊗ w_s
        where sum_s u_s[i,j] * v_s[j,k] * w_s[i,k] = T[i,j,k]
        T[i,j,k] = 1 iff this triple contributes to C[i,k] via A[i,j]*B[j,k]
        """
        terms = [
            # m1 = (a+d)(e+h): u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
            (np.array([1,0,0,1]), np.array([1,0,0,1]), np.array([1,0,0,1])),
            # m2 = (c+d)e: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,-1]
            (np.array([0,0,1,1]), np.array([1,0,0,0]), np.array([0,0,1,-1])),
            # m3 = a(f-h): u=[1,0,0,0], v=[0,1,0,-1], w=[0,1,0,1]
            (np.array([1,0,0,0]), np.array([0,1,0,-1]), np.array([0,1,0,1])),
            # m4 = d(g-c): u=[0,0,0,1], v=[0,0,1,-1], w=[1,0,1,0]
            (np.array([0,0,0,1]), np.array([0,0,1,-1]), np.array([1,0,1,0])),
            # m5 = (a+b)h: u=[1,1,0,0], v=[0,0,0,1], w=[-1,1,0,0]
            (np.array([1,1,0,0]), np.array([0,0,0,1]), np.array([-1,1,0,0])),
            # m6 = (c-a)(e+f): u=[-1,0,1,0], v=[1,1,0,0], w=[0,0,0,1]
            (np.array([-1,0,1,0]), np.array([1,1,0,0]), np.array([0,0,0,1])),
            # m7 = (b-d)(g+h): u=[0,1,0,-1], v=[0,0,1,1], w=[1,0,0,0]
            (np.array([0,1,0,-1]), np.array([0,0,1,1]), np.array([1,0,0,0])),
        ]
        return terms

    def verify_2x2(strassen_terms):
        """Verify the 2x2x2 Strassen decomposition."""
        # Check: for each (i,j,k) in 2x2, sum_s u_s[idx_a] * v_s[idx_b] * w_s[idx_c] = 1 if valid
        # A idx: i*2+j, B idx: j*2+k, C idx: i*2+k
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    total = 0.0
                    for s in range(7):
                        u2, v2, w2 = strassen_terms[s]
                        a_idx = i * 2 + j
                        b_idx = j * 2 + k
                        c_idx = i * 2 + k
                        total += u2[a_idx] * v2[b_idx] * w2[c_idx]
                    if abs(total - 1.0) > 1e-10:
                        return False
        return True

    def build_3x3x3_rank28():
        """
        Build a rank-28 decomposition for 3x3x3 using 4 overlapping 2x2 Strassen blocks.
        Blocks: TL(0,1,0,1), TR(0,1,1,2), BL(1,2,0,1), BR(1,2,1,2)
        """
        rank = 28
        U = np.zeros((9, rank))
        V = np.zeros((9, rank))
        W = np.zeros((9, rank))
        
        strassen_terms = strassen_2x2x2_terms()
        
        blocks = [
            (0, 1, 0, 1),  # TL: A rows 0-1, A cols 0-1
            (0, 1, 1, 2),  # TR: A rows 0-1, A cols 1-2
            (1, 2, 0, 1),  # BL: A rows 1-2, A cols 0-1
            (1, 2, 1, 2),  # BR: A rows 1-2, A cols 1-2
        ]
        
        col = 0
        for r1, r2, c1, c2 in blocks:
            for s in range(7):
                u2, v2, w2 = strassen_terms[s]
                
                # For this 2x2 block:
                # A sub-block: rows {r1,r2}, cols {c1,c2}
                #   2x2 pos (i,j) -> 3x3 pos (r1+i, c1+j), idx = (r1+i)*3 + (c1+j)
                # B sub-block: rows {c1,c2} (B row = A col), cols {c1,c2} (B col = C col)
                #   2x2 pos (j,k) -> 3x3 pos (c1+j, c1+k), idx = (c1+j)*3 + (c1+k)
                # C sub-block: rows {r1,r2}, cols {c1,c2}
                #   2x2 pos (i,k) -> 3x3 pos (r1+i, c1+k), W idx = (c1+k)*3 + (r1+i)
                
                # A mapping: 2x2 idx = i*2+j
                for i in range(2):
                    for j in range(2):
                        idx2 = i * 2 + j
                        a3_row = r1 + i
                        a3_col = c1 + j
                        U[a3_row * 3 + a3_col, col] += u2[idx2]
                
                # B mapping: 2x2 idx = j*2+k (B row j, B col k)
                for j in range(2):
                    for k in range(2):
                        idx2 = j * 2 + k
                        b3_row = c1 + j
                        b3_col = c1 + k
                        V[b3_row * 3 + b3_col, col] += v2[idx2]
                
                # W mapping: 2x2 idx = i*2+k (C row i, C col k)
                # W index convention: W[k*n+i, s] where k is output col, i is output row
                for i in range(2):
                    for k in range(2):
                        idx2 = i * 2 + k
                        c3_row = r1 + i
                        c3_col = c1 + k
                        W[c3_col * 3 + c3_row, col] += w2[idx2]
                
                col += 1
        
        return U, V, W, rank

    def verify_3x3x3(U, V, W):
        """Verify 3x3x3 decomposition: sum_s U[i*3+j,s]*V[j*3+k,s]*W[k*3+i,s] = 1"""
        max_err = 0.0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    total = np.dot(U[i*3+j, :], V[j*3+k, :] * W[k*3+i, :])
                    err = abs(total - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def build_3x3x6_from_3x3x3(U33, V33, W33, rank33):
        """
        Lift a rank-r 3x3x3 decomposition to 3x3x6 with rank 2r.
        B is 3x6: split into two 3x3 halves.
        B_half1 = B[:, 0:3], B_half2 = B[:, 3:6]
        C_half1 = A * B_half1 (3x3), C_half2 = A * B_half2 (3x3)
        C = [C_half1, C_half2] (3x6)
        
        U shape: (9, 2r) - same A for both halves
        V shape: (18, 2r) - V[j*6+k, s]
        W shape: (18, 2r) - W[k*3+i, s]
        """
        rank = 2 * rank33
        U = np.zeros((9, rank))
        V = np.zeros((18, rank))
        W = np.zeros((18, rank))
        
        for s in range(rank33):
            # First half: B cols 0-2, C cols 0-2
            idx1 = 2 * s
            for i in range(3):
                for j in range(3):
                    U[i*3+j, idx1] = U33[i*3+j, s]
            for j in range(3):
                for k in range(3):
                    V[j*6+k, idx1] = V33[j*3+k, s]
            for k in range(3):
                for i in range(3):
                    W[k*3+i, idx1] = W33[k*3+i, s]
            
            # Second half: B cols 3-5, C cols 3-5
            idx2 = 2 * s + 1
            for i in range(3):
                for j in range(3):
                    U[i*3+j, idx2] = U33[i*3+j, s]
            for j in range(3):
                for k in range(3):
                    V[j*6+k+3, idx2] = V33[j*3+k, s]
            for k in range(3):
                for i in range(3):
                    W[(k+3)*3+i, idx2] = W33[k*3+i, s]
        
        return U, V, W, rank

    def verify_3x3x6(U, V, W):
        """Verify 3x3x6 decomposition: sum_s U[i*3+j,s]*V[j*6+k,s]*W[k*3+i,s] = 1"""
        max_err = 0.0
        for i in range(3):
            for j in range(3):
                for k in range(6):
                    total = np.dot(U[i*3+j, :], V[j*6+k, :] * W[k*3+i, :])
                    err = abs(total - 1.0)
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

    best_result = None
    best_rank = float('inf')

    # Method 1: 4 Strassen blocks for 3x3x3 (rank 28), lift to 3x3x6 (rank 56)
    # This is worse than schoolbook (54), so skip if we can do better
    # But let's verify the 3x3x3 part first
    try:
        U33, V33, W33, rank33 = build_3x3x3_rank28()
        err33 = verify_3x3x3(U33, V33, W33)
        if err33 < 1e-9:
            U, V, W, rank = build_3x3x6_from_3x3x3(U33, V33, W33, rank33)
            err = verify_3x3x6(U, V, W)
            if err < 1e-9 and rank < best_rank:
                best_rank = rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception as e:
        pass

    # Method 2: Schoolbook fallback (rank 54)
    U, V, W, rank = schoolbook_3x3x6()
    err = verify_3x3x6(U, V, W)
    if err < 1e-9 and rank < best_rank:
        best_rank = rank
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    if best_result is not None:
        return best_result

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
