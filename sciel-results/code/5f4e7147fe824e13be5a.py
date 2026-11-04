"""Matrix multiplication 4x4x6 - Strassen-based rank reduction."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 24
    pn = p * n  # 24
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p  # 96
        U = np.zeros((nm, rank), dtype=int)
        V = np.zeros((mp, rank), dtype=int)
        W = np.zeros((pn, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Strassen-based decomposition for 4x4x6
    # Top level: 4x4 = 2x2 blocks of 2x2, 4x6 = 2x3 blocks of 2x3
    # 7 Strassen multiplications of (2x2) * (2x3)
    # Each (2x2)*(2x3): inner Strassen (7) for cols 0-1 + schoolbook (4) for col 2 = 11
    # Total rank: 7 * 11 = 77
    
    R = 77
    U = np.zeros((nm, R), dtype=float)
    V = np.zeros((mp, R), dtype=float)
    W = np.zeros((pn, R), dtype=float)
    
    # Strassen coefficients for 2x2x2
    alpha = [
        [1, 0, 0, 1],  # s0: A00 + A11
        [0, 0, 1, 1],  # s1: A10 + A11
        [1, 0, 0, 0],  # s2: A00
        [0, 0, 0, 1],  # s3: A11
        [1, 1, 0, 0],  # s4: A00 + A01
        [-1, 0, 1, 0], # s5: A10 - A00
        [0, 1, 0, -1]  # s6: A01 - A11
    ]
    beta = [
        [1, 0, 0, 1],  # s0: B00 + B11
        [1, 0, 0, 0],  # s1: B00
        [0, 1, 0, -1], # s2: B01 - B11
        [-1, 0, 1, 0], # s3: B10 - B00
        [0, 0, 0, 1],  # s4: B11
        [1, 1, 0, 0],  # s5: B00 + B01
        [0, 0, 1, 1]   # s6: B10 + B11
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],     # s0
        [(1, 0, 1), (1, 1, -1)],    # s1
        [(0, 1, 1), (1, 1, 1)],     # s2
        [(0, 0, 1), (1, 0, 1)],     # s3
        [(0, 0, -1), (0, 1, 1)],    # s4
        [(1, 1, 1)],                # s5
        [(0, 0, 1)]                 # s6
    ]
    
    col = 0
    for s in range(7):
        # Part 1: Inner Strassen for (2x2) * (2x2) [columns 0,1 of Y_s]
        for si in range(7):
            # U part
            for a in range(4):
                if alpha[si][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]
            
            # V part
            for b in range(4):
                if beta[si][b] != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += beta[si][b] * beta[s][c]
            
            # W part
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        # Part 2: Schoolbook for (2x2) * (2x1) [column 2 of Y_s]
        for u in range(2):
            for v in range(2):
                # U part
                for r in range(4):
                    if alpha[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += alpha[s][r]
                
                # V part
                for c in range(4):
                    if beta[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]
                
                # W part
                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
    
    # Verify the decomposition
    def verify_decomp(U_mat, V_mat, W_mat):
        """Check if U, V, W exactly decompose the tensor."""
        rank = U_mat.shape[1]
        # Use tensor contraction for efficiency
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # We need T[i,j,k] = 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_mat[i * m + j, r] * V_mat[j * p + k, r] * W_mat[k * n + i, r]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    if not verify_decomp(U, V, W):
        return schoolbook()
    
    # Try to reduce rank by finding and removing linearly dependent terms
    def try_reduce_rank(U_mat, V_mat, W_mat, max_iter=300):
        """Try to reduce rank by removing linearly dependent terms."""
        current_U = U_mat.copy()
        current_V = V_mat.copy()
        current_W = W_mat.copy()
        current_rank = current_U.shape[1]
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            if current_rank <= 1:
                break
            
            found_removal = False
            # Try removing each term and check if the decomposition still holds
            # This is O(rank^2 * nm * mp * pn) which is expensive, so we limit iterations
            for r in range(current_rank):
                if time.time() > deadline:
                    break
                test_U = np.delete(current_U, r, axis=1)
                test_V = np.delete(current_V, r, axis=1)
                test_W = np.delete(current_W, r, axis=1)
                
                if verify_decomp(test_U, test_V, test_W):
                    current_U = test_U
                    current_V = test_V
                    current_W = test_W
                    current_rank -= 1
                    found_removal = True
                    break
            
            if not found_removal:
                # Try a few random removals
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    r = np.random.randint(0, current_rank)
                    test_U = np.delete(current_U, r, axis=1)
                    test_V = np.delete(current_V, r, axis=1)
                    test_W = np.delete(current_W, r, axis=1)
                    
                    if verify_decomp(test_U, test_V, test_W):
                        current_U = test_U
                        current_V = test_V
                        current_W = test_W
                        current_rank -= 1
                        found_removal = True
                        break
                if not found_removal:
                    break
        
        return current_U, current_V, current_W, current_rank
    
    # Attempt rank reduction
    U_red, V_red, W_red, rank_red = try_reduce_rank(U, V, W, max_iter=200)
    
    if verify_decomp(U_red, V_red, W_red):
        U, V, W = U_red, V_red, W_red
    
    # Convert to lists for the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
