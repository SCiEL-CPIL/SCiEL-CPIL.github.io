"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Shapes: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Tensor T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We want to find the minimal rank r such that this holds for all i,j,k.
    
    # The schoolbook rank is n*m*p = 2*3*9 = 54.
    # Lower bound: max(n*m, m*p, p*n) = max(6, 27, 18) = 27.
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """
        Verify if U, V, W form a valid decomposition.
        U, V, W are numpy arrays or lists.
        Returns True if valid, False otherwise.
        """
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        rank = U_arr.shape[1]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = U_arr[i * m + j, :]
                    v = V_arr[j * p + k, :]
                    w = W_arr[k * n + i, :]
                    val = np.dot(u * v, w)
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def snap_to_half_int(arr):
        """Snap values to nearest half-integer."""
        return [[int(round(x * 2)) / 2.0 for x in row] for row in arr]

    def find_decomp_small(n_s, m_s, p_s, rank_s, max_attempts=20, max_iters=50):
        """
        Find a rank rank_s decomposition for n_s x m_s x p_s multiplication.
        Returns (U, V, W) as lists or None.
        """
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
            
            U_s = np.random.randn(n_s*m_s, rank_s)
            V_s = np.random.randn(m_s*p_s, rank_s)
            W_s = np.random.randn(p_s*n_s, rank_s)
            
            for it in range(max_iters):
                # Solve for U
                U_new = np.zeros((n_s*m_s, rank_s))
                for i in range(n_s):
                    for j in range(m_s):
                        A = np.zeros(rank_s)
                        for k in range(p_s):
                            A += V_s[j*p_s+k, :] * W_s[k*n_s+i, :]
                        norm_sq = np.sum(A*A)
                        if norm_sq > 1e-9:
                            U_new[i*m_s+j, :] = A / norm_sq
                U_s = U_new
                
                # Solve for V
                V_new = np.zeros((m_s*p_s, rank_s))
                for j in range(m_s):
                    for k in range(p_s):
                        A = np.zeros(rank_s)
                        for i in range(n_s):
                            A += U_s[i*m_s+j, :] * W_s[k*n_s+i, :]
                        norm_sq = np.sum(A*A)
                        if norm_sq > 1e-9:
                            V_new[j*p_s+k, :] = A / norm_sq
                V_s = V_new
                
                # Solve for W
                W_new = np.zeros((p_s*n_s, rank_s))
                for k in range(p_s):
                    for i in range(n_s):
                        A = np.zeros(rank_s)
                        for j in range(m_s):
                            A += U_s[i*m_s+j, :] * V_s[j*p_s+k, :]
                        norm_sq = np.sum(A*A)
                        if norm_sq > 1e-9:
                            W_new[k*n_s+i, :] = A / norm_sq
                W_s = W_new
                
                # Check
                valid = True
                for i in range(n_s):
                    for j in range(m_s):
                        for k in range(p_s):
                            val = np.dot(U_s[i*m_s+j, :] * V_s[j*p_s+k, :], W_s[k*n_s+i, :])
                            if abs(val - 1.0) > 1e-6:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                
                if valid:
                    # Snap to half-integers
                    U_snap = snap_to_half_int(U_s.tolist())
                    V_snap = snap_to_half_int(V_s.tolist())
                    W_snap = snap_to_half_int(W_s.tolist())
                    
                    # Verify snapped
                    valid_snap = True
                    for i in range(n_s):
                        for j in range(m_s):
                            for k in range(p_s):
                                val = np.dot(np.array(U_snap[i*m_s+j]) * np.array(V_snap[j*p_s+k]), np.array(W_snap[k*n_s+i]))
                                if abs(val - 1.0) > 1e-9:
                                    valid_snap = False
                                    break
                            if not valid_snap: break
                        if not valid_snap: break
                    
                    if valid_snap:
                        return (U_snap, V_snap, W_snap)
        return None

    def construct_from_small(U_s, V_s, W_s, n_s, m_s, p_s, rank_s, factor):
        """
        Construct a decomposition for n x m x (p_s * factor) from a decomposition 
        for n_s x m_s x p_s.
        Assumes n=n_s, m=m_s, p=p_s*factor.
        """
        if n != n_s or m != m_s:
            return None
        
        p_big = p_s * factor
        rank_big = rank_s * factor
        
        U_big = [[0.0] * rank_big for _ in range(n*m)]
        V_big = [[0.0] * rank_big for _ in range(m*p_big)]
        W_big = [[0.0] * rank_big for _ in range(p_big*n)]
        
        # U_big[i*m+j, factor*c + q] = U_s[i*m+j, c]
        for i in range(n):
            for j in range(m):
                for c in range(rank_s):
                    for q in range(factor):
                        U_big[i*m+j][factor*c + q] = U_s[i*m_s+j][c]
        
        # V_big[j*p_big+k, factor*c + q] = V_s[j*p_s+r, c] if q == k//p_s else 0
        # k = p_s * q + r
        for j in range(m):
            for q in range(factor):
                for r in range(p_s):
                    k = p_s*q + r
                    for c in range(rank_s):
                        V_big[j*p_big + k][factor*c + q] = V_s[j*p_s + r][c]
        
        # W_big[k*n+i, factor*c + q] = W_s[r*n+i, c] if q == k//p_s else 0
        # k = p_s * q + r
        for q in range(factor):
            for r in range(p_s):
                for i in range(n):
                    k = p_s*q + r
                    for c in range(rank_s):
                        W_big[k*n + i][factor*c + q] = W_s[r*n + i][c]
        
        # Verify
        if verify(U_big, V_big, W_big):
            return {"U": U_big, "V": V_big, "W": W_big}
        
        return None

    # Try to find a low rank decomposition for 2x3x3
    # Lower bound for 2x3x3 is max(6, 9, 6) = 9.
    # If we find rank R for 2x3x3, we get rank 3R for 2x3x9.
    # We want 3R to be minimal, so R should be minimal.
    
    n_s, m_s, p_s = 2, 3, 3
    factor = 3
    
    for rank_s in range(9, 18): # Try up to schoolbook for small
        if time.time() > deadline - 5:
            break
        
        res_small = find_decomp_small(n_s, m_s, p_s, rank_s)
        if res_small is not None:
            U_s, V_s, W_s = res_small
            res_big = construct_from_small(U_s, V_s, W_s, n_s, m_s, p_s, rank_s, factor)
            if res_big is not None:
                return res_big

    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
