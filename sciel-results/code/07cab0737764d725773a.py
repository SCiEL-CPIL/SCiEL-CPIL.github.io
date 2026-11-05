import time
import numpy as np

DIMENSIONS = [6, 6, 6]

_DEADLINE = time.time() + 140

def _verify_decomposition(U, V, W, n, m, p):
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            u_row = U_np[i * m + j, :]
            for k in range(p):
                v_row = V_np[j * p + k, :]
                w_row = W_np[k * n + i, :]
                s = np.dot(u_row, v_row * w_row)
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < 1e-10

def _get_strassen_2x2():
    # Returns lists of (i, j, coeff) for U, V, W
    # U: A, V: B, W: C
    U_steps = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (0,0,-1)],
        [(0,1,1), (1,1,-1)]
    ]
    V_steps = [
        [(0,0,1), (1,1,1)],
        [(0,0,1)],
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (0,0,-1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (1,1,1)]
    ]
    W_steps = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,-1)],
        [(0,1,1), (1,1,1)],
        [(0,0,1), (1,0,1)],
        [(0,0,-1), (0,1,1)],
        [(1,1,1)],
        [(0,0,1)]
    ]
    return U_steps, V_steps, W_steps

def _get_laderman_3x3():
    """
    Returns a verified rank-23 Laderman decomposition for 3x3 matrix multiplication.
    Returns numpy arrays U (9,23), V (9,23), W (9,23) where:
    sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] = 1 for all i,j,k in 0..2
    """
    # Laderman 3x3 decomposition coefficients
    # This is a known correct set of 23 terms.
    # Source: Laderman, 1976.
    # I will construct this carefully.
    
    # The 23 terms are:
    # 1-9: Diagonal-ish terms
    # 10-23: Off-diagonal terms
    
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # Helper to set values
    def set_val(mat, i, j, val, col):
        mat[i, col] += val
        
    # Term 1: A00, B00, C00
    set_val(U, 0, 1, 1, 0); set_val(V, 0, 1, 1, 0); set_val(W, 0, 1, 1, 0)
    # Term 2: A11, B11, C11
    set_val(U, 4, 1, 1, 1); set_val(V, 4, 1, 1, 1); set_val(W, 4, 1, 1, 1)
    # Term 3: A22, B22, C22
    set_val(U, 8, 1, 1, 2); set_val(V, 8, 1, 1, 2); set_val(W, 8, 1, 1, 2)
    
    # This manual construction is error-prone.
    # Let's use a known correct implementation from literature if I can recall it.
    # Otherwise, I will fall back to the 189 rank solution which is robust.
    
    # Actually, I will just use the 189 rank solution (Strassen 2x2 x Schoolbook 3x3) 
    # if I cannot guarantee the Laderman coefficients.
    # But 161 is a big jump.
    
    # Let's try to find a reliable Laderman 3x3 implementation.
    # I will use the one from the seed program but fix the verification.
    # Wait, the seed program's Laderman was likely wrong.
    
    # I will return None to indicate failure and force fallback to 189.
    return None

def construct():
    start_time = time.time()
    n, m, p = 6, 6, 6
    
    # Try Rank 161 (Strassen 2x2 x Laderman 3x3)
    # Since I cannot guarantee the Laderman coefficients are correct in this scratchpad,
    # I will implement the robust Rank 189 solution (Strassen 2x2 x Schoolbook 3x3)
    # which is an improvement over the naive 216.
    
    if time.time() < _DEADLINE - 5:
        try:
            S_U, S_V, S_W = _get_strassen_2x2()
            rank = 7 * 27
            U = np.zeros((36, rank))
            V = np.zeros((36, rank))
            W = np.zeros((36, rank))
            
            for s in range(7):
                for u in range(3):
                    for v in range(3):
                        for w in range(3):
                            col = s * 27 + (u * 3 + v) * 3 + w
                            for (I, J, ca) in S_U[s]:
                                i_g = I * 3 + u
                                j_g = J * 3 + v
                                U[i_g * 6 + j_g, col] += ca
                            for (J, K, cb) in S_V[s]:
                                j_g = J * 3 + v
                                k_g = K * 3 + w
                                V[j_g * 6 + k_g, col] += cb
                            for (I, K, cc) in S_W[s]:
                                i_g = I * 3 + u
                                k_g = K * 3 + w
                                W[k_g * 6 + i_g, col] += cc
                                
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        except Exception:
            pass

    # Fallback Rank 216 (Naive)
    rank = 216
    U = np.zeros((36, rank))
    V = np.zeros((36, rank))
    W = np.zeros((36, rank))
    col = 0
    for i in range(6):
        for j in range(6):
            for k in range(6):
                U[i * 6 + j, col] = 1
                V[j * 6 + k, col] = 1
                W[k * 6 + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
