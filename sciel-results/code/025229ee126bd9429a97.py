import time
import numpy as np
import scipy.optimize as opt

def _verify(U, V, W, n, m, p):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                if abs(val - 1.0) > 1e-9:
                    return False
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m+j, j*p+k, k*n+i))
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                if (a, b, c) not in valid:
                    val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                    if abs(val) > 1e-9:
                        return False
    return True

def _schoolbook(n, m, p):
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
    return U, V, W

def _strassen_block(n, m, p):
    n, m, p = 2, 2, 16
    num_blocks = p // 2
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(num_blocks):
        intermediates = [
            {'U': [(0, 1), (3, 1)], 'V': [(2*t, 1), (16+2*t+1, 1)], 'W': [(4*t, 1), (4*t+3, 1)]},
            {'U': [(2, 1), (3, 1)], 'V': [(2*t, 1)], 'W': [(4*t+1, 1), (4*t+3, -1)]},
            {'U': [(0, 1)], 'V': [(2*t+1, 1), (16+2*t+1, -1)], 'W': [(4*t+2, 1), (4*t+3, 1)]},
            {'U': [(3, 1)], 'V': [(16+2*t, 1), (2*t, -1)], 'W': [(4*t, 1), (4*t+1, 1)]},
            {'U': [(0, 1), (1, 1)], 'V': [(16+2*t+1, 1)], 'W': [(4*t, -1), (4*t+2, 1)]},
            {'U': [(2, 1), (0, -1)], 'V': [(2*t, 1), (2*t+1, 1)], 'W': [(4*t+3, 1)]},
            {'U': [(1, 1), (3, -1)], 'V': [(16+2*t, 1), (16+2*t+1, 1)], 'W': [(4*t, 1)]}
        ]
        for inter in intermediates:
            for idx, coeff in inter['U']:
                U[idx][col] += coeff
            for idx, coeff in inter['V']:
                V[idx][col] += coeff
            for idx, coeff in inter['W']:
                W[idx][col] += coeff
            col += 1
    return U, V, W

def _try_rank13_2x2x4():
    n, m, p = 2, 2, 4
    R = 13
    T = np.zeros((4, 8, 8))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    np.random.seed(42)
    U = np.random.randn(4, R)
    V = np.random.randn(8, R)
    W = np.random.randn(8, R)
    
    def objective(x):
        U_ = x[:4*R].reshape(4, R)
        V_ = x[4*R:4*R+8*R].reshape(8, R)
        W_ = x[4*R+8*R:].reshape(8, R)
        T_approx = np.einsum('ar,br,cr->abc', U_, V_, W_)
        return np.sum((T - T_approx)**2)
        
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    try:
        res = opt.minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-15})
        if res.fun < 1e-10:
            U_sol = res.x[:4*R].reshape(4, R)
            V_sol = res.x[4*R:4*R+8*R].reshape(8, R)
            W_sol = res.x[4*R+8*R:].reshape(8, R)
            
            U_round = np.round(U_sol * 2) / 2
            V_round = np.round(V_sol * 2) / 2
            W_round = np.round(W_sol * 2) / 2
            
            if _verify(U_round.tolist(), V_round.tolist(), W_round.tolist(), n, m, p):
                return U_round, V_round, W_round
    except:
        pass
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try to find rank 52 solution (4 * 13)
    if time.time() - start_time < 100:
        block = _try_rank13_2x2x4()
        if block is not None:
            U4, V4, W4 = block
            R4 = U4.shape[1]
            R16 = 4 * R4
            U16 = np.tile(U4, (1, 4))
            V16 = np.zeros((32, R16))
            W16 = np.zeros((32, R16))
            for t in range(4):
                V16[t*8:(t+1)*8, t*R4:(t+1)*R4] = V4
                W16[t*8:(t+1)*8, t*R4:(t+1)*R4] = W4
            
            U_list = U16.tolist()
            V_list = V16.tolist()
            W_list = W16.tolist()
            
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
                
    # Fall back to Strassen rank 56
    U, V, W = _strassen_block(n, m, p)
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
        
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
