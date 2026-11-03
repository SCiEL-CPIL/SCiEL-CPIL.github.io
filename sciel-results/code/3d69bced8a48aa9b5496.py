import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def build_tensor(n, m, p, u_conv, v_conv, w_conv):
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if u_conv == 'row':
                    a = i * m + j
                else:
                    a = j * n + i
                if v_conv == 'row':
                    b = j * p + k
                else:
                    b = k * m + j
                if w_conv == 'row':
                    c = i * p + k
                else:
                    c = k * n + i
                T[a, b, c] = 1
    return T

def verify_decomposition(U, V, W, T):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T_rec = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_rec, T, atol=1e-9), T_rec

def current_solution():
    n, m, p = DIMENSIONS
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_indices, v_indices, w_indices, col):
        for u_idx, val in u_indices:
            U[u_idx][col] += val
        for v_idx, val in v_indices:
            V[v_idx][col] += val
        for w_idx, val in w_indices:
            W[w_idx][col] += val
    
    add_term([(0, 1), (4, 1)], [(0, 1), (4, 1)], [(0, 1), (3, 1)], 0)
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    add_term([(0, 1)], [(1, 1), (4, -1)], [(2, 1), (3, 1)], 2)
    add_term([(4, 1)], [(3, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    add_term([(0, 1), (1, 1)], [(4, 1)], [(0, -1), (2, 1)], 4)
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    add_term([(1, 1), (4, -1)], [(3, 1), (4, 1)], [(0, 1)], 6)
    add_term([(2, 1)], [(6, 1)], [(0, 1)], 7)
    add_term([(5, 1)], [(6, 1)], [(1, 1)], 8)
    add_term([(2, 1)], [(7, 1)], [(2, 1)], 9)
    add_term([(5, 1)], [(7, 1)], [(3, 1)], 10)
    add_term([(0, 1)], [(2, 1)], [(4, 1)], 11)
    add_term([(3, 1)], [(2, 1)], [(5, 1)], 12)
    add_term([(1, 1)], [(5, 1)], [(4, 1)], 13)
    add_term([(4, 1)], [(5, 1)], [(5, 1)], 14)
    add_term([(2, 1)], [(8, 1)], [(4, 1)], 15)
    add_term([(5, 1)], [(8, 1)], [(5, 1)], 16)
    
    return {"U": U, "V": V, "W": W}

sol = current_solution()
for u_conv in ['row', 'col']:
    for v_conv in ['row', 'col']:
        for w_conv in ['row', 'col']:
            T = build_tensor(2, 3, 3, u_conv, v_conv, w_conv)
            valid, T_rec = verify_decomposition(sol['U'], sol['V'], sol['W'], T)
            if valid:
                print(f"VALID: u={u_conv}, v={v_conv}, w={w_conv}")
            else:
                err = np.max(np.abs(T_rec - T))
                print(f"INVALID: u={u_conv}, v={v_conv}, w={w_conv}, max_err={err:.4f}")
