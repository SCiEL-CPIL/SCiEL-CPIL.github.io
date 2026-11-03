import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    start_time = time.time()
    deadline = 145.0

    # Winograd 2x2x2 factors
    U_w = np.array([
        [1, 0, 1, 1, 0, 1, 0],
        [0, 1, 1, 0, 1, -1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 1, 0, 0, 1, 0, -1]
    ], dtype=float)

    V_w = np.array([
        [1, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 0, 1, 0, 0]
    ], dtype=float)

    W_w = np.array([
        [1, 0, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, -1, 1],
        [1, 0, 1, 0, 0, 0, 1]
    ], dtype=float)

    # Schoolbook 2x2x1 factors
    U_s = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=float)
    V_s = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ], dtype=float)
    W_s = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=float)

    # Verify Winograd 2x2x2
    def verify_222(U, V, W):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    a = i * 2 + j
                    b = j * 2 + k
                    c = k * 2 + i
                    s = np.dot(U[a], np.multiply(V[b], W[c]))
                    if abs(s - 1.0) > 1e-9:
                        return False
        for a in range(4):
            ia, ja = divmod(a, 2)
            for b in range(4):
                jb, kb = divmod(b, 2)
                for c in range(4):
                    kc, ic = divmod(c, 2)
                    if ia == ic and ja == jb and kb == kc:
                        continue
                    s = np.dot(U[a], np.multiply(V[b], W[c]))
                    if abs(s) > 1e-9:
                        return False
        return True

    if not verify_222(U_w, V_w, W_w):
        return _schoolbook()

    # Verify Schoolbook 2x2x1
    def verify_221(U, V, W):
        for i in range(2):
            for j in range(2):
                k = 0
                a = i * 2 + j
                b = j * 1 + k
                c = k * 2 + i
                s = np.dot(U[a], np.multiply(V[b], W[c]))
                if abs(s - 1.0) > 1e-9:
                    return False
        for a in range(4):
            ia, ja = divmod(a, 2)
            for b in range(2):
                jb, kb = divmod(b, 1)
                for c in range(2):
                    kc, ic = divmod(c, 2)
                    if ia == ic and ja == jb and kb == kc:
                        continue
                    s = np.dot(U[a], np.multiply(V[b], W[c]))
                    if abs(s) > 1e-9:
                        return False
        return True

    if not verify_221(U_s, V_s, W_s):
        return _schoolbook()

    total_rank = 50
    U = np.zeros((8, total_rank))
    V = np.zeros((28, total_rank))
    W = np.zeros((14, total_rank))

    # Helper to place Winograd block
    def place_winograd(U, V, W, col_start, j_offset, k_offset):
        # j_offset: 0 or 2
        # k_offset: 0, 2, or 4
        # U rows: i in 0,1; j in j_offset, j_offset+1
        # U_w rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        for i in range(2):
            for j_loc in range(2):
                j_glob = j_offset + j_loc
                a = i * 4 + j_glob
                u_row = i * 2 + j_loc
                U[a, col_start:col_start+7] = U_w[u_row]
        
        # V rows: j in j_offset, j_offset+1; k in k_offset, k_offset+1
        # V_w rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        for j_loc in range(2):
            j_glob = j_offset + j_loc
            for k_loc in range(2):
                k_glob = k_offset + k_loc
                b = j_glob * 7 + k_glob
                v_row = j_loc * 2 + k_loc
                V[b, col_start:col_start+7] = V_w[v_row]
                
        # W rows: k in k_offset, k_offset+1; i in 0,1
        # W_w rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        for k_loc in range(2):
            k_glob = k_offset + k_loc
            for i in range(2):
                c = k_glob * 2 + i
                w_row = k_loc * 2 + i
                W[c, col_start:col_start+7] += W_w[w_row]

    # Helper to place Schoolbook block
    def place_schoolbook(U, V, W, col_start, j_offset, k_offset):
        # j_offset: 0 or 2
        # k_offset: 6
        for i in range(2):
            for j_loc in range(2):
                j_glob = j_offset + j_loc
                a = i * 4 + j_glob
                u_row = i * 2 + j_loc
                U[a, col_start:col_start+4] = U_s[u_row]
                
        for j_loc in range(2):
            j_glob = j_offset + j_loc
            k_glob = k_offset
            b = j_glob * 7 + k_glob
            v_row = j_loc
            V[b, col_start:col_start+4] = V_s[v_row]
            
        for i in range(2):
            k_glob = k_offset
            c = k_glob * 2 + i
            w_row = i
            W[c, col_start:col_start+4] += W_s[w_row]

    # Block 0: j in {0, 1}
    col = 0
    place_winograd(U, V, W, col, 0, 0); col += 7
    place_winograd(U, V, W, col, 0, 2); col += 7
    place_winograd(U, V, W, col, 0, 4); col += 7
    place_schoolbook(U, V, W, col, 0, 6); col += 4

    # Block 1: j in {2, 3}
    place_winograd(U, V, W, col, 2, 0); col += 7
    place_winograd(U, V, W, col, 2, 2); col += 7
    place_winograd(U, V, W, col, 2, 4); col += 7
    place_schoolbook(U, V, W, col, 2, 6); col += 4

    # Verify full decomposition
    def check_valid(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = np.dot(U[a], np.multiply(V[b], W[c]))
                    if abs(s - 1.0) > 1e-6:
                        return False
        for a in range(n * m):
            ia = a // m
            ja = a % m
            for b in range(m * p):
                jb = b // p
                kb = b % p
                for c in range(p * n):
                    kc = c // n
                    ic = c % n
                    if ia == ic and ja == jb and kb == kc:
                        continue
                    s = np.dot(U[a], np.multiply(V[b], W[c]))
                    if abs(s) > 1e-6:
                        return False
        return True

    if time.time() < deadline - 5:
        if check_valid(U, V, W, total_rank):
            def clean(x):
                if abs(x - round(x)) < 1e-10:
                    return int(round(x))
                if abs(2*x - round(2*x)) < 1e-10:
                    return round(2*x) / 2
                return float(x)
            return {
                "U": [[clean(U[i, j]) for j in range(total_rank)] for i in range(n*m)],
                "V": [[clean(V[i, j]) for j in range(total_rank)] for i in range(m*p)],
                "W": [[clean(W[i, j]) for j in range(total_rank)] for i in range(p*n)]
            }

    return _schoolbook()

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
