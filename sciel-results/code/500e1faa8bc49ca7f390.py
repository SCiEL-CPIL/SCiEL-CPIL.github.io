"""Matrix multiplication 3x3 @ 3x5 via block-Strassen rank-21 decomposition."""
import time

DIMENSIONS = [3, 3, 5]
T0 = time.time()
DEADLINE = 115.0


def construct():
    # A is 3x3, B is 3x5, C = A @ B is 3x5.
    # Flat indices:
    #   A: a_idx(i,j) = i*3+j, i,j in 0..2  (9 entries)
    #   B: b_idx(j,k) = j*5+k, j in 0..2, k in 0..4  (15 entries)
    #   C: c_idx(k,i) = k*3+i, k in 0..4, i in 0..2  (15 entries)  [W[k*n+i], n=3]
    U_all, V_all, W_all = [], [], []

    def a_idx(i, j):
        return i * 3 + j

    def b_idx(j, k):
        return j * 5 + k

    def c_idx(k, i):
        return k * 3 + i

    def add_term(u, v, w):
        U_all.append(u)
        V_all.append(v)
        W_all.append(w)

    def vec_A11(coeffs):
        v = [0] * 9
        for (i, j), val in coeffs.items():
            v[a_idx(i, j)] = val
        return v

    # ---- A11 @ B11a : 2x2 @ 2x2 (Strassen, rank 7) ----
    def vec_B11a(coeffs):
        v = [0] * 15
        for (j, k), val in coeffs.items():
            v[b_idx(j, k)] = val
        return v

    def vec_C11a(coeffs):
        v = [0] * 15
        for (k, i), val in coeffs.items():
            v[c_idx(k, i)] = val
        return v

    add_term(vec_A11({(0,0):1,(1,1):1}), vec_B11a({(0,1):1,(1,1):-1}), vec_C11a({(0,0):1,(1,1):1}))
    add_term(vec_A11({(1,0):1,(1,1):1}), vec_B11a({(0,0):1}), vec_C11a({(0,1):1}))
    add_term(vec_A11({(0,0):1}), vec_B11a({(1,0):1,(1,1):-1}), vec_C11a({(1,0):1}))
    add_term(vec_A11({(1,1):1}), vec_B11a({(0,0):1,(1,1):1}), vec_C11a({(0,0):1,(0,1):1}))
    add_term(vec_A11({(0,0):1,(0,1):1}), vec_B11a({(1,1):1}), vec_C11a({(0,0):1,(1,0):-1}))
    add_term(vec_A11({(1,0):1,(0,0):-1}), vec_B11a({(0,0):1,(0,1):1}), vec_C11a({(1,1):1}))
    add_term(vec_A11({(0,1):1,(1,0):-1}), vec_B11a({(1,0):1,(1,1):1}), vec_C11a({(0,0):1}))

    # ---- A11 @ B11b : 2x2 @ 2x2 (Strassen, rank 7) ----
    def vec_B11b(coeffs):
        v = [0] * 15
        for (j, k), val in coeffs.items():
            v[b_idx(j, k)] = val
        return v

    def vec_C11b(coeffs):
        v = [0] * 15
        for (k, i), val in coeffs.items():
            v[c_idx(k, i)] = val
        return v

    add_term(vec_A11({(0,0):1,(1,1):1}), vec_B11b({(0,3):1,(1,3):-1}), vec_C11b({(2,0):1,(3,1):1}))
    add_term(vec_A11({(1,0):1,(1,1):1}), vec_B11b({(0,2):1}), vec_C11b({(2,1):1}))
    add_term(vec_A11({(0,0):1}), vec_B11b({(1,2):1,(1,3):-1}), vec_C11b({(3,0):1}))
    add_term(vec_A11({(1,1):1}), vec_B11b({(0,2):1,(1,3):1}), vec_C11b({(2,0):1,(2,1):1}))
    add_term(vec_A11({(0,0):1,(0,1):1}), vec_B11b({(1,3):1}), vec_C11b({(2,0):1,(3,0):-1}))
    add_term(vec_A11({(1,0):1,(0,0):-1}), vec_B11b({(0,2):1,(0,3):1}), vec_C11b({(3,1):1}))
    add_term(vec_A11({(0,1):1,(1,0):-1}), vec_B11b({(1,2):1,(1,3):1}), vec_C11b({(2,0):1}))

    # ---- A11 @ B11c : 2x2 @ 2x1 (rank 2) ----
    U = [0]*9; U[a_idx(0,0)]=1; U[a_idx(1,0)]=1
    V = [0]*15; V[b_idx(0,4)]=1
    W = [0]*15; W[c_idx(4,0)]=1; W[c_idx(4,1)]=1
    add_term(U, V, W)
    U = [0]*9; U[a_idx(0,1)]=1; U[a_idx(1,1)]=1
    V = [0]*15; V[b_idx(1,4)]=1
    W = [0]*15; W[c_idx(4,0)]=1; W[c_idx(4,1)]=1
    add_term(U, V, W)

    # ---- A12 @ B21 : 2x1 @ 1x5 (rank 2) ----
    U = [0]*9; U[a_idx(0,2)]=1
    V = [0]*15
    for k in range(5): V[b_idx(2,k)]=1
    W = [0]*15
    for k in range(5): W[c_idx(k,0)]=1
    add_term(U, V, W)
    U = [0]*9; U[a_idx(1,2)]=1
    V = [0]*15
    for k in range(5): V[b_idx(2,k)]=1
    W = [0]*15
    for k in range(5): W[c_idx(k,1)]=1
    add_term(U, V, W)

    # ---- A21 @ B11 : 1x2 @ 2x5 (rank 2) ----
    U = [0]*9; U[a_idx(2,0)]=1
    V = [0]*15
    for k in range(5): V[b_idx(0,k)]=1
    W = [0]*15
    for k in range(5): W[c_idx(k,2)]=1
    add_term(U, V, W)
    U = [0]*9; U[a_idx(2,1)]=1
    V = [0]*15
    for k in range(5): V[b_idx(1,k)]=1
    W = [0]*15
    for k in range(5): W[c_idx(k,2)]=1
    add_term(U, V, W)

    # ---- A22 @ B21 : 1x1 @ 1x5 (rank 1) ----
    U = [0]*9; U[a_idx(2,2)]=1
    V = [0]*15
    for k in range(5): V[b_idx(2,k)]=1
    W = [0]*15
    for k in range(5): W[c_idx(k,2)]=1
    add_term(U, V, W)

    return {"U": U_all, "V": V_all, "W": W_all}
