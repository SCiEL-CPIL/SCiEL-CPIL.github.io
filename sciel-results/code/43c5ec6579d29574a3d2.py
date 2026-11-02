import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 26, 26

    def A_idx(i, j): return i * m + j
    def B_idx(j, k): return j * p + k
    def C_idx(i, k): return k * n + i

    def build_rank46():
        R = 46
        U = np.zeros((nm, R), dtype=float)
        V = np.zeros((mp, R), dtype=float)
        W = np.zeros((pn, R), dtype=float)

        def add_strassen_block(k0, col_offset):
            prods = [
                ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
                ({(1,0):1,(1,1):1}, {(0,0):1}),
                ({(0,0):1}, {(0,1):1,(1,1):-1}),
                ({(1,1):1}, {(1,0):1,(0,0):-1}),
                ({(0,0):1,(0,1):1}, {(1,1):1}),
                ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}),
                ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}),
            ]
            for t, (a_coeff, b_coeff) in enumerate(prods):
                col = col_offset + t
                for (i, j), c in a_coeff.items():
                    U[A_idx(i, j), col] += c
                for (j, kl), c in b_coeff.items():
                    V[B_idx(j, k0 + kl), col] += c
            c_combos = [
                (0, k0,     [(0,1),(3,1),(4,-1),(6,1)]),
                (0, k0+1,   [(2,1),(4,1)]),
                (1, k0,     [(1,1),(3,1)]),
                (1, k0+1,   [(0,1),(1,-1),(2,1),(5,1)]),
            ]
            for (i, k, terms) in c_combos:
                for (t, coeff) in terms:
                    W[C_idx(i, k), col_offset + t] += coeff

        def add_single_block(k, col_offset):
            prods = [
                ((0,0),(0,k)),
                ((0,1),(1,k)),
                ((1,0),(0,k)),
                ((1,1),(1,k)),
            ]
            for t, ((i,j),(jj,kk)) in enumerate(prods):
                col = col_offset + t
                U[A_idx(i, j), col] += 1
                V[B_idx(jj, kk), col] += 1
            W[C_idx(0, k), col_offset+0] += 1
            W[C_idx(0, k), col_offset+1] += 1
            W[C_idx(1, k), col_offset+2] += 1
            W[C_idx(1, k), col_offset+3] += 1

        col = 0
        for k0 in range(0, 12, 2):
            add_strassen_block(k0, col)
            col += 7
        add_single_block(12, col)
        col += 4
        assert col == R
        return U, V, W, R

    U, V, W, R = build_rank46()

    def verify(U, V, W, tol=1e-9):
        T = np.einsum('ir,jr,kr->ijk', U, V, W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T[A_idx(i,j), B_idx(j,k), C_idx(i,k)] - 1.0) > tol:
                        return False
        for i in range(nm):
            for j in range(mp):
                for k in range(pn):
                    a = i // m; b = i % m
                    b2 = j // p; k2 = j % p
                    k3 = k // n; a2 = k % n
                    if a == a2 and b == b2 and k2 == k3:
                        continue
                    if abs(T[i, j, k]) > tol:
                        return False
        return True

    if not verify(U, V, W):
        R0 = n * m * p
        U0 = np.zeros((nm, R0), dtype=float)
        V0 = np.zeros((mp, R0), dtype=float)
        W0 = np.zeros((pn, R0), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U0[i*m+j, c] = 1.0
                    V0[j*p+k, c] = 1.0
                    W0[k*n+i, c] = 1.0
                    c += 1
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = R

    # Try to find column dependencies to reduce rank
    if time.time() < deadline - 20:
        # Check if any column is linearly dependent on others in the tensor space
        # For each column c, check if removing it and adjusting others still works
        # This is expensive, so only try a few
        for c_try in range(R - 1, -1, -1):
            if time.time() > deadline - 15:
                break
            # Try removing column c_try
            U_test = np.delete(U, c_try, axis=1)
            V_test = np.delete(V, c_try, axis=1)
            W_test = np.delete(W, c_try, axis=1)
            if verify(U_test, V_test, W_test, tol=1e-8):
                best_U, best_V, best_W = U_test, V_test, W_test
                best_rank = R - 1
                U, V, W = U_test, V_test, W_test
                R = best_rank
                break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
