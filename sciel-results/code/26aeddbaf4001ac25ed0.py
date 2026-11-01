import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 4]

def construct():
    start_time = time.time()
    deadline = 145.0

    n, m, p = DIMENSIONS

    # Build the multiplication tensor T as a 3D array
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    T_flat = T.flatten()

    # Try numerical rank-13 search with warm start from rank-14
    best_solution = None
    U14, V14, W14 = _build_rank14_solution()

    max_restarts = 80
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break

        # Warm start from rank 14
        if restart < 15:
            U_init = np.array(U14)[:, :13]
            V_init = np.array(V14)[:, :13]
            W_init = np.array(W14)[:, :13]
        else:
            U_init = None
            V_init = None
            W_init = None

        solution = _search_rank_13(n, m, p, T_flat, restart, deadline, U_init, V_init, W_init)
        if solution is not None:
            best_solution = solution
            break

    if best_solution is not None:
        U, V, W = best_solution
        if _verify_solution(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}

    # Fall back to rank 14
    if _verify_solution(U14, V14, W14, n, m, p):
        return {"U": U14, "V": V14, "W": W14}

    # Fall back to schoolbook rank 16
    rank_fb = n * m * p
    U_fb = np.zeros((n*m, rank_fb))
    V_fb = np.zeros((m*p, rank_fb))
    W_fb = np.zeros((p*n, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1.0
                V_fb[j*p+k, col] = 1.0
                W_fb[k*n+i, col] = 1.0
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

def _build_rank14_solution():
    """Build the rank-14 Strassen-based solution."""
    n, m, p = 2, 2, 4
    rank = 14
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))

    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(0,2), 3=(0,3), 4=(1,0), 5=(1,1), 6=(1,2), 7=(1,3)
    # W indices (k*n+i): C[0,0]=0, C[0,1]=2, C[0,2]=4, C[0,3]=6, C[1,0]=1, C[1,1]=3, C[1,2]=5, C[1,3]=7

    # Left Strassen block (B columns 0,1)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[5, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    U[0, 2] = 1
    V[1, 2] = 1; V[5, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    U[3, 3] = 1
    V[4, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    U[0, 4] = 1; U[1, 4] = 1
    V[5, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    U[1, 6] = 1; U[3, 6] = -1
    V[4, 6] = 1; V[5, 6] = 1
    W[0, 6] = 1

    # Right Strassen block (B columns 2,3)
    U[0, 7] = 1; U[3, 7] = 1
    V[2, 7] = 1; V[7, 7] = 1
    W[4, 7] = 1; W[7, 7] = 1

    U[2, 8] = 1; U[3, 8] = 1
    V[2, 8] = 1
    W[5, 8] = 1; W[7, 8] = -1

    U[0, 9] = 1
    V[3, 9] = 1; V[7, 9] = -1
    W[6, 9] = 1; W[7, 9] = 1

    U[3, 10] = 1
    V[6, 10] = 1; V[2, 10] = -1
    W[4, 10] = 1; W[5, 10] = 1

    U[0, 11] = 1; U[1, 11] = 1
    V[7, 11] = 1
    W[4, 11] = -1; W[6, 11] = 1

    U[2, 12] = 1; U[0, 12] = -1
    V[2, 12] = 1; V[3, 12] = 1
    W[7, 12] = 1

    U[1, 13] = 1; U[3, 13] = -1
    V[6, 13] = 1; V[7, 13] = 1
    W[4, 13] = 1

    return U.tolist(), V.tolist(), W.tolist()

def _search_rank_13(n, m, p, T_flat, seed, deadline, U_init=None, V_init=None, W_init=None):
    rank = 13
    np.random.seed(seed)

    if U_init is not None:
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
    else:
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)

    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

    def residuals(x):
        U_r = x[:n*m*rank].reshape(n*m, rank)
        V_r = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W_r = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
        T_approx = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        return T_approx.flatten() - T_flat

    try:
        result = least_squares(residuals, x0, method='lm', max_nfev=3000)

        if time.time() > deadline - 5:
            return None

        if result.success or result.cost < 1e-8:
            x_opt = result.x
            U_opt = x_opt[:n*m*rank].reshape(n*m, rank)
            V_opt = x_opt[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_opt = x_opt[n*m*rank + m*p*rank:].reshape(p*n, rank)

            U_snap = np.round(U_opt * 2) / 2
            V_snap = np.round(V_opt * 2) / 2
            W_snap = np.round(W_opt * 2) / 2
            if _verify_solution(U_snap, V_snap, W_snap, n, m, p):
                return U_snap, V_snap, W_snap

            U_snap2 = np.round(U_opt)
            V_snap2 = np.round(V_opt)
            W_snap2 = np.round(W_opt)
            if _verify_solution(U_snap2, V_snap2, W_snap2, n, m, p):
                return U_snap2, V_snap2, W_snap2
    except Exception:
        pass

    return None

def _verify_solution(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]

    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            a = i * m + j
                            b = jp * p + k
                            c = kp * n + ip
                            expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            actual = np.sum(U[a] * V[b] * W[c])
                            if abs(actual - expected) > 1e-8:
                                return False
    return True
