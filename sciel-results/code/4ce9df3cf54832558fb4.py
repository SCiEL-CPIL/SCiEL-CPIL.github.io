import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    # Build the target tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Build the rank-28 Strassen-based decomposition
    def build_strassen_28():
        rank = 28
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0

        def add_strassen(i1, i2, j1, j2, k1, k2):
            nonlocal col
            # M1 = (A00 + A11)(B00 + B11)
            U[i1*m+j1, col] = 1; U[i2*m+j2, col] = 1
            V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = 1; W[k2*n+i2, col] = 1
            col += 1
            # M2 = (A10 + A11) B00
            U[i2*m+j1, col] = 1; U[i2*m+j2, col] = 1
            V[j1*p+k1, col] = 1
            W[k1*n+i2, col] = 1; W[k2*n+i2, col] = -1
            col += 1
            # M3 = A00 (B01 - B11)
            U[i1*m+j1, col] = 1
            V[j1*p+k2, col] = 1; V[j2*p+k2, col] = -1
            W[k2*n+i1, col] = 1; W[k2*n+i2, col] = 1
            col += 1
            # M4 = A11 (B10 - B00)
            U[i2*m+j2, col] = 1
            V[j2*p+k1, col] = 1; V[j1*p+k1, col] = -1
            W[k1*n+i1, col] = 1; W[k1*n+i2, col] = 1
            col += 1
            # M5 = (A00 + A01) B11
            U[i1*m+j1, col] = 1; U[i1*m+j2, col] = 1
            V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = -1; W[k2*n+i1, col] = 1
            col += 1
            # M6 = (A10 - A00)(B00 + B01)
            U[i2*m+j1, col] = 1; U[i1*m+j1, col] = -1
            V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
            W[k2*n+i2, col] = 1
            col += 1
            # M7 = (A01 - A11)(B10 + B11)
            U[i1*m+j2, col] = 1; U[i2*m+j2, col] = -1
            V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = 1
            col += 1

        def add_outer(i, j, k):
            nonlocal col
            U[i*m+j, col] = 1
            V[j*p+k, col] = 1
            W[k*n+i, col] = 1
            col += 1

        # 2x2 * 2x2 Strassen blocks for j in {0,1}, k in {0,1} and k in {2,3}
        add_strassen(0, 1, 0, 1, 0, 1)
        add_strassen(0, 1, 0, 1, 2, 3)
        # Schoolbook for remaining
        # A[:,2] * B[2, 0:2]
        for i in range(n):
            for k in range(2):
                add_outer(i, 2, k)
        # A[:,0:2] * B[0:2, 4]
        for j in range(2):
            for i in range(n):
                add_outer(i, j, 4)
        # A[:,2] * B[2, 2:5]
        for i in range(n):
            for k in range(2, 5):
                add_outer(i, 2, k)

        assert col == rank, f"col={col} != {rank}"
        return U, V, W

    def check_decomposition(U, V, W, T):
        T_rec = np.einsum('ir,vr,wr->ivw', U, V, W, optimize=True)
        return np.max(np.abs(T_rec - T)) < 1e-6

    # Build rank-28 and verify
    U28, V28, W28 = build_strassen_28()
    valid_28 = check_decomposition(U28, V28, W28, T)

    if not valid_28:
        # Fallback to schoolbook rank 30
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = 28
    best_result = (U28, V28, W28)

    # Try to find a lower rank decomposition using alternating minimization
    # We will try rank 27 and 26
    for target_rank in [27, 26]:
        if time_left() < 30:
            break
            
        found = False
        for attempt in range(10):
            if time_left() < 10:
                break
                
            # Initialize from best known or random
            if target_rank < 28:
                U = U28[:, :target_rank].copy()
                V = V28[:, :target_rank].copy()
                W = W28[:, :target_rank].copy()
                # Add noise to escape local minima
                U += 0.5 * np.random.randn(nm, target_rank)
                V += 0.5 * np.random.randn(mp, target_rank)
                W += 0.5 * np.random.randn(pn, target_rank)
            else:
                U = np.random.randn(nm, target_rank)
                V = np.random.randn(mp, target_rank)
                W = np.random.randn(pn, target_rank)

            # Alternating minimization
            for it in range(100):
                if time_left() < 5:
                    break
                    
                # Solve for W
                # T[i,j,w] = sum_r U[i,r] V[j,r] W[w,r]
                # For fixed w, T[:, :, w] = sum_r (U[:,r] V[:,r]^T) W[w,r]
                # Let M_r = U[:,r] V[:,r]^T
                # T_w = sum_r M_r W[w,r]
                # Vectorize: T_w_flat = sum_r vec(M_r) W[w,r]
                
                B = np.zeros((nm * mp, target_rank))
                for r in range(target_rank):
                    B[:, r] = np.outer(U[:, r], V[:, r]).flatten()
                
                W_new = np.zeros((pn, target_rank))
                for w in range(pn):
                    b_w = T[:, :, w].flatten()
                    try:
                        x, _, _, _ = np.linalg.lstsq(B, b_w, rcond=None)
                        W_new[w, :] = x
                    except:
                        break
                
                W = W_new

                # Solve for U
                C = np.zeros((mp * pn, target_rank))
                for r in range(target_rank):
                    C[:, r] = np.outer(V[:, r], W[:, r]).flatten()
                
                U_new = np.zeros((nm, target_rank))
                for i in range(nm):
                    b_i = T[i, :, :].flatten()
                    try:
                        x, _, _, _ = np.linalg.lstsq(C, b_i, rcond=None)
                        U_new[i, :] = x
                    except:
                        break
                U = U_new

                # Solve for V
                D = np.zeros((nm * pn, target_rank))
                for r in range(target_rank):
                    D[:, r] = np.outer(U[:, r], W[:, r]).flatten()
                
                V_new = np.zeros((mp, target_rank))
                for v in range(mp):
                    b_v = T[:, v, :].flatten()
                    try:
                        x, _, _, _ = np.linalg.lstsq(D, b_v, rcond=None)
                        V_new[v, :] = x
                    except:
                        break
                V = V_new

            # Check and snap
            if check_decomposition(U, V, W, T):
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                if check_decomposition(U_snap, V_snap, W_snap, T):
                    best_rank = target_rank
                    best_result = (U_snap, V_snap, W_snap)
                    found = True
                    break

    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
