import numpy as np
import time

def construct():
    n, m, p = 2, 3, 8
    nm, mp, pn = n*m, m*p, p*n  # 6, 24, 16

    start_time = time.time()
    deadline = start_time + 115

    # Build the tensor T with non-zero entries
    # T[a, b, d] = 1 if a=i*m+j, b=j*p+k, d=k*n+i for some i,j,k
    nonzero_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                d = k*n + i
                nonzero_entries.append((a, b, d))

    # Initialize with schoolbook rank 48
    best_rank = n*m*p  # 48
    best_U = None
    best_V = None
    best_W = None

    U0 = np.zeros((nm, best_rank))
    V0 = np.zeros((mp, best_rank))
    W0 = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1

    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()

    # Try to reduce rank from 47 down to 1
    for r in range(n*m*p - 1, 0, -1):
        if time.time() > deadline - 15:
            break

        found = False
        for attempt in range(50):
            if time.time() > deadline - 10:
                break

            # Random initialization with small values
            U = np.random.randn(nm, r) * 0.1
            V = np.random.randn(mp, r) * 0.1
            W = np.random.randn(pn, r) * 0.1

            # Alternating minimization
            for iteration in range(200):
                if time.time() > deadline - 10:
                    break

                # Solve for W given U, V
                M = U @ V.T  # nm x mp
                W_new = np.zeros((pn, r))
                success = True
                for d in range(pn):
                    T_d = np.zeros(nm*mp)
                    for a, b, dd in nonzero_entries:
                        if dd == d:
                            T_d[a*mp + b] = 1.0
                    try:
                        x, _, _, _ = np.linalg.lstsq(M, T_d, rcond=None)
                        W_new[d, :] = x
                    except:
                        success = False
                        break
                if not success:
                    break
                W = W_new

                # Check residual
                residual = 0.0
                for a, b, d in nonzero_entries:
                    val = np.sum(U[a,:] * V[b,:] * W[d,:])
                    if abs(val - 1) > 1e-6:
                        residual += 1

                # Check some zero entries
                zero_residual = 0
                for _ in range(50):
                    a = np.random.randint(0, nm)
                    b = np.random.randint(0, mp)
                    d = np.random.randint(0, pn)
                    is_nonzero = False
                    for na, nb, nd in nonzero_entries:
                        if a == na and b == nb and d == nd:
                            is_nonzero = True
                            break
                    if not is_nonzero:
                        val = np.sum(U[a,:] * V[b,:] * W[d,:])
                        if abs(val) > 1e-6:
                            zero_residual += 1

                if residual == 0 and zero_residual == 0:
                    found = True
                    best_rank = r
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break

                # Solve for U given V, W
                M2 = V @ W.T  # mp x pn
                U_new = np.zeros((nm, r))
                success = True
                for a in range(nm):
                    T_a = np.zeros(mp*pn)
                    for aa, b, d in nonzero_entries:
                        if aa == a:
                            T_a[b*pn + d] = 1.0
                    try:
                        x, _, _, _ = np.linalg.lstsq(M2, T_a, rcond=None)
                        U_new[a, :] = x
                    except:
                        success = False
                        break
                if not success:
                    break
                U = U_new

                # Solve for V given U, W
                M3 = U @ W.T  # nm x pn
                V_new = np.zeros((mp, r))
                success = True
                for b in range(mp):
                    T_b = np.zeros(nm*pn)
                    for a, bb, d in nonzero_entries:
                        if bb == b:
                            T_b[a*pn + d] = 1.0
                    try:
                        x, _, _, _ = np.linalg.lstsq(M3, T_b, rcond=None)
                        V_new[b, :] = x
                    except:
                        success = False
                        break
                if not success:
                    break
                V = V_new

            if found:
                break

        if found:
            break

    # Convert to list format
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
