import time
import numpy as np
from fractions import Fraction
from itertools import product
from scipy.optimize import least_squares
from scipy.linalg import lstsq

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 4, 4, 7

    # ------------------------------------------------------------------
    # Step 1: Build a verified rank-97 (seed) decomposition.
    # ------------------------------------------------------------------
    def build_rank97():
        # 2x2 Strassen
        U2 = np.zeros((4, 7))
        V2 = np.zeros((4, 7))
        W2 = np.zeros((4, 7))

        def g(i, j):
            return i * 2 + j

        # r=0: M1 = (A00+A11)(B00+B11)
        U2[g(0,0),0]=1; U2[g(1,1),0]=1
        V2[g(0,0),0]=1; V2[g(1,1),0]=1
        W2[g(0,0),0]=1; W2[g(1,1),0]=1

        # r=1: M2 = (A10+A11)B00
        U2[g(1,0),1]=1; U2[g(1,1),1]=1
        V2[g(0,0),1]=1
        W2[g(1,0),1]=1; W2[g(1,1),1]=-1

        # r=2: M3 = A00(B10-B11)
        U2[g(0,0),2]=1
        V2[g(1,0),2]=1; V2[g(1,1),2]=-1
        W2[g(0,1),2]=1; W2[g(1,1),2]=1

        # r=3: M4 = A11(B01-B00)
        U2[g(1,1),3]=1
        V2[g(0,1),3]=1; V2[g(0,0),3]=-1
        W2[g(0,0),3]=1; W2[g(1,0),3]=1

        # r=4: M5 = (A00+A01)B11
        U2[g(0,0),4]=1; U2[g(0,1),4]=1
        V2[g(1,1),4]=1
        W2[g(0,0),4]=-1; W2[g(0,1),4]=1

        # r=5: M6 = (A10-A00)(B00+B01)
        U2[g(1,0),5]=1; U2[g(0,0),5]=-1
        V2[g(0,0),5]=1; V2[g(0,1),5]=1
        W2[g(1,1),5]=1

        # r=6: M7 = (A01-A11)(B10+B11)
        U2[g(0,1),6]=1; U2[g(1,1),6]=-1
        V2[g(1,0),6]=1; V2[g(1,1),6]=1
        W2[g(0,0),6]=1

        R44 = 49
        U44 = np.zeros((16, R44))
        V44 = np.zeros((16, R44))
        W44 = np.zeros((16, R44))

        for r1 in range(7):
            for r2 in range(7):
                r = r1*7 + r2
                nz_u1 = np.nonzero(U2[:,r1])[0]
                nz_u2 = np.nonzero(U2[:,r2])[0]
                nz_v1 = np.nonzero(V2[:,r1])[0]
                nz_v2 = np.nonzero(V2[:,r2])[0]
                nz_w1 = np.nonzero(W2[:,r1])[0]
                nz_w2 = np.nonzero(W2[:,r2])[0]

                for a in nz_u1:
                    for b in nz_u2:
                        i1, j1 = a//2, a%2
                        i2, j2 = b//2, b%2
                        i = i1*2 + i2
                        j = j1*2 + j2
                        U44[i*4+j, r] += U2[a,r1]*U2[b,r2]

                for a in nz_v1:
                    for b in nz_v2:
                        j1, k1 = a//2, a%2
                        j2, k2 = b//2, b%2
                        j = j1*2 + j2
                        k = k1*2 + k2
                        V44[j*4+k, r] += V2[a,r1]*V2[b,r2]

                for a in nz_w1:
                    for b in nz_w2:
                        i1, k1 = a//2, a%2
                        i2, k2 = b//2, b%2
                        i = i1*2 + i2
                        k = k1*2 + k2
                        W44[k*4+i, r] += W2[a,r1]*W2[b,r2]

        # Schoolbook 4x4 * 4x3
        R43 = 48
        U43 = np.zeros((16, R43))
        V43 = np.zeros((12, R43))
        W43 = np.zeros((12, R43))
        c = 0
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    U43[i*4+j, c] = 1
                    V43[j*3+k, c] = 1
                    W43[k*4+i, c] = 1
                    c += 1

        R_total = R44 + R43
        U = np.zeros((16, R_total))
        V = np.zeros((28, R_total))
        W = np.zeros((28, R_total))

        U[:, 0:49] = U44
        for j in range(4):
            for k in range(4):
                V[j*7+k, 0:49] = V44[j*4+k, :]
        W[0:16, 0:49] = W44

        U[:, 49:97] = U43
        for j in range(4):
            for kl in range(3):
                V[j*7+(4+kl), 49:97] = V43[j*3+kl, :]
        for kl in range(3):
            for i in range(4):
                W[(4+kl)*4+i, 49:97] = W43[kl*4+i, :]

        return U, V, W

    U97, V97, W97 = build_rank97()

    def verify(U, V, W):
        # target: C = A B, A is 4x4, B is 4x7
        # U shape: 16 x R  (i*4+j)
        # V shape: 28 x R  (j*7+k)
        # W shape: 28 x R  (k*4+i)
        target = set()
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    target.add((i*4+j, j*7+k, k*4+i))

        Un = np.asarray(U, dtype=float)
        Vn = np.asarray(V, dtype=float)
        Wn = np.asarray(W, dtype=float)

        # Check target entries are 1
        for (u, v, w) in target:
            val = float(np.dot(Un[u]*Vn[v], Wn[w]))
            if abs(val - 1.0) > 1e-9:
                return False

        # Check non-target entries are 0
        for u in range(16):
            for v in range(28):
                for w in range(28):
                    if (u, v, w) in target:
                        continue
                    val = float(np.dot(Un[u]*Vn[v], Wn[w]))
                    if abs(val) > 1e-9:
                        return False
        return True

    if verify(U97, V97, W97):
        result = (U97, V97, W97)
    else:
        # Fallback to pure schoolbook rank-112
        R = 112
        U = np.zeros((16, R))
        V = np.zeros((28, R))
        W = np.zeros((28, R))
        c = 0
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    U[i*4+j, c] = 1
                    V[j*7+k, c] = 1
                    W[k*4+i, c] = 1
                    c += 1
        result = (U, V, W)

    # ------------------------------------------------------------------
    # Step 2: Try to improve via numerical optimization to find a
    #         rank < 97 decomposition with integer/half-integer entries.
    # ------------------------------------------------------------------
    # Build the target tensor (sparse)
    T = np.zeros((16, 28, 28))
    for i in range(4):
        for j in range(4):
            for k in range(7):
                T[i*4+j, j*7+k, k*4+i] = 1.0

    # Try rank 96, 95, ... down to a floor. Each attempt is time-bounded.
    # We use a structured initialization: take the rank-97 solution, drop one
    # rank column, and try to re-optimize to close the gap.
    #
    # However, a cleaner approach: for each target rank R, initialize U,V,W
    # by projecting the rank-97 solution to rank R (keep first R columns),
    # then run least_squares to minimize ||U V^T W^T - T||_F, then round to
    # nearest 0.5 and verify.
    #
    # This is expensive; we only try a few ranks with a tight time budget.
    best_result = result

    for R_target in [96, 95, 94]:
        if time.time() > deadline - 30:
            break

        # Initialize from rank-97, keeping first R_target columns
        U_init = U97[:, :R_target].astype(float).copy()
        V_init = V97[:, :R_target].astype(float).copy()
        W_init = W97[:, :R_target].astype(float).copy()

        # Add small random perturbation to break symmetry
        np.random.seed(R_target)
        U_init += np.random.randn(16, R_target) * 0.01
        V_init += np.random.randn(28, R_target) * 0.01
        W_init += np.random.randn(28, R_target) * 0.01

        def residuals(x):
            U_p = x[0:16*R_target].reshape(16, R_target)
            V_p = x[16*R_target:16*R_target+28*R_target].reshape(28, R_target)
            W_p = x[16*R_target+28*R_target:].reshape(28, R_target)
            T_recon = np.einsum('ur,vr,wr->uvw', U_p, V_p, W_p)
            return (T_recon - T).ravel()

        x0 = np.concatenate([U_init.ravel(), V_init.ravel(), W_init.ravel()])

        try:
            t0 = time.time()
            res = least_squares(residuals, x0, method='lm', max_nfev=300,
                                 xtol=1e-10, ftol=1e-10)
            elapsed = time.time() - t0

            if res.cost < 1e-8:
                U_opt = res.x[0:16*R_target].reshape(16, R_target)
                V_opt = res.x[16*R_target:16*R_target+28*R_target].reshape(28, R_target)
                W_opt = res.x[16*R_target+28*R_target:].reshape(28, R_target)

                # Round to nearest 0.5
                U_r = np.round(U_opt * 2) / 2
                V_r = np.round(V_opt * 2) / 2
                W_r = np.round(W_opt * 2) / 2

                T_check = np.einsum('ur,vr,wr->uvw', U_r, V_r, W_r)
                if np.allclose(T_check, T, atol=1e-9):
                    if verify(U_r, V_r, W_r):
                        best_result = (U_r, V_r, W_r)
                        break  # Found a valid lower rank, stop
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Step 3: Return best result
    # ------------------------------------------------------------------
    U_best, V_best, W_best = best_result
    return {
        "U": U_best.tolist(),
        "V": V_best.tolist(),
        "W": W_best.tolist()
    }
