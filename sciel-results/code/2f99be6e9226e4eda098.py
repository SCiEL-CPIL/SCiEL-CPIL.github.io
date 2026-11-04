import time
import numpy as np

def construct():
    n, m, p = 2, 4, 4
    nm, mp, pn = n * m, m * p, p * n  # 8, 16, 8
    start_time = time.time()
    deadline = start_time + 130.0

    # Rank 28 solution via 4 independent Strassen 2x2 blocks
    def build_rank28():
        rank = 28
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]

        def add_strassen(c0, xr, yr, zr):
            # M1: (x0+x3)(y0+y3) -> z0+z3
            U[xr[0]][c0+0] += 1; U[xr[3]][c0+0] += 1
            V[yr[0]][c0+0] += 1; V[yr[3]][c0+0] += 1
            W[zr[0]][c0+0] += 1; W[zr[3]][c0+0] += 1
            # M2: (x2+x3)(y0) -> z2-z3
            U[xr[2]][c0+1] += 1; U[xr[3]][c0+1] += 1
            V[yr[0]][c0+1] += 1
            W[zr[2]][c0+1] += 1; W[zr[3]][c0+1] -= 1
            # M3: (x0)(y1-y3) -> z1+z3
            U[xr[0]][c0+2] += 1
            V[yr[1]][c0+2] += 1; V[yr[3]][c0+2] -= 1
            W[zr[1]][c0+2] += 1; W[zr[3]][c0+2] += 1
            # M4: (x3)(y2-y0) -> z0+z2
            U[xr[3]][c0+3] += 1
            V[yr[2]][c0+3] += 1; V[yr[0]][c0+3] -= 1
            W[zr[0]][c0+3] += 1; W[zr[2]][c0+3] += 1
            # M5: (x0+x1)(y3) -> -z0+z1
            U[xr[0]][c0+4] += 1; U[xr[1]][c0+4] += 1
            V[yr[3]][c0+4] += 1
            W[zr[0]][c0+4] -= 1; W[zr[1]][c0+4] += 1
            # M6: (x2-x0)(y0+y1) -> z3
            U[xr[2]][c0+5] += 1; U[xr[0]][c0+5] -= 1
            V[yr[0]][c0+5] += 1; V[yr[1]][c0+5] += 1
            W[zr[3]][c0+5] += 1
            # M7: (x1-x3)(y2+y3) -> z0
            U[xr[1]][c0+6] += 1; U[xr[3]][c0+6] -= 1
            V[yr[2]][c0+6] += 1; V[yr[3]][c0+6] += 1
            W[zr[0]][c0+6] += 1

        add_strassen(0,  [0,1,4,5], [0,1,4,5], [0,2,1,3])   # A_L*B_TL -> C_L
        add_strassen(7,  [2,3,6,7], [8,9,12,13], [0,2,1,3])  # A_R*B_BL -> C_L
        add_strassen(14, [0,1,4,5], [2,3,6,7], [4,6,5,7])    # A_L*B_TR -> C_R
        add_strassen(21, [2,3,6,7], [10,11,14,15], [4,6,5,7])# A_R*B_BR -> C_R
        return U, V, W

    U28, V28, W28 = build_rank28()

    # Verify correctness of rank 28
    def verify(U, V, W):
        r = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        # T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
        # Expected: T[i,j,k] = 1 if i//4*4 + j//4 == k, else 0
        # Actually for matrix mult: (AB)[k, i] = sum_j A[i,j]*B[j,k]
        # U index i (row of A), V index j (col of A = row of B), W index k (col of B)
        # W[k*n+i] convention: W index = k*n + i where i is output col, k is output row
        # A is n x m = 2x4, B is m x p = 4x4, C is n x p = 2x4
        # C[i, k] = sum_j A[i, j] B[j, k]
        # Tensor: T[i, j, k] = 1 for all i,j,k (it's the identity of matmul)
        # U[i, r] * V[j, r] * W[k*n+i, r] should sum to 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(r):
                        s += Ua[i, r] * Va[j, r] * Wa[k*n + i, r]
                    if abs(s - 1.0) > 1e-10:
                        return False, (i, j, k, s)
        return True, None

    ok, info = verify(U28, V28, W28)
    if not ok:
        # Fallback: return schoolbook rank 32 (always valid)
        rank = 32
        U = [[0]*rank for _ in range(nm)]
        V = [[0]*rank for _ in range(mp)]
        W = [[0]*rank for _ in range(pn)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*j + j*p + k
                    U[i][r] = 1
                    V[j][r] = 1
                    W[k*n+i][r] = 1
        return {"U": U, "V": V, "W": W}

    # Now try to find rank 27 by searching for a shared decomposition
    # Strategy: Try to find rank-13 for "two blocks sharing left factor"
    # Block pair 1: A_L*(B_TL + B_TR) with outputs C_L and C_R
    # This is equivalent to: given A (2x2), B1 (2x2), B2 (2x2),
    # compute A*B1 and A*B2 simultaneously.
    # Tensor: T1 in R^4 x R^4 x R^4 (A, B1, C1) + T2 in R^4 x R^4 x R^4 (A, B2, C2)
    # Total: R^4 x R^8 x R^8
    # Standard: 7+7=14. Can we do 13?

    # Try a structured approach: look for a rank-13 solution
    # by allowing one shared term between the two Strassen decompositions.
    # This is a non-linear optimization problem. Let's use a local search.

    if time.time() > deadline - 30:
        return {"U": U28, "V": V28, "W": W28}

    # Attempt: use scipy to try to find a rank-27 solution via
    # minimizing the Frobenius norm of (T - sum of rank-1 tensors)
    # where T is the 8x16x8 matmul tensor.
    # This is a hard non-convex problem. Let's try a few random restarts.

    # Build the full tensor T[i,j,k] = 1 for all i in [0,8), j in [0,16), k in [0,8)
    # Wait - the matmul tensor has T[i,j,k]=1 for all valid (i,j,k) where
    # i indexes A rows (0..7), j indexes the shared dimension (0..15), k indexes C cols (0..7)
    # Actually the matmul tensor for (n,m,p) has entries T[i,j,k] = delta_{i,j,k}
    # No - the matmul tensor is T[i,j,k] = 1 for ALL i,j,k. It's the identity tensor.
    # C[i,k] = sum_j A[i,j] B[j,k], so the tensor is all-ones.

    # Actually no. The tensor T is defined by: for any A, B,
    # C = T(A,B) means C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    # For standard matmul, T[i,j,k] = 1 for all i,j,k.

    # So T is the all-ones tensor of shape (8, 16, 8).

    # A rank-r decomposition: T = sum_{l=1}^r u_l ⊗ v_l ⊗ w_l
    # where u_l in R^8, v_l in R^16, w_l in R^8.
    # T[i,j,k] = sum_l U[i,l] V[j,l] W[k,l] = 1 for all i,j,k.

    # This means: for all (i,j,k), sum_l U[i,l] V[j,l] W[k,l] = 1.
    # Equivalently, for each k, the matrix M_k = sum_l V[:,l] * (U * W[k,:])^T ... no.
    # For fixed k: sum_l (U[:,l] * W[k,l]) * V[:,l]^T = ones_{8x16}
    # i.e., sum_l W[k,l] * (U[:,l] ⊗ V[:,l]) = ones_{8x16} (reshaped)

    # Let's try to find rank 27 via optimization.
    # Variables: U (8x27), V (16x27), W (8x27)
    # Objective: sum_{i,j,k} (sum_l U[i,l]V[j,l]W[k,l] - 1)^2

    # This has 8*27 + 16*27 + 8*27 = 216*3 = 648 variables.
    # The objective has 8*16*8 = 1024 terms.

    # Let's try a few random initializations near the rank-28 solution.
    # We can project the rank-28 solution to rank 27 by zeroing one column
    # and see if we can optimize back to satisfy the constraints.

    import random
    best_rank = 28
    best_U, best_V, best_W = U28, V28, W28

    # Try: take rank-28 solution, zero out column r0, then optimize
    # the remaining 27 columns to minimize error.
    for attempt in range(10):
        if time.time() > deadline - 20:
            break

        r0 = attempt * 2  # try different columns to zero
        U_init = np.array(U28, dtype=float)[:, :r0 + 1] + np.array(U28, dtype=float)[:, r0 + 1:]
        # Actually, just drop column r0
        U_init = np.concatenate([np.array(U28, dtype=float)[:, :r0], np.array(U28, dtype=float)[:, r0+1:]], axis=1)
        V_init = np.concatenate([np.array(V28, dtype=float)[:, :r0], np.array(V28, dtype=float)[:, r0+1:]], axis=1)
        W_init = np.concatenate([np.array(W28, dtype=float)[:, :r0], np.array(W28, dtype=float)[:, r0+1:]], axis=1)

        # Add small random perturbation
        U_init += 0.01 * np.random.randn(8, 27)
        V_init += 0.01 * np.random.randn(16, 27)
        W_init += 0.01 * np.random.randn(8, 27)

        # Use scipy optimize
        try:
            from scipy.optimize import minimize

            def objective(params):
                U_flat, V_flat, W_flat = params[:216], params[216:432], params[432:648]
                Uf = U_flat.reshape(8, 27)
                Vf = V_flat.reshape(16, 27)
                Wf = W_flat.reshape(8, 27)
                # Compute T[i,j,k] = sum_l Uf[i,l]*Vf[j,l]*Wf[k,l]
                # Vectorize: for each l, add Uf[:,l] ⊗ Vf[:,l] ⊗ Wf[:,l]
                # This is expensive. Use einsum.
                # T = np.einsum('il,jl,kl->ijk', Uf, Vf, Wf)
                # That's 8*16*8*27 operations = 27648, fine.
                T_approx = np.einsum('il,jl,kl->ijk', Uf, Vf, Wf)
                err = np.sum((T_approx - 1.0)**2)
                return err

            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            result = minimize(objective, x0, method='L-BFGS-B',
                            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})

            if result.fun < 1e-8:
                # Found an exact rank-27 solution!
                Uf = result.x[:216].reshape(8, 27)
                Vf = result.x[216:432].reshape(16, 27)
                Wf = result.x[432:648].reshape(8, 27)

                # Check if entries are close to integer/half-integer
                all_2x = np.concatenate([Uf.flatten()*2, Vf.flatten()*2, Wf.flatten()*2])
                rounded = np.round(all_2x)
                if np.max(np.abs(all_2x - rounded)) < 1e-6:
                    # Convert to half-integers
                    Uf = rounded[:216].reshape(8, 27) / 2.0
                    Vf = rounded[216:432].reshape(16, 27) / 2.0
                    Wf = rounded[432:648].reshape(8, 27) / 2.0
                    best_rank = 27
                    best_U = Uf.tolist()
                    best_V = Vf.tolist()
                    best_W = Wf.tolist()
                    break
                else:
                    # Not half-integer, skip
                    pass

        except Exception:
            pass

    if best_rank == 27:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U28, "V": V28, "W": W28}
