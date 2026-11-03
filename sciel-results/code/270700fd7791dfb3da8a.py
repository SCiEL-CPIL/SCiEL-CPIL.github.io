"""Matrix multiplication 3,4,4 – rank-minimal exact tensor decomposition."""

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    deadline = time.time() + 110.0

    # Schoolbook construction
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    def to_list(mat):
        return mat.tolist()

    # Try to find a lower-rank decomposition using ALS
    def als_search(rank, max_iters=50, n_trials=5, tol=1e-8):
        for trial in range(n_trials):
            if time.time() > deadline:
                return None
            rng = random.Random(42 + trial)
            # Random initialization
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1

            for it in range(max_iters):
                if time.time() > deadline:
                    return None

                # Update U: for each row a = i*m+j
                for a in range(n * m):
                    i = a // m
                    j = a % m
                    # The tensor slice T[:, j, :] = W[:, k*n+i] for each k
                    # We want: sum_r U[a,r] * (V[:,r] outer W[:,r]) to match T
                    # For row a of U, we solve: U[a,:] * (V^T W elementwise) = T_a
                    # T_a[k] = 1 if k == j else 0 ... actually T[i,j,k] = 1 for all i,j,k in schoolbook
                    # Wait, the tensor for matrix multiplication is T[i,j,k] = 1 (scalar 1)
                    # The decomposition is: T = sum_r U[:,r] outer V[:,r] outer W[:,r]
                    # So T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    
                    # To update U[a, :]:
                    # T[i,j,k] = sum_r U[a,r] * V[j*p+k, r] * W[k*n+i, r]
                    # Let M_a = V[j*p:(j+1)*p, :] * W[:, :] (elementwise, then reshape)
                    # Actually for fixed i,j: T[i,j,k] for k=0..p-1
                    # T[i,j,k] = sum_r U[a,r] * V[j*p+k, r] * W[k*n+i, r]
                    
                    # Build matrix B of shape (p, rank): B[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    # Then T_a = U[a,:] @ B^T
                    # So U[a,:] = T_a @ (B^T)^+  where T_a is the known target row
                    
                    B = np.zeros((p, rank))
                    for k in range(p):
                        B[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    # Target: T_a[k] = 1 for all k (since it's matrix mult tensor)
                    target = np.ones(p)
                    # Solve U[a,:] @ B^T = target => U[a,:] = target @ pinv(B^T)
                    if rank <= p:
                        # U[a,:] = target @ (B^T)^+ 
                        # B^T has shape (rank, p), so (B^T)^+ has shape (p, rank)
                        # target is (p,), result is (rank,)
                        try:
                            U[a, :] = target @ np.linalg.pinv(B.T)
                        except:
                            pass
                    else:
                        # Use least squares: minimize ||U[a,:] @ B^T - target||
                        # U[a,:] @ B^T = target  =>  B * U[a,:]^T = target
                        try:
                            U[a, :], _, _, _ = np.linalg.lstsq(B, target, rcond=None)
                        except:
                            pass

                # Update V: for each row b = j*p+k
                for b in range(m * p):
                    j = b // p
                    k = b % p
                    # T[i,j,k] = sum_r U[i*m+j, r] * V[b, r] * W[k*n+i, r]
                    # For fixed j,k: sum over i
                    # T[i,j,k] = V[b,r] * (U[i*m+j, r] * W[k*n+i, r])
                    # Build C of shape (n, rank): C[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    C = np.zeros((n, rank))
                    for i in range(n):
                        C[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    target = np.ones(n)
                    # V[b,:] @ C^T = target
                    if rank <= n:
                        try:
                            V[b, :] = target @ np.linalg.pinv(C.T)
                        except:
                            pass
                    else:
                        try:
                            V[b, :], _, _, _ = np.linalg.lstsq(C, target, rcond=None)
                        except:
                            pass

                # Update W: for each row c = k*n+i
                for c in range(p * n):
                    k = c // n
                    i = c % n
                    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[c, r]
                    # For fixed k,i: sum over j
                    # T[i,j,k] = W[c,r] * (U[i*m+j, r] * V[j*p+k, r])
                    D = np.zeros((m, rank))
                    for j in range(m):
                        D[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    target = np.ones(m)
                    # W[c,:] @ D^T = target
                    if rank <= m:
                        try:
                            W[c, :] = target @ np.linalg.pinv(D.T)
                        except:
                            pass
                    else:
                        try:
                            W[c, :], _, _, _ = np.linalg.lstsq(D, target, rcond=None)
                        except:
                            pass

                # Check error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                            err += (val - 1.0) ** 2
                if err < tol:
                    return U, V, W

        return None

    # Try ranks from 40 down to 25
    best_result = None
    for rank in range(40, 24, -1):
        if time.time() > deadline:
            break
        result = als_search(rank, max_iters=30, n_trials=3)
        if result is not None:
            best_result = (rank, result)
            break

    if best_result is not None:
        rank, (U, V, W) = best_result
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    else:
        return {"U": to_list(U_school), "V": to_list(V_school), "W": to_list(W_school)}
# EVOLVE-BLOCK-END
