"""Rewritten program for rank-minimal exact tensor decomposition of 3x3 * 3x7 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n, m, p = 3, 3, 7
    nU = n * m  # 9
    nV = m * p  # 21
    nW = p * n  # 21

    # The tensor T_{(i,j),(j',k),(k',i')} = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    # We want T = sum_r U_r \otimes V_r \otimes W_r
    # where U is nU x rank, V is nV x rank, W is nW x rank

    # Schoolbook decomposition: rank = n*m*p = 63
    rank_schoolbook = n * m * p
    U_school = np.zeros((nU, rank_schoolbook))
    V_school = np.zeros((nV, rank_schoolbook))
    W_school = np.zeros((nW, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    best_rank = rank_schoolbook
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()

    def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        """Verify that the decomposition is correct."""
        rank = U.shape[1]
        nU, nV, nW = n * m, m * p, p * n

        # Build the full tensor from the decomposition
        # T_{a,b,c} = sum_r U[a,r] * V[b,r] * W[c,r]
        # where a = i*m+j, b = j'*p+k, c = k'*n+i'
        # Expected: T_{a,b,c} = 1 if a=(i,j), b=(j,k), c=(k,i) for same (i,j,k), else 0
        # i.e., T_{i*m+j, j*p+k, k*n+i} = 1 for all valid (i,j,k), else 0

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    # This entry should be 1
                    val = 0.0
                    for r in range(rank):
                        val += U[a, r] * V[b, r] * W[c, r]
                    if abs(val - 1.0) > tol:
                        return False

        # Check that all other entries are 0
        # There are nU*nV*nW - n*m*p entries that should be 0
        # But checking all of them is expensive. Let's check a sample.
        # Actually, let's check all entries for correctness since nU*nV*nW = 9*21*21 = 3969
        for a in range(nU):
            i = a // m
            j = a % m
            for b in range(nV):
                j2 = b // p
                k2 = b % p
                for c in range(nW):
                    k = c // n
                    i2 = c % n
                    # This entry should be 1 if (i,j,k) = (i2,j2,k2), else 0
                    expected = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                    val = 0.0
                    for r in range(rank):
                        val += U[a, r] * V[b, r] * W[c, r]
                    if abs(val - expected) > tol:
                        return False
        return True

    def try_als(U_init, V_init, W_init, n, m, p, max_iter=100, tol=1e-12):
        """Try to refine a rank-r decomposition using ALS. Returns (success, U, V, W)."""
        rank = U_init.shape[1]
        nU, nV, nW = n * m, m * p, p * n
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()

        for iteration in range(max_iter):
            if time.time() > deadline - 20.0:
                return False, U, V, W

            # Update U: for each row a, solve U[a,:] = argmin sum_{b,c} (T_{a,b,c} - sum_r U[a,r]*V[b,r]*W[c,r])^2
            # This is a linear system: (V^T W_c * V W_c^T) U[a,:]^T = sum_c T_{a,*,c} V W_c^T
            # More precisely, fix V and W, for each a:
            # min_U sum_{b,c} (T_{a,b,c} - (U[a,:] \otimes V[b,:] \otimes W[c,:]))^2
            # = min_U sum_{b,c} (T_{a,b,c} - U[a,:]^T (V[b,:] \otimes W[c,:]))^2
            # Let M_{a,b,c} = V[b,:] \otimes W[c,:] (a vector of length rank)
            # Then we want U[a,:] to minimize sum_{b,c} (T_{a,b,c} - U[a,:]^T M_{a,b,c})^2
            # This is a least squares problem: U[a,:] M^T = T_{a,:}^T
            # where M is a (nV*nW) x rank matrix with rows M_{a,b,c} = V[b,:] \otimes W[c,:]
            # and T_{a,:} is the vector of T_{a,b,c} over all b,c.

            # Build the matrix M: for each (b,c), M_{b,c} = V[b,:] \otimes W[c,:]
            # M has shape (nV*nW, rank)
            M = np.zeros((nV * nW, rank))
            for b in range(nV):
                for c in range(nW):
                    idx = b * nW + c
                    M[idx, :] = np.kron(V[b, :], W[c, :])

            # For each a, T_a is a vector of length nV*nW
            # T_{a,b,c} = 1 if a=(i,j), b=(j,k), c=(k,i) for the same (i,j,k), else 0
            # i.e., T_{i*m+j, j*p+k, k*n+i} = 1

            # We need to solve: U[a,:] M^T = T_a^T
            # Using least squares: U[a,:] = T_a M (M^T M)^{-1}
            # Or using np.linalg.lstsq

            MMT = M.T @ M  # rank x rank
            # Add small regularization for numerical stability
            MMT += 1e-10 * np.eye(rank)

            for a in range(nU):
                i = a // m
                j = a % m
                # Build T_a: vector of length nV*nW
                T_a = np.zeros(nV * nW)
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    T_a[b * nW + c] = 1.0
                # Solve: U[a,:] M^T = T_a^T
                # => U[a,:] = T_a M (M^T M)^{-1}
                # Using lstsq:
                try:
                    U[a, :], _, _, _ = np.linalg.lstsq(M.T, T_a, rcond=None)
                except np.linalg.LinAlgError:
                    return False, U, V, W

            # Update V: similar, fix U and W
            # For each b: V[b,:] = argmin sum_{a,c} (T_{a,b,c} - U[a,:]^T (V[b,:] \otimes W[c,:]))^2
            # Let N_{a,c} = U[a,:] \otimes W[c,:]
            # N has shape (nU*nW, rank)
            N = np.zeros((nU * nW, rank))
            for a in range(nU):
                for c in range(nW):
                    idx = a * nW + c
                    N[idx, :] = np.kron(U[a, :], W[c, :])

            NNt = N.T @ N
            NNt += 1e-10 * np.eye(rank)

            for b in range(nV):
                j2 = b // p
                k2 = b % p
                T_b = np.zeros(nU * nW)
                for i in range(n):
                    a = i * m + j2
                    c = k2 * n + i
                    T_b[a * nW + c] = 1.0
                try:
                    V[b, :], _, _, _ = np.linalg.lstsq(N.T, T_b, rcond=None)
                except np.linalg.LinAlgError:
                    return False, U, V, W

            # Update W: similar, fix U and V
            # For each c: W[c,:] = argmin sum_{a,b} (T_{a,b,c} - U[a,:]^T (V[b,:] \otimes W[c,:]))^2
            # Let P_{a,b} = U[a,:] \otimes V[b,:]
            # P has shape (nU*nV, rank)
            P = np.zeros((nU * nV, rank))
            for a in range(nU):
                for b in range(nV):
                    idx = a * nV + b
                    P[idx, :] = np.kron(U[a, :], V[b, :])

            PPt = P.T @ P
            PPt += 1e-10 * np.eye(rank)

            for c in range(nW):
                k = c // n
                i = c % n
                T_c = np.zeros(nU * nV)
                for j in range(m):
                    a = i * m + j
                    b = j * p + k
                    T_c[a * nV + b] = 1.0
                try:
                    W[c, :], _, _, _ = np.linalg.lstsq(P.T, T_c, rcond=None)
                except np.linalg.LinAlgError:
                    return False, U, V, W

            # Check convergence
            if verify_decomposition(U, V, W, n, m, p, tol=1e-8):
                # Verify with tighter tolerance
                if verify_decomposition(U, V, W, n, m, p, tol=1e-12):
                    return True, U, V, W

        # Final check
        if verify_decomposition(U, V, W, n, m, p, tol=1e-10):
            return True, U, V, W
        return False, U, V, W

    # Try ranks from the theoretical lower bound upward
    # Lower bound: max(nU, nV, nW) = max(9, 21, 21) = 21
    # But for matrix multiplication tensors, the rank is known to be higher.
    # For 3x3*3x3, best known upper bound is 23, lower bound is 19.
    # For 3x3*3x7, let's try from 21 upward.

    for target_rank in range(21, rank_schoolbook + 1):
        if time.time() > deadline - 25.0:
            break

        # Try multiple random initializations
        for attempt in range(10):
            if time.time() > deadline - 25.0:
                break

            np.random.seed(42 + target_rank * 100 + attempt)
            U_init = np.random.randn(nU, target_rank)
            V_init = np.random.randn(nV, target_rank)
            W_init = np.random.randn(nW, target_rank)

            success, U, V, W = try_als(U_init, V_init, W_init, n, m, p, max_iter=50, tol=1e-12)

            if success:
                best_rank = target_rank
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                break  # Found a valid decomposition at this rank, try lower next

        # If we found a decomposition at this rank, we can stop trying higher ranks
        # but we should try to go lower. Actually we're going from low to high,
        # so if we find one, the first success is the lowest rank.
        if best_rank < rank_schoolbook:
            break

    # Convert to lists for return
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()

    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
