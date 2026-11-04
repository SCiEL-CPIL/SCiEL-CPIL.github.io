import numpy as np
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 96

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115

    # Build target tensor (sparse, only 96 non-zeros)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Precompute T reshapes ONCE (fixes major performance bug)
    T_U = np.zeros((n * m, m * p * p * n))
    for i in range(n * m):
        for j in range(m * p):
            for k in range(p * n):
                T_U[i, j * p * n + k] = T[i, j, k]

    T_V = np.zeros((m * p, n * m * p * n))
    for i in range(n * m):
        for j in range(m * p):
            for k in range(p * n):
                T_V[j, i * p * n + k] = T[i, j, k]

    T_W = np.zeros((p * n, n * m * m * p))
    for i in range(n * m):
        for j in range(m * p):
            for k in range(p * n):
                T_W[k, i * m * p + j] = T[i, j, k]

    # Schoolbook fallback
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_schoolbook

    def fast_als(rank, U_init, V_init, W_init, max_iters=200, tol=1e-10):
        """Fast ALS using precomputed T reshapes."""
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()

        for it in range(max_iters):
            if time.time() > deadline:
                return False, U, V, W

            # Update U: T_U = U @ M_U^T, where M_U[r, (j,k)] = V[j,r]*W[k,r]
            M_U = np.einsum('jr,kr->rjk', V, W).reshape(rank, m * p * p * n)
            try:
                U_new = np.linalg.lstsq(M_U.T, T_U.T, rcond=None)[0].T
                U = U_new
            except:
                break

            # Update V: T_V = V @ M_V^T, where M_V[r, (i,k)] = U[i,r]*W[k,r]
            M_V = np.einsum('ir,kr->rik', U, W).reshape(rank, n * m * p * n)
            try:
                V_new = np.linalg.lstsq(M_V.T, T_V.T, rcond=None)[0].T
                V = V_new
            except:
                break

            # Update W: T_W = W @ M_W^T, where M_W[r, (i,j)] = U[i,r]*V[j,r]
            M_W = np.einsum('ir,jr->rij', U, V).reshape(rank, n * m * m * p)
            try:
                W_new = np.linalg.lstsq(M_W.T, T_W.T, rcond=None)[0].T
                W = W_new
            except:
                break

            # Check convergence
            S = np.einsum('ir,jr,kr->ijk', U, V, W)
            if np.linalg.norm(T - S) < tol:
                return True, U, V, W

        S = np.einsum('ir,jr,kr->ijk', U, V, W)
        if np.linalg.norm(T - S) < 1e-8:
            return True, U, V, W
        return False, U, V, W

    def try_polish(U, V, W, rank, max_polish_iters=50):
        """High-precision polish pass with tighter tolerance."""
        U_p = U.copy()
        V_p = V.copy()
        W_p = W.copy()
        for _ in range(max_polish_iters):
            if time.time() > deadline:
                break
            M_U = np.einsum('jr,kr->rjk', V_p, W_p).reshape(rank, m * p * p * n)
            try:
                U_p = np.linalg.lstsq(M_U.T, T_U.T, rcond=None)[0].T
            except:
                break
            M_V = np.einsum('ir,kr->rik', U_p, W_p).reshape(rank, n * m * p * n)
            try:
                V_p = np.linalg.lstsq(M_V.T, T_V.T, rcond=None)[0].T
            except:
                break
            M_W = np.einsum('ir,jr->rij', U_p, V_p).reshape(rank, n * m * m * p)
            try:
                W_p = np.linalg.lstsq(M_W.T, T_W.T, rcond=None)[0].T
            except:
                break
            S = np.einsum('ir,jr,kr->ijk', U_p, V_p, W_p)
            if np.linalg.norm(T - S) < 1e-12:
                break
        S = np.einsum('ir,jr,kr->ijk', U_p, V_p, W_p)
        err = np.linalg.norm(T - S)
        if err < 1e-10:
            # Round to half-integers
            U_r = np.round(U_p * 2) / 2
            V_r = np.round(V_p * 2) / 2
            W_r = np.round(W_p * 2) / 2
            S_r = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
            if np.allclose(S_r, T, atol=1e-10):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            # Try rounding to integers
            U_i = np.round(U_p)
            V_i = np.round(V_p)
            W_i = np.round(W_p)
            S_i = np.einsum('ir,jr,kr->ijk', U_i, V_i, W_i)
            if np.allclose(S_i, T, atol=1e-10):
                return {"U": U_i.tolist(), "V": V_i.tolist(), "W": W_i.tolist()}
            return {"U": U_p.tolist(), "V": V_p.tolist(), "W": W_p.tolist()}
        return None

    def kronecker_init(rank, seed):
        """Kronecker-structured init: factor p=8 as 4*2, m=4 as 4*1.
        
        Model: T_{3x4x8} = T_{3x4x4} ⊗ T_{1x1x2} (approximately)
        This gives a structured initialization that may converge faster.
        """
        np.random.seed(seed)
        
        # Base 3x4x4 multiplication tensor (schoolbook rank 48)
        n1, m1, p1 = 3, 4, 4
        # Second factor: 1x1x2 identity-like
        n2, m2, p2 = 1, 1, 2
        
        # Build 3x4x4 schoolbook
        U1 = np.zeros((n1 * m1, n1 * m1 * p1))
        V1 = np.zeros((m1 * p1, n1 * m1 * p1))
        W1 = np.zeros((p1 * n1, n1 * m1 * p1))
        col = 0
        for i in range(n1):
            for j in range(m1):
                for k in range(p1):
                    U1[i * m1 + j, col] = 1
                    V1[j * p1 + k, col] = 1
                    W1[k * n1 + i, col] = 1
                    col += 1
        
        # Build 1x1x2 "identity" tensor
        U2 = np.zeros((n2 * m2, n2 * m2 * p2))
        V2 = np.zeros((m2 * p2, n2 * m2 * p2))
        W2 = np.zeros((p2 * n2, n2 * m2 * p2))
        col2 = 0
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    U2[i * m2 + j, col2] = 1
                    V2[j * p2 + k, col2] = 1
                    W2[k * n2 + i, col2] = 1
                    col2 += 1
        
        # Kronecker product of factor matrices
        # U = U1 ⊗ U2, V = V1 ⊗ V2, W = W1 ⊗ W2
        # But we need to reshape to correct dimensions
        # U1: (12, 48), U2: (1, 2) -> U_kron: (12, 96)
        # V1: (16, 48), V2: (2, 2) -> V_kron: (16, 96)  
        # W1: (12, 48), W2: (2, 2) -> W_kron: (24, 96)
        
        U_kron = np.kron(U1, U2)  # (12*1, 48*2) = (12, 96)
        V_kron = np.kron(V1, V2)  # (16*2, 48*2) = (32, 96)
        W_kron = np.kron(W1, W2)  # (12*2, 48*2) = (24, 96)
        
        # Verify dimensions
        assert U_kron.shape == (n * m, 96), f"U_kron shape {U_kron.shape}"
        assert V_kron.shape == (m * p, 96), f"V_kron shape {V_kron.shape}"
        assert W_kron.shape == (p * n, 96), f"W_kron shape {W_kron.shape}"
        
        # Now we have rank 96. To get lower rank, we need to merge columns.
        # Add small random perturbation and try to compress.
        U_init = U_kron + np.random.randn(n * m, 96) * 0.01
        V_init = V_kron + np.random.randn(m * p, 96) * 0.01
        W_init = W_kron + np.random.randn(p * n, 96) * 0.01
        
        # If rank < 96, select subset of columns
        if rank < 96:
            # Use SVD to find best low-rank approximation
            # For each matrix, compute SVD and keep top 'rank' components
            # But we need consistent column selection across U, V, W
            # Use a random projection approach instead
            P = np.random.randn(96, rank)
            U_init = U_kron @ P
            V_init = V_kron @ P
            W_init = W_kron @ P
        
        return U_init, V_init, W_init

    # Try structured initializations based on Kronecker products
    target_ranks = [88, 84, 80, 76, 72, 68, 64, 60, 56, 52, 48]

    for rank in target_ranks:
        if time.time() > deadline:
            break

        # Kronecker-structured initializations
        for seed in range(25):
            if time.time() > deadline:
                break

            U_init, V_init, W_init = kronecker_init(rank, seed)
            success, U, V, W = fast_als(rank, U_init, V_init, W_init, max_iters=120)

            if success:
                result = try_polish(U, V, W, rank)
                if result is not None:
                    return result

        # Multiple random initializations
        for seed in range(30):
            if time.time() > deadline:
                break

            np.random.seed(10000 + seed)
            U_init = np.random.randn(n * m, rank) * 0.5
            V_init = np.random.randn(m * p, rank) * 0.5
            W_init = np.random.randn(p * n, rank) * 0.5

            success, U, V, W = fast_als(rank, U_init, V_init, W_init, max_iters=100)

            if success:
                result = try_polish(U, V, W, rank)
                if result is not None:
                    return result

        # Try starting from schoolbook with column merging
        if rank < rank_schoolbook:
            for trial in range(10):
                if time.time() > deadline:
                    break
                np.random.seed(20000 + trial)
                # Random projection from schoolbook
                P = np.random.randn(rank_schoolbook, rank)
                U_init = U_school @ P
                V_init = V_school @ P
                W_init = W_school @ P

                success, U, V, W = fast_als(rank, U_init, V_init, W_init, max_iters=80)
                if success:
                    result = try_polish(U, V, W, rank)
                    if result is not None:
                        return result

    return best_result
# EVOLVE-BLOCK-END
