"""Optimized matrix multiplication decomposition for 4x4 * 4x9."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145  # safety margin before 150s limit

    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36

    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p  # 144
    U_fb = np.zeros((nm, rank_schoolbook))
    V_fb = np.zeros((mp, rank_schoolbook))
    W_fb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1

    best_rank = rank_schoolbook
    best_U = U_fb.copy()
    best_V = V_fb.copy()
    best_W = W_fb.copy()

    def tensor_residual(U, V, W):
        """Compute the residual tensor R[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1"""
        rank = U.shape[1]
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)

        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        UV = np.einsum('ijr,jkr->ijkr', U_3d, V_3d)
        T = np.einsum('ijkr,kir->ijk', UV, W_3d)
        return T - 1.0

    def tensor_grads(U, V, W):
        """Compute gradients of loss = sum_{i,j,k} (T[i,j,k]-1)^2 with respect to U, V, W"""
        rank = U.shape[1]
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)

        # T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
        UV = np.einsum('ijr,jkr->ijkr', U_3d, V_3d)
        T = np.einsum('ijkr,kir->ijk', UV, W_3d)
        err = T - 1.0

        # grad_U[i,j,r] = 2 * sum_k err[i,j,k]*V[j,k,r]*W[k,i,r]
        grad_U = 2.0 * np.einsum('ijk,jkr,kir->ijr', err, V_3d, W_3d)

        # grad_V[j,k,r] = 2 * sum_i err[i,j,k]*U[i,j,r]*W[k,i,r]
        grad_V = 2.0 * np.einsum('ijk,ijr,kir->jkr', err, U_3d, W_3d)

        # grad_W[k,i,r] = 2 * sum_j err[i,j,k]*U[i,j,r]*V[j,k,r]
        grad_W = 2.0 * np.einsum('ijk,ijr,jkr->kir', err, U_3d, V_3d)

        return grad_U.reshape(nm, rank), grad_V.reshape(mp, rank), grad_W.reshape(pn, rank)

    def try_optimization(target_rank, U_init, V_init, W_init, max_steps=3000, lr=0.01):
        """Try to optimize from given initialization. Returns (success, U, V, W)."""
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()

        # Adam optimizer
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        m_U = np.zeros_like(U)
        v_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        v_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)
        t = 0

        for step in range(max_steps):
            if time.time() > deadline:
                return False, U, V, W

            t += 1
            gU, gV, gW = tensor_grads(U, V, W)

            # Adam update
            m_U = beta1 * m_U + (1 - beta1) * gU
            v_U = beta2 * v_U + (1 - beta2) * gU**2
            m_U_hat = m_U / (1 - beta1**t)
            v_U_hat = v_U / (1 - beta2**t)
            U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)

            m_V = beta1 * m_V + (1 - beta1) * gV
            v_V = beta2 * v_V + (1 - beta2) * gV**2
            m_V_hat = m_V / (1 - beta1**t)
            v_V_hat = v_V / (1 - beta2**t)
            V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)

            m_W = beta1 * m_W + (1 - beta1) * gW
            v_W = beta2 * v_W + (1 - beta2) * gW**2
            m_W_hat = m_W / (1 - beta1**t)
            v_W_hat = v_W / (1 - beta2**t)
            W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)

            # Check convergence periodically
            if step % 200 == 0:
                T = tensor_residual(U, V, W)
                max_err = np.max(np.abs(T))
                if max_err < 1e-8:
                    return True, U, V, W

        # Final check
        T = tensor_residual(U, V, W)
        max_err = np.max(np.abs(T))
        if max_err < 1e-8:
            return True, U, V, W

        return False, U, V, W

    def strassen_4x4_block_decomp():
        """
        Use Strassen's algorithm structure for 4x4 matrix multiplication.
        Split 4x4 into 2x2 blocks, use 7 multiplications per 2x2x2 block.
        For 4x4 * 4x9, we need to handle the rectangular case.
        
        Strategy: Split A (4x4) into 2x2 blocks A11,A12,A21,A22
                 Split B (4x9) into 2x2 blocks B11,B12,B21,B22 where each is 2x9
        
        Actually, let's use a different approach: embed into square case.
        """
        pass

    def block_circulant_decomposition():
        """
        Try block-circulant embedding approach.
        Partition matrices into blocks and use circulant structure.
        """
        pass

    # Try various ranks from lower to higher
    # Start with structured initializations for better convergence
    
    # Try ranks that are more likely to work
    rank_candidates = list(range(50, 145))
    
    for target_rank in rank_candidates:
        if time.time() > deadline:
            break

        # Try multiple initializations
        for trial in range(20):
            if time.time() > deadline:
                break

            # Different initialization strategies
            if trial == 0:
                # Random small values
                U = np.random.randn(nm, target_rank) * 0.1
                V = np.random.randn(mp, target_rank) * 0.1
                W = np.random.randn(pn, target_rank) * 0.1
            elif trial == 1:
                # Initialize with schoolbook-like structure (sparse)
                U = np.zeros((nm, target_rank))
                V = np.zeros((mp, target_rank))
                W = np.zeros((pn, target_rank))
                # Randomly assign some entries
                for r in range(target_rank):
                    i = np.random.randint(n)
                    j = np.random.randint(m)
                    k = np.random.randint(p)
                    U[i * m + j, r] = np.random.choice([-1, 1])
                    V[j * p + k, r] = np.random.choice([-1, 1])
                    W[k * n + i, r] = np.random.choice([-1, 1])
            elif trial == 2:
                # Use Kronecker product structure
                U = np.zeros((nm, target_rank))
                V = np.zeros((mp, target_rank))
                W = np.zeros((pn, target_rank))
                # Try to use tensor product of smaller decompositions
                for r in range(target_rank):
                    i = r % n
                    j = (r // n) % m
                    k = (r // (n * m)) % p
                    U[i * m + j, r] = 1
                    V[j * p + k, r] = 1
                    W[k * n + i, r] = 1
            else:
                # Random with different scales
                scale = 0.1 ** (trial * 0.1)
                U = np.random.randn(nm, target_rank) * scale
                V = np.random.randn(mp, target_rank) * scale
                W = np.random.randn(pn, target_rank) * scale

            success, U_opt, V_opt, W_opt = try_optimization(target_rank, U, V, W, max_steps=2000)

            if success:
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_U = U_opt.copy()
                    best_V = V_opt.copy()
                    best_W = W_opt.copy()
                break

    # Return the best found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
