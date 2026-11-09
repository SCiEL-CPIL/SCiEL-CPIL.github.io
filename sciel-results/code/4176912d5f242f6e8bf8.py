import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 40
    
    start_time = time.time()
    deadline = 120.0
    safety_margin = 5.0
    
    def check_deadline():
        return time.time() + safety_margin > deadline

    # Step 1: Try to find a rank-10 decomposition for 2x2x5
    # If found, use it to construct a rank-20 decomposition for 2x4x5
    
    n_small, m_small, p_small = 2, 2, 5
    rank_small = 10
    
    # Build target tensor for 2x2x5
    T_small = np.zeros((n_small * m_small, m_small * p_small, p_small * n_small), dtype=np.float64)
    for i in range(n_small):
        for j in range(m_small):
            for k in range(p_small):
                a = i * m_small + j
                b = j * p_small + k
                c = k * n_small + i
                T_small[a, b, c] = 1.0

    best_U_small = None
    best_V_small = None
    best_W_small = None

    # Search for rank-10 decomposition
    for restart in range(50):
        if check_deadline():
            break
            
        np.random.seed(restart)
        U = np.random.randn(n_small * m_small, rank_small)
        V = np.random.randn(m_small * p_small, rank_small)
        W = np.random.randn(p_small * n_small, rank_small)
        
        lr = 0.01
        for step in range(5000):
            if check_deadline():
                break
                
            # Reconstruction
            recon = np.zeros_like(T_small)
            for r in range(rank_small):
                recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
            err = recon - T_small
            loss = np.sum(err**2)
            
            if loss < 1e-10:
                break
            
            # Gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for r in range(rank_small):
                grad_U[:, r] = np.einsum('abc,b->a', err, V[:, r]) * W[:, r]
                grad_V[:, r] = np.einsum('abc,a->b', err, U[:, r]) * W[:, r]
                grad_W[:, r] = np.einsum('abc,a->c', err, U[:, r]) * V[:, r]
                
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            if step % 100 == 0 and loss < 1e-8:
                break
        
        # Check result
        recon = np.zeros_like(T_small)
        for r in range(rank_small):
            recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        max_err = np.max(np.abs(recon - T_small))
        
        if max_err < 1e-6:
            # Try to clean up to integers/half-integers
            U_clean = None
            V_clean = None
            W_clean = None
            for scale in [1, 2]:
                U_s = np.round(U * scale)
                V_s = np.round(V * scale)
                W_s = np.round(W * scale)
                recon_s = np.zeros_like(T_small)
                for r in range(rank_small):
                    recon_s += np.einsum('a,b,c->abc', U_s[:, r], V_s[:, r], W_s[:, r])
                if np.max(np.abs(recon_s - T_small)) < 1e-9:
                    U_clean = U_s / scale
                    V_clean = V_s / scale
                    W_clean = W_s / scale
                    break
            
            if U_clean is not None:
                best_U_small = U_clean
                best_V_small = V_clean
                best_W_small = W_clean
                break

    # Step 2: Construct 2x4x5 solution
    # If rank-10 small solution found, total rank is 20
    # Else, fall back to schoolbook rank 40
    
    if best_U_small is not None:
        total_rank = 20
        U_total = np.zeros((n * m, total_rank), dtype=np.float64)
        V_total = np.zeros((m * p, total_rank), dtype=np.float64)
        W_total = np.zeros((p * n, total_rank), dtype=np.float64)
        
        # Block 1: j in {0, 1}
        for i in range(n_small):
            for j in range(m_small):
                a_local = i * m_small + j
                a_global = i * m + j  # m=4, so a_global = i*4 + j
                for r in range(rank_small):
                    U_total[a_global, r] = best_U_small[a_local, r]
                    
        for j in range(m_small):
            for k in range(p_small):
                b_local = j * p_small + k
                b_global = j * p + k  # p=5, so b_global = j*5 + k
                for r in range(rank_small):
                    V_total[b_global, r] = best_V_small[b_local, r]
                    
        for k in range(p_small):
            for i in range(n_small):
                c_local = k * n_small + i
                c_global = k * n + i  # n=2, so c_global = k*2 + i
                for r in range(rank_small):
                    W_total[c_global, r] = best_W_small[c_local, r]
                    
        # Block 2: j in {2, 3}
        # Offset in V by 10 (since p=5, and j starts at 2, so j*5 = 10)
        # U and W indices are the same as block 1?
        # U: a_global = i*4 + j. For j in {2,3}, a_global is different.
        # Local U index a_local = i*2 + j' where j' = j-2.
        for i in range(n_small):
            for j_prime in range(m_small):
                j = j_prime + 2
                a_local = i * m_small + j_prime
                a_global = i * m + j
                for r in range(rank_small):
                    U_total[a_global, r + rank_small] = best_U_small[a_local, r]
                    
        for j_prime in range(m_small):
            j = j_prime + 2
            for k in range(p_small):
                b_local = j_prime * p_small + k
                b_global = j * p + k
                for r in range(rank_small):
                    V_total[b_global, r + rank_small] = best_V_small[b_local, r]
                    
        for k in range(p_small):
            for i in range(n_small):
                c_local = k * n_small + i
                c_global = k * n + i
                for r in range(rank_small):
                    W_total[c_global, r + rank_small] = best_W_small[c_local, r]
                    
        # Verify
        recon = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(total_rank):
            recon += np.einsum('a,b,c->abc', U_total[:, r], V_total[:, r], W_total[:, r])
            
        target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m + j, j*p + k, k*n + i] = 1.0
                    
        if np.max(np.abs(recon - target)) < 1e-6:
            def clean_matrix(mat):
                mat = np.round(mat * 2) / 2
                return mat.tolist()
            
            return {"U": clean_matrix(U_total), "V": clean_matrix(V_total), "W": clean_matrix(W_total)}
        else:
            # Verification failed, fall back
            pass

    # Fallback: Schoolbook
    U_sb = np.zeros((n*m, rank_schoolbook), dtype=np.float64)
    V_sb = np.zeros((m*p, rank_schoolbook), dtype=np.float64)
    W_sb = np.zeros((p*n, rank_schoolbook), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1.0
                V_sb[j*p + k, col] = 1.0
                W_sb[k*n + i, col] = 1.0
                col += 1
                
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END
