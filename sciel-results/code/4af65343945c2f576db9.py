import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    W indexed by k*n + i
    """
    try:
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        if U_arr.ndim != 2 or V_arr.ndim != 2 or W_arr.ndim != 2:
            return False
        
        rank = U_arr.shape[1]
        
        # Check valid entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U_arr[a, :], np.dot(V_arr[b, :], W_arr[c, :]))
                    if np.abs(val - 1.0) > tol:
                        return False
        
        # Check invalid entries should be 0
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    
                    if not (j_a == j_b and k_b == k_c and i_c == i_a):
                        val = np.dot(U_arr[a, :], np.dot(V_arr[b, :], W_arr[c, :]))
                        if np.abs(val) > tol:
                            return False
        
        return True
    except Exception:
        return False

def _strassen_2x2x2():
    """
    Return rank-7 decomposition for 2x2x2 matrix multiplication.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10 + A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B10 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B01 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00 + A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2, 5] = 1; U[3, 5] = -1
    V[1, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1, 6] = 1; U[2, 6] = -1
    V[0, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _replicate_strassen(n=2, m=2, p=16):
    """
    Replicate the 2x2x2 Strassen decomposition 8 times for 2x2x16.
    Rank = 7 * 8 = 56
    """
    U2, V2, W2 = _strassen_2x2x2()
    
    num_blocks = p // 2
    rank = 7 * num_blocks
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for block in range(num_blocks):
        for r in range(7):
            col = block * 7 + r
            U[:, col] = U2[:, r]
            for j in range(m):
                for k in range(2):
                    b_idx = j * p + (block * 2 + k)
                    v_idx = j * 2 + k
                    V[b_idx, col] = V2[v_idx, r]
            for i in range(n):
                for k_local in range(2):
                    k = block * 2 + k_local
                    w_idx = k * n + i
                    w2_idx = k_local * n + i
                    W[w_idx, col] = W2[w2_idx, r]
    
    return U, V, W

def _round_to_half_int(arr, tol=1e-3):
    """Round array to nearest half-integer if close."""
    result = arr.copy()
    rounded = np.round(result * 2) / 2
    if np.max(np.abs(result - rounded)) < tol:
        return rounded
    return result

def _try_find_decomposition(n, m, p, rank, seed, time_budget, init=None):
    """
    Try to find a rank-rank decomposition using gradient-based optimization.
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    # Collect all valid and invalid triples
    valid_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_triples.append((a, b, c))
    valid_triples = np.array(valid_triples)
    
    invalid_triples = []
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                if not (j_a == j_b and k_b == k_c and i_c == i_a):
                    invalid_triples.append((a, b, c))
    invalid_triples = np.array(invalid_triples) if invalid_triples else np.empty((0, 3), dtype=int)
    
    # Sample invalid triples to keep computation fast
    max_invalid = 1000
    if len(invalid_triples) > max_invalid:
        idx = rng.choice(len(invalid_triples), max_invalid, replace=False)
        invalid_triples = invalid_triples[idx]
    
    all_triples = np.vstack([valid_triples, invalid_triples])
    targets = np.concatenate([np.ones(len(valid_triples)), np.zeros(len(invalid_triples))])
    a_idx = all_triples[:, 0]
    b_idx = all_triples[:, 1]
    c_idx = all_triples[:, 2]
    
    # Initialize
    if init is not None:
        U, V, W = init
        U = np.array(U, dtype=np.float64).copy()
        V = np.array(V, dtype=np.float64).copy()
        W = np.array(W, dtype=np.float64).copy()
    else:
        U = rng.randn(n * m, rank) * 0.5
        V = rng.randn(m * p, rank) * 0.5
        W = rng.randn(p * n, rank) * 0.5
    
    # Adam optimizer
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m_U = np.zeros_like(U)
    v_U = np.zeros_like(U)
    m_V = np.zeros_like(V)
    v_V = np.zeros_like(V)
    m_W = np.zeros_like(W)
    v_W = np.zeros_like(W)
    t = 0
    lr = 0.05
    
    best_loss = float('inf')
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    num_triples = len(all_triples)
    
    for step in range(5000):
        if time.time() - start_time > time_budget:
            break
        t += 1
        
        # Compute predictions
        U_sub = U[a_idx, :]
        V_sub = V[b_idx, :]
        W_sub = W[c_idx, :]
        preds = np.sum(U_sub * V_sub * W_sub, axis=1)
        residuals = preds - targets
        
        # Compute gradients
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # Batch processing
        batch_size = 200
        for batch_start in range(0, num_triples, batch_size):
            batch_end = min(batch_start + batch_size, num_triples)
            a_b = a_idx[batch_start:batch_end]
            b_b = b_idx[batch_start:batch_end]
            c_b = c_idx[batch_start:batch_end]
            res_b = residuals[batch_start:batch_end]
            
            U_b = U[a_b, :]
            V_b = V[b_b, :]
            W_b = W[c_b, :]
            
            np.add.at(grad_U, a_b, res_b[:, None] * V_b * W_b)
            np.add.at(grad_V, b_b, res_b[:, None] * U_b * W_b)
            np.add.at(grad_W, c_b, res_b[:, None] * U_b * V_b)
        
        # Adam update
        lr_t = lr * np.sqrt(1 - beta2**t) / (1 - beta1**t)
        
        m_U = beta1 * m_U + (1 - beta1) * grad_U
        v_U = beta2 * v_U + (1 - beta2) * grad_U**2
        U -= lr_t * m_U / (np.sqrt(v_U) + eps)
        
        m_V = beta1 * m_V + (1 - beta1) * grad_V
        v_V = beta2 * v_V + (1 - beta2) * grad_V**2
        V -= lr_t * m_V / (np.sqrt(v_V) + eps)
        
        m_W = beta1 * m_W + (1 - beta1) * grad_W
        v_W = beta2 * v_W + (1 - beta2) * grad_W**2
        W -= lr_t * m_W / (np.sqrt(v_W) + eps)
        
        # Compute loss
        preds = np.sum(U[a_idx, :] * V[b_idx, :] * W[c_idx, :], axis=1)
        loss = np.sum((preds - targets) ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
            if best_loss < 1e-8:
                break
    
    # Round to half-integers
    U_round = _round_to_half_int(best_U)
    V_round = _round_to_half_int(best_V)
    W_round = _round_to_half_int(best_W)
    
    if _verify_decomposition(U_round.tolist(), V_round.tolist(), W_round.tolist(), n, m, p, tol=1e-4):
        return U_round.tolist(), V_round.tolist(), W_round.tolist()
    
    if _verify_decomposition(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p, tol=1e-4):
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    def _schoolbook():
        rank_school = n * m * p
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    try:
        # Start with replicated Strassen decomposition (rank 56)
        U_str, V_str, W_str = _replicate_strassen(n, m, p)
        best_result = {"U": U_str.tolist(), "V": V_str.tolist(), "W": W_str.tolist()}
        best_rank = 56
        
        if not _verify_decomposition(best_result["U"], best_result["V"], best_result["W"], n, m, p, tol=1e-6):
            best_result = _schoolbook()
            best_rank = n * m * p
        
        # Try to find lower rank decompositions
        seed = 42
        target_ranks = [54, 52, 50, 48]
        
        for rank in target_ranks:
            if time.time() > deadline:
                break
            
            remaining_time = deadline - time.time()
            if remaining_time < 15:
                break
            
            # Try perturbing Strassen with this rank
            # Initialize by taking a subset of columns
            init_U = U_str[:, :rank].copy()
            init_V = V_str[:, :rank].copy()
            init_W = W_str[:, :rank].copy()
            
            # Add noise
            rng = np.random.RandomState(seed)
            init_U += rng.randn(n * m, rank) * 0.01
            init_V += rng.randn(m * p, rank) * 0.01
            init_W += rng.randn(p * n, rank) * 0.01
            
            result = _try_find_decomposition(n, m, p, rank, seed, remaining_time * 0.6, init=(init_U, init_V, init_W))
            seed += 1
            
            if result is not None:
                U, V, W = result
                if _verify_decomposition(U, V, W, n, m, p, tol=1e-4):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    continue
        
        return best_result
    except Exception:
        return _schoolbook()

# EVOLVE-BLOCK-END
