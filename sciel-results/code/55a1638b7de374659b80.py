"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100.0

    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, r):
        """Check if U, V, W form a valid tensor decomposition. Returns max error."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        u_vals = Ua[u_idx, :]  # (n*m*p, r)
        v_vals = Va[v_idx, :]  # (n*m*p, r)
        w_vals = Wa[w_idx, :]  # (n*m*p, r)
        
        prod = u_vals * v_vals * w_vals
        result = prod.sum(axis=1)
        
        return np.max(np.abs(result - 1.0))

    # Precompute index arrays for efficient verification
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx

    best_result = make_schoolbook()
    best_rank = n * m * p

    # Structural change: Use block-structured decomposition.
    # Split the p dimension (7) into p1=4 and p2=3.
    # For the (4,4,4) block, use a known low-rank decomposition if possible.
    # For the (4,4,3) block, use schoolbook or a low-rank decomposition.
    #
    # We will construct a candidate decomposition by combining:
    # 1. A decomposition for the first 4 columns of W (corresponding to k=0..3)
    # 2. A decomposition for the last 3 columns of W (corresponding to k=4..6)
    #
    # The full decomposition is the concatenation of these two parts.
    #
    # For the (4,4,4) part, we can try to find a rank-49 decomposition (known to exist).
    # For the (4,4,3) part, we can use schoolbook (rank 48) or try to find a lower rank.
    #
    # Total rank = rank_444 + rank_443.
    #
    # To find a rank-49 decomposition for 4x4x4, we can use a block Strassen-like approach
    # or search for one. However, implementing a known 49-rank decomposition is complex.
    #
    # Alternative structural approach: Use a "mixed" initialization that is close to a
    # known low-rank structure, then refine.
    #
    # Let's try to construct a candidate with rank = 49 + 48 = 97.
    # We initialize the first 49 columns with a random structure that might be refined
    # to a 4x4x4 decomposition, and the last 48 columns with a schoolbook 4x4x3 decomposition.

    def make_hybrid_init():
        """Create an initial guess for a hybrid 49+48 rank decomposition."""
        r = 97
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        # Part 1: 4x4x4 block (k=0..3)
        # Initialize with random values, hoping to refine to a low-rank 4x4x4 decomposition
        np.random.seed(42)
        U[:, :49] = np.random.randn(n * m, 49) * 0.5
        V[:, :49] = np.random.randn(m * p, 49) * 0.5
        W[:, :49] = np.random.randn(p * n, 49) * 0.5
        
        # Part 2: 4x4x3 block (k=4..6)
        # Use schoolbook initialization
        col = 49
        for i in range(n):
            for j in range(m):
                for k in range(4, 7):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        return U, V, W, r

    def refine_hybrid(U, V, W, r, max_iter=300):
        """
        Refine the hybrid decomposition.
        Key insight: The first 49 columns should only affect k=0..3,
        and the last 48 columns should only affect k=4..6.
        We can enforce this by masking during refinement.
        """
        Ua = U.copy()
        Va = V.copy()
        Wa = W.copy()
        
        mU = np.zeros_like(Ua)
        mV = np.zeros_like(Va)
        mW = np.zeros_like(Wa)
        beta = 0.9
        lr = 0.1
        best_err = np.inf
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            u_vals = Ua[u_idx, :]
            v_vals = Va[v_idx, :]
            w_vals = Wa[w_idx, :]
            
            u_4d = u_vals.reshape(n, m, p, r)
            v_4d = v_vals.reshape(n, m, p, r)
            w_4d = w_vals.reshape(n, m, p, r)
            
            prod = u_4d * v_4d * w_4d
            result = prod.sum(axis=3)
            error = result - 1.0
            err_norm = np.linalg.norm(error)
            
            if err_norm < 1e-8:
                break
            
            if err_norm < best_err:
                best_err = err_norm
            
            err_4d = error[:, :, :, np.newaxis]
            
            grad_U_4d = 2 * err_4d * v_4d * w_4d
            grad_U = np.sum(grad_U_4d, axis=2).reshape(n * m, r)
            
            grad_V_4d = 2 * err_4d * u_4d * w_4d
            grad_V_sum = np.sum(grad_V_4d, axis=0)
            grad_V = grad_V_sum.reshape(m * p, r)
            
            grad_W_4d = 2 * err_4d * u_4d * v_4d
            grad_W_sum = np.sum(grad_W_4d, axis=1)
            grad_W_sum = np.transpose(grad_W_sum, (1, 0, 2))
            grad_W = grad_W_sum.reshape(p * n, r)
            
            # Enforce block structure:
            # First 49 columns: only affect k=0..3
            # Last 48 columns: only affect k=4..6
            # For U and V, this means we can update all columns.
            # For W, the first 49 columns should only have non-zero entries for k=0..3,
            # and the last 48 columns should only have non-zero entries for k=4..6.
            
            # Mask for W:
            # W has shape (p*n, r) = (7*4, r) = (28, r)
            # For k=0..3 (first 4 values of k), the indices are 0..15 (k*n + i for k=0..3, i=0..3)
            # For k=4..6 (last 3 values of k), the indices are 16..27 (k*n + i for k=4..6, i=0..3)
            
            # Create masks
            mask_W_first = np.zeros((p * n, r), dtype=bool)
            mask_W_first[0:16, 0:49] = True  # k=0..3, first 49 columns
            
            mask_W_second = np.zeros((p * n, r), dtype=bool)
            mask_W_second[16:28, 49:97] = True  # k=4..6, last 48 columns
            
            # Apply masks to gradients
            grad_W[mask_W_first] *= 1.0
            grad_W[~mask_W_first & (np.arange(r)[None, :] < 49)] = 0.0
            
            grad_W[mask_W_second] *= 1.0
            grad_W[~mask_W_second & (np.arange(r)[None, :] >= 49)] = 0.0
            
            # Update with momentum
            mU = beta * mU + (1 - beta) * grad_U
            mV = beta * mV + (1 - beta) * grad_V
            mW = beta * mW + (1 - beta) * grad_W
            
            Ua -= lr * mU
            Va -= lr * mV
            Wa -= lr * mW
            
            # Re-apply masks to W after update
            Wa[~mask_W_first & (np.arange(r)[None, :] < 49)] = 0.0
            Wa[~mask_W_second & (np.arange(r)[None, :] >= 49)] = 0.0
            
            new_err = verify(Ua.tolist(), Va.tolist(), Wa.tolist(), r)
            if new_err > best_err * 1.5:
                lr *= 0.5
            elif new_err < best_err * 0.5:
                lr *= 1.1
                lr = min(lr, 0.5)
        
        return Ua, Va, Wa

    # Try the hybrid approach
    U_init, V_init, W_init, r_hybrid = make_hybrid_init()
    U_ref, V_ref, W_ref = refine_hybrid(U_init, V_init, W_init, r_hybrid, max_iter=200)
    
    err = verify(U_ref.tolist(), V_ref.tolist(), W_ref.tolist(), r_hybrid)
    if err < 1e-6:
        # Try to round to exact integers/halves
        U_round = np.round(U_ref * 2) / 2
        V_round = np.round(V_ref * 2) / 2
        W_round = np.round(W_ref * 2) / 2
        err_round = verify(U_round.tolist(), V_round.tolist(), W_round.tolist(), r_hybrid)
        if err_round < 1e-9:
            best_rank = r_hybrid
            best_result = {
                "U": U_round.tolist(),
                "V": V_round.tolist(),
                "W": W_round.tolist()
            }
    else:
        # Fallback: Try random search for lower ranks
        candidate_ranks = list(range(97, 112))
        for r in candidate_ranks:
            if time.time() > deadline - 10:
                break
            for trial in range(2):
                if time.time() > deadline - 5:
                    break
                np.random.seed(trial * 10000 + r * 100)
                U = np.random.randn(n * m, r) * 0.3
                V = np.random.randn(m * p, r) * 0.3
                W = np.random.randn(p * n, r) * 0.3
                
                # Simple refinement
                Ua, Va, Wa = U.copy(), V.copy(), W.copy()
                mU = np.zeros_like(Ua)
                mV = np.zeros_like(Va)
                mW = np.zeros_like(Wa)
                beta = 0.9
                lr = 0.1
                for it in range(100):
                    if time.time() > deadline - 5:
                        break
                    u_vals = Ua[u_idx, :]
                    v_vals = Va[v_idx, :]
                    w_vals = Wa[w_idx, :]
                    u_4d = u_vals.reshape(n, m, p, r)
                    v_4d = v_vals.reshape(n, m, p, r)
                    w_4d = w_vals.reshape(n, m, p, r)
                    prod = u_4d * v_4d * w_4d
                    result = prod.sum(axis=3)
                    error = result - 1.0
                    err_norm = np.linalg.norm(error)
                    if err_norm < 1e-8:
                        break
                    err_4d = error[:, :, :, np.newaxis]
                    grad_U_4d = 2 * err_4d * v_4d * w_4d
                    grad_U = np.sum(grad_U_4d, axis=2).reshape(n * m, r)
                    grad_V_4d = 2 * err_4d * u_4d * w_4d
                    grad_V = np.sum(grad_V_4d, axis=0).reshape(m * p, r)
                    grad_W_4d = 2 * err_4d * u_4d * v_4d
                    grad_W_sum = np.sum(grad_W_4d, axis=1)
                    grad_W = np.transpose(grad_W_sum, (1, 0, 2)).reshape(p * n, r)
                    mU = beta * mU + (1 - beta) * grad_U
                    mV = beta * mV + (1 - beta) * grad_V
                    mW = beta * mW + (1 - beta) * grad_W
                    Ua -= lr * mU
                    Va -= lr * mV
                    Wa -= lr * mW
                    new_err = verify(Ua.tolist(), Va.tolist(), Wa.tolist(), r)
                    if new_err > err_norm * 1.5:
                        lr *= 0.5
                    elif new_err < err_norm * 0.5:
                        lr *= 1.1
                        lr = min(lr, 0.5)
                
                err = verify(Ua.tolist(), Va.tolist(), Wa.tolist(), r)
                if err < 1e-6:
                    U_round = np.round(Ua * 2) / 2
                    V_round = np.round(Va * 2) / 2
                    W_round = np.round(Wa * 2) / 2
                    err_round = verify(U_round.tolist(), V_round.tolist(), W_round.tolist(), r)
                    if err_round < 1e-9:
                        best_rank = r
                        best_result = {
                            "U": U_round.tolist(),
                            "V": V_round.tolist(),
                            "W": W_round.tolist()
                        }
                        break
            if best_rank < r:
                break
    
    return best_result
# EVOLVE-BLOCK-END
