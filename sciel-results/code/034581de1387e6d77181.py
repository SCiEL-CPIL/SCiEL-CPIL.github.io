import time
import numpy as np
from scipy.linalg import lstsq

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # Safety margin for 150s limit

    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10

    # Build target tensor
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def validate(U, V, W, atol=1e-6):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)

    def snap_to_half(arr):
        return np.round(arr * 2) / 2

    def snap_to_quarter(arr):
        return np.round(arr * 4) / 4

    def try_snap_validate(U, V, W):
        # Try half-integers first (most likely for Strassen-based)
        U_s = snap_to_half(U)
        V_s = snap_to_half(V)
        W_s = snap_to_half(W)
        if validate(U_s, V_s, W_s):
            return True, U_s, V_s, W_s
        # Try integers
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if validate(U_i, V_i, W_i):
            return True, U_i, V_i, W_i
        # Try quarters
        U_q = snap_to_quarter(U)
        V_q = snap_to_quarter(V)
        W_q = snap_to_quarter(W)
        if validate(U_q, V_q, W_q):
            return True, U_q, V_q, W_q
        return False, None, None, None

    def build_rank46():
        rank = 46
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))

        def add_strassen(j_off, k_off, col_start):
            c = col_start
            # P1 = (A00 + A11)(B00 + B11)
            U[0 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1

            # P2 = (A10 + A11) B00
            U[1 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            W[k_off * n + 1][c] += 1
            W[(k_off + 1) * n + 1][c] -= 1
            c += 1

            # P3 = A00 (B01 - B11)
            U[0 * m + j_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1

            # P4 = A11 (B10 - B00)
            U[1 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[j_off * p + k_off][c] -= 1
            W[k_off * n + 0][c] += 1
            W[k_off * n + 1][c] += 1
            c += 1

            # P5 = (A00 + A01) B11
            U[0 * m + j_off][c] += 1
            U[0 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            c += 1

            # P6 = (A10 - A00)(B00 + B01)
            U[1 * m + j_off][c] += 1
            U[0 * m + j_off][c] -= 1
            V[j_off * p + k_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1

            # P7 = (A01 - A11)(B10 + B11)
            U[0 * m + (j_off + 1)][c] += 1
            U[1 * m + (j_off + 1)][c] -= 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1

        add_strassen(0, 0, 0)
        add_strassen(0, 2, 7)
        add_strassen(2, 0, 14)
        add_strassen(2, 2, 21)

        col = 28
        for i in range(2):
            for k in range(4):
                U[i * m + 4][col] = 1
                V[4 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                if time.time() > deadline - 10:
                    return U, V, W

        for i in range(2):
            for j in range(4):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1
                if time.time() > deadline - 10:
                    return U, V, W

        for i in range(2):
            U[i * m + 4][col] = 1
            V[4 * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            if time.time() > deadline - 10:
                return U, V, W

        return U, V, W

    def gauss_newton_polish(U, V, W, max_iter=10, tol=1e-12):
        """Gauss-Newton polish for high precision."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        R = U.shape[1]
        
        # Flatten parameters
        def flatten_params():
            return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def unflatten_params(x):
            U_flat = x[:U.size]
            V_flat = x[U.size:U.size+V.size]
            W_flat = x[U.size+V.size:]
            return U_flat.reshape(U.shape), V_flat.reshape(V.shape), W_flat.reshape(W.shape)
            
        x = flatten_params()
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            U_cur, V_cur, W_cur = unflatten_params(x)
            prod = np.einsum('ar,br,cr->abc', U_cur, V_cur, W_cur, optimize=True)
            diff = prod - T
            diff_flat = diff.flatten()
            loss = np.sum(diff_flat * diff_flat)
            
            if loss < tol:
                break
                
            # Build Jacobian
            # J is (N_res, N_params)
            # N_res = 10*25*10 = 2500
            # N_params = 10*R + 25*R + 10*R = 45*R
            # We construct J by computing gradients for each parameter.
            # This is expensive if done naively, but R is small (<=46).
            
            J = np.zeros((diff.size, x.size))
            
            # dR/dU
            # R_{ijk} = sum_r U_{ir} V_{jr} W_{kr}
            # dR_{ijk}/dU_{ar} = V_{jr} W_{kr}
            # We can compute this efficiently with einsum or loops.
            # Given small dims, loops are fine.
            
            # Reshape diff to (i, j, k)
            # U is (i, r), V is (j, r), W is (k, r)
            
            # We'll fill J column by column for efficiency in memory access patterns if needed,
            # but here we can just compute the blocks.
            
            # Block for U: shape (i, r) -> maps to col index in x
            # For each i, r:
            #   dR_{i,j,k} / dU_{i,r} = V_{j,r} W_{k,r}
            #   This is a vector of length 2500 (j,k).
            
            # Let's use einsum to compute the Jacobian blocks directly if possible,
            # or just loop over r.
            
            for r in range(R):
                # U block: cols corresponding to U[:, r]
                # V block: cols corresponding to V[:, r]
                # W block: cols corresponding to W[:, r]
                
                # dR/dU[:, r]
                # For each i:
                #   dR_{i, j, k} / dU_{i, r} = V_{j, r} W_{k, r}
                # This is outer product of V[:, r] and W[:, r]
                # V[:, r] is shape (25,), W[:, r] is shape (10,)
                # Outer is (25, 10) -> flatten to 250
                # But we need it for each i.
                # Actually, for fixed r, the derivative w.r.t U_{i, r} is the same for all i? No.
                # dR_{i, j, k} / dU_{i, r} = V_{j, r} W_{k, r}.
                # This value does not depend on i!
                # So for a fixed r, the column block for U[:, r] is:
                # For i=0: row block is V[:, r] outer W[:, r]
                # For i=1: row block is V[:, r] outer W[:, r]
                # ...
                
                vec_V = V_cur[:, r] # shape (25,)
                vec_W = W_cur[:, r] # shape (10,)
                outer_VW = np.outer(vec_V, vec_W) # shape (25, 10)
                outer_VW_flat = outer_VW.flatten() # shape (250,)
                
                # Fill J for U columns
                for i in range(U_dim):
                    col_start = i * R + r
                    # The rows in J corresponding to i are i*250 to (i+1)*250
                    row_start = i * 250
                    J[row_start:row_start+250, col_start] = outer_VW_flat
                    
                # Fill J for V columns
                # dR_{i, j, k} / dV_{j, r} = U_{i, r} W_{k, r}
                # For fixed j, r:
                #   dR_{i, j, k} / dV_{j, r} = U_{i, r} W_{k, r}
                # This depends on i and k.
                # For a fixed j, the column in J corresponds to V_{j, r}.
                # The rows are (i, k) pairs.
                # Value is U_{i, r} W_{k, r}.
                
                vec_U = U_cur[:, r] # shape (10,)
                # For each i, k:
                #   val = U_{i, r} W_{k, r}
                #   This is outer product of U[:, r] and W[:, r]
                outer_UW = np.outer(vec_U, vec_W) # shape (10, 10)
                outer_UW_flat = outer_UW.flatten() # shape (100,)
                
                for j in range(V_dim):
                    col_start = U.size + j * R + r
                    # Rows for fixed j: j*100 to (j+1)*100
                    row_start = j * 100
                    J[row_start:row_start+100, col_start] = outer_UW_flat
                    
                # Fill J for W columns
                # dR_{i, j, k} / dW_{k, r} = U_{i, r} V_{j, r}
                # For fixed k, r:
                #   Value is U_{i, r} V_{j, r}
                # Outer product of U[:, r] and V[:, r]
                outer_UV = np.outer(vec_U, vec_V) # shape (10, 25)
                outer_UV_flat = outer_UV.flatten() # shape (250,)
                
                for k in range(W_dim):
                    col_start = U.size + V.size + k * R + r
                    # Rows for fixed k: k*100 to (k+1)*100
                    # Wait, row index is i*250 + j*10 + k?
                    # T is (i, j, k).
                    # Flatten order: i is slowest, k is fastest.
                    # Index = i * (25*10) + j * 10 + k
                    # So for fixed k, the rows are not contiguous in a simple block?
                    # No, for fixed k, j varies, i varies.
                    # The rows corresponding to a specific k are scattered.
                    # J[row, col] where row = i*250 + j*10 + k.
                    # We need to scatter the values.
                    
                    # This is inefficient. Let's stick to the previous method or optimize.
                    # Given the small size, we can just loop i, j.
                    for i in range(U_dim):
                        for j in range(V_dim):
                            row_idx = i * 250 + j * 10 + k
                            J[row_idx, col_start] = vec_U[i] * vec_V[j]
                
                # Optimization: We can vectorize the W part too if we construct the row indices.
                # But let's see if the time is okay.
                # 46 iterations * (10 + 25 + 10*25) operations for filling.
                # 46 * (10 + 25 + 250) ~ 13000 ops. Very fast.
                
            # Solve J delta = -diff_flat
            # Use least squares
            delta, _, _, _ = lstsq(J, -diff_flat)
            
            # Damping
            step_norm = np.linalg.norm(delta)
            if step_norm > 1.0:
                delta = delta / step_norm * 1.0
            elif step_norm > 0.1:
                delta = delta / step_norm * 0.1
                
            x = x + delta
            U, V, W = unflatten_params(x)
            
        return U, V, W

    def fast_optimize(U_init, V_init, W_init, target_rank, max_iter=150, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8, lr_decay=0.999):
        """Adam optimizer with adaptive restarts and final Gauss-Newton polish."""
        R = target_rank
        U = U_init[:, :R].copy()
        V = V_init[:, :R].copy()
        W = W_init[:, :R].copy()

        mU = np.zeros_like(U)
        vU = np.zeros_like(U)
        mV = np.zeros_like(V)
        vV = np.zeros_like(V)
        mW = np.zeros_like(W)
        vW = np.zeros_like(W)

        prev_loss = float('inf')
        no_improve = 0
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        current_lr = lr
        restart_count = 0

        for it in range(max_iter):
            if time.time() > deadline - 20:
                break

            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            loss = np.sum(diff * diff)

            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

            if loss < 1e-10:
                break

            # Stagnation detection with restart
            if loss > prev_loss * 1.005 and no_improve > 12:
                restart_count += 1
                if restart_count > 4:
                    break
                # Perturb to escape local minimum
                U += np.random.randn(*U.shape) * 0.01
                V += np.random.randn(*V.shape) * 0.01
                W += np.random.randn(*W.shape) * 0.01
                mU = np.zeros_like(U)
                vU = np.zeros_like(U)
                mV = np.zeros_like(V)
                vV = np.zeros_like(V)
                mW = np.zeros_like(W)
                vW = np.zeros_like(W)
                no_improve = 0
                prev_loss = loss
                continue
            else:
                no_improve += 1
                prev_loss = loss

            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)

            t = it + 1
            mU = beta1 * mU + (1 - beta1) * gradU
            vU = beta2 * vU + (1 - beta2) * (gradU ** 2)
            mU_hat = mU / (1 - beta1 ** t)
            vU_hat = vU / (1 - beta2 ** t)
            U -= current_lr * mU_hat / (np.sqrt(vU_hat) + eps)

            mV = beta1 * mV + (1 - beta1) * gradV
            vV = beta2 * vV + (1 - beta2) * (gradV ** 2)
            mV_hat = mV / (1 - beta1 ** t)
            vV_hat = vV / (1 - beta2 ** t)
            V -= current_lr * mV_hat / (np.sqrt(vV_hat) + eps)

            mW = beta1 * mW + (1 - beta1) * gradW
            vW = beta2 * vW + (1 - beta2) * (gradW ** 2)
            mW_hat = mW / (1 - beta1 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            W -= current_lr * mW_hat / (np.sqrt(vW_hat) + eps)

            current_lr *= lr_decay

        # Always try to polish and snap, even if Adam didn't converge tightly
        U2, V2, W2 = gauss_newton_polish(best_U, best_V, best_W, max_iter=20, tol=1e-14)
        valid, U_s, V_s, W_s = try_snap_validate(U2, V2, W2)
        if valid:
            return U_s, V_s, W_s

        # Also try snapping the Adam result directly in case Gauss-Newton diverged
        valid, U_s, V_s, W_s = try_snap_validate(best_U, best_V, best_W)
        if valid:
            return U_s, V_s, W_s

        return None, None, None

    # Build and verify rank-46
    U46, V46, W46 = build_rank46()
    if not validate(U46, V46, W46):
        # Fallback to standard rank 50 if something went wrong
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }

    best_result = None

    np.random.seed(42)

    # Score columns by "removability" - columns that share more support are more redundant
    col_nonzeros_U = np.count_nonzero(U46, axis=0)
    col_nonzeros_V = np.count_nonzero(V46, axis=0)
    col_nonzeros_W = np.count_nonzero(W46, axis=0)

    # Create support sets for each column
    support_U = [set(np.nonzero(U46[:, c])[0]) for c in range(46)]
    support_V = [set(np.nonzero(V46[:, c])[0]) for c in range(46)]
    support_W = [set(np.nonzero(W46[:, c])[0]) for c in range(46)]

    # Score: lower = more likely to be removable (smaller support, more overlap with others)
    col_scores = []
    for c in range(46):
        support_size = len(support_U[c]) + len(support_V[c]) + len(support_W[c])
        # Count how many other columns share significant support
        overlap_count = 0
        for c2 in range(46):
            if c2 == c:
                continue
            overlap = len(support_U[c] & support_U[c2]) + len(support_V[c] & support_V[c2]) + len(support_W[c] & support_W[c2])
            if overlap > 3:
                overlap_count += 1
        score = support_size + overlap_count * 0.5
        col_scores.append(score)

    col_order = np.argsort(col_scores)

    # Strategy 1: Try removing single columns (rank 45)
    max_attempts = min(20, len(col_order))
    for attempt in range(max_attempts):
        if time.time() > deadline - 60:
            break

        col_to_zero = col_order[attempt]
        mask = [i for i in range(46) if i != col_to_zero]
        U_test = U46[:, mask].copy()
        V_test = V46[:, mask].copy()
        W_test = W46[:, mask].copy()

        # Try multiple noise scales and seeds
        found = False
        for seed in [42, 123, 999, 777, 555, 333]:
            if time.time() > deadline - 55:
                break
            np.random.seed(seed + attempt * 100)
            for noise_scale in [0.003, 0.01, 0.03, 0.06]:
                if time.time() > deadline - 52:
                    break
                U_try = U_test.copy()
                V_try = V_test.copy()
                W_try = W_test.copy()
                U_try += np.random.randn(*U_try.shape) * noise_scale
                V_try += np.random.randn(*V_try.shape) * noise_scale
                W_try += np.random.randn(*W_try.shape) * noise_scale

                U_opt, V_opt, W_opt = fast_optimize(U_try, V_try, W_try, 45, max_iter=100, lr=0.04)

                if U_opt is not None:
                    best_result = (U_opt, V_opt, W_opt)
                    found = True
                    break

            if found:
                break

        if found:
            break

    # Strategy 2: If rank 45 failed, try removing pairs of columns (rank 44)
    if best_result is None and time.time() < deadline - 70:
        # Try pairs of columns with high overlap
        pair_candidates = []
        for i in range(len(col_order)):
            for j in range(i+1, len(col_order)):
                c1 = col_order[i]
                c2 = col_order[j]
                overlap = len(support_U[c1] & support_U[c2]) + len(support_V[c1] & support_V[c2]) + len(support_W[c1] & support_W[c2])
                pair_candidates.append((overlap, c1, c2))
        pair_candidates.sort(reverse=True)

        for idx in range(min(15, len(pair_candidates))):
            if time.time() > deadline - 65:
                break
            _, c1, c2 = pair_candidates[idx]
            mask = [i for i in range(46) if i != c1 and i != c2]
            U_test = U46[:, mask].copy()
            V_test = V46[:, mask].copy()
            W_test = W46[:, mask].copy()

            for seed in [42, 123, 999, 555]:
                if time.time() > deadline - 62:
                    break
                np.random.seed(seed + idx * 10)
                for noise_scale in [0.008, 0.03, 0.06]:
                    if time.time() > deadline - 60:
                        break
                    U_try = U_test.copy()
                    V_try = V_test.copy()
                    W_try = W_test.copy()
                    U_try += np.random.randn(*U_try.shape) * noise_scale
                    V_try += np.random.randn(*V_try.shape) * noise_scale
                    W_try += np.random.randn(*W_try.shape) * noise_scale

                    U_opt, V_opt, W_opt = fast_optimize(U_try, V_try, W_try, 44, max_iter=80, lr=0.035)

                    if U_opt is not None:
                        best_result = (U_opt, V_opt, W_opt)
                        break
                if best_result is not None:
                    break
            if best_result is not None:
                break

    # Strategy 3: Try rank 43 with structured perturbation
    if best_result is None and time.time() < deadline - 80:
        # Remove 3 columns with highest mutual overlap
        triple_candidates = []
        for i in range(len(col_order)):
            for j in range(i+1, len(col_order)):
                for k in range(j+1, len(col_order)):
                    c1, c2, c3 = col_order[i], col_order[j], col_order[k]
                    overlap = (len(support_U[c1] & support_U[c2]) + len(support_V[c1] & support_V[c2]) + len(support_W[c1] & support_W[c2]) +
                              len(support_U[c1] & support_U[c3]) + len(support_V[c1] & support_V[c3]) + len(support_W[c1] & support_W[c3]) +
                              len(support_U[c2] & support_U[c3]) + len(support_V[c2] & support_V[c3]) + len(support_W[c2] & support_W[c3]))
                    triple_candidates.append((overlap, c1, c2, c3))
        triple_candidates.sort(reverse=True)

        for idx in range(min(8, len(triple_candidates))):
            if time.time() > deadline - 75:
                break
            _, c1, c2, c3 = triple_candidates[idx]
            mask = [i for i in range(46) if i not in (c1, c2, c3)]
            U_test = U46[:, mask].copy()
            V_test = V46[:, mask].copy()
            W_test = W46[:, mask].copy()

            for seed in [42, 123, 999]:
                if time.time() > deadline - 72:
                    break
                np.random.seed(seed + idx * 20)
                for noise_scale in [0.02, 0.05, 0.08]:
                    if time.time() > deadline - 70:
                        break
                    U_try = U_test.copy()
                    V_try = V_test.copy()
                    W_try = W_test.copy()
                    U_try += np.random.randn(*U_try.shape) * noise_scale
                    V_try += np.random.randn(*V_try.shape) * noise_scale
                    W_try += np.random.randn(*W_try.shape) * noise_scale

                    U_opt, V_opt, W_opt = fast_optimize(U_try, V_try, W_try, 43, max_iter=60, lr=0.03)

                    if U_opt is not None:
                        best_result = (U_opt, V_opt, W_opt)
                        break
                if best_result is not None:
                    break
            if best_result is not None:
                break

    if best_result is not None:
        U_r, V_r, W_r = best_result
        # Verify the result
        if validate(U_r, V_r, W_r):
            return {
                "U": [[float(x) for x in row] for row in U_r],
                "V": [[float(x) for x in row] for row in V_r],
                "W": [[float(x) for x in row] for row in W_r]
            }

    return {
        "U": [[float(x) for x in row] for row in U46],
        "V": [[float(x) for x in row] for row in V46],
        "W": [[float(x) for x in row] for row in W46]
    }
# EVOLVE-BLOCK-END
