import time
import numpy as np
import scipy.optimize

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin for 150s limit

    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10

    # Build target multiplication tensor
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def validate(U, V, W, atol=1e-6):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return bool(np.allclose(prod, T, atol=atol))

    def build_rank46():
        rank = 46
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))

        def add_strassen(j_off, k_off, c0):
            c = c0
            # P1 = (A00 + A11)(B00 + B11)
            U[0 * m + j_off, c] += 1; U[1 * m + (j_off + 1), c] += 1
            V[j_off * p + k_off, c] += 1; V[(j_off + 1) * p + (k_off + 1), c] += 1
            W[k_off * n + 0, c] += 1; W[(k_off + 1) * n + 1, c] += 1
            c += 1
            # P2 = (A10 + A11) B00
            U[1 * m + j_off, c] += 1; U[1 * m + (j_off + 1), c] += 1
            V[j_off * p + k_off, c] += 1
            W[k_off * n + 1, c] += 1; W[(k_off + 1) * n + 1, c] -= 1
            c += 1
            # P3 = A00 (B01 - B11)
            U[0 * m + j_off, c] += 1
            V[j_off * p + (k_off + 1), c] += 1; V[(j_off + 1) * p + (k_off + 1), c] -= 1
            W[(k_off + 1) * n + 0, c] += 1; W[(k_off + 1) * n + 1, c] += 1
            c += 1
            # P4 = A11 (B10 - B00)
            U[1 * m + (j_off + 1), c] += 1
            V[(j_off + 1) * p + k_off, c] += 1; V[j_off * p + k_off, c] -= 1
            W[k_off * n + 0, c] += 1; W[k_off * n + 1, c] += 1
            c += 1
            # P5 = (A00 + A01) B11
            U[0 * m + j_off, c] += 1; U[0 * m + (j_off + 1), c] += 1
            V[(j_off + 1) * p + (k_off + 1), c] += 1
            W[k_off * n + 0, c] -= 1; W[(k_off + 1) * n + 0, c] += 1
            c += 1
            # P6 = (A10 - A00)(B00 + B01)
            U[1 * m + j_off, c] += 1; U[0 * m + j_off, c] -= 1
            V[j_off * p + k_off, c] += 1; V[j_off * p + (k_off + 1), c] += 1
            W[(k_off + 1) * n + 1, c] += 1
            c += 1
            # P7 = (A01 - A11)(B10 + B11)
            U[0 * m + (j_off + 1), c] += 1; U[1 * m + (j_off + 1), c] -= 1
            V[(j_off + 1) * p + k_off, c] += 1; V[(j_off + 1) * p + (k_off + 1), c] += 1
            W[k_off * n + 0, c] += 1
            return c

        c = add_strassen(0, 0, 0)
        c = add_strassen(0, 2, c)
        c = add_strassen(2, 0, c)
        c = add_strassen(2, 2, c)

        # tail: column j = 4
        for i in range(n):
            for k in range(4):
                U[i * m + 4, c] = 1; V[4 * p + k, c] = 1; W[k * n + i, c] = 1; c += 1
        # tail: row k = 4
        for i in range(n):
            for j in range(4):
                U[i * m + j, c] = 1; V[j * p + 4, c] = 1; W[4 * n + i, c] = 1; c += 1
        # corner (4,4)
        for i in range(n):
            U[i * m + 4, c] = 1; V[4 * p + 4, c] = 1; W[4 * n + i, c] = 1; c += 1

        return U, V, W

    U46, V46, W46 = build_rank46()
    base_valid = validate(U46, V46, W46)

    def try_rank_reduction(U_base, V_base, W_base, target_rank):
        R = target_rank
        num_cols = U_base.shape[1]
        
        # Calculate a score for each column to decide which to drop first
        # Prefer dropping columns with simpler structure (fewer non-zeros) or those in the tail
        col_scores = []
        for c in range(num_cols):
            nz_u = np.count_nonzero(U_base[:, c])
            nz_v = np.count_nonzero(V_base[:, c])
            nz_w = np.count_nonzero(W_base[:, c])
            # Tail columns are typically simpler (1 non-zero in each)
            # Strassen columns have 2 or 3 non-zeros
            score = nz_u + nz_v + nz_w
            col_scores.append((score, c))
        
        # Sort by score (ascending), so we try dropping the simplest ones first
        # Actually, dropping a "simple" column leaves a "simple" hole.
        # Dropping a "complex" column might be harder to fill.
        # Let's try dropping the tail columns first (indices 28..45)
        # They are the last 18 columns.
        
        # Create a list of candidate columns to drop, prioritizing tail
        candidates = [c for c in range(28, 46)] + [c for c in range(0, 28)]
        
        for drop_idx in candidates:
            if time.time() > deadline - 20:
                break
            
            mask = [i for i in range(num_cols) if i != drop_idx]
            U_init = U_base[:, mask]
            V_init = V_base[:, mask]
            W_init = W_base[:, mask]
            
            # Flatten for scipy
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def loss_and_grad(x):
                U = x[:U_dim*R].reshape(U_dim, R)
                V = x[U_dim*R:U_dim*R+V_dim*R].reshape(V_dim, R)
                W = x[U_dim*R+V_dim*R:].reshape(W_dim, R)
                
                prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                diff = prod - T
                loss = float(np.sum(diff * diff))
                
                # Gradients
                gU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
                gV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
                gW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
                
                grad = np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
                return loss, grad
            
            try:
                res = scipy.optimize.minimize(
                    loss_and_grad, x0, method='L-BFGS-B', jac=True,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.success or res.fun < 1e-8:
                    # Extract best solution
                    x_best = res.x
                    U_best = x_best[:U_dim*R].reshape(U_dim, R)
                    V_best = x_best[U_dim*R:U_dim*R+V_dim*R].reshape(V_dim, R)
                    W_best = x_best[U_dim*R+V_dim*R:].reshape(W_dim, R)
                    
                    # Snap to half-integers
                    U_snap = np.round(U_best * 2) / 2
                    V_snap = np.round(V_best * 2) / 2
                    W_snap = np.round(W_best * 2) / 2
                    
                    if validate(U_snap, V_snap, W_snap):
                        return U_snap, V_snap, W_snap
                        
            except Exception:
                continue
                
        return None

    if base_valid:
        # Attempt Rank 45
        if time.time() < deadline - 30:
            result_45 = try_rank_reduction(U46, V46, W46, 45)
            if result_45 is not None:
                U, V, W = result_45
                return {
                    "U": [[float(x) for x in row] for row in U],
                    "V": [[float(x) for x in row] for row in V],
                    "W": [[float(x) for x in row] for row in W],
                }
        
        # Fallback to Rank 46
        return {
            "U": [[float(x) for x in row] for row in U46],
            "V": [[float(x) for x in row] for row in V46],
            "W": [[float(x) for x in row] for row in W46],
        }
    else:
        # Fallback to Naive Rank 50
        rank = n * m * p
        U = np.zeros((U_dim, rank)); V = np.zeros((V_dim, rank)); W = np.zeros((W_dim, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1; V[j * p + k, c] = 1; W[k * n + i, c] = 1; c += 1
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W],
        }
# EVOLVE-BLOCK-END
