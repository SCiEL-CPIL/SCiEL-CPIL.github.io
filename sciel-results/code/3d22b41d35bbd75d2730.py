"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120.0
    
    # Precompute target indices for efficient checking and target vector construction
    # T[a, b, c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i
    target_set = set()
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_set.add((a, b, c))
                target_indices.append((a, b, c))
    
    # Schoolbook solution (Rank 45)
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, rank

    U_best, V_best, W_best, rank_best = get_schoolbook()

    def check_decomposition(U, V, W, rank, tol=1e-6):
        """Verify the decomposition exactly."""
        # Check targets
        for a, b, c in target_indices:
            val = np.dot(U[a], np.multiply(V[b], W[c]))
            if abs(val - 1.0) > tol:
                return False
        # Check non-targets (sample for speed, or all if small)
        # Total triples: 9*15*15 = 2025. This is small enough to check all.
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    if (a, b, c) in target_set:
                        continue
                    val = np.dot(U[a], np.multiply(V[b], W[c]))
                    if abs(val) > tol:
                        return False
        return True

    def try_rank(r, max_iter=200, seed=0):
        """
        Alternating Least Squares (ALS) approach.
        We solve for U, V, W sequentially.
        """
        if time.time() > deadline - 5:
            return None
        
        np.random.seed(seed)
        U = np.random.randn(n*m, r) * 0.01
        V = np.random.randn(m*p, r) * 0.01
        W = np.random.randn(p*n, r) * 0.01
        
        # Precompute design matrices structure? 
        # For U[a]: Design matrix A_U has rows (b, c). A_U[b, c, :] = V[b, :] * W[c, :]
        # We can construct A_U as a 2D array of shape (m*p*p*n, r)
        # Row index idx corresponds to (b, c).
        
        # Helper to construct design matrix for U
        def build_A_U(V, W):
            # Shape (m*p, p*n, r) -> flatten to (m*p*p*n, r)
            # V[b, r] * W[c, r]
            # We can use broadcasting
            # V is (m*p, r), W is (p*n, r)
            # V[:, np.newaxis, :] is (m*p, 1, r)
            # W[np.newaxis, :, :] is (1, p*n, r)
            # Product is (m*p, p*n, r)
            A = V[:, np.newaxis, :] * W[np.newaxis, :, :]
            return A.reshape(-1, r)

        # Helper to construct target vector for U[a]
        def build_t_U(a, V, W):
            # t[idx] = 1 if (a, b, c) is target, else 0
            # idx = b * (p*n) + c
            t = np.zeros(m*p*p*n)
            for b in range(m*p):
                for c in range(p*n):
                    if (a, b, c) in target_set:
                        t[b * (p*n) + c] = 1.0
            return t

        # Similar for V and W
        def build_A_V(U, W):
            # A_V[a, c, :] = U[a, :] * W[c, :]
            A = U[:, np.newaxis, :] * W[np.newaxis, :, :]
            return A.reshape(-1, r)

        def build_t_V(b, U, W):
            # t[idx] = 1 if (a, b, c) is target
            # idx = a * (p*n) + c
            t = np.zeros(n*m*p*n)
            for a in range(n*m):
                for c in range(p*n):
                    if (a, b, c) in target_set:
                        t[a * (p*n) + c] = 1.0
            return t

        def build_A_W(U, V):
            # A_W[a, b, :] = U[a, :] * V[b, :]
            A = U[:, np.newaxis, :] * V[np.newaxis, :, :]
            return A.reshape(-1, r)

        def build_t_W(c, U, V):
            # t[idx] = 1 if (a, b, c) is target
            # idx = a * (m*p) + b
            t = np.zeros(n*m*m*p)
            for a in range(n*m):
                for b in range(m*p):
                    if (a, b, c) in target_set:
                        t[a * (m*p) + b] = 1.0
            return t

        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Update U
            A_U = build_A_U(V, W)
            for a in range(n*m):
                t_U = build_t_U(a, V, W)
                # Solve A_U @ U[a] = t_U
                # A_U is (N, r), U[a] is (r,)
                try:
                    U[a], _, _, _ = np.linalg.lstsq(A_U, t_U, rcond=None)
                except:
                    pass

            # Update V
            A_V = build_A_V(U, W)
            for b in range(m*p):
                t_V = build_t_V(b, U, W)
                try:
                    V[b], _, _, _ = np.linalg.lstsq(A_V, t_V, rcond=None)
                except:
                    pass

            # Update W
            A_W = build_A_W(U, V)
            for c in range(p*n):
                t_W = build_t_W(c, U, V)
                try:
                    W[c], _, _, _ = np.linalg.lstsq(A_W, t_W, rcond=None)
                except:
                    pass
            
            # Check convergence periodically
            if it % 10 == 0:
                if check_decomposition(U, V, W, r, tol=1e-4):
                    return U, V, W, r
        
        # Final check
        if check_decomposition(U, V, W, r, tol=1e-6):
            return U, V, W, r
            
        return None

    # Try to find a lower rank. 
    # Known lower bounds for 3x3x5 are likely close to schoolbook or slightly less.
    # We try a few random restarts for ranks < 45.
    # To save time, we only try a few specific ranks.
    ranks_to_try = [44, 43, 42, 41, 40]
    
    for r in ranks_to_try:
        if time.time() > deadline - 20:
            break
            
        found = False
        for seed in range(5): # 5 restarts per rank
            if time.time() > deadline - 10:
                break
            res = try_rank(r, max_iter=50, seed=seed)
            if res is not None:
                U, V, W, rank = res
                # Final high-precision verification
                if check_decomposition(U, V, W, rank, tol=1e-8):
                    U_best, V_best, W_best, rank_best = U, V, W, rank
                    found = True
                    break
        if found:
            break # Found a valid lower rank, stop searching lower ranks
    
    # If we didn't find a lower rank, we stick with the schoolbook (rank 45)
    # which is guaranteed to work.
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    return {"U": to_list(U_best), "V": to_list(V_best), "W": to_list(W_best)}
# EVOLVE-BLOCK-END
