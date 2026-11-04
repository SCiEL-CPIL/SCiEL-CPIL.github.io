"""Improved program for matrix multiplication tensor decomposition (5,5,5)."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Deadline: 140s out of 150s
    t0 = time.time()
    deadline = t0 + 140.0

    # First, get the schoolbook decomposition as a fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    def verify(U, V, W):
        """Verify that the tensor decomposition is exact."""
        r = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        T_computed = np.zeros((n, m, p))
        for c in range(r):
            Uc = Ua[:, c].reshape(n, m)
            Vc = Va[:, c].reshape(m, p)
            Wc = Wa[:, c].reshape(p, n)
            # T[i,j,k] += U[i,j,c] * V[j,k,c] * W[k,i,c]
            # For fixed i,k: sum_j U[i,j,c] * V[j,k,c] * W[k,i,c]
            # We can compute the full tensor via einsum
            # Actually, let's just use the direct loop but vectorized over c
            # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
        # Use einsum for efficiency
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        U_r = Ua.reshape(n, m, r)
        V_r = Va.reshape(m, p, r)
        W_r = Wa.reshape(p, n, r)
        
        # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
        T_computed = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        T_target = np.ones((n, m, p))
        return np.allclose(T_computed, T_target, atol=1e-10)

    def compute_tensor(U, V, W):
        """Compute the tensor from factors."""
        r = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        U_r = Ua.reshape(n, m, r)
        V_r = Va.reshape(m, p, r)
        W_r = Wa.reshape(p, n, r)
        return np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)

    def try_structured_search(rank, max_iters=200):
        """Try to find a decomposition of given rank using structured random search 
        with entries from {0, 0.5, 1, -0.5, -1} and local improvement."""
        nonlocal best_result, best_rank, deadline
        
        if rank >= best_rank:
            return False
        
        vals = [0, 0.5, 1, -0.5, -1]
        np.random.seed(rank * 1000 + 42)
        
        for attempt in range(30):
            if time.time() > deadline:
                return False
            
            # Random initialization with structured values
            U = np.random.choice(vals, size=(n * m, rank))
            V = np.random.choice(vals, size=(m * p, rank))
            W = np.random.choice(vals, size=(p * n, rank))
            
            # Quick check
            T_comp = compute_tensor(U.tolist(), V.tolist(), W.tolist())
            error = np.sum((T_comp - 1.0) ** 2)
            
            if error < 1e-8:
                # Found a solution!
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                if verify(U_list, V_list, W_list):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    return True
                continue
            
            # Local search: try flipping individual entries
            for step in range(max_iters):
                if time.time() > deadline:
                    return False
                
                # Try flipping a random entry in U, V, or W
                factor = np.random.randint(0, 3)
                if factor == 0:
                    row = np.random.randint(0, n * m)
                    col = np.random.randint(0, rank)
                    old_val = U[row, col]
                    new_val = np.random.choice([v for v in vals if v != old_val])
                    U[row, col] = new_val
                    T_comp = compute_tensor(U.tolist(), V.tolist(), W.tolist())
                    new_error = np.sum((T_comp - 1.0) ** 2)
                    if new_error >= error:
                        U[row, col] = old_val
                    else:
                        error = new_error
                elif factor == 1:
                    row = np.random.randint(0, m * p)
                    col = np.random.randint(0, rank)
                    old_val = V[row, col]
                    new_val = np.random.choice([v for v in vals if v != old_val])
                    V[row, col] = new_val
                    T_comp = compute_tensor(U.tolist(), V.tolist(), W.tolist())
                    new_error = np.sum((T_comp - 1.0) ** 2)
                    if new_error >= error:
                        V[row, col] = old_val
                    else:
                        error = new_error
                else:
                    row = np.random.randint(0, p * n)
                    col = np.random.randint(0, rank)
                    old_val = W[row, col]
                    new_val = np.random.choice([v for v in vals if v != old_val])
                    W[row, col] = new_val
                    T_comp = compute_tensor(U.tolist(), V.tolist(), W.tolist())
                    new_error = np.sum((T_comp - 1.0) ** 2)
                    if new_error >= error:
                        W[row, col] = old_val
                    else:
                        error = new_error
                
                if error < 1e-8:
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    if verify(U_list, V_list, W_list):
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = rank
                        return True
                    break
        
        return False

    def try_gradient_search(rank, max_iters=100):
        """Try to find a decomposition using gradient descent with smooth objective."""
        nonlocal best_result, best_rank, deadline
        
        if rank >= best_rank:
            return False
        
        np.random.seed(rank * 7 + 13)
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        # Normalize
        for c in range(rank):
            for mat in [U, V, W]:
                norm = np.linalg.norm(mat[:, c])
                if norm > 1e-10:
                    mat[:, c] /= norm
        
        lr = 0.05
        best_error = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute tensor and gradients
            U_r = U.reshape(n, m, rank)
            V_r = V.reshape(m, p, rank)
            W_r = W.reshape(p, n, rank)
            
            T_comp = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
            diff = T_comp - 1.0
            error = np.sum(diff ** 2)
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if error < 1e-8:
                # Found a solution!
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                # Round to nearest 0.5
                U_rounded = [[round(x * 2) / 2 for x in row] for row in U_list]
                V_rounded = [[round(x * 2) / 2 for x in row] for row in V_list]
                W_rounded = [[round(x * 2) / 2 for x in row] for row in W_list]
                if verify(U_rounded, V_rounded, W_rounded):
                    best_result = {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                    best_rank = rank
                    return True
                else:
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    return True
            
            # Compute gradients
            dU_r = np.einsum('ijk,jkc,kic->ijc', diff, V_r, W_r)
            dV_r = np.einsum('ijk,ijc,kic->jkc', diff, U_r, W_r)
            dW_r = np.einsum('ijk,ijc,jkc->kic', diff, U_r, V_r)
            
            dU = 2 * dU_r.reshape(n * m, rank)
            dV = 2 * dV_r.reshape(m * p, rank)
            dW = 2 * dW_r.reshape(p * n, rank)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Periodic renormalization
            if it % 20 == 0 and it > 0:
                for c in range(rank):
                    for mat in [U, V, W]:
                        norm = np.linalg.norm(mat[:, c])
                        if norm > 1e-10:
                            mat[:, c] /= norm
        
        # Check best found
        if best_error < 1e-6:
            U_list = best_U.tolist()
            V_list = best_V.tolist()
            W_list = best_W.tolist()
            U_rounded = [[round(x * 2) / 2 for x in row] for row in U_list]
            V_rounded = [[round(x * 2) / 2 for x in row] for row in V_list]
            W_rounded = [[round(x * 2) / 2 for x in row] for row in W_list]
            if verify(U_rounded, V_rounded, W_rounded):
                best_result = {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                best_rank = rank
                return True
        
        return False

    # Try various ranks from lower to higher
    # Known bounds: rank(5,5,5) is between 49 (lower bound) and 125 (schoolbook)
    ranks_to_try = [70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 124]
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        if r >= best_rank:
            continue
        
        # Try structured search first (faster per attempt)
        success = try_structured_search(r, max_iters=50)
        if success:
            break
        
        if time.time() > deadline:
            break
        
        # Try gradient search
        success = try_gradient_search(r, max_iters=50)
        if success:
            break
    
    return best_result
# EVOLVE-BLOCK-END
