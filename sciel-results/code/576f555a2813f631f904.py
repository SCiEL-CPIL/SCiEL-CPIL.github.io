import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = 150.0 - 15.0  # 15-second safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T as a 3D array
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W form a valid rank-r decomposition."""
        r = U.shape[1]
        T_rec = np.zeros((nm, mp, pn))
        for r_idx in range(r):
            T_rec += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        return np.allclose(T_rec, T, atol=tol)
    
    def compute_error(U, V, W):
        """Compute the max error of the reconstruction."""
        r = U.shape[1]
        T_rec = np.zeros((nm, mp, pn))
        for r_idx in range(r):
            T_rec += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        return np.max(np.abs(T_rec - T))
    
    def try_structured_search(r, max_attempts=2000):
        """
        Try to find a rank-r decomposition using structured search.
        Uses the natural grouping of the tensor indices.
        """
        if time.time() > deadline:
            return None
        
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + attempt * 13)
            
            # Strategy: Create structured patterns based on the tensor's natural grouping
            # i in 0..2 (output row), j in 0..2 (shared index), k in 0..6 (output col)
            
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            W = np.zeros((pn, r))
            
            # For each rank-1 term, assign a pattern
            for r_idx in range(r):
                # Choose which (i,j,k) "block" this term covers
                # Use a structured assignment
                
                # Pattern 1: Cover one (i,j) pair for all k
                if attempt % 4 == 0:
                    i = r_idx % n
                    j = (r_idx // n) % m
                    k_start = (r_idx // (n*m)) % p
                    k_end = min(p, k_start + 1 + (attempt % 3))
                    
                    U[i*m+j, r_idx] = 1
                    for k in range(k_start, k_end):
                        V[j*p+k, r_idx] = 1
                        W[k*n+i, r_idx] = 1
                
                # Pattern 2: Cover one (j,k) pair for all i
                elif attempt % 4 == 1:
                    j = r_idx % m
                    k = (r_idx // m) % p
                    i_start = (r_idx // (m*p)) % n
                    i_end = min(n, i_start + 1 + (attempt % 2))
                    
                    for i in range(i_start, i_end):
                        U[i*m+j, r_idx] = 1
                        V[j*p+k, r_idx] = 1
                        W[k*n+i, r_idx] = 1
                
                # Pattern 3: Cover one (i,k) pair for all j
                elif attempt % 4 == 2:
                    i = r_idx % n
                    k = (r_idx // n) % p
                    j_start = (r_idx // (n*p)) % m
                    j_end = min(m, j_start + 1 + (attempt % 2))
                    
                    for j in range(j_start, j_end):
                        U[i*m+j, r_idx] = 1
                        V[j*p+k, r_idx] = 1
                        W[k*n+i, r_idx] = 1
                
                # Pattern 4: Random sparse assignment
                else:
                    # Choose a random subset of indices
                    num_terms = np.random.randint(1, 4)
                    for _ in range(num_terms):
                        i = np.random.randint(0, n)
                        j = np.random.randint(0, m)
                        k = np.random.randint(0, p)
                        U[i*m+j, r_idx] += 1
                        V[j*p+k, r_idx] += 1
                        W[k*n+i, r_idx] += 1
            
            # Compute error
            error = compute_error(U, V, W)
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if error < 1e-8:
                return (U, V, W)
            
            # Local refinement if error is small
            if error < 0.5 and attempt % 5 == 0:
                for _ in range(30):
                    if time.time() > deadline:
                        break
                    
                    # Try flipping a random entry
                    mat_choice = np.random.randint(0, 3)
                    if mat_choice == 0:
                        row, col = np.random.randint(0, nm), np.random.randint(0, r)
                        old_val = U[row, col]
                        U[row, col] = old_val + np.random.choice([-1, 1])
                    elif mat_choice == 1:
                        row, col = np.random.randint(0, mp), np.random.randint(0, r)
                        old_val = V[row, col]
                        V[row, col] = old_val + np.random.choice([-1, 1])
                    else:
                        row, col = np.random.randint(0, pn), np.random.randint(0, r)
                        old_val = W[row, col]
                        W[row, col] = old_val + np.random.choice([-1, 1])
                    
                    new_error = compute_error(U, V, W)
                    
                    if new_error < error:
                        error = new_error
                        if error < 1e-8:
                            return (U, V, W)
                    else:
                        # Revert
                        if mat_choice == 0:
                            U[row, col] = old_val
                        elif mat_choice == 1:
                            V[row, col] = old_val
                        else:
                            W[row, col] = old_val
        
        if best_U is not None and check_decomposition(best_U, best_V, best_W):
            return (best_U, best_V, best_W)
        
        return None
    
    def try_gradient_descent(r, max_iters=5000, lr=0.05):
        """Try to find a rank-r decomposition using gradient descent."""
        if time.time() > deadline:
            return None
        
        np.random.seed(42)
        U = np.random.randn(nm, r) * 0.5
        V = np.random.randn(mp, r) * 0.5
        W = np.random.randn(pn, r) * 0.5
        
        best_error = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute T_rec
            T_rec = np.zeros((nm, mp, pn))
            for r_idx in range(r):
                T_rec += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
            
            error = T_rec - T
            max_error = np.max(np.abs(error))
            
            if max_error < best_error:
                best_error = max_error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if max_error < 1e-8:
                return (U, V, W)
            
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for r_idx in range(r):
                # grad_U[:, r_idx] = sum_{b,c} error[a,b,c] * V[b,r_idx] * W[c,r_idx]
                grad_U[:, r_idx] = np.sum(error * np.einsum('b,c->bc', V[:, r_idx], W[:, r_idx])[None, :, :], axis=(1, 2))
                # grad_V[:, r_idx] = sum_{a,c} error[a,b,c] * U[a,r_idx] * W[c,r_idx]
                grad_V[:, r_idx] = np.sum(error * np.einsum('a,c->ac', U[:, r_idx], W[:, r_idx])[None, :, :], axis=(0, 2))
                # grad_W[:, r_idx] = sum_{a,b} error[a,b,c] * U[a,r_idx] * V[b,r_idx]
                grad_W[:, r_idx] = np.sum(error * np.einsum('a,b->ab', U[:, r_idx], V[:, r_idx])[None, :, :], axis=(0, 1))
            
            # Adaptive learning rate
            current_lr = lr * (0.95 ** (it // 100))
            
            U -= current_lr * grad_U
            V -= current_lr * grad_V
            W -= current_lr * grad_W
            
            # Occasionally restart if stuck
            if it % 500 == 0 and max_error > 0.1:
                np.random.seed(42 + it)
                U = np.random.randn(nm, r) * 0.5
                V = np.random.randn(mp, r) * 0.5
                W = np.random.randn(pn, r) * 0.5
        
        if check_decomposition(best_U, best_V, best_W):
            return (best_U, best_V, best_W)
        
        return None
    
    # Try to find the minimum rank
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    # Try ranks from a reasonable upper bound down to lower bound
    # Theoretical lower bound: max(nm, mp, pn) = 21
    # Known results suggest rank around 30-50 for this size
    
    ranks_to_try = [45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30]
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Try structured search first (faster)
        result = try_structured_search(r, max_attempts=1500)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_result = (U, V, W)
            break
        
        # Try gradient descent if structured search fails
        if time.time() < deadline - 30:
            result = try_gradient_descent(r, max_iters=3000)
            if result is not None:
                U, V, W = result
                best_rank = r
                best_result = (U, V, W)
                break
    
    U, V, W = best_result
    
    # Convert to lists of lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
