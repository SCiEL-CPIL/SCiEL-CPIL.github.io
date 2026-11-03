"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 75
    best_rank = rank_schoolbook
    best_decomp = None

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

    # Start with schoolbook as fallback
    best_decomp = make_schoolbook()

    def compute_error(U, V, W):
        """
        Compute the sum of squared errors for the tensor decomposition.
        U: (n*m, r)
        V: (m*p, r)
        W: (p*n, r)
        """
        r = U.shape[1]
        # Reshape to 3D
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        
        # We want S[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
        # This is a bit tricky with standard einsum because indices wrap around.
        # S[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        
        # Let's compute it directly for small sizes
        # Or use einsum:
        # U[i,j,c] * V[j,k,c] -> M[i,j,k,c]
        # M[i,j,k,c] * W[k,i,c] -> S[i,j,k]
        
        # M = np.einsum('ijc,jkc->ijkc', U3, V3)
        # S = np.einsum('ijkc,kic->ijk', M, W3)
        
        # Check dimensions:
        # U3: (n, m, r)
        # V3: (m, p, r)
        # W3: (p, n, r)
        
        # M[i,j,k,c] = U3[i,j,c] * V3[j,k,c]
        # S[i,j,k] = sum_c M[i,j,k,c] * W3[k,i,c]
        
        # Note: W3 is indexed by [k, i, c]. In W3, first index is k (0..p-1), second is i (0..n-1).
        # So W3[k,i,c] is correct.
        
        M = np.einsum('ijc,jkc->ijkc', U3, V3)
        S = np.einsum('ijkc,kic->ijk', M, W3)
        
        # Target is 1 for all i,j,k
        diff = S - 1.0
        return np.sum(diff**2)

    def round_to_half(x):
        # Round to nearest 0.5
        return np.round(x * 2) / 2.0

    def discrete_local_search(U, V, W, max_steps=2000, seed=0):
        """
        Try to reduce error to 0 by modifying entries to be in {-1, -0.5, 0, 0.5, 1}.
        """
        np.random.seed(seed)
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        # Initial rounding
        U = round_to_half(U)
        V = round_to_half(V)
        W = round_to_half(W)
        
        current_error = compute_error(U, V, W)
        
        if current_error < 1e-6:
            return U, V, W, current_error
        
        best_error = current_error
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Precompute indices for faster error update? 
        # For now, recompute. It's O(n*m*p*r). 
        # 3*5*5*60 = 4500 ops. 2000 steps = 9M ops. Fast enough.
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            # Pick a random matrix and entry
            mat_id = np.random.randint(3)
            if mat_id == 0:
                mat = U
                shape = U.shape
            elif mat_id == 1:
                mat = V
                shape = V.shape
            else:
                mat = W
                shape = W.shape
            
            idx = np.random.randint(shape[0])
            col = np.random.randint(shape[1])
            
            old_val = mat[idx, col]
            
            # Try changing to a different half-integer
            # Possible values: -1, -0.5, 0, 0.5, 1
            candidates = [-1.0, -0.5, 0.0, 0.5, 1.0]
            if old_val in candidates:
                candidates.remove(old_val)
            
            # Pick a random candidate
            new_val = np.random.choice(candidates)
            
            mat[idx, col] = new_val
            
            new_error = compute_error(U, V, W)
            
            if new_error < best_error:
                best_error = new_error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if best_error < 1e-6:
                    return best_U, best_V, best_W, best_error
            else:
                # Revert
                mat[idx, col] = old_val
        
        return best_U, best_V, best_W, best_error

    # Search for lower ranks
    # Start from a reasonable lower bound
    # Known lower bound for n*m*p is often close to n*m*p.
    # Let's try from 60 up to 74.
    
    for target_rank in range(60, rank_schoolbook):
        if time.time() > deadline - 20:
            break
        
        # Try a few random initializations for this rank
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(target_rank * 1000 + attempt)
            
            # Initialize with small random values
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Perform a few steps of gradient descent to minimize error
            # This helps get close to a solution before rounding
            lr = 0.01
            for gd_step in range(50):
                if time.time() > deadline - 10:
                    break
                
                # Compute gradients
                # S = sum_c U V W
                # dE/dU = 2 * (S-1) * V * W
                # This is complex to compute directly with einsums for gradients.
                # Let's just use a simple random perturbation or skip GD and go straight to rounding + local search.
                # Given the small size, maybe just random restarts with local search is sufficient.
                pass
            
            # Round to half-integers
            U_r = round_to_half(U)
            V_r = round_to_half(V)
            W_r = round_to_half(W)
            
            # Local search
            U_f, V_f, W_f, err = discrete_local_search(U_r, V_r, W_f, max_steps=1000, seed=target_rank*100+attempt)
            
            if err < 1e-6:
                # Found a valid decomposition
                best_rank = target_rank
                best_decomp = {
                    "U": U_f.tolist(),
                    "V": V_f.tolist(),
                    "W": W_f.tolist()
                }
                break
        
        if best_rank < rank_schoolbook:
            break

    if best_rank == rank_schoolbook:
        best_decomp = make_schoolbook()

    return best_decomp
# EVOLVE-BLOCK-END
