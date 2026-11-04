"""Combined program: structural rank search with exact rational recovery."""
import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build target tensor T[i,j,k] = 1 if consistent, else 0
    # T is shape (n, m, p)
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    
    # Flattened target for verification
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # Product sum: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal T[i,j,k]
    
    N, M, P = n, m, p
    
    def verify(U, V, W):
        """
        Check if U, V, W exactly decompose T.
        U: (N*M, r), V: (M*P, r), W: (P*N, r)
        """
        r = U.shape[1]
        U_reshaped = U.reshape(N, M, r)
        V_reshaped = V.reshape(M, P, r)
        W_reshaped = W.reshape(P, N, r)
        
        # T_rec[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Using einsum: 'ijc,jkc,kic->ijk'
        T_rec = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        # Check exactness with small tolerance for floating point
        # Since we aim for integer/half-integer, 1e-9 is safe
        return np.allclose(T_rec, T, atol=1e-9)
    
    def objective_and_grad(x, r):
        size_U = N * M * r
        size_V = M * P * r
        size_W = P * N * r
        
        U_flat = x[:size_U]
        V_flat = x[size_U:size_U+size_V]
        W_flat = x[size_U+size_V:]
        
        U = U_flat.reshape(N, M, r)
        V = V_flat.reshape(M, P, r)
        W = W_flat.reshape(P, N, r)
        
        T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        err = T_rec - T
        loss = np.sum(err**2)
        
        # Gradients
        # dL/dU[i,j,c] = 2 * sum_k err[i,j,k] * V[j,k,c] * W[k,i,c]
        grad_U = 2 * np.einsum('ijk,jkc,kic->ijc', err, V, W)
        # dL/dV[j,k,c] = 2 * sum_i err[i,j,k] * U[i,j,c] * W[k,i,c]
        grad_V = 2 * np.einsum('ijk,ijc,kic->jkc', err, U, W)
        # dL/dW[k,i,c] = 2 * sum_j err[i,j,k] * U[i,j,c] * V[j,k,c]
        grad_W = 2 * np.einsum('ijk,ijc,jkc->kic', err, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad
    
    def try_rank(r, seed=0):
        if time.time() > deadline:
            return None
            
        np.random.seed(seed)
        size_U = N * M * r
        size_V = M * P * r
        size_W = P * N * r
        total_size = size_U + size_V + size_W
        
        # Initialize with small random values
        x0 = np.random.randn(total_size) * 0.5
        
        # Use L-BFGS-B for optimization
        try:
            res = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, 
                           options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 20})
        except Exception:
            return None
            
        if time.time() > deadline:
            return None
            
        # Check if converged to a low error
        if res.fun < 1e-6:
            # Extract optimized tensors
            U_opt = res.x[:size_U].reshape(N, M, r)
            V_opt = res.x[size_U:size_U+size_V].reshape(M, P, r)
            W_opt = res.x[size_U+size_V:].reshape(P, N, r)
            
            # Try to snap to rational values (integers or halves)
            # Strategy 1: Round to nearest integer
            U_int = np.round(U_opt)
            V_int = np.round(V_opt)
            W_int = np.round(W_opt)
            
            if verify(U_int, V_int, W_int):
                return U_int.flatten(axis=1), V_int.flatten(axis=1), W_int.flatten(axis=1)
            
            # Strategy 2: Round to nearest half-integer
            U_half = np.round(U_opt * 2) / 2.0
            V_half = np.round(V_opt * 2) / 2.0
            W_half = np.round(W_opt * 2) / 2.0
            
            if verify(U_half, V_half, W_half):
                return U_half.flatten(axis=1), V_half.flatten(axis=1), W_half.flatten(axis=1)
            
            # Strategy 3: If close but not exact, try a local refinement or different snap
            # For now, if it's very close, we might try a few more seeds or return failure
            if res.fun < 1e-8:
                # It's likely a valid solution but floating point noise prevents exact snap.
                # We can try to solve for the exact rational values if the structure allows.
                # However, given time constraints, we return None to try other ranks/seeds.
                pass
        
        return None
    
    # Search for minimal rank
    # Theoretical lower bound is max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # But for matrix multiplication, Strassen gives lower bounds.
    # For 3x4 * 4x6, schoolbook is 72.
    # Let's search from a reasonable lower bound upwards, or downwards from schoolbook?
    # Downwards is safer as we know schoolbook works.
    # However, finding a valid decomposition at a specific rank is hard.
    # Let's try ranks from 30 up to 71? Or 71 down to 30?
    # Given the difficulty, let's try a few specific ranks known to be potential candidates or just scan.
    # Actually, scanning all ranks is expensive.
    # Let's start from rank_schoolbook - 1 downwards, but stop if we find one.
    # Or start from a lower bound and go up?
    # The prompt asks for rank-minimal.
    # Let's try ranks in range [24, 72].
    # To save time, we'll prioritize lower ranks but limit the number of tries per rank.
    
    # We know rank 72 works (schoolbook).
    # Let's try to find a lower rank.
    # We'll iterate r from 24 to 71.
    # For each r, we'll try a few seeds.
    
    best_result = None
    
    # First, ensure we have a fallback (schoolbook)
    U_fb = np.zeros((N*M, rank_schoolbook))
    V_fb = np.zeros((M*P, rank_schoolbook))
    W_fb = np.zeros((P*N, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1.0
                V_fb[j*p+k, col] = 1.0
                W_fb[k*n+i, col] = 1.0
                col += 1
    best_result = (U_fb, V_fb, W_fb)
    best_rank = rank_schoolbook
    
    # Search for lower rank
    # We'll try a subset of ranks to stay within time.
    # Let's try every 2nd rank or so? Or just go down from 71?
    # Let's go down from 71 to 24.
    # But optimization is slow.
    # Let's limit total optimization calls.
    
    max_opt_calls = 10
    opt_calls = 0
    
    for r in range(rank_schoolbook - 1, 23, -1):
        if time.time() > deadline or opt_calls >= max_opt_calls:
            break
            
        # Try a few seeds for this rank
        for seed in range(3):
            if time.time() > deadline or opt_calls >= max_opt_calls:
                break
                
            opt_calls += 1
            result = try_rank(r, seed)
            if result:
                U, V, W = result
                best_result = (U, V, W)
                best_rank = r
                # If we found a lower rank, we can break out of seed loop
                # and continue to lower ranks to see if we can go even lower
                break
                
    # Convert best_result to list of lists for output
    U_out, V_out, W_out = best_result
    U_list = U_out.tolist()
    V_list = V_out.tolist()
    W_list = W_out.tolist()
    
    # Ensure entries are integers or half-integers (floats are fine, but let's clean up)
    # The problem statement says "integer/half-integer entries".
    # Our snapping ensures this.
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
