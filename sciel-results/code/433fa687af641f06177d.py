<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Schoolbook baseline (always valid)
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U,V,W form a valid decomposition.
        For all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should equal 1.
        """
        # Build the tensor
        # U shape: (n*m, r)
        # V shape: (m*p, r)
        # W shape: (p*n, r)
        r = U.shape[1]
        
        # For each (i,j,k), compute sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # This is equivalent to: for each c, the outer product structure
        # Let's compute it efficiently
        
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        
        # Compute the tensor T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Use einsum
        # First compute A[i,j,k] = sum_c U[i,j,c] * V[j,k,c] -> shape (n, m, p)
        # Then T[i,j,k] = sum_k' ... wait no
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        
        # Let's compute step by step
        # For fixed i,j: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # = sum_c U[i,j,c] * (V[j,:] * W[:,i])_k  ... no that's not right either
        # Actually for fixed i,j,k: it's a scalar
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        
        # Using einsum:
        # T[i,j,k] = U[i,j,c] * V[j,k,c] * W[k,i,c] summed over c
        # = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        # Check if T is all ones
        return np.allclose(T, np.ones((n, m, p)), atol=tol)
    
    def try_rank_optimization(target_rank, num_trials=10, max_iter=50):
        """Try to find a decomposition of given rank using iterative optimization."""
        if time.time() > deadline:
            return None
        
        n, m, p = DIMENSIONS
        nm = n * m
        mp = m * p
        pn = p * n
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Random initialization
            U = np.random.randn(nm, target_rank) * 0.5
            V = np.random.randn(mp, target_rank) * 0.5
            W = np.random.randn(pn, target_rank) * 0.5
            
            # Iterative optimization: minimize ||T - 1||^2 where T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
            # Use gradient descent with learning rate scheduling
            
            lr = 0.01
            best_loss = float('inf')
            best_state = (U.copy(), V.copy(), W.copy())
            
            for it in range(max_iter):
                if time.time() > deadline:
                    return None
                
                # Compute current tensor
                U_r = U.reshape(n, m, target_rank)
                V_r = V.reshape(m, p, target_rank)
                W_r = W.reshape(p, n, target_rank)
                
                T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                loss = np.sum((T - 1.0)**2)
                
                if loss < best_loss:
                    best_loss = loss
                    best_state = (U.copy(), V.copy(), W.copy())
                
                if loss < 1e-10:
                    return (U.copy(), V.copy(), W.copy())
                
                # Compute gradients
                # dL/dU[i,j,c] = 2*(T[i,j,k]-1) * V[j,k,c] * W[k,i,c] summed over k
                # dL/dV[j,k,c] = 2*(T[i,j,k]-1) * U[i,j,c] * W[k,i,c] summed over i
                # dL/dW[k,i,c] = 2*(T[i,j,k]-1) * U[i,j,c] * V[j,k,c] summed over j
                
                diff = 2.0 * (T - 1.0)
                
                # Gradient for U: shape (n, m, target_rank)
                # dU[i,j,c] = sum_k diff[i,j,k] * V[j,k,c] * W[k,i,c]
                grad_U = np.einsum('ijk,jkc,kic->ijc', diff, V_r, W_r)
                grad_V = np.einsum('ijk,ijc,kic->jkc', diff, U_r, W_r)
                grad_W = np.einsum('ijk,ijc,jkc->kic', diff, U_r, V_r)
                
                # Update
                U -= lr * grad_U.reshape(nm, target_rank)
                V -= lr * grad_V.reshape(mp, target_rank)
                W -= lr * grad_W.reshape(pn, target_rank)
                
                # Decay learning rate
                lr *= 0.99
            
            # Check best state
            if check_decomposition(best_state[0], best_state[1], best_state[2]):
                return best_state
        
        return None
    
    # Try various ranks
    for target_rank in [30, 28, 25, 22, 20, 18, 15]:
        if time.time() > deadline:
            break
        
        result = try_rank_optimization(target_rank, num_trials=3, max_iter=100)
        if result is not None:
            U, V, W = result
            if check_decomposition(U, V, W):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break  # Found a valid decomposition, try lower ranks
    
    # Final check
    if not check_decomposition(best_U, best_V, best_W):
        # Fallback to schoolbook
        best_rank = rank_sb
        best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(), 
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s limit
    
    # Schoolbook baseline (always valid)
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if U,V,W form a valid decomposition."""
        r = U.shape[1]
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        return np.allclose(T, np.ones((n, m, p)), atol=tol)
    
    def compute_loss(U, V, W):
        """Compute ||T - 1||^2."""
        r = U.shape[1]
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.sum((T - 1.0)**2), T
    
    def polish(U, V, W, max_iter=200, tol=1e-10):
        """High-precision polish: alternating least squares + gradient descent."""
        r = U.shape[1]
        nm, mp, pn = n*m, m*p, p*n
        
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        
        # Alternating least squares: fix two, solve for third
        # For U: for each (i,j), U[i,j,:] should satisfy sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all k
        # This is a linear system in U[i,j,:]
        # Actually, let's do a simpler approach: gradient-based polish with adaptive LR
        
        lr = 0.05
        best_loss = float('inf')
        best_state = (U.copy(), V.copy(), W.copy())
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            U_r = U.reshape(n, m, r)
            V_r = V.reshape(m, p, r)
            W_r = W.reshape(p, n, r)
            
            T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
            loss = np.sum((T - 1.0)**2)
            
            if loss < best_loss:
                best_loss = loss
                best_state = (U.copy(), V.copy(), W.copy())
            
            if loss < tol:
                break
            
            diff = 2.0 * (T - 1.0)
            grad_U = np.einsum('ijk,jkc,kic->ijc', diff, V_r, W_r)
            grad_V = np.einsum('ijk,ijc,kic->jkc', diff, U_r, W_r)
            grad_W = np.einsum('ijk,ijc,jkc->kic', diff, U_r, V_r)
            
            # Adaptive learning rate based on gradient norms
            gU_norm = np.linalg.norm(grad_U)
            gV_norm = np.linalg.norm(grad_V)
            gW_norm = np.linalg.norm(grad_W)
            
            if gU_norm > 1e-12:
                U -= lr * grad_U.reshape(nm, r)
            if gV_norm > 1e-12:
                V -= lr * grad_V.reshape(mp, r)
            if gW_norm > 1e-12:
                W -= lr * grad_W.reshape(pn, r)
            
            # Decay LR
            lr *= 0.995
        
        return best_state
    
    def kronecker_init(target_rank, seed=None):
        """Structured initialization exploiting symmetry between the two 5-dimensions.
        Uses Kronecker products of identity matrices to create a near-solution structure.
        """
        rng = np.random.RandomState(seed)
        nm, mp, pn = n*m, m*p, p*n
        
        # Strategy: embed the 3x3x3 core (rank 27 schoolbook) and extend with
        # symmetric structure for the 5x5 part
        # 
        # Key insight: m=p=5, so we can exploit the symmetry.
        # Initialize with block structure:
        # - For the "diagonal" part of the 5x5 multiplication, use identity-like patterns
        # - For off-diagonal, use permutation patterns
        
        # First, try a structured initialization based on the schoolbook
        # but with random rotations to escape local minima
        
        # Approach 1: Random initialization with scaling
        scale = 1.0 / np.sqrt(target_rank)
        U = rng.randn(nm, target_rank) * scale
        V = rng.randn(mp, target_rank) * scale
        W = rng.randn(pn, target_rank) * scale
        
        # Approach 2: Kronecker structure
        # Since m=p=5, we can think of the 5x5 multiplication as having
        # a "core" of size min(n, m, p) = 3 and extensions
        
        # Let's try: initialize using the schoolbook pattern but with random mixing
        # to allow rank reduction
        
        # Create a base pattern from schoolbook and add random perturbations
        base_U = np.zeros((nm, target_rank))
        base_V = np.zeros((mp, target_rank))
        base_W = np.zeros((pn, target_rank))
        
        # Fill first min(target_rank, n*m*p) columns with schoolbook-like pattern
        fill = min(target_rank, n*m*p)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= target_rank:
                        break
                    # Add random perturbation to break exact schoolbook structure
                    base_U[i*m+j, col] = 1.0 + rng.randn() * 0.1
                    base_V[j*p+k, col] = 1.0 + rng.randn() * 0.1
                    base_W[k*n+i, col] = 1.0 + rng.randn() * 0.1
                    col += 1
        
        # For remaining columns, use random values
        if col < target_rank:
            base_U[:, col:] = rng.randn(nm, target_rank - col) * 0.3
            base_V[:, col:] = rng.randn(mp, target_rank - col) * 0.3
            base_W[:, col:] = rng.randn(pn, target_rank - col) * 0.3
        
        # Choose between random and structured init
        if rng.random() < 0.5:
            return U, V, W
        else:
            return base_U, base_V, base_W
    
    def try_rank_optimization(target_rank, num_trials=5, max_iter=300):
        """Try to find a decomposition of given rank using iterative optimization."""
        if time.time() > deadline:
            return None
        
        nm, mp, pn = n*m, m*p, p*n
        best_result = None
        best_loss = float('inf')
        
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            seed = trial * 7919 + int(target_rank * 131)
            U, V, W = kronecker_init(target_rank, seed=seed)
            
            # Phase 1: Coarse optimization with larger LR
            lr = 0.1
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                U_r = U.reshape(n, m, target_rank)
                V_r = V.reshape(m, p, target_rank)
                W_r = W.reshape(p, n, target_rank)
                
                T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                loss = np.sum((T - 1.0)**2)
                
                if loss < best_loss:
                    best_loss = loss
                    best_result = (U.copy(), V.copy(), W.copy())
                
                if loss < 1e-12:
                    return (U.copy(), V.copy(), W.copy())
                
                diff = 2.0 * (T - 1.0)
                grad_U = np.einsum('ijk,jkc,kic->ijc', diff, V_r, W_r)
                grad_V = np.einsum('ijk,ijc,kic->jkc', diff, U_r, W_r)
                grad_W = np.einsum('ijk,ijc,jkc->kic', diff, U_r, V_r)
                
                U -= lr * grad_U.reshape(nm, target_rank)
                V -= lr * grad_V.reshape(mp, target_rank)
                W -= lr * grad_W.reshape(pn, target_rank)
                
                lr *= 0.998
            
            # Phase 2: Polish
            if best_result is not None and best_loss < 1.0:
                polished = polish(best_result[0], best_result[1], best_result[2], max_iter=100)
                pol_loss, _ = compute_loss(polished[0], polished[1], polished[2])
                if pol_loss < best_loss:
                    best_loss = pol_loss
                    best_result = polished
        
        if best_result is not None and best_loss < 1e-6:
            return best_result
        return None
    
    # Try various ranks in decreasing order
    # Theoretical: rank >= max(n*m, m*p, p*n) / something... 
    # For 3x5x5: lower bound is at least max(15,25,15)=25 for trivial bounds
    # But matrix multiplication rank is known to be less than n*m*p
    rank_candidates = [40, 35, 30, 28, 25, 23, 21, 20]
    
    for target_rank in rank_candidates:
        if time.time() > deadline:
            break
        
        result = try_rank_optimization(target_rank, num_trials=4, max_iter=250)
        if result is not None:
            U, V, W = result
            if check_decomposition(U, V, W):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                # Try to go lower
                continue
        # If we can't find rank R, try R+2 next
    
    # Final check
    if not check_decomposition(best_U, best_V, best_W):
        best_rank = rank_sb
        best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(), 
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
