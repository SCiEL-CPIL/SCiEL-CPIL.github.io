import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify that U, V, W form a valid tensor decomposition using vectorized operations."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i,j,k] = 1 for all i,j,k
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Use einsum: 'ijr,jkr,kir->ijk'
    T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    # Check if T[i,j,k] = 1 for all i,j,k
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _verify_complex(U, V, W):
    """Verify complex-valued tensor decomposition."""
    U = np.array(U, dtype=np.complex128)
    V = np.array(V, dtype=np.complex128)
    W = np.array(W, dtype=np.complex128)
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline with margin
    
    # Start with the trivial decomposition (rank = n*m*p = 24)
    rank = n * m * p
    U0 = np.zeros((n * m, rank))
    V0 = np.zeros((m * p, rank))
    W0 = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1
    
    best_rank = rank
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    # Strategy 1: Try to find lower-rank decompositions using structured search
    # The key insight is that for matrix multiplication, there may be algebraic
    # structures that allow rank reduction.
    
    # For 2x3 * 3x4, the schoolbook rank is 24.
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    # Upper bound from schoolbook: 24
    # There are known constructions that can beat schoolbook for some dimensions.
    
    # Strategy: Try random search with structured values for ranks below 24
    # Use values from a small set to increase chance of exact solutions
    
    # Try ranks from 23 down to 12
    for target_rank in range(23, 11, -1):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random attempts for this rank
        max_attempts = 50
        found = False
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Generate random U, V, W with the target rank
            # Use values from a structured set
            rng = np.random.RandomState(attempt * 1000 + target_rank)
            vals = np.array([-1, -0.5, 0, 0.5, 1])
            
            U_rand = np.zeros((n * m, target_rank))
            V_rand = np.zeros((m * p, target_rank))
            W_rand = np.zeros((p * n, target_rank))
            
            for r in range(target_rank):
                U_rand[:, r] = vals[rng.randint(0, len(vals), n * m)]
                V_rand[:, r] = vals[rng.randint(0, len(vals), m * p)]
                W_rand[:, r] = vals[rng.randint(0, len(vals), p * n)]
            
            if _verify(U_rand, V_rand, W_rand):
                best_rank = target_rank
                best_U, best_V, best_W = U_rand.copy(), V_rand.copy(), W_rand.copy()
                found = True
                break
        
        # If we found a valid decomposition at this rank, we can stop
        if found:
            break
    
    # Strategy 2: Try complex-valued search (Gaussian integers approach)
    # This explores a larger search space that may contain solutions
    # not accessible with real-valued entries
    
    if best_rank == rank and time.time() < deadline - 40:
        for target_rank in range(23, 11, -1):
            if time.time() > deadline - 20:
                break
            
            max_attempts = 30
            found = False
            for attempt in range(max_attempts):
                if time.time() > deadline - 10:
                    break
                
                rng = np.random.RandomState(attempt * 2000 + target_rank + 777)
                # Use Gaussian integers: a + bi where a, b in {-1, -0.5, 0, 0.5, 1}
                vals_real = np.array([-1, -0.5, 0, 0.5, 1])
                vals_imag = np.array([-1, -0.5, 0, 0.5, 1])
                
                U_rand = np.zeros((n * m, target_rank), dtype=np.complex128)
                V_rand = np.zeros((m * p, target_rank), dtype=np.complex128)
                W_rand = np.zeros((p * n, target_rank), dtype=np.complex128)
                
                for r in range(target_rank):
                    Ur = vals_real[rng.randint(0, len(vals_real), n * m)]
                    Ui = vals_imag[rng.randint(0, len(vals_imag), n * m)]
                    U_rand[:, r] = Ur + 1j * Ui
                    
                    Vr = vals_real[rng.randint(0, len(vals_real), m * p)]
                    Vi = vals_imag[rng.randint(0, len(vals_imag), m * p)]
                    V_rand[:, r] = Vr + 1j * Vi
                    
                    Wr = vals_real[rng.randint(0, len(vals_real), p * n)]
                    Wi = vals_imag[rng.randint(0, len(vals_imag), p * n)]
                    W_rand[:, r] = Wr + 1j * Wi
                
                if _verify_complex(U_rand, V_rand, W_rand):
                    # Convert to real representation if possible
                    # For now, just record that we found a complex solution
                    # But the output needs to be real-valued or half-integer
                    # So we need to check if the solution can be made real
                    best_rank = target_rank
                    # Convert complex to real by taking real parts if imaginary parts are ~0
                    if np.allclose(U_rand.imag, 0, atol=1e-9) and \
                       np.allclose(V_rand.imag, 0, atol=1e-9) and \
                       np.allclose(W_rand.imag, 0, atol=1e-9):
                        best_U = U_rand.real.copy()
                        best_V = V_rand.real.copy()
                        best_W = W_rand.real.copy()
                    else:
                        # Complex solution found but can't directly use
                        # Keep the best real solution
                        pass
                    found = True
                    break
            
            if found:
                break
    
    # Strategy 3: Local search / perturbation from trivial decomposition
    if best_rank == rank and time.time() < deadline - 15:
        # Try to perturb pairs of columns and see if we can merge them
        # This is a heuristic approach
        
        U_curr = U0.copy()
        V_curr = V0.copy()
        W_curr = W0.copy()
        
        # For each pair of columns, try small perturbations
        for r1 in range(rank):
            if time.time() > deadline - 5:
                break
            for r2 in range(r1 + 1, rank):
                if time.time() > deadline - 5:
                    break
                
                # Try to express column r2 as a linear combination involving column r1
                # with small perturbation
                for delta in [-0.5, 0.5]:
                    U_test = U_curr.copy()
                    V_test = V_curr.copy()
                    W_test = W_curr.copy()
                    
                    # Perturb column r2 by delta * column r1
                    U_test[:, r2] += delta * U_test[:, r1]
                    V_test[:, r2] += delta * V_test[:, r1]
                    W_test[:, r2] += delta * W_test[:, r1]
                    
                    # Check if the perturbed decomposition is still valid
                    # (it won't be in general, but we're looking for lucky cases)
                    if _verify(U_test, V_test, W_test):
                        best_rank = rank - 1
                        best_U = U_test.copy()
                        best_V = V_test.copy()
                        best_W = W_test.copy()
                        break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
