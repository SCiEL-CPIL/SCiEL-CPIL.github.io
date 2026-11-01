"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    # Build the multiplication tensor from the decomposition
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For matrix multiplication: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So T should satisfy: for all i,j,k, T[i,j,k] = 1 if we're computing C[i,k] += A[i,j]*B[j,k]
    # Actually, the tensor M_{n,m,p}[i,j,k] = 1 for all i,j,k in the standard basis
    # The decomposition says: M[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We need this to equal 1 for all (i,j,k) triples that correspond to valid matrix mult terms.
    
    # Let's verify by checking that the reconstructed tensor matches the identity tensor
    # For 2x2 * 2x15: n=2, m=2, p=15
    # The multiplication tensor has entries M[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Wait, that's not right. The tensor for matrix multiplication C = A*B where A is n x m, B is m x p
    # has structure: C[i,k] = sum_j A[i,j] B[j,k]
    # As a tensor: T[i,j,k] where the multiplication is C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i,j,k.
    
    # Check: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_rank_reduction(base_U, base_V, base_W, target_rank, n, m, p, max_iter=500):
    """Try to reduce rank by finding linear dependencies among the rank terms."""
    U = np.array(base_U, dtype=float)
    V = np.array(base_V, dtype=float)
    W = np.array(base_W, dtype=float)
    current_rank = U.shape[1]
    
    if current_rank <= target_rank:
        return None
    
    # Try to eliminate terms by finding linear combinations
    # We can try: for each term c, see if we can express its contribution as a combination of others
    # This is hard in general, so we try a greedy approach with random perturbations
    
    best = None
    deadline = time.time() + 30  # 30 second budget for this attempt
    
    for iter_num in range(max_iter):
        if time.time() > deadline:
            break
        
        # Randomly select a term to try to eliminate
        c_elim = random.randint(0, current_rank - 1)
        
        # Try to find coefficients alpha_c such that for all (i,j,k):
        # sum_{c != c_elim} alpha_c * U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = U[i*m+j,c_elim] * V[j*p+k,c_elim] * W[k*n+i,c_elim]
        
        # This is a system of equations. Let's set up the linear system.
        # Unknowns: alpha_c for c != c_elim (current_rank - 1 unknowns)
        # Equations: one for each (i,j,k) triple (n*m*p = 60 equations)
        
        n_eq = n * m * p
        n_unknowns = current_rank - 1
        
        if n_unknowns >= n_eq:
            # Underdetermined or square, try least squares
            A_mat = np.zeros((n_eq, n_unknowns))
            b_vec = np.zeros(n_eq)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        rhs = U[i*m+j, c_elim] * V[j*p+k, c_elim] * W[k*n+i, c_elim]
                        b_vec[eq_idx] = rhs
                        
                        unknown_idx = 0
                        for c in range(current_rank):
                            if c == c_elim:
                                continue
                            A_mat[eq_idx, unknown_idx] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            unknown_idx += 1
                        eq_idx += 1
            
            # Solve least squares
            try:
                alpha, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                
                # Check if the solution is exact (within tolerance)
                if residuals.size > 0:
                    res = np.sqrt(residuals[0])
                else:
                    res = np.linalg.norm(A_mat @ alpha - b_vec)
                
                if res < 1e-8:
                    # Build new matrices without term c_elim
                    mask = [c for c in range(current_rank) if c != c_elim]
                    new_U = U[:, mask] * (1.0 + alpha)  # Scale remaining terms
                    # Actually, we need: new decomposition = sum_{c in mask} (1 + alpha_c) * original_c
                    # Wait, the equation was: sum_{c in mask} alpha_c * term_c = term_{c_elim}
                    # So: term_{c_elim} = sum_{c in mask} alpha_c * term_c
                    # New decomposition: sum_{c in mask} (original_c) + term_{c_elim}
                    #   = sum_{c in mask} (original_c) + sum_{c in mask} alpha_c * term_c
                    #   = sum_{c in mask} (1 + alpha_c) * term_c
                    # But we want to just drop c_elim and adjust the others.
                    
                    new_U = np.zeros((U.shape[0], n_unknowns))
                    new_V = np.zeros((V.shape[0], n_unknowns))
                    new_W = np.zeros((W.shape[0], n_unknowns))
                    
                    for idx, c in enumerate(mask):
                        scale = 1.0 + alpha[idx]
                        new_U[:, idx] = U[:, c] * scale
                        new_V[:, idx] = V[:, c] * scale
                        new_W[:, idx] = W[:, c] / scale  # Keep product the same? No...
                        # Actually, we need U_new * V_new * W_new = U_old * V_old * W_old for each term
                        # If we scale U by s, V by 1, W by 1/s, the product is preserved.
                        # But we're scaling all three? That changes the product by s^3.
                        # We should scale only one matrix per term.
                        # Let's just scale U:
                        new_U[:, idx] = U[:, c] * (1.0 + alpha[idx])
                        new_V[:, idx] = V[:, c]
                        new_W[:, idx] = W[:, c]
                    
                    # Verify
                    if _verify(new_U.tolist(), new_V.tolist(), new_W.tolist(), n, m, p):
                        U, V, W = new_U, new_V, new_W
                        current_rank = n_unknowns
                        if best is None or current_rank < best[0]:
                            best = (current_rank, U.copy(), V.copy(), W.copy())
                        if current_rank <= target_rank:
                            break
            except Exception:
                pass
    
    if best is not None:
        return best[1], best[2], best[3]
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for the 120s limit
    
    # Start with schoolbook
    sb = _schoolbook()
    best_result = sb
    best_rank = len(sb["U"][0])
    
    # Try to reduce rank using the elimination method
    U, V, W = np.array(sb["U"], dtype=float), np.array(sb["V"], dtype=float), np.array(sb["W"], dtype=float)
    current_rank = U.shape[1]
    
    while current_rank > 1 and time.time() < deadline:
        result = _try_rank_reduction(U.tolist(), V.tolist(), W.tolist(), current_rank - 1, n, m, p, max_iter=100)
        if result is not None:
            U, V, W = result
            current_rank = U.shape[1]
            best_rank = current_rank
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Can't reduce further with this method, break
            break
    
    # Convert to lists if numpy arrays
    if isinstance(best_result["U"], np.ndarray):
        best_result["U"] = best_result["U"].tolist()
    if isinstance(best_result["V"], np.ndarray):
        best_result["V"] = best_result["V"].tolist()
    if isinstance(best_result["W"], np.ndarray):
        best_result["W"] = best_result["W"].tolist()
    
    return best_result
# EVOLVE-BLOCK-END
