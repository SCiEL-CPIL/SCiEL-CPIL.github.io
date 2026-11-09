"""Matrix multiplication 3x4x8 tensor decomposition search."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    rank_trivial = n * m * p  # 96
    triples = [(i, k, j) for i in range(n) for k in range(m) for j in range(p)]
    T = len(triples)  # 96
    
    idx_U = np.array([i*m+k for i,k,j in triples])
    idx_V = np.array([k*p+j for i,k,j in triples])
    idx_W = np.array([j*n+i for i,k,j in triples])
    
    def _verify_fast(U, V, W, tol=1e-9):
        """Verify decomposition using vectorized operations."""
        r = U.shape[1]
        U_exp = U[idx_U, :]
        V_exp = V[idx_V, :]
        W_exp = W[idx_W, :]
        # Compute sum of products for each triple
        vals = np.einsum('tr,tr,tr->t', U_exp, V_exp, W_exp)
        return bool(np.all(np.abs(vals - 1.0) < tol))
    
    def _verify_exact(U, V, W):
        """Exact verification using Fraction arithmetic."""
        U_frac = [[Fraction(int(round(x*2)), 2) for x in row] for row in U]
        V_frac = [[Fraction(int(round(x*2)), 2) for x in row] for row in V]
        W_frac = [[Fraction(int(round(x*2)), 2) for x in row] for row in W]
        
        for t in range(T):
            s = Fraction(0)
            for r in range(U_frac[0][0].denominator * 0 + len(U_frac[0])):
                s += U_frac[idx_U[t]][r] * V_frac[idx_V[t]][r] * W_frac[idx_W[t]][r]
            if s != Fraction(1):
                return False
        return True
    
    def _is_half_integer(arr, tol=1e-6):
        return bool(np.all(np.abs(arr * 2 - np.round(arr * 2)) < tol))
    
    def _to_output(U, V, W):
        U_out = [[float(U[i,r]) for r in range(U.shape[1])] for i in range(U.shape[0])]
        V_out = [[float(V[i,r]) for r in range(V.shape[1])] for i in range(V.shape[0])]
        W_out = [[float(W[i,r]) for r in range(W.shape[1])] for i in range(W.shape[0])]
        return {"U": U_out, "V": V_out, "W": W_out}
    
    def _clean_output(U, V, W):
        """Convert to clean integer/half-integer representation."""
        def _clean_mat(M):
            result = []
            for row in M:
                new_row = []
                for val in row:
                    v2 = round(val * 2)
                    if v2 % 2 == 0:
                        new_row.append(int(v2 // 2))
                    else:
                        new_row.append(v2 / 2.0)
                result.append(new_row)
            return result
        return {"U": _clean_mat(U), "V": _clean_mat(V), "W": _clean_mat(W)}
    
    best_result = None
    best_rank = rank_trivial
    
    # Strategy 1: Try structured decompositions first
    # For 3x4x8, we can exploit that 8 = 2*4
    
    # Try a Kronecker-like structure: treat the 4x8 as two 4x4 blocks
    # This might allow rank reduction if the blocks share structure
    
    # Strategy 2: Gradient-based search with half-integer constraint projection
    # Strategy 3: Integer programming / exhaustive search for small ranks
    
    # Lower bounds:
    # - At least max(n*m, m*p, n*p) = max(12, 32, 24) = 32
    # - At least n*m*p / max(n,m,p) is not a valid bound
    # - Strassen-style bounds don't directly apply here
    
    lower_bound = max(n*m, m*p, n*p)  # 32
    
    # Try to find decompositions starting from lower bound upward
    # Use gradient descent with periodic projection to half-integers
    
    for rank in range(lower_bound, rank_trivial + 1):
        if time.time() > deadline - 5.0:
            break
        
        found = False
        # More attempts for smaller ranks (harder to find)
        attempts = 20 if rank <= 40 else 12 if rank <= 60 else 8
        
        for attempt in range(attempts):
            if time.time() > deadline - 5.0:
                break
            
            # Initialize with random half-integer values
            rng = np.random.default_rng(seed=rank * 1000 + attempt)
            
            # Start with small random values in {-1, -0.5, 0, 0.5, 1}
            U = rng.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, rank)).astype(float)
            V = rng.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, rank)).astype(float)
            W = rng.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, rank)).astype(float)
            
            # Gradient descent with projection
            lr = 0.01
            for it in range(200):
                if time.time() > deadline - 3.0:
                    break
                
                U_exp = U[idx_U, :]
                V_exp = V[idx_V, :]
                W_exp = W[idx_W, :]
                
                # Compute errors
                prod = U_exp * V_exp * W_exp
                vals = np.sum(prod, axis=1)
                errors = vals - 1.0
                
                # Gradients
                grad_U_exp = errors[:, None] * (V_exp * W_exp)
                grad_V_exp = errors[:, None] * (U_exp * W_exp)
                grad_W_exp = errors[:, None] * (U_exp * V_exp)
                
                # Accumulate gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                np.add.at(grad_U, idx_U, grad_U_exp)
                np.add.at(grad_V, idx_V, grad_V_exp)
                np.add.at(grad_W, idx_W, grad_W_exp)
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Project to nearest half-integers periodically
                if it % 20 == 0:
                    U = np.round(U * 2) / 2
                    V = np.round(V * 2) / 2
                    W = np.round(W * 2) / 2
                    
                    # Check if we've found a valid solution
                    if _verify_fast(U, V, W, 1e-8):
                        if _verify_exact(U, V, W):
                            best_result = _clean_output(U, V, W)
                            best_rank = rank
                            found = True
                            break
                
                # Early termination if error is small
                if it % 10 == 0 and np.max(np.abs(errors)) < 1e-6:
                    U = np.round(U * 2) / 2
                    V = np.round(V * 2) / 2
                    W = np.round(W * 2) / 2
                    if _verify_fast(U, V, W, 1e-8):
                        if _verify_exact(U, V, W):
                            best_result = _clean_output(U, V, W)
                            best_rank = rank
                            found = True
                            break
            
            if found:
                break
        
        if found:
            break
    
    # Strategy 3: If gradient search didn't find anything better, try a more 
    # systematic approach: fix U and V, solve for W
    
    if best_result is None or best_rank == rank_trivial:
        if time.time() < deadline - 10.0:
            # Try to solve for W given U and V
            # For each triple (i,k,j), we need sum_r U[i*m+k,r]*V[k*p+j,r]*W[j*n+i,r] = 1
            # This is a linear system in W
            
            for rank in range(lower_bound, min(rank_trivial, 50) + 1):
                if time.time() > deadline - 5.0:
                    break
                
                # Try several random U, V pairs
                for attempt in range(10):
                    if time.time() > deadline - 5.0:
                        break
                    
                    rng = np.random.default_rng(seed=rank * 5000 + attempt)
                    U = rng.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, rank)).astype(float)
                    V = rng.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, rank)).astype(float)
                    
                    # Build linear system for W
                    # Unknowns: W[j*n+i, r] for all j,i,r
                    # Equations: for each (i,k,j): sum_r U[i*m+k,r]*V[k*p+j,r]*W[j*n+i,r] = 1
                    
                    # Number of unknowns: p*n*rank
                    # Number of equations: T = n*m*p
                    
                    num_unknowns = p * n * rank
                    A = np.zeros((T, num_unknowns))
                    b = np.ones(T)
                    
                    for t, (i, k, j) in enumerate(triples):
                        for r in range(rank):
                            col = j * n + i  # row index in W
                            col = col * rank + r  # flatten
                            A[t, col] = U[i*m+k, r] * V[k*p+j, r]
                    
                    # Solve least squares
                    try:
                        W_sol, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
                        W_sol = W_sol.reshape(p*n, rank)
                        
                        # Project to half-integers
                        W_proj = np.round(W_sol * 2) / 2
                        
                        # Verify
                        if _verify_fast(U, V, W_proj, 1e-8):
                            if _verify_exact(U, V, W_proj):
                                best_result = _clean_output(U, V, W_proj)
                                best_rank = rank
                                break
                    except np.linalg.LinAlgError:
                        continue
                
                if best_result is not None and best_rank == rank:
                    break
    
    # Fallback: return schoolbook decomposition
    if best_result is None:
        U_out = [[0]*rank_trivial for _ in range(n*m)]
        V_out = [[0]*rank_trivial for _ in range(m*p)]
        W_out = [[0]*rank_trivial for _ in range(p*n)]
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    r = i*(m*p) + k*p + j
                    U_out[i*m+k][r] = 1
                    V_out[k*p+j][r] = 1
                    W_out[j*n+i][r] = 1
        best_result = {"U": U_out, "V": V_out, "W": W_out}
    
    return best_result
# EVOLVE-BLOCK-END
