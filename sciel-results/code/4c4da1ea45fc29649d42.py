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
    
    lower_bound = max(n*m, m*p, n*p)  # 32
    
    # Strategy: For each rank, try random U,V from a small set, solve for W exactly
    # The key insight: if we fix U and V, the equations are linear in W
    # We need W to be half-integer, so we can try to find U,V such that the
    # least-squares solution for W is close to half-integers
    
    # Use a smarter approach: try U,V with entries from {-1, 0, 1} and check
    # if the resulting W (from least squares) can be rounded to half-integers
    
    for rank in range(lower_bound, min(rank_trivial + 1, 60)):
        if time.time() > deadline - 5.0:
            break
        
        found = False
        # Number of attempts depends on rank
        if rank <= 35:
            attempts = 50
        elif rank <= 40:
            attempts = 30
        elif rank <= 50:
            attempts = 20
        else:
            attempts = 15
        
        for attempt in range(attempts):
            if time.time() > deadline - 5.0:
                break
            
            rng = np.random.default_rng(seed=rank * 10000 + attempt * 7)
            
            # Use values from {-1, 0, 1} for U and V
            U = rng.choice([-1, 0, 1], size=(n*m, rank)).astype(float)
            V = rng.choice([-1, 0, 1], size=(m*p, rank)).astype(float)
            
            # Build linear system for W
            num_unknowns = p * n * rank
            A = np.zeros((T, num_unknowns))
            b = np.ones(T)
            
            for t, (i, k, j) in enumerate(triples):
                for r in range(rank):
                    col = (j * n + i) * rank + r
                    A[t, col] = U[i*m+k, r] * V[k*p+j, r]
            
            # Check if system is consistent (rank of A should be at least T for unique solution)
            # Solve least squares
            try:
                W_sol, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
                W_sol = W_sol.reshape(p*n, rank)
                
                # Check if W_sol is close to half-integers
                W_proj = np.round(W_sol * 2) / 2
                
                # Verify with tight tolerance
                if _verify_fast(U, V, W_proj, 1e-8):
                    if _verify_exact(U, V, W_proj):
                        best_result = _clean_output(U, V, W_proj)
                        best_rank = rank
                        found = True
                        break
            except np.linalg.LinAlgError:
                continue
        
        if found:
            break
    
    # If the above didn't work, try a different approach:
    # Use structured U and V based on the 4x8 = 4x(2*4) structure
    if best_result is None or best_rank == rank_trivial:
        if time.time() < deadline - 10.0:
            # Try Kronecker-like structure
            # Treat the 8 columns as two groups of 4
            # This might allow sharing of computation
            
            for rank in range(lower_bound, min(rank_trivial + 1, 50)):
                if time.time() > deadline - 5.0:
                    break
                
                found = False
                attempts = 20
                
                for attempt in range(attempts):
                    if time.time() > deadline - 5.0:
                        break
                    
                    rng = np.random.default_rng(seed=rank * 20000 + attempt * 13)
                    
                    # Create structured U: for each (i,k), use random values
                    U = rng.choice([-1, 0, 1], size=(n*m, rank)).astype(float)
                    
                    # Create structured V: for each (k,j), use values that
                    # are the same for j and j+4 (exploiting 8=2*4)
                    V = np.zeros((m*p, rank))
                    for k in range(m):
                        for j in range(4):
                            val = rng.choice([-1, 0, 1], size=(rank,))
                            V[k*p+j, :] = val
                            V[k*p+(j+4), :] = val  # Same for the second block
                    
                    # Solve for W
                    num_unknowns = p * n * rank
                    A = np.zeros((T, num_unknowns))
                    b = np.ones(T)
                    
                    for t, (i, k, j) in enumerate(triples):
                        for r in range(rank):
                            col = (j * n + i) * rank + r
                            A[t, col] = U[i*m+k, r] * V[k*p+j, r]
                    
                    try:
                        W_sol, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
                        W_sol = W_sol.reshape(p*n, rank)
                        W_proj = np.round(W_sol * 2) / 2
                        
                        if _verify_fast(U, V, W_proj, 1e-8):
                            if _verify_exact(U, V, W_proj):
                                best_result = _clean_output(U, V, W_proj)
                                best_rank = rank
                                found = True
                                break
                    except np.linalg.LinAlgError:
                        continue
                
                if found:
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
