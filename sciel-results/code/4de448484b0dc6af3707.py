import numpy as np
import itertools
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will solve for M_{n, m, p'} where p' divides p
    # If we find rank r for M_{n, m, p'}, we can repeat p/p' times to get rank r*(p/p') for M_{n, m, p}
    
    # Target: M_{2, 3, 2}
    n_sub, m_sub, p_sub = 2, 3, 2
    
    # Check if p is divisible by p_sub
    if p % p_sub == 0:
        repeats = p // p_sub
        # Try to find minimal rank for M_{2,3,2}
        best_rank = None
        best_U = None
        best_V = None
        best_W = None
        
        deadline = time.time() + 100 # 100 seconds budget for search
        
        # First, compute the provable lower bound via bilinear coefficient matrix rank
        lower_bound = _compute_bilinear_rank_lower_bound(n_sub, m_sub, p_sub)
        
        # Try ranks from lower_bound upward
        # Schoolbook for sub-problem is n_sub*m_sub*p_sub = 12
        for r in range(lower_bound, 13): # Up to 12 (schoolbook for sub-problem)
            if time.time() > deadline:
                break
            if _find_decomposition(n_sub, m_sub, p_sub, r, deadline):
                best_rank = r
                best_U, best_V, best_W = _last_solution
                break
        
        if best_rank is not None:
            # Construct full U, V, W
            full_rank = best_rank * repeats
            U_full = [[0] * full_rank for _ in range(n * m)]
            V_full = [[0] * full_rank for _ in range(m * p)]
            W_full = [[0] * full_rank for _ in range(p * n)]
            
            for rep in range(repeats):
                offset = rep * best_rank
                # Map indices
                # U indices: i*m + j (0..5) -> same in sub
                # V indices: j*p + k. In sub, k is 0..1. In full, k = rep*2 + k_sub
                # W indices: k*n + i. In sub, k is 0..1. In full, k = rep*2 + k_sub
                
                for i in range(n_sub):
                    for j in range(m_sub):
                        u_idx = i * m_sub + j
                        u_full_idx = i * m + j
                        for c in range(best_rank):
                            U_full[u_full_idx][offset + c] = best_U[u_idx][c]
                
                for j in range(m_sub):
                    for k_sub in range(p_sub):
                        v_idx = j * p_sub + k_sub
                        k_full = rep * p_sub + k_sub
                        v_full_idx = j * p + k_full
                        for c in range(best_rank):
                            V_full[v_full_idx][offset + c] = best_V[v_idx][c]
                
                for k_sub in range(p_sub):
                    for i in range(n_sub):
                        w_idx = k_sub * n_sub + i
                        k_full = rep * p_sub + k_sub
                        w_full_idx = k_full * n + i
                        for c in range(best_rank):
                            W_full[w_full_idx][offset + c] = best_W[w_idx][c]
                
            return {"U": U_full, "V": V_full, "W": W_full}
    
    # Fallback: Schoolbook
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

def _compute_bilinear_rank_lower_bound(n, m, p):
    """
    Compute a lower bound on the rank of the matrix multiplication tensor
    by analyzing the bilinear coefficient structure.
    
    The multiplication tensor T has entries T[i,j,k] where:
    - i indexes rows of A (0..n-1)
    - j indexes columns of A / rows of B (0..m-1)  
    - k indexes columns of B (0..p-1)
    
    A rank-r decomposition means T = sum_{c=0}^{r-1} u_c ⊗ v_c ⊗ w_c
    where u_c ∈ R^{n*m}, v_c ∈ R^{m*p}, w_c ∈ R^{p*n}.
    
    For each fixed (j,k), we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i.
    
    The lower bound comes from: for each (j,k), the vectors {U[:,j] * W[:,k] * V_col : c} 
    must span the all-ones vector in R^n.
    
    More precisely, consider the matrix B_{j,k} of size n × r with entries B[i,c] = U[i*m+j,c] * W[k*n+i,c].
    We need B_{j,k} @ V[j*p+k, :] = 1_n.
    
    The rank of the full multiplication tensor is at least the maximum over all (j,k) of the
    minimum number of vectors needed. But a tighter bound: consider all (j,k) pairs together.
    
    Actually, the standard lower bound is max(n*m, m*p, p*n) = max(6, 6, 4) = 6 for (2,3,2).
    But we can do better by examining the actual coefficient matrix.
    
    Let's construct the full bilinear system. For the tensor to have rank r, we need
    r vectors in each space such that the trilinear form works.
    
    A key observation: fix j. Then for each k, we need to solve for V[j*p+k,:].
    The constraint is: for each i, sum_c U[i*m+j,c] * W[k*n+i,c] * V[j*p+k,c] = 1.
    
    Let's define for each (j,k) the n×r matrix A_{j,k}[i,c] = U[i*m+j,c] * W[k*n+i,c].
    We need A_{j,k} @ v_{j,k} = 1_n where v_{j,k} = V[j*p+k,:].
    
    This is always solvable if r >= n (since we have n equations and r unknowns).
    But the entries must be half-integers, which constrains things.
    
    For a pure rank lower bound (ignoring integrality), the answer is max(n, m, p) in some cases,
    but for matrix multiplication it's known to be at least max(n*m, m*p, p*n) / min(n,m,p) in some bounds.
    
    Let's just use the simple lower bound max(n*m, m*p, p*n) and also try to verify
    whether rank 6 is achievable for (2,3,2) via the coefficient matrix analysis.
    """
    # Simple lower bound
    lb = max(n * m, m * p, p * n)
    
    # For (2,3,2): lb = max(6, 6, 4) = 6
    # We know rank 6 is the schoolbook for (2,3,2) if we use 6 terms... wait, schoolbook is 12.
    # Actually schoolbook for (2,3,2) is 2*3*2 = 12 terms.
    # But the lower bound from "number of nonzero entries" type arguments gives 6.
    
    return lb

_last_solution = None

def _find_decomposition(n, m, p, r, deadline):
    global _last_solution
    import random
    import numpy as np
    
    # Strategy: 
    # 1. Try structured patterns first
    # 2. Then random search with wider value ranges
    
    # First, try structured decompositions
    if _try_structured(n, m, p, r):
        return True
    
    # Random search with multiple seeds and wider ranges
    for seed in range(50):
        if time.time() > deadline:
            return False
        
        random.seed(seed)
        for attempt in range(200):
            if time.time() > deadline:
                return False
            
            # Use wider range for better coverage
            val_range = 3
            U = np.array([[random.randint(-val_range, val_range) for _ in range(r)] for _ in range(n*m)], dtype=float)
            W = np.array([[random.randint(-val_range, val_range) for _ in range(r)] for _ in range(p*n)], dtype=float)
            
            V_sol = np.zeros((m*p, r), dtype=float)
            valid = True
            
            for j in range(m):
                for k in range(p):
                    # Build linear system for V[j*p+k, c]
                    # Eq for each i: sum_c U[i*m+j, c] W[k*n+i, c] V[j*p+k, c] = 1
                    A_sys = np.zeros((n, r), dtype=float)
                    for i in range(n):
                        for c in range(r):
                            A_sys[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    
                    # Solve A_sys @ V_col = ones(n)
                    try:
                        V_col = np.linalg.lstsq(A_sys, np.ones(n), rcond=None)[0]
                    except:
                        valid = False
                        break
                        
                    # Check if V_col has integer or half-integer entries
                    V_col_check = V_col * 2
                    if not np.allclose(V_col_check, np.round(V_col_check), atol=1e-6):
                        valid = False
                        break
                    V_sol[j*p+k, :] = V_col
                    
            if valid:
                # Verify the solution
                ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for c in range(r):
                                val += U[i*m+j, c] * V_sol[j*p+k, c] * W[k*n+i, c]
                            if not np.isclose(val, 1, atol=1e-6):
                                ok = False
                                break
                        if not ok: break
                    if not ok: break
                
                if ok:
                    _last_solution = (U.tolist(), V_sol.tolist(), W.tolist())
                    return True
    
    return False

def _try_structured(n, m, p, r):
    """Try structured decompositions with known patterns"""
    global _last_solution
    import numpy as np
    
    # Try pattern 1: U and W as permutation-like matrices
    # For rank r = n*m*p (schoolbook), this always works
    if r == n * m * p:
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        _last_solution = (U.tolist(), V.tolist(), W.tolist())
        return True
    
    # Try pattern 2: Kronecker-like structure
    # For M_{2,3,2}, try to use rank 6 with specific structure
    if n == 2 and m == 3 and p == 2 and r == 6:
        # Try a specific structured approach
        # U: 6x6, V: 6x6, W: 4x6
        # We need sum_c U[i*3+j, c] * V[j*2+k, c] * W[k*2+i, c] = 1
        
        # Try using orthogonal-like patterns
        # Let's try a few specific patterns
        
        # Pattern: use identity in some sense
        U = np.zeros((6, 6))
        W = np.zeros((4, 6))
        
        # Try to set up so that for each (j,k), the system is well-conditioned
        # Use different patterns for different (j,k)
        
        # Attempt 1: Simple alternating pattern
        for c in range(6):
            for i in range(2):
                for j in range(3):
                    U[i*3+j, c] = 1 if (i + j + c) % 2 == 0 else -1
            for k in range(2):
                for i in range(2):
                    W[k*2+i, c] = 1 if (k + i + c) % 2 == 0 else -1
        
        V_sol = np.zeros((6, 6))
        valid = True
        for j in range(3):
            for k in range(2):
                A_sys = np.zeros((2, 6))
                for i in range(2):
                    for c in range(6):
                        A_sys[i, c] = U[i*3+j, c] * W[k*2+i, c]
                try:
                    V_col = np.linalg.lstsq(A_sys, np.ones(2), rcond=None)
                    V_col = V_col[0]
                    V_col_check = V_col * 2
                    if not np.allclose(V_col_check, np.round(V_col_check), atol=1e-6):
                        valid = False
                        break
                    V_sol[j*2+k, :] = V_col
                except:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Verify
            ok = True
            for i in range(2):
                for j in range(3):
                    for k in range(2):
                        val = sum(U[i*3+j, c] * V_sol[j*2+k, c] * W[k*2+i, c] for c in range(6))
                        if not np.isclose(val, 1, atol=1e-6):
                            ok = False
                            break
                    if not ok: break
                if not ok: break
            
            if ok:
                _last_solution = (U.tolist(), V_sol.tolist(), W.tolist())
                return True
    
    # Try rank 7 for (2,3,2) with a more systematic approach
    if n == 2 and m == 3 and p == 2 and r == 7:
        # Use a structured approach: try to find U, W such that all
        # A_{j,k} matrices have rank exactly 2 and the solutions are half-integers
        import itertools
        
        # Try small integer values for U and W
        # U is 6x7, W is 4x7
        # For each (j,k), A_{j,k}[i,c] = U[i*3+j,c] * W[k*2+i,c]
        # We need A_{j,k} to be 2x7 with rank 2, and the solution to A@v=1 to be half-integer
        
        # Strategy: fix U and W to have entries in {-1, 0, 1} and check
        # There are 3^42 possibilities which is too many, so use random sampling
        
        import random
        for trial in range(2000):
            if time.time() > deadline:
                return False
            
            random.seed(1000 + trial)
            U = np.array([[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n*m)])
            W = np.array([[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(p*n)])
            
            V_sol = np.zeros((m*p, r))
            valid = True
            
            for j in range(m):
                for k in range(p):
                    A_sys = np.zeros((n, r))
                    for i in range(n):
                        for c in range(r):
                            A_sys[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    
                    try:
                        V_col, residuals, rank, sv = np.linalg.lstsq(A_sys, np.ones(n), rcond=None)
                    except:
                        valid = False
                        break
                    
                    # Check half-integer
                    V_col_check = V_col * 2
                    if not np.allclose(V_col_check, np.round(V_col_check), atol=1e-6):
                        valid = False
                        break
                    V_sol[j*p+k, :] = V_col
            
            if valid:
                # Verify
                ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U[i*m+j, c] * V_sol[j*p+k, c] * W[k*n+i, c] for c in range(r))
                            if not np.isclose(val, 1, atol=1e-6):
                                ok = False
                                break
                        if not ok: break
                    if not ok: break
                
                if ok:
                    _last_solution = (U.tolist(), V_sol.tolist(), W.tolist())
                    return True
    
    return False
# EVOLVE-BLOCK-END
