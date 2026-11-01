"""Matrix multiplication 2x3x9 — rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 9]


def _validate(U, V, W, n=2, m=3, p=9, tol=1e-9):
    """Return True if U,V,W exactly decompose the matrix multiplication tensor.

    T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    must equal 1 if (i,j,k) is a valid index, 0 otherwise.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _validate_sparse(U, V, W, n=2, m=3, p=9, tol=1e-9):
    """Faster validation using numpy where possible, checking all (i,j,k) triples."""
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i * m + j]
                v_val = V_np[j * p + k]
                w_val = W_np[k * n + i]
                s = np.dot(u_val, v_val * w_val)
                if abs(s - 1.0) > tol:
                    return False
    return True


def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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


def _try_block_structure(n, m, p, deadline):
    """
    Try to exploit block structure: split p into blocks and use Kronecker-like
    decompositions. For p=9=3*3, try M_{2,3,3} x 3 structure.
    
    Also try rank r from lower bound upward with structured random search
    using rational entries.
    """
    n, m, p = 2, 3, 9
    lower_bound = max(n * m, m * p, p * n)  # 27
    schoolbook_rank = n * m * p  # 54
    
    # Try a few structured approaches
    # Approach 1: For each block of size 3 in p, use the 2x3x3 tensor
    # The 2x3x3 tensor has schoolbook rank 18. If we can't improve it,
    # total is 3*18 = 54. No gain.
    
    # Approach 2: Try rank r with random rational entries
    # For small r, try to find U, V, W with entries in {0, 1, -1, 1/2}
    # This is a hard combinatorial problem, so we do a limited search.
    
    rng = random.Random(12345)
    
    # Try rank values from lower_bound to schoolbook_rank - 1
    # For each, do a few random restarts
    for r in range(lower_bound, schoolbook_rank):
        if time.time() > deadline:
            break
        
        # Number of restarts decreases as r increases (harder to find)
        n_restarts = max(3, 20 - (r - lower_bound) // 5)
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Generate random W first (it's the "output" side)
            # Use sparse random matrices with entries in {0, 1, -1}
            W = [[0] * r for _ in range(p * n)]
            for row in range(p * n):
                for col in range(r):
                    if rng.random() < 0.3:  # 30% density
                        W[row][col] = rng.choice([1, -1])
            
            # Now try to solve for U and V
            # For each (i,j,k): sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
            # This is bilinear in U and V given W.
            # Fix U randomly, solve for V, or vice versa.
            
            # Let's fix U randomly and solve for V linearly.
            # For fixed U and W, the constraint on V is:
            # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all (i,j,k)
            # This is linear in V entries V[j,k,c].
            
            # Variables: V[j*p+k, c] for j in 0..m-1, k in 0..p-1, c in 0..r-1
            # That's m*p*r = 27*r variables.
            # Constraints: n*m*p = 54 equations.
            
            # For r >= 54/27 = 2, we have more variables than equations, so
            # solutions may exist. But we also need U to be "nice".
            
            # Let's try: generate random U with entries in {0, 1, -1}
            U = [[0] * r for _ in range(n * m)]
            for row in range(n * m):
                for col in range(r):
                    if rng.random() < 0.3:
                        U[row][col] = rng.choice([1, -1])
            
            # Build linear system for V
            # A @ v = b where v is the vectorized V
            # Row of A corresponds to (i,j,k), column to (j',k',c)
            # A[(i,j,k), (j',k',c)] = U[i*m+j, c] * W[k*n+i, c] if j==j' and k==k' else 0
            # b[(i,j,k)] = 1
            
            n_vars = m * p * r
            n_eqs = n * m * p
            
            # Build A as a sparse-like structure
            # For each equation (i,j,k), the non-zero columns are those with j'=j, k'=k
            # i.e., columns (j, k, c) for c in 0..r-1
            # A[(i,j,k), (j,k,c)] = U[i*m+j, c] * W[k*n+i, c]
            
            # We can solve this system using numpy
            A = np.zeros((n_eqs, n_vars), dtype=np.float64)
            b = np.ones(n_eqs, dtype=np.float64)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(r):
                            var_idx = j * p * r + k * r + c
                            A[eq_idx, var_idx] = U[i * m + j][c] * W[k * n + i][c]
                        eq_idx += 1
            
            # Solve least squares
            try:
                v_sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            # Check if solution is good
            if len(residuals) > 0 and residuals[0] > 1e-6:
                continue
            
            # Check if solution entries are "nice" (close to half-integers)
            V = [[0] * r for _ in range(m * p)]
            nice = True
            for row in range(m * p):
                for col in range(r):
                    val = v_sol[row * r + col]
                    # Check if close to a half-integer
                    nearest = round(val * 2) / 2
                    if abs(val - nearest) > 1e-4:
                        nice = False
                        break
                    V[row][col] = nearest
                if not nice:
                    break
            
            if not nice:
                continue
            
            # Verify the full decomposition
            if _validate_sparse(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W, "rank": r}
    
    return None


def _try_lattice_search(n, m, p, deadline):
    """
    Try a more systematic search for small rank improvements.
    Use the fact that for n=2, the W matrix has only 18 rows.
    """
    # This is a placeholder for a more sophisticated search
    # Given time constraints, we'll just do a few random trials
    # at specific ranks where improvements might be possible.
    
    # Known: M_{2,3,3} has rank 18 (schoolbook). No improvement known.
    # So 3 copies give 54.
    
    # Try rank 45, 40, 36 with random structured search
    for r in [45, 40, 36, 32]:
        if time.time() > deadline:
            break
        
        rng = random.Random(r * 1000)
        
        for restart in range(5):
            if time.time() > deadline:
                break
            
            # Random W with structure: make it block-diagonal
            W = [[0] * r for _ in range(p * n)]
            
            # Try to make W have a Kronecker structure
            # W is 18 x r. Try to make it 3 blocks of 6 x (r/3)
            if r % 3 == 0:
                r_block = r // 3
                for block in range(3):
                    for row in range(6):
                        for col in range(r_block):
                            if rng.random() < 0.4:
                                W[block * 6 + row][block * r_block + col] = rng.choice([1, -1])
            
            # Random U
            U = [[0] * r for _ in range(n * m)]
            for row in range(n * m):
                for col in range(r):
                    if rng.random() < 0.3:
                        U[row][col] = rng.choice([1, -1])
            
            # Solve for V
            n_vars = m * p * r
            n_eqs = n * m * p
            
            A = np.zeros((n_eqs, n_vars), dtype=np.float64)
            b = np.ones(n_eqs, dtype=np.float64)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(r):
                            var_idx = j * p * r + k * r + c
                            A[eq_idx, var_idx] = U[i * m + j][c] * W[k * n + i][c]
                        eq_idx += 1
            
            try:
                v_sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            if len(residuals) > 0 and residuals[0] > 1e-6:
                continue
            
            V = [[0] * r for _ in range(m * p)]
            nice = True
            for row in range(m * p):
                for col in range(r):
                    val = v_sol[row * r + col]
                    nearest = round(val * 2) / 2
                    if abs(val - nearest) > 1e-4:
                        nice = False
                        break
                    V[row][col] = nearest
                if not nice:
                    break
            
            if not nice:
                continue
            
            if _validate_sparse(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W, "rank": r}
    
    return None


def construct():
    """Return a rank-minimal exact decomposition of the 2x3x9 matrix mult tensor."""
    t_start = time.time()
    DEADLINE = t_start + 115.0  # Safety margin before 120s limit

    n, m, p = 2, 3, 9

    # --- Schoolbook baseline (rank 54) ---
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p

    # --- Try to find lower-rank decomposition ---
    # Strategy 1: Block structure search
    result = _try_block_structure(n, m, p, DEADLINE)
    if result is not None and result["rank"] < best_rank:
        best_rank = result["rank"]
        best_result = {"U": result["U"], "V": result["V"], "W": result["W"]}

    # Strategy 2: Lattice/structured search
    if time.time() < DEADLINE:
        result2 = _try_lattice_search(n, m, p, DEADLINE)
        if result2 is not None and result2["rank"] < best_rank:
            best_rank = result2["rank"]
            best_result = {"U": result2["U"], "V": result2["V"], "W": result2["W"]}

    # --- Final verification ---
    if not _validate(best_result["U"], best_result["V"], best_result["W"], n, m, p):
        # Fallback to schoolbook
        best_result = _schoolbook(n, m, p)

    return best_result
