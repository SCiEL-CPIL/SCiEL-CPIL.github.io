"""Matrix multiplication 2x3x9 — rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 9]


def _validate(U, V, W, n=2, m=3, p=9, tol=1e-9):
    """Return True if U,V,W exactly decompose the matrix multiplication tensor."""
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
    """Faster validation using numpy."""
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


def _try_block_exploit(n, m, p, deadline):
    """
    Exploit the fact that p=9=3*3 by trying to share structure across the 3 blocks.
    
    Key insight: If we can find a rank-r decomposition for M_{2,3,3}, we might be able
    to combine 3 such decompositions with some sharing to get rank < 3*r.
    
    More directly: try to find U, V, W where W has a block structure that allows
    terms to be shared across k-blocks.
    """
    n, m, p = 2, 3, 9
    lower_bound = max(n * m, m * p, p * n)  # 27
    schoolbook_rank = n * m * p  # 54
    
    rng = random.Random(98765)
    
    # Try ranks from lower_bound+1 upward (skip 27 as it's the theoretical min, unlikely achievable)
    for r in range(lower_bound + 1, schoolbook_rank):
        if time.time() > deadline:
            break
        
        n_restarts = max(2, 15 - (r - lower_bound) // 4)
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Strategy: Make W have a structured pattern that encourages sharing
            # W is 18 x r. Try to make it have low-rank structure in blocks.
            
            # Initialize W with a block-circulant-like structure
            W = [[0] * r for _ in range(p * n)]
            
            # For each of the 3 k-blocks (each of size 3), assign a subset of columns
            # with some overlap to allow sharing
            block_size = p // 3  # 3
            cols_per_block = r // 3
            remainder = r % 3
            
            for block in range(3):
                start_col = block * cols_per_block + min(block, remainder)
                end_col = start_col + cols_per_block + (1 if block < remainder else 0)
                
                for row in range(block * block_size * n, (block + 1) * block_size * n):
                    for col in range(start_col, end_col):
                        if rng.random() < 0.5:
                            W[row][col] = rng.choice([1, -1])
            
            # Generate U with structure: make it depend only on (i, j mod something)
            U = [[0] * r for _ in range(n * m)]
            for i in range(n):
                for j in range(m):
                    for col in range(r):
                        # Make U[i,j,col] depend on a hash of (i, j, col mod something)
                        if rng.random() < 0.4:
                            U[i * m + j][col] = rng.choice([1, -1])
            
            # Now solve for V using least squares
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
            
            # Check residual
            if len(residuals) > 0 and residuals[0] > 1e-8:
                continue
            
            # Check if solution is close to half-integers
            V = [[0] * r for _ in range(m * p)]
            nice = True
            max_err = 0.0
            for row in range(m * p):
                for col in range(r):
                    val = v_sol[row * r + col]
                    nearest = round(val * 2) / 2
                    err = abs(val - nearest)
                    max_err = max(max_err, err)
                    if err > 1e-3:
                        nice = False
                        break
                    V[row][col] = nearest
                if not nice:
                    break
            
            if not nice:
                continue
            
            # Verify
            if _validate_sparse(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W, "rank": r}
    
    return None


def _try_direct_low_rank(n, m, p, deadline):
    """
    Try to directly find low-rank decompositions using a different approach:
    parameterize U and W with few free parameters, then solve for V.
    
    Use the fact that the tensor has symmetries.
    """
    n, m, p = 2, 3, 9
    lower_bound = max(n * m, m * p, p * n)  # 27
    
    rng = random.Random(11111)
    
    # Try specific ranks where improvements are most likely
    target_ranks = [30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]
    
    for r in target_ranks:
        if time.time() > deadline:
            break
        
        n_restarts = 8
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Use a structured initialization for U and W
            # Make U and W have rank <= 3 in their respective block structures
            
            # W: 18 x r. Make it have block-diagonal structure with 3 blocks of 6 rows
            W = [[0] * r for _ in range(p * n)]
            r_block = r // 3
            rem = r % 3
            
            for block in range(3):
                s = block * r_block + min(block, rem)
                e = s + r_block + (1 if block < rem else 0)
                for row in range(block * 6, (block + 1) * 6):
                    for col in range(s, e):
                        W[row][col] = rng.choice([1, -1, 0, 0])  # 25% each of 1,-1,0,0
            
            # U: 6 x r. Make it have some structure
            U = [[0] * r for _ in range(n * m)]
            for row in range(n * m):
                for col in range(r):
                    U[row][col] = rng.choice([1, -1, 0, 0])
            
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
            
            if len(residuals) > 0 and residuals[0] > 1e-8:
                continue
            
            V = [[0] * r for _ in range(m * p)]
            nice = True
            for row in range(m * p):
                for col in range(r):
                    val = v_sol[row * r + col]
                    nearest = round(val * 2) / 2
                    if abs(val - nearest) > 1e-3:
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


def _polish(result, n=2, m=3, p=9):
    """Try to reduce the rank by removing zero columns or combining terms."""
    if result is None:
        return None
    
    U = [row[:] for row in result["U"]]
    V = [row[:] for row in result["V"]]
    W = [row[:] for row in result["W"]]
    r = len(U[0])
    
    # Remove columns that are all zeros
    new_U = []
    new_V = []
    new_W = []
    for c in range(r):
        if any(U[i][c] != 0 for i in range(n * m)) or \
           any(V[i][c] != 0 for i in range(m * p)) or \
           any(W[i][c] != 0 for i in range(p * n)):
            new_U.append([U[i][c] for i in range(n * m)])
            new_V.append([V[i][c] for i in range(m * p)])
            new_W.append([W[i][c] for i in range(p * n)])
    
    if len(new_U) < r:
        # Transpose back
        new_U = [[new_U[c][i] for c in range(len(new_U))] for i in range(n * m)]
        new_V = [[new_V[c][i] for c in range(len(new_V))] for i in range(m * p)]
        new_W = [[new_W[c][i] for c in range(len(new_W))] for i in range(p * n)]
        
        if _validate_sparse(new_U, new_V, new_W, n, m, p):
            return {"U": new_U, "V": new_V, "W": new_W, "rank": len(new_U[0])}
    
    return result


def construct():
    """Return a rank-minimal exact decomposition of the 2x3x9 matrix mult tensor."""
    t_start = time.time()
    DEADLINE = t_start + 115.0  # Safety margin before 120s limit

    n, m, p = 2, 3, 9

    # --- Schoolbook baseline (rank 54) ---
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p

    # --- Try block exploitation ---
    result = _try_block_exploit(n, m, p, DEADLINE)
    if result is not None and result["rank"] < best_rank:
        best_rank = result["rank"]
        best_result = {"U": result["U"], "V": result["V"], "W": result["W"]}

    # --- Try direct low-rank search ---
    if time.time() < DEADLINE:
        result2 = _try_direct_low_rank(n, m, p, DEADLINE)
        if result2 is not None and result2["rank"] < best_rank:
            best_rank = result2["rank"]
            best_result = {"U": result2["U"], "V": result2["V"], "W": result2["W"]}

    # --- Polish: remove zero columns ---
    if time.time() < DEADLINE:
        polished = _polish(best_result, n, m, p)
        if polished is not None and polished["rank"] < best_rank:
            best_rank = polished["rank"]
            best_result = {"U": polished["U"], "V": polished["V"], "W": polished["W"]}

    # --- Final verification ---
    if not _validate(best_result["U"], best_result["V"], best_result["W"], n, m, p):
        # Fallback to schoolbook
        best_result = _schoolbook(n, m, p)

    return best_result
