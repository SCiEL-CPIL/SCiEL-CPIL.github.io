"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    
    Strategy:
    1. Start with schoolbook (rank 36) as fallback.
    2. Use structured random search with better initialization (Strassen-inspired).
    3. Use vectorized NumPy for smooth error calculation to speed up local search.
    4. Use coordinate descent / hill climbing on the smooth surrogate.
    5. Final polish with exact rational arithmetic.
    """
    start_time = time.time()
    deadline = start_time + 135  # 135s budget, leaving 15s margin for 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(dim_U)]
        V = [[0] * rank for _ in range(dim_V)]
        W = [[0] * rank for _ in range(dim_W)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Active triples: (a, b, c) where a=i*m+j, b=j*p+k, c=k*n+i
    active_triples = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                active_triples.add((a, b, c))
    
    # Precompute all triples that need to be checked
    all_triples = list(itertools.product(range(dim_U), range(dim_V), range(dim_W)))
    active_set = set(active_triples)
    
    # Convert to numpy arrays for vectorized operations
    # T[a, b, c] is 1 if active, 0 otherwise
    T = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
    for a, b, c in active_set:
        T[a, b, c] = 1.0
        
    # Precompute indices for faster access
    # We will compute the tensor product U[:,:,None] * V[None,:,:] * W[None,None,:] 
    # But that's O(R * dim_U * dim_V * dim_W) which is R * 9 * 12 * 12 = 1296 R.
    # For R=24, that's ~31k operations per error calc. Very fast in numpy.
    
    def verify(U, V, W, r):
        """Check if U, V, W with rank r satisfy all constraints using exact arithmetic."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        
        # Calculate the tensor product
        # U shape: (dim_U, r)
        # V shape: (dim_V, r)
        # W shape: (dim_W, r)
        
        # We need sum_r U[a,r] V[b,r] W[c,r]
        # This is equivalent to contracting the batch dimension r.
        # Let's do it in chunks or just loop over r if r is small?
        # Actually, einsum is best: 'ar,br,cr->abc'
        prod = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Check against T
        # T is 1 for active, 0 for inactive
        # prod should match T exactly
        if not np.allclose(prod, T, atol=0, rtol=0):
            return False
        return True

    def smooth_error_np(U, V, W, r):
        """
        Compute smooth squared error using NumPy.
        U, V, W are numpy arrays.
        """
        # prod[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        prod = np.einsum('ar,br,cr->abc', U, V, W)
        diff = prod - T
        return np.sum(diff * diff)

    def local_search_smooth_np(U, V, W, r, max_iters=10000, time_limit=10.0):
        """
        Local search using the smooth squared-error objective.
        U, V, W are numpy arrays.
        """
        start_ls = time.time()
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        current_err = smooth_error_np(U, V, W, r)
        best_err = current_err
        best_U = U.copy()
        best_V = V.copy()
        best_W = W.copy()
        
        # Temperature for simulated annealing
        T = 5.0
        T_min = 0.001
        cooling = 0.999
        
        # Pre-allocate for speed? No, just use standard ops.
        
        for it in range(max_iters):
            if time.time() > time_limit or time.time() > deadline:
                break
            
            # Pick a random matrix, row, and column to perturb
            mat_choice = random.choice(['U', 'V', 'W'])
            if mat_choice == 'U':
                mat = U
                nrows = dim_U
            elif mat_choice == 'V':
                mat = V
                nrows = dim_V
            else:
                mat = W
                nrows = dim_W
            
            row = random.randint(0, nrows - 1)
            col = random.randint(0, r - 1)
            
            # Try small integer perturbations
            # Since we want integer/half-integer, let's perturb by 1 or 0.5?
            # The problem statement says "integer/half-integer entries".
            # Let's stick to integers for the main search, then allow 0.5 in polish?
            # Or just search in integers, as most optimal decompositions for small matrices are integer.
            delta = random.choice([-2, -1, 1, 2])
            
            old_val = mat[row, col]
            new_val = old_val + delta
            
            # Keep values in a reasonable range
            if abs(new_val) > 10:
                continue
            
            mat[row, col] = new_val
            new_err = smooth_error_np(U, V, W, r)
            
            # Accept if better, or with probability exp(-(new_err - current_err)/T)
            if new_err <= current_err or random.random() < np.exp(-(new_err - current_err) / max(T, 1e-10)):
                current_err = new_err
                if new_err < best_err:
                    best_err = new_err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
            else:
                mat[row, col] = old_val
            
            T = max(T_min, T * cooling)
        
        return best_U, best_V, best_W, best_err

    def polish(U, V, W, r):
        """
        Final polish: try to fix any remaining small errors by adjusting entries.
        Use exact integer arithmetic.
        """
        # Convert to lists for easy mutation in verify
        U_list = [row[:] for row in U]
        V_list = [row[:] for row in V]
        W_list = [row[:] for row in W]
        
        if verify(U_list, V_list, W_list, r):
            return U_list, V_list, W_list, True
        
        # If smooth error is very low, we are close.
        # Try to snap to nearest integer or half-integer?
        # The problem allows half-integers.
        # Let's try to find an exact solution near the current floating point solution.
        
        # Strategy: For each entry, try rounding to nearest integer, and nearest half-integer.
        # This is a huge search space if done globally.
        # Instead, do a local coordinate search on the exact error count.
        
        # First, let's see if we can just round everything.
        U_round = [[round(x) for x in row] for row in U_list]
        V_round = [[round(x) for x in row] for row in V_list]
        W_round = [[round(x) for x in row] for row in W_list]
        if verify(U_round, V_round, W_round, r):
            return U_round, V_round, W_round, True
            
        # Try half-integers
        U_half = [[int(x*2)/2 for x in row] for row in U_list]
        V_half = [[int(x*2)/2 for x in row] for row in V_list]
        W_half = [[int(x*2)/2 for x in row] for row in W_list]
        if verify(U_half, V_half, W_half, r):
            return U_half, V_half, W_half, True
            
        # Local search on exact error
        def error_count_exact(U, V, W, r):
            errors = 0
            for a, b, c in all_triples:
                s = 0
                for ci in range(r):
                    s += U[a][ci] * V[b][ci] * W[c][ci]
                expected = 1 if (a, b, c) in active_set else 0
                if s != expected:
                    errors += 1
            return errors

        curr_err = error_count_exact(U_list, V_list, W_list, r)
        if curr_err == 0:
            return U_list, V_list, W_list, True

        # Hill climbing on exact error
        for _ in range(1000):
            if time.time() > deadline:
                break
            
            improved = False
            # Pick a random matrix, row, col
            mat_choice = random.choice(['U', 'V', 'W'])
            if mat_choice == 'U':
                mat = U_list
                nrows = dim_U
            elif mat_choice == 'V':
                mat = V_list
                nrows = dim_V
            else:
                mat = W_list
                nrows = dim_W
            
            row = random.randint(0, nrows - 1)
            col = random.randint(0, r - 1)
            
            # Try small integer deltas
            for delta in [-2, -1, 1, 2]:
                old_val = mat[row][col]
                mat[row][col] = old_val + delta
                new_err = error_count_exact(U_list, V_list, W_list, r)
                if new_err < curr_err:
                    curr_err = new_err
                    improved = True
                    break
                elif new_err == curr_err and random.random() < 0.1:
                    # Plateau move
                    pass
                else:
                    mat[row][col] = old_val
            
            if not improved and curr_err == 0:
                break
                
        if curr_err == 0:
            return U_list, V_list, W_list, True
            
        return U_list, V_list, W_list, False

    best_result = schoolbook()
    best_rank = n * m * p
    
    def try_rank_structured(r, max_attempts=200):
        """Try to find a rank-r decomposition using structured random search with smooth local search."""
        if time.time() > deadline:
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            strategy = attempt % 4
            
            if strategy == 0:
                # Sparse random: mostly zeros with some ±1
                U = np.zeros((dim_U, r), dtype=np.float64)
                V = np.zeros((dim_V, r), dtype=np.float64)
                W = np.zeros((dim_W, r), dtype=np.float64)
                for a in range(dim_U):
                    num_nonzero = random.randint(1, min(4, r))
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        U[a, ci] = random.choice([-1, 1])
                for b in range(dim_V):
                    num_nonzero = random.randint(1, min(4, r))
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        V[b, ci] = random.choice([-1, 1])
                for c in range(dim_W):
                    num_nonzero = random.randint(1, min(4, r))
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        W[c, ci] = random.choice([-1, 1])
                        
            elif strategy == 1:
                # Dense random with small values
                scale = 2
                U = np.random.randint(-scale, scale + 1, size=(dim_U, r)).astype(np.float64)
                V = np.random.randint(-scale, scale + 1, size=(dim_V, r)).astype(np.float64)
                W = np.random.randint(-scale, scale + 1, size=(dim_W, r)).astype(np.float64)
                
            elif strategy == 2:
                # Strassen-inspired: use block structure
                # Initialize with random superpositions of known low-rank patterns
                U = np.zeros((dim_U, r), dtype=np.float64)
                V = np.zeros((dim_V, r), dtype=np.float64)
                W = np.zeros((dim_W, r), dtype=np.float64)
                
                for ci in range(r):
                    # Random signs for a "tensor"
                    u_vals = np.random.choice([-1, 0, 1], size=dim_U)
                    v_vals = np.random.choice([-1, 0, 1], size=dim_V)
                    w_vals = np.random.choice([-1, 0, 1], size=dim_W)
                    U[:, ci] = u_vals
                    V[:, ci] = v_vals
                    W[:, ci] = w_vals
            
            else:
                # Hybrid: start from a partial schoolbook and merge columns
                U = np.zeros((dim_U, r), dtype=np.float64)
                V = np.zeros((dim_V, r), dtype=np.float64)
                W = np.zeros((dim_W, r), dtype=np.float64)
                
                for ci in range(r):
                    # Pick 2-3 random schoolbook triples and superpose
                    num_terms = random.randint(1, 3)
                    for _ in range(num_terms):
                        i = random.randint(0, n - 1)
                        j = random.randint(0, m - 1)
                        k = random.randint(0, p - 1)
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        sign = random.choice([-1, 1])
                        U[a, ci] += sign
                        V[b, ci] += sign
                        W[c, ci] += sign
            
            # Quick check: verify active triples first
            active_ok = True
            for (a, b, c) in active_triples:
                s = sum(U[a, ci] * V[b, ci] * W[c, ci] for ci in range(r))
                if abs(s - 1) > 1e-9:
                    active_ok = False
                    break
            
            if not active_ok:
                # Try smooth local search to fix
                time_remaining = deadline - time.time()
                if time_remaining > 3:
                    U, V, W, err = local_search_smooth_np(U, V, W, r, max_iters=2000, time_limit=min(3.0, time_remaining * 0.3))
                    # Re-check
                    active_ok = True
                    for (a, b, c) in active_triples:
                        s = sum(U[a, ci] * V[b, ci] * W[c, ci] for ci in range(r))
                        if abs(s - 1) > 1e-9:
                            active_ok = False
                            break
                    if not active_ok:
                        continue
            
            # Full verification
            if verify(U, V, W, r):
                return {"U": [row.tolist() for row in U], "V": [row.tolist() for row in V], "W": [row.tolist() for row in W]}
            
            # If close, try polish
            # Calculate error count
            prod = np.einsum('ar,br,cr->abc', U, V, W)
            diff = prod - T
            err_count = np.sum(np.abs(diff) > 1e-9)
            
            if err_count <= 5 and time.time() < deadline - 2:
                U_list = [row.tolist() for row in U]
                V_list = [row.tolist() for row in V]
                W_list = [row.tolist() for row in W]
                U_list, V_list, W_list, ok = polish(U_list, V_list, W_list, r)
                if ok:
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try to find lower rank decompositions
    # Start from a reasonable guess. Rank 24 is a good target.
    for r in range(22, best_rank):
        if time.time() > deadline:
            break
        
        time_remaining = deadline - time.time()
        if time_remaining < 5:
            break
        
        if r <= 24:
            max_attempts = min(1000, int(time_remaining * 20))
        elif r <= 28:
            max_attempts = min(500, int(time_remaining * 10))
        else:
            max_attempts = min(200, int(time_remaining * 5))
        
        result = try_rank_structured(r, max_attempts)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    return best_result
# EVOLVE-BLOCK-END
