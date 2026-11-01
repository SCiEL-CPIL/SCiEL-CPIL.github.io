import time
import numpy as np
from fractions import Fraction
import itertools

DIMENSIONS = [2, 2, 11]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # n*m = 4, m*p = 22, p*n = 22
    
    # Build the multiplication tensor entries
    # T[i, j, k] = 1 if C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor form: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                tensor_entries.append((u_idx, v_idx, w_idx))
    
    # Pre-compute indices for fast ALS
    entries_by_u = [[] for _ in range(n * m)]
    entries_by_v = [[] for _ in range(m * p)]
    entries_by_w = [[] for _ in range(p * n)]
    for (u_idx, v_idx, w_idx) in tensor_entries:
        entries_by_u[u_idx].append((v_idx, w_idx))
        entries_by_v[v_idx].append((u_idx, w_idx))
        entries_by_w[w_idx].append((u_idx, v_idx))
        
    # Set of non-zero entries for quick lookup
    nz_set = set(tensor_entries)
    # All possible (u,v,w) indices
    all_u = range(n * m)
    all_v = range(m * p)
    all_w = range(p * n)
    
    # Known rank-39 decomposition (Strassen-like for 2x2 * 2x11)
    def make_rank39():
        rank = 39
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        r = 0
        # 5 blocks of size 2 in p dimension, using 7 mults each = 35
        for block in range(5):
            k0 = block * 2
            k1 = block * 2 + 1
            
            v_e = k0      # B[0, k0]
            v_f = k1      # B[0, k1]
            v_g = 11 + k0 # B[1, k0]
            v_h = 11 + k1 # B[1, k1]
            
            w_c00 = 2 * k0     # C[0, k0]
            w_c10 = 2 * k0 + 1 # C[1, k0]
            w_c01 = 2 * k1     # C[0, k1]
            w_c11 = 2 * k1 + 1 # C[1, k1]
            
            # Strassen-like 2x2 * 2x2 block decomposition
            # m1 = (a+d)(e+h)
            U[0][r] = 1; U[3][r] = 1
            V[v_e][r] = 1; V[v_h][r] = 1
            W[w_c00][r] = 1; W[w_c11][r] = 1
            r += 1
            # m2 = (c+d)e
            U[2][r] = 1; U[3][r] = 1
            V[v_e][r] = 1
            W[w_c10][r] = 1; W[w_c11][r] = -1
            r += 1
            # m3 = a(f-h)
            U[0][r] = 1
            V[v_f][r] = 1; V[v_h][r] = -1
            W[w_c01][r] = 1; W[w_c11][r] = 1
            r += 1
            # m4 = d(g-e)
            U[3][r] = 1
            V[v_g][r] = 1; V[v_e][r] = -1
            W[w_c00][r] = 1; W[w_c10][r] = 1
            r += 1
            # m5 = (a+b)h
            U[0][r] = 1; U[1][r] = 1
            V[v_h][r] = 1
            W[w_c00][r] = -1; W[w_c01][r] = 1
            r += 1
            # m6 = (c-a)(e+f)
            U[2][r] = 1; U[0][r] = -1
            V[v_e][r] = 1; V[v_f][r] = 1
            W[w_c11][r] = 1
            r += 1
            # m7 = (b-d)(g+h)
            U[1][r] = 1; U[3][r] = -1
            V[v_g][r] = 1; V[v_h][r] = 1
            W[w_c00][r] = 1
            r += 1
        
        # Last block k=10 (only 1 column, not 2)
        # C[:, 10] = A * B[:, 10]
        # 2x2 * 2x1 = 4 mults
        k2 = 10
        v_e2 = k2      # B[0, 10]
        v_g2 = 11 + k2 # B[1, 10]
        w_c02 = 2 * k2     # C[0, 10]
        w_c12 = 2 * k2 + 1 # C[1, 10]
        
        U[0][r] = 1; V[v_e2][r] = 1; W[w_c02][r] = 1; r += 1
        U[1][r] = 1; V[v_g2][r] = 1; W[w_c02][r] = 1; r += 1
        U[2][r] = 1; V[v_e2][r] = 1; W[w_c12][r] = 1; r += 1
        U[3][r] = 1; V[v_g2][r] = 1; W[w_c12][r] = 1; r += 1
        
        return {"U": U, "V": V, "W": W}
    
    def verify_full(U, V, W):
        """
        Exact verification using integers (multiplied by 2 to avoid floats)
        U, V, W are lists of lists of half-integers (or integers)
        """
        rank = len(U[0])
        # Convert to integers by multiplying by 2
        U_int = [[int(x * 2) for x in row] for row in U]
        V_int = [[int(x * 2) for x in row] for row in V]
        W_int = [[int(x * 2) for x in row] for row in W]
        
        # Check non-zero entries: sum U*V*W = 1 => sum (2U)(2V)(2W) = 8
        for (u_idx, v_idx, w_idx) in tensor_entries:
            total = 0
            for r in range(rank):
                total += U_int[u_idx][r] * V_int[v_idx][r] * W_int[w_idx][r]
            if total != 8:
                return False
                
        # Check zero entries: sum U*V*W = 0 => sum (2U)(2V)(2W) = 0
        # There are 4*22*22 = 1936 total entries. 44 are non-zero. 1892 are zero.
        # To speed up, we can check all. 1892 * 39 multiplications is fast enough in Python?
        # 1892 * 39 * 3 ~ 200k ops. Yes.
        for u_idx in range(n * m):
            for v_idx in range(m * p):
                for w_idx in range(p * n):
                    if (u_idx, v_idx, w_idx) in nz_set:
                        continue
                    total = 0
                    for r in range(rank):
                        total += U_int[u_idx][r] * V_int[v_idx][r] * W_int[w_idx][r]
                    if total != 0:
                        return False
        return True

    def verify_nz(U, V, W):
        """Fast check for non-zero entries only (for search loop)"""
        rank = len(U[0])
        for (u_idx, v_idx, w_idx) in tensor_entries:
            total = 0.0
            for r in range(rank):
                total += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
            if abs(total - 1) > 1e-9:
                return False
        return True

    def try_als_rank(target_rank, max_iters=100, seed=42, init_type='random'):
        if time.time() > deadline - 5:
            return None
        
        np.random.seed(seed)
        
        if init_type == 'random':
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
        elif init_type == 'strassen_perturb':
            r39 = make_rank39()
            U_full = np.array(r39['U'], dtype=float)
            V_full = np.array(r39['V'], dtype=float)
            W_full = np.array(r39['W'], dtype=float)
            # Truncate to target_rank and perturb
            U = U_full[:, :target_rank].copy() + np.random.randn(n * m, target_rank) * 0.1
            V = V_full[:, :target_rank].copy() + np.random.randn(m * p, target_rank) * 0.1
            W = W_full[:, :target_rank].copy() + np.random.randn(p * n, target_rank) * 0.1
        else:
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
        
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # ALS Step
            # Update U
            for u_idx in range(n * m):
                e_list = entries_by_u[u_idx]
                if not e_list: continue
                A_mat = np.array([V[v_idx] * W[w_idx] for (v_idx, w_idx) in e_list])
                b_vec = np.ones(len(e_list))
                try:
                    x, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    U[u_idx] = x
                except: pass
            
            # Update V
            for v_idx in range(m * p):
                e_list = entries_by_v[v_idx]
                if not e_list: continue
                A_mat = np.array([U[u_idx] * W[w_idx] for (u_idx, w_idx) in e_list])
                b_vec = np.ones(len(e_list))
                try:
                    x, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    V[v_idx] = x
                except: pass
            
            # Update W
            for w_idx in range(p * n):
                e_list = entries_by_w[w_idx]
                if not e_list: continue
                A_mat = np.array([U[u_idx] * V[v_idx] for (u_idx, v_idx) in e_list])
                b_vec = np.ones(len(e_list))
                try:
                    x, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    W[w_idx] = x
                except: pass
            
            # Compute error on non-zero entries
            err = 0.0
            for (u_idx, v_idx, w_idx) in tensor_entries:
                val = np.dot(U[u_idx] * V[v_idx], W[w_idx])
                err += (val - 1.0) ** 2
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-12:
                break
        
        if best_U is None:
            return None
        
        # Round to half-integers
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        U_list = U_rounded.tolist()
        V_list = V_rounded.tolist()
        W_list = W_rounded.tolist()
        
        # Full verification
        if verify_full(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}
        
        # If direct rounding failed, try Discrete Local Search (DLS)
        # This is the key improvement: search nearby half-integer points
        if time.time() < deadline - 10:
            dls_result = discrete_local_search(U_list, V_list, W_list, target_rank)
            if dls_result:
                return dls_result
        
        return None

    def discrete_local_search(U, V, W, rank):
        """
        Perform a local search on the half-integer grid around the initial guess.
        We try flipping single elements in U, V, or W to see if it improves validity.
        """
        U_curr = [row[:] for row in U]
        V_curr = [row[:] for row in V]
        W_curr = [row[:] for row in W]
        
        # Pre-compute current error on non-zero entries for quick check
        # We only care if it becomes valid.
        
        def is_valid(U_l, V_l, W_l):
            return verify_full(U_l, V_l, W_l)
            
        # Heuristic: Try to reduce the error on non-zero entries or satisfy zero entries.
        # A simple hill-climbing approach:
        # 1. Check if current is valid.
        if is_valid(U_curr, V_curr, W_curr):
            return {"U": U_curr, "V": V_curr, "W": W_curr}
            
        # 2. Try single-step moves: change one half-integer to another nearby half-integer.
        # Possible values: ..., -1, -0.5, 0, 0.5, 1, ...
        # We restrict to small integers for simplicity: {-1, -0.5, 0, 0.5, 1}
        
        # To keep it fast, we limit the number of iterations.
        max_dls_iters = 500
        current_err = float('inf')
        
        # Calculate a proxy error: sum of squares of non-zero entries (should be 1) 
        # + sum of squares of zero entries (should be 0)
        # This is expensive to compute fully every step. 
        # Instead, we just check validity periodically or use a cheaper metric.
        # Given the small size, let's just try random perturbations and check validity.
        
        # Strategy: Random restarts of DLS
        for restart in range(5):
            if time.time() > deadline - 5:
                return None
                
            U_t = [row[:] for row in U]
            V_t = [row[:] for row in V]
            W_t = [row[:] for row in W]
            
            for step in range(max_dls_iters):
                if time.time() > deadline - 5:
                    return None
                    
                # Pick a random factor and a random element
                factor = np.random.choice(3)
                if factor == 0:
                    idx = np.random.choice(n * m)
                    col = np.random.choice(rank)
                    old_val = U_t[idx][col]
                elif factor == 1:
                    idx = np.random.choice(m * p)
                    col = np.random.choice(rank)
                    old_val = V_t[idx][col]
                else:
                    idx = np.random.choice(p * n)
                    col = np.random.choice(rank)
                    old_val = W_t[idx][col]
                
                # Pick a new value from neighbors
                # Neighbors of x are x-0.5, x+0.5, and maybe x (stay)
                # We want to move towards integers or 0.5
                candidates = []
                # Round to nearest 0.5
                nearest_half = np.round(old_val * 2) / 2
                candidates.append(nearest_half - 0.5)
                candidates.append(nearest_half)
                candidates.append(nearest_half + 0.5)
                
                # Filter to reasonable range [-2, 2]
                candidates = [c for c in candidates if -2 <= c <= 2]
                if not candidates:
                    candidates = [0.0]
                    
                new_val = np.random.choice(candidates)
                
                if new_val == old_val:
                    continue
                    
                # Apply change
                if factor == 0:
                    U_t[idx][col] = new_val
                elif factor == 1:
                    V_t[idx][col] = new_val
                else:
                    W_t[idx][col] = new_val
                    
                # Check validity
                if is_valid(U_t, V_t, W_t):
                    return {"U": U_t, "V": V_t, "W": W_t}
                    
                # If not valid, revert (or keep if we were doing simulated annealing, 
                # but simple hill climbing on validity is binary. 
                # Since validity is rare, we might need to use an error metric to guide the search.)
                
                # Let's use an error metric to guide the search.
                # Error = sum (val_nz - 1)^2 + sum (val_zero)^2
                # This is expensive. Let's approximate with just non-zero error?
                # No, zero constraints are crucial.
                
                # Optimization: Only check full validity every N steps?
                # Or compute error and keep if error decreases?
                
                # Let's compute the error.
                err = 0.0
                # Non-zero
                for (u_idx, v_idx, w_idx) in tensor_entries:
                    val = 0
                    for r in range(rank):
                        val += U_t[u_idx][r] * V_t[v_idx][r] * W_t[w_idx][r]
                    err += (val - 1.0) ** 2
                # Zero (sample 100 random zero entries for speed)
                if np.random.rand() < 0.1: # Check zero constraints less frequently
                    for _ in range(50):
                        u_idx = np.random.choice(n * m)
                        v_idx = np.random.choice(m * p)
                        w_idx = np.random.choice(p * n)
                        if (u_idx, v_idx, w_idx) not in nz_set:
                            val = 0
                            for r in range(rank):
                                val += U_t[u_idx][r] * V_t[v_idx][r] * W_t[w_idx][r]
                            err += val ** 2
                            
                # If error is 0, it's valid.
                if err < 1e-9:
                    # Verify fully to be sure
                    if is_valid(U_t, V_t, W_t):
                        return {"U": U_t, "V": V_t, "W": W_t}
                
                # Simple heuristic: if error increased, revert.
                # We don't have previous error easily accessible in this loop structure without storing it.
                # Let's just keep the change. Random walk.
                
        return None

    # Search for lower ranks
    # Try ranks 38, 37, 36
    for target_rank in [38, 37, 36]:
        if time.time() > deadline - 15:
            break
            
        # Multiple seeds
        for seed in range(10):
            if time.time() > deadline - 15:
                break
                
            # Alternate between random and strassen-perturb
            init_type = 'random' if seed % 2 == 0 else 'strassen_perturb'
            
            result = try_als_rank(target_rank, max_iters=50, seed=seed, init_type=init_type)
            if result is not None:
                return result
                
    # Fallback
    return make_rank39()
