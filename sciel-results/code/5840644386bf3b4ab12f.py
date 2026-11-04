import time
import numpy as np
import sys

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact decomposition for matrix multiplication 2x4 by 4x7.
    
    Strategy:
    1. Discrete Search: Since the solution must have integer or half-integer entries,
       we search directly on the lattice {0, 0.5, 1.0, ...}.
    2. We use a Simulated Annealing style approach. We start with a random valid 
       decomposition (or a known valid one like schoolbook) and perturb it.
    3. The objective is to minimize the "rank" (number of non-zero columns) while 
       maintaining exactness. 
    4. To make this tractable, we fix a target rank R and try to find a valid 
       decomposition of that rank. We start from low R (lower bound) and increase.
    5. For a fixed R, we use a local search:
       - Initialize U, V, W with small random integers/half-integers.
       - Calculate Error = sum (T_target - T_recon)^2.
       - If Error == 0, we have a solution.
       - If not, we perform random moves: change one entry in U, V, or W to a 
         nearby integer/half-integer value.
       - Accept move if it reduces Error, or with probability exp(-delta_E/T) 
         (Simulated Annealing) to escape local minima.
    6. We also try a "pruning" strategy: Start with a known valid decomposition 
       of higher rank (e.g., 56), and try to zero out columns one by one while 
       re-optimizing the remaining entries to maintain exactness.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # Safety margin
    
    # Lower bound for rank is max(n*m, m*p, n*p) = max(8, 28, 14) = 28.
    # Theoretical rank of M_{2,4,7} is likely around 30-40.
    
    def verify_exact(U, V, W, rank, n, m, p, tol=1e-9):
        """Check if the decomposition is exactly correct."""
        U_arr = np.array(U)[:, :rank]
        V_arr = np.array(V)[:, :rank]
        W_arr = np.array(W)[:, :rank]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U_arr[u_idx], V_arr[v_idx] * W_arr[w_idx])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def make_schoolbook(n, m, p):
        """Standard rank n*m*p decomposition."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank

    # Fallback: Schoolbook
    U_fb, V_fb, W_fb, rank_fb = make_schoolbook(n, m, p)
    best_result = {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    best_rank = rank_fb

    # Discrete Search Parameters
    # We will try ranks from 28 up to 50.
    # For each rank, we run a limited number of iterations of local search.
    
    target_ranks = [28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    
    # Values allowed: -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2
    # We restrict to small values to keep the search space manageable.
    # Actually, let's allow a slightly wider range but prioritize small ones.
    # Let's stick to integers and half-integers in [-3, 3] for now.
    # To be efficient, we only propose moves to "nearby" values.
    
    def compute_error(U, V, W, n, m, p):
        """Compute sum of squared errors."""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx], V[v_idx] * W[w_idx])
                    err += (val - 1.0) ** 2
        return err

    def random_init_discrete(n, m, p, R, rng, scale=1.0):
        """Initialize with random discrete values."""
        # Values: 0, +/-0.5, +/-1, +/-1.5, +/-2
        # We want them to be small.
        possible_vals = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
        U = np.array(rng.choice(possible_vals, (n*m, R)))
        V = np.array(rng.choice(possible_vals, (m*p, R)))
        W = np.array(rng.choice(possible_vals, (p*n, R)))
        # Scale to ensure products are not too large
        # If we pick random values, the sum of R products might be large.
        # We want sum ~ 1.
        # Let's just let the search handle it, but maybe start with sparser matrices.
        # 80% zeros, 20% non-zeros.
        mask_U = rng.random((n*m, R)) < 0.2
        mask_V = rng.random((m*p, R)) < 0.2
        mask_W = rng.random((p*n, R)) < 0.2
        U[~mask_U] = 0.0
        V[~mask_V] = 0.0
        W[~mask_W] = 0.0
        return U, V, W

    # We will use a specific "Move" strategy for the local search.
    # A move consists of:
    # 1. Selecting a matrix (U, V, or W)
    # 2. Selecting a row
    # 3. Selecting a column (index < R)
    # 4. Changing the value to a nearby discrete value.
    
    # To speed up error calculation, we can update only the affected terms.
    # T[i,j,k] depends on U[i*m+j, :], V[j*p+k, :], W[k*n+i, :]
    # If we change U[u_idx, c], it affects all (i,j,k) such that i*m+j == u_idx.
    # There are p such (i,j,k) tuples for a fixed (i,j).
    # Specifically, for fixed i,j, k varies from 0 to p-1.
    # So changing U[i*m+j, c] affects T[i,j,k] for all k.
    
    def get_affected_indices(matrix_name, row_idx, n, m, p):
        """
        Returns list of (i, j, k) tuples affected by changing matrix_name[row_idx, c].
        Also returns the other two matrix indices involved.
        """
        affected = []
        if matrix_name == 'U':
            # U[i*m+j, c]
            u_idx = row_idx
            i = u_idx // m
            j = u_idx % m
            for k in range(p):
                v_idx = j * p + k
                w_idx = k * n + i
                affected.append((i, j, k, v_idx, w_idx))
        elif matrix_name == 'V':
            # V[j*p+k, c]
            v_idx = row_idx
            j = v_idx // p
            k = v_idx % p
            for i in range(n):
                u_idx = i * m + j
                w_idx = k * n + i
                affected.append((i, j, k, u_idx, w_idx))
        elif matrix_name == 'W':
            # W[k*n+i, c]
            w_idx = row_idx
            k = w_idx // n
            i = w_idx % n
            for j in range(m):
                u_idx = i * m + j
                v_idx = j * p + k
                affected.append((i, j, k, u_idx, v_idx))
        return affected

    # Since n, m, p are small, we can just recompute the full error for simplicity 
    # in the first pass, but incremental is better. 
    # Given the small size (56 entries), full recompute is ~56 dot products of length R.
    # R ~ 30. 56 * 30 = 1680 ops. Very fast.
    # So we can just recompute full error for each move.
    
    for R in target_ranks:
        if time.time() > deadline:
            break
            
        if R >= best_rank:
            continue
            
        # Try multiple restarts for this R
        found = False
        max_restarts = 50
        max_steps_per_restart = 2000
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            rng = np.random.default_rng(42 + restart)
            U, V, W = random_init_discrete(n, m, p, R, rng)
            
            # Initial error
            curr_err = compute_error(U, V, W, n, m, p)
            
            # If we start with 0 error, we are done
            if curr_err < 1e-9:
                found = True
                break
                
            # Simulated Annealing
            T = 10.0
            cooling_rate = 0.99
            
            for step in range(max_steps_per_restart):
                if time.time() > deadline:
                    break
                    
                # Propose a move
                mat_choice = rng.integers(0, 3) # 0:U, 1:V, 2:W
                if mat_choice == 0:
                    mat = 'U'
                    arr = U
                    n_rows = n * m
                elif mat_choice == 1:
                    mat = 'V'
                    arr = V
                    n_rows = m * p
                else:
                    mat = 'W'
                    arr = W
                    n_rows = p * n
                
                row = rng.integers(0, n_rows)
                col = rng.integers(0, R)
                
                old_val = arr[row, col]
                
                # Propose new value: nearby discrete value
                # Let's pick a random value from a small set around old_val
                # Or just pick a random value from the allowed set.
                # To be efficient, let's pick from [-2, 2] step 0.5
                # But to avoid huge jumps, let's pick from old_val +/- 0.5, +/- 1.0
                # Or just random from the set of allowed values.
                # Let's use a random walk on the lattice.
                # Possible values: ... -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5 ...
                # Let's just pick a random integer or half-integer in [-3, 3]
                # This might be too random. Let's try "local" moves first.
                
                # Local move: +/- 0.5 or +/- 1.0
                delta = rng.choice([-1.0, -0.5, 0.5, 1.0])
                new_val = old_val + delta
                # Clamp to [-3, 3]
                if new_val < -3.0: new_val = -3.0
                if new_val > 3.0: new_val = 3.0
                # Ensure it's a multiple of 0.5
                new_val = np.round(new_val * 2) / 2
                
                if new_val == old_val:
                    continue
                    
                # Apply move
                arr[row, col] = new_val
                
                # Calculate new error
                new_err = compute_error(U, V, W, n, m, p)
                
                # Acceptance criterion
                delta_err = new_err - curr_err
                if delta_err < 0:
                    # Accept
                    curr_err = new_err
                else:
                    # Accept with probability exp(-delta_err / T)
                    if T > 1e-6:
                        prob = np.exp(-delta_err / T)
                        if rng.random() < prob:
                            curr_err = new_err
                        else:
                            # Reject
                            arr[row, col] = old_val
                    else:
                        # Reject
                        arr[row, col] = old_val
                        
                T *= cooling_rate
                
                if curr_err < 1e-9:
                    found = True
                    break
                    
            if found:
                break
                
        if found:
            # Verify exactness
            if verify_exact(U.tolist(), V.tolist(), W.tolist(), R, n, m, p):
                best_rank = R
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                # Since we iterate R upwards, the first one found is the lowest.
                # We can break early.
                break
            else:
                # Floating point issue? Force rounding
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                if verify_exact(U.tolist(), V.tolist(), W.tolist(), R, n, m, p):
                    best_rank = R
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break

    return best_result
# EVOLVE-BLOCK-END
