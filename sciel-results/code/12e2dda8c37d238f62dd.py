import time
import numpy as np

def construct(q, n, radius):
    t_start = time.time()
    deadline = 118.0  # Safety margin before 120s limit

    space_size = q ** n
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i - 1] * q

    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * q + d
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        return w

    # Precompute neighbors (radius=1: self + all 1-change neighbors)
    nb_count = 1 + n * (q - 1)
    neighbors = np.empty((space_size, nb_count), dtype=np.int32)
    for idx in range(space_size):
        w = idx_to_word(idx)
        nb_list = [idx]
        for pos in range(n):
            for new_d in range(q):
                if new_d == w[pos]:
                    continue
                delta = (new_d - w[pos]) * q_powers[n - 1 - pos]
                nb_list.append(idx + delta)
        neighbors[idx] = nb_list

    def build_cov(code_indices):
        cov = np.zeros(space_size, dtype=np.int32)
        for c in code_indices:
            cov[neighbors[c]] += 1
        return cov

    def greedy_cover():
        """Fast greedy set cover to get a valid initial solution."""
        uncovered = np.ones(space_size, dtype=bool)
        code = []
        while np.any(uncovered) and (time.time() - t_start) < deadline - 60.0:
            # Gain: number of currently uncovered points covered by this codeword
            gains = np.sum(uncovered[neighbors], axis=1)
            best = np.argmax(gains)
            if gains[best] == 0:
                break
            code.append(int(best))
            uncovered[neighbors[best]] = False
        return code

    def lagrangian_optimization(initial_code, max_time=60.0):
        """
        Solve the dual of the set cover LP using subgradient method.
        This often finds a solution smaller than greedy.
        """
        t_lag = time.time()
        
        # Dual variables: one for each point (constraint)
        # Initialize to 1.0
        dual = np.ones(space_size, dtype=np.float64)
        
        # Current best solution
        best_solution = initial_code[:]
        best_cost = len(best_solution)
        
        # Precompute neighbor indices for fast gain calculation
        # We need to compute: gain_i = sum_{j in neighbors[i]} dual_j - 1
        # We can precompute the neighbor sums efficiently using matrix multiplication
        # or simple loops if space is small enough.
        # space_size = 2187, nb_count = 19. 
        # neighbors is (2187, 19).
        # dual is (2187,).
        # gains = neighbors @ dual - 1.
        
        step_size = 1.0
        last_update = 0
        
        while (time.time() - t_start) < deadline - 10.0 and (time.time() - t_lag) < max_time:
            # 1. Compute gains
            # gains[i] = sum(dual[j] for j in neighbors[i]) - 1
            gains = neighbors @ dual - 1.0
            
            # 2. Update solution: include codeword if gain > 0
            solution_mask = gains > 0
            solution_indices = np.where(solution_mask)[0]
            
            # 3. Check validity and cost
            if len(solution_indices) == 0:
                # If empty, fallback to best known or greedy
                break
                
            # Check if it covers everything
            # We can check coverage by summing neighbor counts
            # But checking coverage of a specific set is O(N * K).
            # Let's verify coverage efficiently.
            # If we just take all i with gain > 0, does it cover everything?
            # Not necessarily. The LP relaxation is a relaxation.
            # We need to ensure the set {i : gain_i > 0} is a valid cover.
            # Actually, the condition for a point j to be covered is sum_{i: j in N(i)} x_i >= 1.
            # If we set x_i = 1 if gain_i > 0, we don't guarantee coverage.
            # However, often the set of "positive gain" nodes forms a cover or is close.
            # Let's verify coverage.
            
            # Efficient coverage check:
            # cov = np.zeros(space_size, dtype=np.int8)
            # for c in solution_indices:
            #     cov[neighbors[c]] += 1
            # if not np.all(cov > 0):
            #     # Repair needed
            #     pass
            
            # Instead of full repair which is expensive, let's just check if it's valid.
            # If valid, update best.
            # If not, we can try to repair it or just skip this iteration's solution.
            
            # For speed, let's only update best if the solution is valid.
            # To check validity:
            cov = np.zeros(space_size, dtype=np.int8)
            for c in solution_indices:
                cov[neighbors[c]] += 1
            
            is_valid = np.all(cov > 0)
            
            if is_valid and len(solution_indices) < best_cost:
                best_solution = solution_indices.tolist()
                best_cost = len(best_solution)
                last_update = time.time()
                
            # 4. Compute subgradient for dual update
            # Subgradient for dual_j = sum_{i: j in N(i)} x_i - 1
            # Where x_i = 1 if gain_i > 0 else 0
            # This is the same as cov computed above.
            # subgrad_j = cov_j - 1
            
            subgrad = cov.astype(np.float64) - 1.0
            
            # 5. Update dual variables
            # dual = max(0, dual - step * subgrad)
            # Adaptive step size
            if time.time() - last_update > 5.0:
                step_size *= 0.9 # Reduce step if no improvement for a while
            
            dual = np.maximum(0, dual - step_size * subgrad)
            
            # Optional: Normalize or scale dual to prevent drift
            # if np.max(dual) > 10:
            #     dual /= 10.0

        return best_solution

    # --- Phase 1: Initial Greedy Cover ---
    init_code = greedy_cover()
    
    if len(init_code) == 0:
        # Fallback for very small spaces or errors
        return [idx_to_word(i) for i in range(space_size)]
        
    best_code = init_code[:]
    best_size = len(best_code)
    
    # --- Phase 2: Lagrangian Optimization ---
    if (time.time() - t_start) < deadline - 20.0:
        lag_code = lagrangian_optimization(init_code, max_time=50.0)
        if len(lag_code) < best_size:
            best_code = lag_code
            best_size = len(best_code)
            
    # --- Phase 3: Local Search Cleanup (Deterministic Removal) ---
    # This is fast and often shaves off a few more words.
    if (time.time() - t_start) < deadline - 5.0:
        cur_code = best_code[:]
        cur_cov = build_cov(cur_code)
        
        changed = True
        while changed and (time.time() - t_start) < deadline - 2.0:
            changed = False
            for i in range(len(cur_code)):
                c = cur_code[i]
                # If min coverage in neighborhood is >= 2, we can remove c
                # and still have coverage >= 1 everywhere.
                if np.min(cur_cov[neighbors[c]]) >= 2:
                    cur_code.pop(i)
                    cur_cov[neighbors[c]] -= 1
                    changed = True
                    break
        
        if len(cur_code) < best_size:
            best_code = cur_code
            best_size = len(best_code)

    # Final Verification
    if len(best_code) == 0:
        return [idx_to_word(i) for i in range(space_size)]

    final_cov = build_cov(best_code)
    if not np.all(final_cov > 0):
        # Fallback: rebuild from greedy
        fc = greedy_cover()
        fcov = build_cov(fc)
        if np.all(fcov > 0):
            return [idx_to_word(c) for c in fc]
        # Absolute fallback
        return [idx_to_word(i) for i in range(space_size)]

    return [idx_to_word(c) for c in best_code]
