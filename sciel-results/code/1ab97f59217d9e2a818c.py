"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit (150s kill)

    # Helper functions
    def calc_cost_full(perm):
        return int(np.sum(a * b[np.ix_(perm, perm)]))

    def two_opt_delta(p, i, j):
        """Compute delta cost for swapping facilities i and j in permutation p."""
        li = p[i]
        lj = p[j]
        if li == lj:
            return 0
        d = b[lj, p] - b[li, p]
        d[i] = 0
        d[j] = 0
        a_diff = a[i, :] - a[j, :] + a[:, i] - a[:, j]
        a_diff[i] = 0
        a_diff[j] = 0
        return int(np.dot(a_diff, d))

    def linearized_cost_matrix(p):
        """
        Compute the linearized cost matrix for Hungarian algorithm.
        cost[i,l] = sum_j a[i,j] * b[l, p[j]] + sum_j a[j,i] * b[p[j], l]
        """
        C1 = b[:, p]  # C1[l, j] = b[l, p[j]]
        C2 = b[p, :]  # C2[j, l] = b[p[j], l]
        term1 = a @ C1.T  # term1[i,l] = sum_j a[i,j] * b[l, p[j]]
        term2 = a.T @ C2  # term2[i,l] = sum_j a[j,i] * b[p[j], l]
        return term1 + term2

    def two_opt_polish(p, max_time):
        """Greedy 2-opt local search. Returns (p, cost)."""
        cost = calc_cost_full(p)
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for i in range(n):
                if time.time() > max_time:
                    break
                for j in range(i + 1, n):
                    if time.time() > max_time:
                        break
                    delta = two_opt_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        break
                if improved:
                    break
        return p, cost

    # --- Phase 1: Initial Solution via Linear Assignment on Row Sums ---
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    cost_matrix = np.outer(a_row_sums, b_row_sums)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    p = p.copy()

    # Initial polish
    p, current_cost = two_opt_polish(p, max_time=start_time + 15.0)
    best_p = p.copy()
    best_cost = current_cost

    # Check symmetry
    a_sym = np.array_equal(a, a.T)
    b_sym = np.array_equal(b, b.T)

    if a_sym and b_sym:
        # --- Phase 2: Tabu Search ---
        tabu_end_time = start_time + time_limit * 0.6 if 'time_limit' in locals() else start_time + 69.0
        # Actually, let's just use a fixed fraction of the remaining time
        # Total time is 115s. Phase 1 took ~15s. Remaining ~100s.
        # Let's allocate 60s to Tabu, 40s to Hungarian/Polish loop.
        tabu_end_time = start_time + 75.0
        
        # Tabu Search Structures
        # Using a 2D array for tabu status is faster than dict
        tabu = np.zeros((n, n), dtype=np.int64)
        tabu_tenure = 50
        iteration = 0
        
        # Precompute diagonal for faster delta? 
        # The two_opt_delta is already quite fast.
        
        while time.time() < tabu_end_time:
            improved = False
            # Randomize start index to avoid bias
            start_i = random.randint(0, n - 1)
            
            for k in range(n):
                i = (start_i + k) % n
                if time.time() > tabu_end_time:
                    break
                u = p[i]
                
                # Randomize start j
                start_j = random.randint(0, n - 1)
                for m in range(n - 1):
                    j = (start_j + m) % n
                    if i == j:
                        continue
                    if time.time() > tabu_end_time:
                        break
                    
                    # Check tabu
                    is_tabu = tabu[i, j] > iteration or tabu[j, i] > iteration
                    
                    delta = two_opt_delta(p, i, j)
                    
                    # Aspiration: accept if it improves global best even if tabu
                    if delta < 0 or (is_tabu and current_cost + delta < best_cost):
                        # Perform swap
                        v = p[j]
                        p[i], p[j] = v, u
                        current_cost += delta
                        
                        # Update tabu
                        tabu[i, j] = iteration + tabu_tenure
                        tabu[j, i] = iteration + tabu_tenure
                        
                        # Update best
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_p = p.copy()
                        
                        improved = True
                        break # First improvement found, restart pass
                
                if improved:
                    break
                    
            iteration += 1
            
            if not improved:
                # Local optimum reached (no improving non-tabu moves)
                # Diversification: Random kick
                if time.time() > tabu_end_time - 5.0:
                    break
                    
                # Choose a random move, even if it increases cost
                i_kick = random.randint(0, n - 1)
                j_kick = random.randint(0, n - 1)
                if i_kick != j_kick:
                    delta = two_opt_delta(p, i_kick, j_kick)
                    v = p[j_kick]
                    p[i_kick], p[j_kick] = v, p[i_kick]
                    current_cost += delta
                    
                    # Update tabu
                    tabu[i_kick, j_kick] = iteration + tabu_tenure
                    tabu[j_kick, i_kick] = iteration + tabu_tenure
                    
                    # Update best if improved (unlikely)
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    # We don't clear tabu, let it expire

    # --- Phase 3: Hungarian Reassignment Loop with Polish ---
    # Use remaining time for global reassignment
    # This allows escaping local minima found by Tabu/2-opt
    
    hungarian_count = 0
    no_improve_rounds = 0
    max_no_improve = 5
    
    # Start from the best solution found so far
    p = best_p.copy()
    current_cost = best_cost
    
    while time.time() < deadline - 10.0:
        # Compute linearized cost and solve assignment
        cost_mat = linearized_cost_matrix(p)
        
        # Add small random perturbation to break ties and escape plateaus
        if hungarian_count > 0 and no_improve_rounds > 0:
            noise_scale = 1.0 + no_improve_rounds * 0.5
            cost_mat = cost_mat + np.random.randn(n, n) * noise_scale
        
        try:
            row_ind, col_ind = linear_sum_assignment(cost_mat)
        except Exception:
            break
            
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        
        new_cost = calc_cost_full(new_p)
        
        if new_cost < best_cost:
            p = new_p
            best_p = p.copy()
            best_cost = new_cost
            hungarian_count += 1
            no_improve_rounds = 0
            
            # Polish after a good Hungarian step
            # Allocate a small amount of time for polish, say 5s max per iteration
            polish_end = min(deadline - 5.0, time.time() + 5.0)
            p, p_cost = two_opt_polish(p, max_time=polish_end)
            if p_cost < best_cost:
                best_p = p.copy()
                best_cost = p_cost
        else:
            no_improve_rounds += 1
            
            # If stuck, try a perturbed restart from best
            if no_improve_rounds >= max_no_improve:
                p = best_p.copy()
                # Randomly swap a few pairs to escape local minimum
                num_swaps = random.randint(3, 10)
                for _ in range(num_swaps):
                    i_s = random.randint(0, n - 1)
                    j_s = random.randint(0, n - 1)
                    if i_s != j_s:
                        p[i_s], p[j_s] = p[j_s], p[i_s]
                
                # Polish the perturbed solution
                polish_end = min(deadline - 5.0, time.time() + 5.0)
                p, p_cost = two_opt_polish(p, max_time=polish_end)
                
                if p_cost < best_cost:
                    best_p = p.copy()
                    best_cost = p_cost
                    no_improve_rounds = 0
                else:
                    # Reset to best if perturbation didn't help
                    p = best_p.copy()
                    no_improve_rounds = 0 # Reset counter to try again

    # Final safety check: ensure best_p is a valid permutation
    if sorted(best_p.tolist()) != list(range(n)):
        valid_p = np.arange(n, dtype=np.int64)
        return valid_p.tolist()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
