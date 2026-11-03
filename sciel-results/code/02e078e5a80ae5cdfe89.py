"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit

    # Check symmetry (QAPLIB instances are symmetric)
    a_sym = np.array_equal(a, a.T)
    b_sym = np.array_equal(b, b.T)
    
    if not (a_sym and b_sym):
        # Fallback for non-symmetric (unlikely for this task, but safe)
        # We'll just use a generic swap delta which is slower
        def calc_delta_generic(i, j, perm):
            u = perm[i]
            v = perm[j]
            # Cost change
            # This is O(n) but without symmetry shortcuts
            old_cost = 0
            new_cost = 0
            # We only need the difference, but for simplicity in fallback, 
            # we might just do a full recalc if n is small, but n=256 is too big for full recalc per swap.
            # However, the task guarantees QAPLIB tai256c which is symmetric.
            # We assume symmetry for performance.
            return 0 
            
    # Precompute symmetric parts for speed
    a_diag = np.diag(a)
    b_diag = np.diag(b)

    def calculate_cost(perm):
        return int(np.sum(a * b[np.ix_(perm, perm)]))

    def calc_delta(i, j, perm):
        u = perm[i]
        v = perm[j]
        
        # Delta = (A_ii - A_jj)(B_vv - B_uu) + 2 * sum_{k!=i,j} (A_ik - A_jk)(B_vk - B_uk)
        # Note: B_vk - B_uk is (B_v - B_u)[k]
        # A_ik - A_jk is (A_i - A_j)[k]
        
        diff_A = a[i, :] - a[j, :]
        diff_B = b[v, :] - b[u, :]
        
        # Dot product over all k
        total_dot = np.dot(diff_A, diff_B)
        
        # Subtract terms where k=i and k=j
        # Term k=i: diff_A[i] * diff_B[i]
        # Term k=j: diff_A[j] * diff_B[j]
        
        exclude_i = diff_A[i] * diff_B[i]
        exclude_j = diff_A[j] * diff_B[j]
        
        off_diag_sum = total_dot - exclude_i - exclude_j
        
        diag_part = (a_diag[i] - a_diag[j]) * (b_diag[v] - b_diag[u])
        
        delta = diag_part + 2 * off_diag_sum
        return int(delta)

    # 1. Initial Solution: Greedy Construction + 2-Opt Polish
    
    # Greedy Construction
    # p[i] is the location assigned to facility i
    # We assign facilities 0 to n-1.
    # Maintain list of available locations.
    perm = np.empty(n, dtype=np.int64)
    available = set(range(n))
    
    # To make it faster, we can pre-sort facilities by some heuristic? 
    # Or just do 0..n-1.
    # Incremental cost for adding facility i to location l:
    # Cost contribution of i with already assigned facilities j:
    # sum_{j assigned} A[i,j] * B[l, p[j]]
    # Plus A[i,i] * B[l,l]
    
    assigned = np.zeros(n, dtype=bool)
    
    for i in range(n):
        best_loc = -1
        best_cost = np.inf
        # Iterate through available locations
        # To speed up, we could limit candidates, but n=256 is small.
        for l in available:
            # Calculate incremental cost
            # sum_{j in assigned} A[i,j] * B[l, p[j]]
            # We can maintain a running sum vector S[l] = sum_{j assigned} A[i,j] * B[l, p[j]]?
            # No, A[i,j] depends on i.
            # So for each i, we compute cost for all l in O(n * n_assigned).
            # Total O(n^3).
            
            # Efficient calculation:
            # Let C_i[l] = A[i,i]*B[l,l] + sum_{j assigned} A[i,j]*B[l, p[j]]
            # We can compute the sum part as:
            # sum_j A[i,j] * B[l, p[j]]
            # Let v = B[:, p[assigned_indices]]? No, B is fixed matrix.
            # B[l, p[j]] is just B[l, :][p[j]]
            
            # Vectorized:
            # assigned_indices = np.where(assigned)[0]
            # costs = a[i, assigned_indices] * b[l, perm[assigned_indices]]
            # sum_costs = np.sum(costs)
            
            # This is O(n) per location, O(n^2) per facility.
            
            if i == 0:
                cost = a[0,0] * b[l,l]
            else:
                assigned_indices = np.where(assigned)[0]
                # a[i, assigned_indices] is shape (k,)
                # b[l, perm[assigned_indices]] is shape (k,)
                cost = a[i, i] * b[l, l] + np.dot(a[i, assigned_indices], b[l, perm[assigned_indices]])
            
            if cost < best_cost:
                best_cost = cost
                best_loc = l
        
        perm[i] = best_loc
        available.remove(best_loc)
        assigned[i] = True

    # 2-Opt Polish on initial solution
    current_p = perm.copy()
    current_cost = calculate_cost(current_p)
    best_p = current_p.copy()
    best_cost = current_cost
    
    # Simple 2-opt pass
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i+1, n):
                d = calc_delta(i, j, current_p)
                if d < 0:
                    current_p[i], current_p[j] = current_p[j], current_p[i]
                    current_cost += d
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = current_p.copy()
                    improved = True
    
    # 2. Tabu Search
    tabu_tenure = 50
    max_iterations = 5000
    iteration = 0
    no_improve_count = 0
    stagnation_limit = 100
    
    # Tabu list: store (i, j) and expiry iteration
    # Using a dictionary might be slow. 
    # Better: Two 2D arrays or a list of tuples?
    # Since n=256, n^2/2 = 32k pairs.
    # We can use a 2D array of int for tabu expiry.
    tabu_expiry = np.zeros((n, n), dtype=np.int32)
    
    while time.time() - start_time < time_limit - 5.0:
        best_delta_in_pass = 0
        best_move_i = -1
        best_move_j = -1
        
        # Randomize start index to avoid bias
        start_i = random.randint(0, n-1)
        
        for k in range(n):
            i = (start_i + k) % n
            u = current_p[i]
            
            # Check time
            if time.time() - start_time > time_limit - 5.0:
                break
                
            for j in range(i + 1, n):
                # Check tabu
                is_tabu = tabu_expiry[i, j] > iteration
                
                # Aspiration: if it improves best_cost, accept even if tabu
                # We need delta to check aspiration.
                d = calc_delta(i, j, current_p)
                
                # Acceptance criteria:
                # 1. Not tabu and d < best_delta_in_pass (which starts at 0, so d<0)
                # 2. Tabu but d + current_cost < best_cost (Aspiration)
                
                if not is_tabu:
                    if d < best_delta_in_pass:
                        best_delta_in_pass = d
                        best_move_i = i
                        best_move_j = j
                else:
                    if current_cost + d < best_cost:
                        best_delta_in_pass = d
                        best_move_i = i
                        best_move_j = j
                        
        # Apply best move found
        if best_move_i != -1:
            i, j = best_move_i, best_move_j
            current_p[i], current_p[j] = current_p[j], current_p[i]
            current_cost += best_delta_in_pass
            
            # Update tabu
            tabu_expiry[i, j] = iteration + tabu_tenure
            tabu_expiry[j, i] = iteration + tabu_tenure
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = current_p.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                
        else:
            # No improving move found. Local optimum or all moves tabu.
            # Diversification: Random Kick
            if time.time() - start_time > time_limit - 5.0:
                break
            
            # Perform a random kick of size 2-5 swaps
            kick_size = random.randint(2, 5)
            for _ in range(kick_size):
                i_kick = random.randint(0, n-1)
                j_kick = random.randint(0, n-1)
                if i_kick != j_kick:
                    d = calc_delta(i_kick, j_kick, current_p)
                    current_p[i_kick], current_p[j_kick] = current_p[j_kick], current_p[i_kick]
                    current_cost += d
                    tabu_expiry[i_kick, j_kick] = iteration + tabu_tenure
                    tabu_expiry[j_kick, i_kick] = iteration + tabu_tenure
            
            no_improve_count += 1

        iteration += 1
        
        # Restart if stagnating
        if no_improve_count > stagnation_limit:
            # Reset tabu list
            tabu_expiry = np.zeros((n, n), dtype=np.int32)
            no_improve_count = 0
            # Optionally, perturb best_p and restart from there?
            # Or just continue from current.
            # Let's try restarting from best_p with a small perturbation
            # But current_p is already perturbed.
            pass

    # 3. Final Polish (Steepest Descent) on best_p
    # Use remaining time to polish best_p
    if time.time() - start_time < time_limit - 1.0:
        current_p = best_p.copy()
        current_cost = best_cost
        improved = True
        while improved and time.time() - start_time < time_limit - 1.0:
            improved = False
            for i in range(n):
                if time.time() - start_time > time_limit - 1.0:
                    break
                for j in range(i+1, n):
                    d = calc_delta(i, j, current_p)
                    if d < 0:
                        current_p[i], current_p[j] = current_p[j], current_p[i]
                        current_cost += d
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_p = current_p.copy()
                        improved = True
                        # Continue searching for more improvements in this pass
                        # or break? Steepest descent usually takes the best move.
                        # But doing all negative moves is a "first descent" which is faster.
                        # Let's stick to first descent for speed in the polish phase.

    # Final safety check
    if sorted(best_p.tolist()) != list(range(n)):
        valid_p = np.arange(n, dtype=np.int64)
        return valid_p.tolist()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
