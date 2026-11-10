"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 85.0
    n = 512
    
    # Pre-compute all 4-term arithmetic progressions (4-APs)
    # A 4-AP is (a, a+d, a+2d, a+3d)
    # We represent each 4-AP as a tuple of indices in [1, n]
    # We need to ensure no 4-AP is fully contained in the selected set.
    # This is equivalent to finding a Maximum Independent Set in a hypergraph,
    # or a Maximum 4-SAT instance where each clause is "not(all 4 in set)".
    # Alternatively, we can view it as a graph problem where vertices are numbers 1..n
    # and we want to maximize the number of vertices such that no hyperedge is fully covered.
    
    # For Simulated Annealing, it's easier to work with the complement:
    # Let x_i = 1 if i is in the set, 0 otherwise.
    # We want to maximize sum(x_i) subject to: for all 4-APs (a,b,c,d), x_a+x_b+x_c+x_d <= 3.
    
    # Let's generate the list of 4-APs
    ap_list = []
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            ap_list.append((a, b, c, e))
            
    num_aps = len(ap_list)
    
    # We will use Simulated Annealing on the binary vector x of length n.
    # Objective: Maximize sum(x)
    # Constraint: For each AP (a,b,c,d), not (x[a-1] and x[b-1] and x[c-1] and x[d-1])
    
    # To make SA efficient, we need a fast way to evaluate the penalty for violations.
    # However, checking all APs for every move is O(num_aps) which is ~10k.
    # n=512, num_aps is roughly 512 * 170 / 3 ~ 30k? 
    # Actually sum_{a=1}^{512} floor((512-a)/3) is about 512*170/3 ~ 29000.
    # 29000 checks per move is too slow for many iterations in 60s.
    
    # Better approach: Local update.
    # When we flip x_i, which APs are affected?
    # Only APs that contain i.
    # Each number i is in at most O(n) APs.
    # Specifically, i can be the 1st, 2nd, 3rd, or 4th term.
    # Let's precompute for each index i (0-based for numpy array, 1-based for logic),
    # the list of AP indices that contain it.
    
    # Index mapping: AP index k corresponds to tuple (a,b,c,d).
    # We store APs as 0-based indices for numpy access.
    ap_indices = []
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            ap_indices.append((a-1, b-1, c-1, e-1))
            
    num_aps = len(ap_indices)
    
    # Build incidence list: for each element idx (0..n-1), list of AP indices containing it.
    element_to_aps = [[] for _ in range(n)]
    for k, ap in enumerate(ap_indices):
        for idx in ap:
            element_to_aps[idx].append(k)
            
    # State: x (numpy bool array of length n)
    # Penalty: Number of violated APs. An AP is violated if all 4 elements are 1.
    # We want to maximize (count_ones - penalty * lambda) or just maximize count_ones subject to penalty=0.
    # Standard SA: Maximize Objective = count_ones.
    # Move: Flip a random bit.
    # Delta Objective = +1 if 0->1, -1 if 1->0.
    # But we must ensure the final result has 0 violations.
    # Strategy: Use a penalty function that heavily penalizes violations.
    # Score = count_ones - M * (number_of_violations).
    # If M is large enough, the algorithm will prefer feasible solutions.
    # M should be > n. Let M = 1000.
    
    # To update penalty efficiently:
    # When flipping x[i]:
    # For each AP in element_to_aps[i]:
    #   Check if the AP was violated before and after the flip.
    #   If it was violated (all 4 were 1) and now one is 0, penalty decreases by 1.
    #   If it was not violated, but now all 4 are 1, penalty increases by 1.
    #   Note: An AP can only become violated if we flip 0->1 and the other 3 are already 1.
    #   An AP can only cease to be violated if we flip 1->0 and the other 3 are still 1.
    
    # Let's maintain:
    # x: np.array of bools
    # violations: np.array of ints (0 or 1) for each AP? Or just count?
    # To check if an AP is violated, we need to know the state of the other 3.
    # If we maintain a count of how many 1s are in each AP, we can do this in O(1) per AP affected.
    
    ap_counts = np.zeros(num_aps, dtype=np.int8)
    
    def update_ap_counts(x, ap_counts, i, new_val):
        """
        Update ap_counts when x[i] changes to new_val (0 or 1).
        Returns delta_violations (change in number of violated APs).
        """
        delta_violations = 0
        for ap_idx in element_to_aps[i]:
            a, b, c, d = ap_indices[ap_idx]
            # Current count includes old x[i]
            old_count = ap_counts[ap_idx]
            
            # Calculate new count
            if new_val == 1:
                new_count = old_count + 1
            else:
                new_count = old_count - 1
            
            # Check violation status
            # Violated if count == 4
            was_violated = (old_count == 4)
            is_violated = (new_count == 4)
            
            if was_violated and not is_violated:
                delta_violations -= 1
            elif not was_violated and is_violated:
                delta_violations += 1
            
            ap_counts[ap_idx] = new_count
        return delta_violations

    def get_score(x, ap_counts):
        """
        Score = count_ones - M * violations
        """
        count_ones = np.sum(x)
        violations = np.sum(ap_counts == 4)
        M = 1000
        return count_ones - M * violations

    # Initial state: Random subset with density ~0.4
    # Or start with greedy? Greedy is fast and gives a good starting point.
    # Let's do a quick greedy first to get a feasible set, then SA to improve.
    
    # Greedy initialization
    x_greedy = np.zeros(n, dtype=bool)
    ap_counts_greedy = np.zeros(num_aps, dtype=np.int8)
    
    # Simple greedy: iterate 1..n, add if no conflict
    # This is O(n * avg_degree). avg_degree is small.
    for i in range(n):
        # Check if adding i creates a violation
        # We need to check all APs containing i.
        # If any AP containing i has the other 3 already set to 1, we cannot add i.
        can_add = True
        for ap_idx in element_to_aps[i]:
            a, b, c, d = ap_indices[ap_idx]
            # Count how many of the other 3 are already 1
            # We can use ap_counts, but ap_counts currently reflects x_greedy.
            # If we add i, the count for this AP increases by 1.
            # If current count is 3, adding i makes it 4 (violation).
            if ap_counts_greedy[ap_idx] == 3:
                can_add = False
                break
        
        if can_add:
            x_greedy[i] = True
            # Update ap_counts_greedy
            for ap_idx in element_to_aps[i]:
                ap_counts_greedy[ap_idx] += 1
                
    # Now start SA from this state
    x = x_greedy.copy()
    ap_counts = ap_counts_greedy.copy()
    
    current_score = get_score(x, ap_counts)
    best_score = current_score
    best_x = x.copy()
    
    # SA Parameters
    T = 50.0  # Initial temperature
    cooling_rate = 0.9999
    min_T = 1e-6
    M = 1000
    
    # We will try to maximize score.
    # Moves: Flip a random bit.
    # To be efficient, we can prioritize flipping bits that are in violated APs (to fix them)
    # or bits that are 0 but could be added without violation (to increase size).
    # Random flipping is simple but might be slow to find large improvements.
    # Let's stick to random flipping for robustness, but use a good schedule.
    
    start_time = time.time()
    
    # Pre-compute list of valid indices
    indices = np.arange(n)
    
    while time.time() < deadline - 2.0:
        # Pick a random index to flip
        i = random.randint(0, n - 1)
        
        old_val = x[i]
        new_val = not old_val
        
        # Calculate delta score
        delta_violations = update_ap_counts(x, ap_counts, i, int(new_val))
        
        # x[i] is not updated yet in the function call above? 
        # Wait, update_ap_counts updates ap_counts but does NOT update x[i].
        # We need to update x[i] to reflect the move for the next iteration's logic?
        # Actually, get_score uses x and ap_counts.
        # Let's manually calculate delta_score.
        # Score = sum(x) - M * violations
        # delta_sum_x = 1 if new_val==1 else -1
        # delta_violations is the change in violation count.
        
        if new_val:
            delta_sum_x = 1
        else:
            delta_sum_x = -1
            
        delta_score = delta_sum_x - M * delta_violations
        
        # Acceptance probability
        if delta_score > 0:
            accept = True
        else:
            # Accept with probability exp(delta_score / T)
            # If T is very small, this is 0.
            if T > 0:
                prob = np.exp(delta_score / T)
                accept = (random.random() < prob)
            else:
                accept = False
                
        if accept:
            x[i] = new_val
            current_score += delta_score
            if current_score > best_score:
                best_score = current_score
                best_x = x.copy()
        else:
            # Revert ap_counts
            update_ap_counts(x, ap_counts, i, int(old_val))
            
        # Cool down
        T *= cooling_rate
        if T < min_T:
            T = min_T

    # Ensure the best solution is valid (0 violations)
    # If best_x has violations, we need to fix them.
    # Since M=1000, the SA should strongly prefer valid solutions.
    # However, if it ends in an invalid state, we must repair it.
    
    # Repair: Remove elements from violated APs until no violations remain.
    # This is a local search / greedy repair.
    
    final_x = best_x.copy()
    final_ap_counts = np.zeros(num_aps, dtype=np.int8)
    for i in range(n):
        if final_x[i]:
            for ap_idx in element_to_aps[i]:
                final_ap_counts[ap_idx] += 1
                
    # Iterate until no violations
    changed = True
    while changed:
        changed = False
        # Find all violated APs
        violated_aps = np.where(final_ap_counts == 4)[0]
        if len(violated_aps) == 0:
            break
            
        # Pick a random violated AP and remove a random element from it
        ap_idx = random.choice(violated_aps)
        a, b, c, d = ap_indices[ap_idx]
        candidates = [a, b, c, d]
        # Prefer removing an element that is in many other violated APs?
        # For simplicity, just pick one randomly.
        # Or pick the one with the highest degree in the conflict graph?
        # Let's just pick randomly to avoid bias.
        idx_to_remove = random.choice(candidates)
        
        # Remove it
        final_x[idx_to_remove] = False
        for ap_idx_2 in element_to_aps[idx_to_remove]:
            final_ap_counts[ap_idx_2] -= 1
        changed = True
        
    # Convert to list of integers (1-based)
    result_indices = np.where(final_x)[0] + 1
    result_list = list(result_indices)
    
    # Final sanity check
    # If result is empty (shouldn't happen), return [1]
    if not result_list:
        return [1]
        
    return sorted(result_list)

# EVOLVE-BLOCK-END
