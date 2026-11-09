import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    N = 512
    start_time = time.time()
    time_limit = 55.0  # Leave 5s margin for evaluator overhead

    # Precompute all 4-APs in [1, N]
    # A 4-AP is defined by (a, d) where a + 3d <= N
    # We store them as indices in a 0-based array of size N
    # indices: i, i+d, i+2d, i+3d
    
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            # a, a+d, a+2d, a+3d
            # Convert to 0-based indices for numpy operations
            i0 = a - 1
            i1 = i0 + d
            i2 = i0 + 2 * d
            i3 = i0 + 3 * d
            aps.append((i0, i1, i2, i3))
    
    # Convert to numpy array for vectorized violation counting
    # Shape: (num_aps, 4)
    aps_arr = np.array(aps, dtype=np.int32)
    
    # To efficiently count violations for a given set S (represented as a boolean array S_bool),
    # we can check S_bool[i0] & S_bool[i1] & S_bool[i2] & S_bool[i3]
    # This is a boolean AND operation across the 4 columns.
    
    def count_violations(S_bool):
        """
        S_bool: numpy array of shape (N,), dtype bool
        Returns: number of 4-APs fully contained in S
        """
        if len(aps_arr) == 0:
            return 0
        # Extract rows corresponding to the APs
        # S_bool[aps_arr] gives shape (num_aps, 4)
        # We want to know if all 4 are True
        # np.all(..., axis=1)
        return np.sum(np.all(S_bool[aps_arr], axis=1))

    def get_violating_aps(S_bool):
        """
        Returns list of indices of APs that are violated (all 4 elements in S)
        """
        if len(aps_arr) == 0:
            return []
        return np.where(np.all(S_bool[aps_arr], axis=1))[0]

    # Initial Construction: Greedy to get a decent starting point
    # This is faster than pure random start and provides a high-quality seed for local search
    def greedy_init():
        S = set()
        # Order by value, or random? Random order often yields better diversity for local search start
        # But let's stick to a deterministic greedy first to ensure a valid baseline
        for x in range(1, N + 1):
            # Check if adding x creates a 4-AP
            # We can use the vectorized check on the current set
            S_bool = np.zeros(N, dtype=bool)
            for s in S:
                S_bool[s-1] = True
            
            # Temporarily add x
            idx = x - 1
            S_bool[idx] = True
            
            if count_violations(S_bool) == 0:
                S.add(x)
            else:
                S_bool[idx] = False
        return S

    # WalkSAT / Local Search
    # State: S_bool (numpy bool array), S_set (python set for easy manipulation)
    # Objective: Maximize |S| subject to count_violations(S_bool) == 0
    
    best_S = set()
    best_size = 0
    
    # Run multiple iterations/restarts to escape local optima
    # Each iteration:
    # 1. Start from a seed (greedy or perturbed previous best)
    # 2. Local search loop:
    #    a. If violations > 0, pick a violated AP and flip one of its elements (prefer removal)
    #    b. If violations == 0, try to add a random element not in S. If it causes violation, 
    #       either reject or accept with small probability (simulated annealing style) or just reject.
    #       To maximize size, we primarily want to add. If we can't add, we might need to remove 
    #       something to make space. This is the "swap" logic.
    
    # For simplicity and speed, we will use a "Min-Conflicts" variant:
    # - If violations > 0: Flip a variable in a random violated clause. 
    #   Bias: If the variable is currently 1 (in S), flip to 0 (remove). If 0, flip to 1 (add).
    #   However, adding usually increases violations. Removing decreases.
    #   So if violated, we MUST remove at least one element from the AP.
    #   We pick a random element from the violated AP that is currently in S, and remove it.
    # - If violations == 0: We are in a valid state. We want to increase size.
    #   Pick a random element x not in S. Try to add it.
    #   If adding x keeps violations == 0, keep it.
    #   If adding x creates violations, we have a choice:
    #      1. Reject (stay put).
    #      2. Accept the violation and let the "fix" phase handle it (this allows moving through invalid states to find larger valid sets).
    
    # Let's implement the "accept violation" strategy (WalkSAT style).
    # This allows the search to traverse the space of sets, not just the valid sub-space.
    
    def local_search(S_init, time_budget):
        S = set(S_init)
        S_bool = np.zeros(N, dtype=bool)
        for s in S:
            S_bool[s-1] = True
        
        current_violations = count_violations(S_bool)
        
        # Track best valid solution found during this search
        best_local_S = set()
        best_local_size = 0
        if current_violations == 0:
            best_local_S = set(S)
            best_local_size = len(S)
        
        # Parameters for WalkSAT
        # When in violation, we pick a clause (AP) and a variable to flip.
        # Standard WalkSAT picks a random variable in the clause.
        # We want to minimize violations. Flipping a 1->0 reduces violations for that AP.
        # Flipping a 0->1 increases violations.
        # So, if we are in violation, we should prefer 1->0 flips.
        
        # When valid, we want to increase size. We try 0->1 flips.
        
        while time.time() - start_time < time_budget:
            if current_violations > 0:
                # Pick a random violated AP
                violated_indices = get_violating_aps(S_bool)
                if len(violated_indices) == 0:
                    current_violations = 0
                    continue
                
                v_idx = random.choice(violated_indices)
                ap = aps_arr[v_idx]
                
                # Find elements in this AP that are currently in S (True)
                # We must flip at least one of them to 0 to fix this specific AP.
                # It might fix others too.
                candidates_to_remove = [idx for idx in ap if S_bool[idx]]
                
                if not candidates_to_remove:
                    # This shouldn't happen if logic is correct, but safety check
                    # If no element is True, the AP isn't violated. 
                    # Re-calculate violations? 
                    current_violations = count_violations(S_bool)
                    continue
                
                # Pick one to remove
                # Strategy: Remove the one that is part of the most other violated APs? 
                # Or just random? Random is simpler and often sufficient for small N.
                # Let's try random first.
                idx_to_flip = random.choice(candidates_to_remove)
                
                # Flip it
                S_bool[idx_to_flip] = False
                S.remove(idx_to_flip + 1)
                
                # Update violations
                # Instead of recounting all, we can update incrementally, but N=512 and ~10k APs is small.
                # Full recount is O(num_aps * 4) ~ 40k ops. Very fast in numpy.
                current_violations = count_violations(S_bool)
                
            else:
                # Valid state. Try to add an element.
                # Pick a random element not in S
                # To be efficient, maintain a list of candidates? 
                # Or just pick random x in [1, N] and check if in S.
                # Since S is likely dense, random picking is okay.
                
                x = random.randint(1, N)
                if x in S:
                    # Already in S. Try a different one? Or just loop?
                    # Let's try a few times to find a candidate
                    found_candidate = False
                    for _ in range(10):
                        x = random.randint(1, N)
                        if x not in S:
                            found_candidate = True
                            break
                    if not found_candidate:
                        # S is full? Unlikely.
                        # If S is full and valid, we are at N (impossible for 4-AP free).
                        # So there must be candidates.
                        continue
                
                idx = x - 1
                # Try adding x
                S_bool[idx] = True
                new_violations = count_violations(S_bool)
                
                if new_violations == 0:
                    # Success! We increased size.
                    S.add(x)
                    current_violations = 0
                    # Update best
                    if len(S) > best_local_size:
                        best_local_size = len(S)
                        best_local_S = set(S)
                else:
                    # Failure. We have a choice:
                    # 1. Revert (stay valid, no growth)
                    # 2. Keep the violation (move to invalid state, might lead to larger valid set later)
                    
                    # WalkSAT heuristic: With some probability, accept the bad move.
                    # Probability can depend on the increase in violations.
                    # Here, let's use a simple probability.
                    # If we always revert, we are stuck in local optima of the valid subgraph.
                    # Accepting violations allows "tunneling".
                    
                    # Let's accept with probability 0.5 if it doesn't increase violations too much?
                    # Or just fixed probability.
                    if random.random() < 0.3:
                        # Accept the violation
                        S.add(x)
                        current_violations = new_violations
                    else:
                        # Revert
                        S_bool[idx] = False
                        current_violations = 0
        
        return best_local_S

    # Main Loop: Multiple restarts with perturbation
    # Start with greedy
    S_greedy = greedy_init()
    best_S = set(S_greedy)
    best_size = len(S_greedy)
    
    # Update best if greedy is good
    # It is valid by construction
    
    # Now do local search from this seed
    # We will run local search for a certain time, then perturb and restart.
    
    num_restarts = 5
    time_per_restart = (time_limit - (time.time() - start_time)) / num_restarts
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Prepare seed
        if r == 0:
            seed_S = set(S_greedy)
        else:
            # Perturb the best solution found so far
            # Remove k random elements and add k random others?
            # Or just start from best_S and do local search.
            # To ensure diversity, let's remove a few random elements from best_S
            seed_S = set(best_S)
            # Remove 5% of elements
            num_to_remove = max(1, int(0.05 * len(seed_S)))
            to_remove = random.sample(list(seed_S), min(num_to_remove, len(seed_S)))
            for x in to_remove:
                seed_S.remove(x)
        
        # Run local search
        # Give it a time slice
        t_start = time.time()
        t_end = t_start + time_per_restart
        # Ensure we don't exceed global limit
        if t_end > start_time + time_limit:
            t_end = start_time + time_limit
            
        result_S = local_search(seed_S, t_end)
        
        if len(result_S) > best_size:
            best_size = len(result_S)
            best_S = set(result_S)
            
    return sorted(best_S)

# EVOLVE-BLOCK-END
