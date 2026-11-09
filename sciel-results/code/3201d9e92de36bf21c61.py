"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import random

def construct():
    start_time = time.time()
    deadline = start_time + 58  # Safety margin for 60s limit
    
    N = 256
    # Candidates are 1..256
    # We want a subset S of {1..256} such that gcd(a,b)=1 for all a!=b in S.
    # Maximize sum(S).
    
    # 1. Precompute conflict information
    # conflict[i] is a set of indices j such that gcd(i+1, j+1) > 1
    # Note: i, j are 0-based indices corresponding to values i+1, j+1.
    # Actually, it's easier to work with values directly or 1-based indices.
    # Let's use values 1..256 directly in sets for clarity, or 0-based for bitmasks?
    # Given N=256, a set of ints is fast enough for local search.
    
    # Let's use 1-based values (1..256) as the keys.
    # conflicts[v] = set of u in 1..256, u != v, such that gcd(u, v) > 1
    
    conflicts = [set() for _ in range(N + 1)]
    for v in range(1, N + 1):
        for u in range(1, N + 1):
            if u != v and gcd(u, v) > 1:
                conflicts[v].add(u)
    
    # 2. Greedy Initialization
    # Sort values by descending magnitude.
    # This biases the initial solution towards high sums.
    # We can also try multiple random orders to get diverse starting points.
    
    def greedy_init(order):
        selected = set()
        # For each value in the given order, add it if it doesn't conflict with existing selected
        for v in order:
            # Check if v conflicts with any in selected
            # Optimization: check intersection
            if not (selected & conflicts[v]):
                selected.add(v)
        return selected
    
    # Generate a few diverse initial solutions
    # 1. Descending order
    # 2. Random permutations (a few)
    
    best_set = set()
    best_sum = 0
    
    # Initial greedy with descending order
    init_order = list(range(N, 0, -1))
    s = greedy_init(init_order)
    current_sum = sum(s)
    if current_sum > best_sum:
        best_sum = current_sum
        best_set = set(s)
    
    # 3. Local Search (Hill Climbing)
    # Move: Add a number, Remove a number, or Swap a number.
    # To be efficient, we should only consider moves that are "potentially" improving.
    
    # Precompute list of all values for random selection
    all_values = list(range(1, N + 1))
    
    # To speed up conflict checking, we maintain the current set.
    # Checking if a new value v can be added: 
    #   v not in selected AND (selected & conflicts[v]) is empty
    
    # Let's run a local search loop until deadline or no improvement.
    
    # We will alternate between:
    # A. Try to add a random unselected number.
    # B. Try to swap a selected number with an unselected number.
    # C. Try to remove a selected number (only if it allows adding two or more others? 
    #    Simple removal is rarely beneficial unless it blocks many high-value additions.
    #    However, standard hill climbing usually considers single moves. 
    #    Let's stick to Add and Swap. Swap is equivalent to Remove+Add.
    
    # Actually, a "Swap" is: remove u from S, add v to S.
    # Condition: u in S, v not in S.
    # New S is valid if (S \ {u} & conflicts[v]) is empty.
    # Gain = v - u. We only want positive gain.
    
    # "Add" is: add v to S.
    # Condition: v not in S, (S & conflicts[v]) is empty.
    # Gain = v.
    
    # We can combine these.
    
    # To make search faster, we can limit the number of random restarts or iterations.
    
    # Let's do a loop:
    # 1. Pick a random unselected v. Try to add it.
    # 2. If that fails or isn't improving, pick a random selected u and random unselected v. Try swap.
    
    # Better: 
    # While time < deadline:
    #   - Pick a random move type (Add or Swap)
    #   - Execute it if valid and improving.
    #   - If no move found in a large batch, break or restart.
    
    # Since N=256 is small, we can actually check ALL possible adds and swaps in each step?
    # 256 adds + 256*256 swaps = ~65k checks.
    # Each check is a set intersection. 
    # 65k * (time for set intersection) might be too slow if done in Python loops naively.
    # Set intersection of size ~30 is very fast in C.
    # Let's try a "Best Improvement" strategy per step, or "First Improvement".
    # First Improvement is usually faster per step but might get stuck.
    # Best Improvement finds the best move in the neighborhood.
    
    # Given the deadline, let's try a hybrid:
    # 1. Greedy Init.
    # 2. Local Search with First Improvement.
    #    - Randomly pick a candidate move from a large sample.
    #    - Or, iterate through all possible adds/swaps but break early on first improvement.
    
    # Let's use a robust local search:
    # At each step, generate a random permutation of possible moves? No, too slow.
    # Just pick random candidates.
    
    # Strategy:
    # 1. Start with greedy set.
    # 2. Loop:
    #    a. Try to add a random unselected number.
    #    b. If fail, try to swap a random selected with a random unselected.
    #    c. If fail, try to remove a random selected number (to escape local minima? 
    #       Removal reduces sum, so only do it if it enables a large add later? 
    #       Standard hill climbing doesn't do this. 
    #       However, "2-opt" style moves (remove 1, add 2) are powerful.
    #       Let's implement "Add 1" and "Swap 1" first. If stuck, try "Remove 1, Add 2".
    
    # Let's implement a function to check validity.
    # is_valid_add(v, S): v not in S and not (S & conflicts[v])
    # is_valid_swap(u, v, S): u in S, v not in S, and not ((S - {u}) & conflicts[v])
    
    # Optimization for is_valid_swap:
    # (S - {u}) & conflicts[v] is empty
    # This is equivalent to: (S & conflicts[v]) is subset of {u}
    # i.e., S & conflicts[v] is either empty or contains only u.
    
    # Let's code the main loop.
    
    current_set = set(best_set)
    current_sum = best_sum
    
    # Precompute list of values for random picking
    # We need to pick from unselected.
    # Maintaining a list of unselected values dynamically is O(N) to update.
    # Since N=256, just filtering is fine.
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try to find an improving ADD
        # Sample a few random unselected numbers
        unselected = [v for v in all_values if v not in current_set]
        if unselected:
            # Try a random sample first to avoid full scan if possible?
            # With N=256, full scan is ~256 ops. Very fast.
            # Let's just scan all unselected for an add.
            for v in unselected:
                if not (current_set & conflicts[v]):
                    # Valid add
                    current_set.add(v)
                    current_sum += v
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_set = set(current_set)
                    improved = True
                    break # First improvement, restart loop to re-evaluate state
        
        # 2. If no add found, try SWAPS
        if not improved:
            selected_list = list(current_set)
            unselected_list = [v for v in all_values if v not in current_set]
            
            # To avoid O(N^2) in Python loops being too slow if we check all pairs,
            # we can sample. But 256*256 = 65k. 
            # 65k set intersections might take ~0.05s?
            # If we do this every 0.01s, we do ~5 full scans per 0.05s.
            # It's likely fast enough.
            
            # However, we can be smarter.
            # Only swap if v > u (since we want to increase sum).
            
            # Let's iterate u in selected, v in unselected.
            # If v <= u, skip (no gain).
            
            # Optimization: Sort unselected descending. 
            # If v <= u, break inner loop?
            
            # Let's just do a random sampling approach for swaps to keep it fast and stochastic.
            # Sample 100 random pairs.
            
            found_swap = False
            # Try a few random swaps
            for _ in range(100):
                if not selected_list or not unselected_list:
                    break
                u = random.choice(selected_list)
                v = random.choice(unselected_list)
                
                if v <= u:
                    continue # No gain
                
                # Check validity of swapping u with v
                # Remove u from current_set temporarily for check?
                # Condition: (current_set - {u}) & conflicts[v] is empty
                # Which is: (current_set & conflicts[v]) is empty OR (current_set & conflicts[v]) == {u}
                
                common = current_set & conflicts[v]
                if not common or common == {u}:
                    # Valid swap
                    current_set.remove(u)
                    current_set.add(v)
                    current_sum += (v - u)
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_set = set(current_set)
                    found_swap = True
                    break
            
            if found_swap:
                improved = True
        
        # 3. If no add or swap found, try a "2-1" move: Remove 1, Add 2.
        # This allows escaping local minima where a single block prevents a large add.
        # Example: S contains u. v1, v2 are high value.
        # v1 conflicts with u. v2 conflicts with u.
        # v1 and v2 are coprime.
        # Removing u allows adding v1 and v2.
        # Gain = v1 + v2 - u.
        
        if not improved:
            # This is more expensive.
            # Sample a random u from selected.
            # Find all v in unselected that conflict with u? No, v that DON'T conflict with S\{u}.
            # Actually, we need v such that (S\{u} & conflicts[v]) is empty.
            # Then we can add v.
            
            # Let's try:
            # Pick random u from S.
            # Compute S_minus_u = S - {u}
            # Find all v in unselected such that (S_minus_u & conflicts[v]) is empty.
            # Let's call this set V_addable.
            # If |V_addable| >= 2, pick two distinct v1, v2 from V_addable that are coprime.
            # If gain = v1 + v2 - u > 0, perform move.
            
            selected_list = list(current_set)
            if selected_list:
                u = random.choice(selected_list)
                S_minus_u = current_set - {u}
                
                # Find addable v's
                addable_vs = []
                for v in all_values:
                    if v in current_set:
                        continue
                    if not (S_minus_u & conflicts[v]):
                        addable_vs.append(v)
                
                if len(addable_vs) >= 2:
                    # We need two coprime numbers from addable_vs
                    # Sort by value desc to maximize gain
                    addable_vs.sort(reverse=True)
                    # Check top few for coprime pair
                    found_pair = False
                    for i in range(min(10, len(addable_vs))):
                        v1 = addable_vs[i]
                        for j in range(i+1, min(10, len(addable_vs))):
                            v2 = addable_vs[j]
                            if gcd(v1, v2) == 1:
                                gain = v1 + v2 - u
                                if gain > 0:
                                    # Perform move
                                    current_set.remove(u)
                                    current_set.add(v1)
                                    current_set.add(v2)
                                    current_sum += gain
                                    if current_sum > best_sum:
                                        best_sum = current_sum
                                        best_set = set(current_set)
                                    found_pair = True
                                    break
                        if found_pair:
                            break
                    if found_pair:
                        improved = True
        
        # If no improvement found in this round, we might be stuck in a local optimum.
        # To escape, we can do a random restart with a perturbed solution?
        # Or just keep trying random moves?
        # The "First Improvement" with random sampling in swaps already provides stochasticity.
        # If we truly get stuck (no valid add, no valid swap, no valid 2-1), 
        # the current set is a local optimum for these moves.
        # We can try a "kick": remove a random element and re-greedy fill?
        
        if not improved:
            # Stuck. Let's try a kick.
            # Remove a random element from current_set.
            # Then run greedy fill again on the remaining set?
            # Or just add random elements if possible?
            
            # Simpler: Just break and do a fresh restart with a different random seed/order?
            # Since we have time, let's do a few fresh greedy inits with random orders.
            
            # Generate a random permutation
            rand_order = all_values[:]
            random.shuffle(rand_order)
            
            # Greedy init on this order
            new_s = greedy_init(rand_order)
            new_sum = sum(new_s)
            
            if new_sum > best_sum:
                best_sum = new_sum
                best_set = set(new_s)
                current_set = set(new_s)
                current_sum = new_sum
            else:
                # Maybe current_set is better, but let's continue searching from current_set
                # The "not improved" flag will loop again.
                # To avoid infinite loop if truly stuck, we rely on the random sampling 
                # in the swap/2-1 steps to eventually find a move if one exists.
                # If no move exists, it's a true local optimum.
                # We can force a move by removing a random item (decreasing sum) to allow a big add?
                # Let's try a "forced" move: remove a random u, add a random v if valid.
                # Even if sum decreases, it might open up.
                
                # For now, just continue. The random sampling in the next iteration 
                # might pick a different u/v pair.
                # If we are truly stuck, we waste time.
                # Let's add a counter. If stuck for 1000 iterations, do a random restart.
                # But checking time is cheaper.
                pass

    # Final check
    result = list(best_set)
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
