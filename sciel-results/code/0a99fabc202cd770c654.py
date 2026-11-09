"""
Optimized solver for the 4-term progression coloring problem.
Uses vectorized precomputation and fast local search (Simulated Annealing) 
to minimize the number of colors used.
"""
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # 1. Precompute all 4-term arithmetic progressions
    # Indices are 0 to 127
    progressions = []
    for i in range(128):
        for d in range(1, (128 - i) // 3 + 1):
            j = i + d
            k = i + 2 * d
            l = i + 3 * d
            if l < 128:
                progressions.append((i, j, k, l))
    
    n_progressions = len(progressions)
    
    # Map position to list of progression indices
    pos_to_progs = [[] for _ in range(128)]
    for idx, (i, j, k, l) in enumerate(progressions):
        pos_to_progs[i].append(idx)
        pos_to_progs[j].append(idx)
        pos_to_progs[k].append(idx)
        pos_to_progs[l].append(idx)

    def count_violations(colors):
        """Count progressions with < 3 distinct colors."""
        count = 0
        for i, j, k, l in progressions:
            if len({colors[i], colors[j], colors[k], colors[l]}) < 3:
                count += 1
        return count

    def is_valid(colors):
        return count_violations(colors) == 0

    def search_k_colors(k, time_budget):
        """
        Try to find a valid coloring using exactly k colors (0..k-1).
        Returns a list of colors if found, else None.
        """
        start_search = time.time()
        max_restarts = 50
        T_initial = 1.0
        T_final = 0.01
        cooling_rate = 0.95
        
        for restart in range(max_restarts):
            if time.time() > start_search + time_budget:
                break
            
            # Initialize random coloring
            colors = [random.randint(0, k - 1) for _ in range(128)]
            
            # Initialize violation state
            # viol[i] is True if progression i is violated
            viol = [False] * n_progressions
            total_viol = 0
            for idx, (i, j, k_idx, l) in enumerate(progressions):
                if len({colors[i], colors[j], colors[k_idx], colors[l]}) < 3:
                    viol[idx] = True
                    total_viol += 1
            
            if total_viol == 0:
                return colors
            
            T = T_initial
            steps = 0
            max_steps = 50000 # Max steps per restart
            
            while total_viol > 0 and steps < max_steps:
                if time.time() > start_search + time_budget:
                    break
                
                # Pick a random position
                pos = random.randint(0, 127)
                old_color = colors[pos]
                
                # Pick a new color
                new_color = random.randint(0, k - 1)
                if new_color == old_color:
                    continue
                
                # Calculate delta in violations
                delta = 0
                affected_progs = pos_to_progs[pos]
                
                # Save old colors for rollback if needed? 
                # We just compute delta. If accepted, we commit.
                
                # We need to check the status of each affected progression.
                # A progression is violated if it has < 3 distinct colors.
                
                for p_idx in affected_progs:
                    i, j, k_idx, l = progressions[p_idx]
                    
                    # Get colors of the 4 positions
                    c1 = colors[i]
                    c2 = colors[j]
                    c3 = colors[k_idx]
                    c4 = colors[l]
                    
                    # If pos is one of them, replace with new_color to check new state
                    # But we haven't updated colors[pos] yet.
                    
                    # Let's create a temporary set of colors for this progression
                    # with the new color at 'pos'.
                    
                    # Optimization: 
                    # Old status is known: viol[p_idx]
                    # New status:
                    new_c1 = new_color if i == pos else c1
                    new_c2 = new_color if j == pos else c2
                    new_c3 = new_color if k_idx == pos else c3
                    new_c4 = new_color if l == pos else c4
                    
                    new_violated = len({new_c1, new_c2, new_c3, new_c4}) < 3
                    
                    if viol[p_idx] and not new_violated:
                        delta -= 1
                    elif not viol[p_idx] and new_violated:
                        delta += 1
                
                # Simulated Annealing acceptance
                if delta <= 0:
                    accept = True
                else:
                    # Accept worse move with probability exp(-delta / T)
                    if T > 1e-9:
                        prob = math.exp(-delta / T)
                        accept = random.random() < prob
                    else:
                        accept = False
                
                if accept:
                    # Commit the move
                    colors[pos] = new_color
                    # Update violation states
                    for p_idx in affected_progs:
                        i, j, k_idx, l = progressions[p_idx]
                        new_c1 = new_color if i == pos else colors[i]
                        new_c2 = new_color if j == pos else colors[j]
                        new_c3 = new_color if k_idx == pos else colors[k_idx]
                        new_c4 = new_color if l == pos else colors[l]
                        
                        was_violated = viol[p_idx]
                        is_violated = len({new_c1, new_c2, new_c3, new_c4}) < 3
                        
                        if was_violated and not is_violated:
                            total_viol -= 1
                        elif not was_violated and is_violated:
                            total_viol += 1
                        viol[p_idx] = is_violated
                    
                    steps += 1
                    T *= cooling_rate
                else:
                    # Don't commit, but still cool down slightly? 
                    # Standard SA cools regardless of acceptance.
                    T *= cooling_rate
                    steps += 1
            
            if total_viol == 0:
                return colors
        
        return None

    # Try to find the minimum number of colors
    # Start with k=3, then 4, etc.
    # We allocate more time to smaller k as they are harder to find but better.
    
    # Heuristic time allocation:
    # k=3: 30s
    # k=4: 20s
    # k=5: 15s
    # k=6: 10s
    # ...
    # Total ~ 80s
    
    time_budgets = {
        3: 30.0,
        4: 20.0,
        5: 15.0,
        6: 10.0,
        7: 5.0,
        8: 5.0
    }
    
    best_colors = None
    best_count = 128 # Fallback
    
    for k in range(3, 9):
        if time.time() > deadline - 2.0:
            break
        
        budget = time_budgets.get(k, 2.0)
        # Ensure we don't exceed deadline
        remaining = deadline - time.time() - 2.0
        if budget > remaining:
            budget = remaining
        
        if budget < 0.5:
            break
            
        result = search_k_colors(k, budget)
        
        if result is not None:
            # Verify
            if is_valid(result):
                # Count distinct colors actually used
                # The search used colors in 0..k-1, but maybe not all are used?
                # The objective is "number of labels actually used".
                # So we should remap to 0..m-1 where m is the count.
                used_colors = sorted(list(set(result)))
                mapping = {c: i for i, c in enumerate(used_colors)}
                optimized_result = [mapping[c] for c in result]
                
                m = len(used_colors)
                if m < best_count:
                    best_count = m
                    best_colors = optimized_result
                    # If we found 3, it's likely optimal. 
                    # But we should check if we can do better? No, 3 is min possible for non-constant 4-APs?
                    # Actually, 2 colors is impossible for large N. 3 is the target.
                    # If we find 3, we can stop? 
                    # Yes, 3 is the theoretical minimum for this type of problem (van der Waerden related).
                    if m == 3:
                        return optimized_result

    # If no small k worked, try a greedy fallback to ensure validity
    # Greedy with many colors is always valid.
    if best_colors is None:
        # Simple greedy: assign smallest color that doesn't create a violation
        # This will use many colors but is guaranteed to be valid.
        colors = []
        for x in range(128):
            assigned = False
            for c in range(128):
                # Check if adding c at x creates a violation
                # Check all progressions involving x
                valid = True
                for p_idx in pos_to_progs[x]:
                    i, j, k_idx, l = progressions[p_idx]
                    # Build the set of colors for this progression
                    # Note: positions < x are already colored, x is c, positions > x are not colored yet?
                    # Wait, the greedy approach must ensure that *all* progressions are valid.
                    # If we only look at progressions involving x, we might miss progressions 
                    # that involve future positions.
                    # However, if we assign colors sequentially, we can only check progressions 
                    # where x is the LAST element? No, that's not sufficient.
                    # A valid coloring requires ALL progressions to be valid.
                    # A simple greedy "smallest color that doesn't violate any *completed* progression" 
                    # doesn't work because future progressions might be violated.
                    
                    # Correct Greedy:
                    # This is a constraint satisfaction problem.
                    # Backtracking is too slow.
                    # Randomized local search with k=128 (or large k) will converge to 0 violations quickly.
                    pass
                
                # Let's just use the local search with a large k (e.g., 10) as a fallback
                # It should find a valid coloring quickly.
                pass
            
            # Fallback: Use the result from a quick search with k=10
            pass
        
        # If the loop above didn't find anything, run a quick search with k=10
        result = search_k_colors(10, 5.0)
        if result is not None and is_valid(result):
            used_colors = sorted(list(set(result)))
            mapping = {c: i for i, c in enumerate(used_colors)}
            best_colors = [mapping[c] for c in result]
        else:
            # Last resort: identity coloring (128 colors) is always valid
            best_colors = list(range(128))

    return best_colors
