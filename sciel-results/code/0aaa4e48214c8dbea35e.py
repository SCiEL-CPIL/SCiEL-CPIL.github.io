"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    import sys
    deadline = time.time() + 55  # safety margin before 60s limit
    N = 128

    # Precompute all 4-term APs
    ap_list = []
    for d in range(1, N):
        for start in range(0, N - 3 * d):
            ap_list.append((start, start + d, start + 2 * d, start + 3 * d))
    
    # For fast conflict counting, map index to list of AP indices it belongs to
    pos_aps = [[] for _ in range(N)]
    for ai, ap in enumerate(ap_list):
        for idx in ap:
            pos_aps[idx].append(ai)

    def is_conflict(colors, ap):
        """Check if AP has < 3 distinct colors."""
        return len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}) < 3

    def min_conflicts_search(num_colors, max_restarts=1000, max_steps_per_restart=20000):
        """
        Min-Conflicts local search to find a valid coloring with 'num_colors' colors.
        Returns a valid coloring list or None.
        """
        rng = random.Random(42)
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Random initialization
            colors = [rng.randint(0, num_colors - 1) for _ in range(N)]
            current_conflicts = sum(1 for ap in ap_list if is_conflict(colors, ap))
            
            if current_conflicts == 0:
                return colors

            stagnation_counter = 0
            
            for step in range(max_steps_per_restart):
                if time.time() > deadline:
                    return None
                
                if current_conflicts == 0:
                    return colors

                # Pick a random position
                pos = rng.randint(0, N - 1)
                
                # Check if this position is part of any conflict
                # We need to find a conflicting AP involving pos
                # Optimization: If we pick a pos not in a conflict, we might increase conflicts.
                # Standard Min-Conflicts: Pick a variable in a conflict.
                # To find one efficiently, we can just check the APs for pos.
                
                in_conflict = False
                for ai in pos_aps[pos]:
                    if is_conflict(colors, ap_list[ai]):
                        in_conflict = True
                        break
                
                if not in_conflict:
                    # This position is not in a conflict.
                    # Skip or restart? Skipping leads to no progress if many non-conflicting pos.
                    # Let's just restart if we have many skips.
                    stagnation_counter += 1
                    if stagnation_counter > 50:
                        break
                    continue

                # Change color of pos to a random different color
                old_color = colors[pos]
                new_color = rng.randint(0, num_colors - 1)
                while new_color == old_color:
                    new_color = rng.randint(0, num_colors - 1)
                
                colors[pos] = new_color
                
                # Calculate delta in conflicts
                # Only APs containing pos can change status
                delta = 0
                for ai in pos_aps[pos]:
                    ap = ap_list[ai]
                    # Check status before (revert temporarily or store?)
                    # We already changed colors[pos]. 
                    # We need to know if it was a conflict BEFORE.
                    # We didn't store that. 
                    # Let's just recalculate conflict count for these APs.
                    # But we need the "before" state.
                    # Let's store the "before" conflict status for APs involving pos before changing.
                    
                    # This logic is flawed in the current loop. 
                    # Let's restructure: 
                    # 1. Identify APs involving pos.
                    # 2. For each AP, check if it's a conflict with OLD color.
                    # 3. Change color.
                    # 4. For each AP, check if it's a conflict with NEW color.
                    # 5. Delta = new_conflicts - old_conflicts.
                    
                    # Let's redo this part correctly.
                # Roll back to calculate delta properly
                colors[pos] = old_color
                old_conflict_count = 0
                for ai in pos_aps[pos]:
                    if is_conflict(colors, ap_list[ai]):
                        old_conflict_count += 1
                
                colors[pos] = new_color
                new_conflict_count = 0
                for ai in pos_aps[pos]:
                    if is_conflict(colors, ap_list[ai]):
                        new_conflict_count += 1
                
                delta = new_conflict_count - old_conflict_count
                current_conflicts += delta
                
                if delta <= 0:
                    # Accept move
                    stagnation_counter = 0
                else:
                    # Revert move
                    colors[pos] = old_color
                    current_conflicts -= delta
                    stagnation_counter += 1
                    if stagnation_counter > 100:
                        break # Restart

            # End of steps for this restart
            # If we didn't find solution, try next restart

        return None

    # Try 3 colors first
    result = min_conflicts_search(3, max_restarts=500, max_steps_per_restart=10000)
    if result is not None:
        return result

    # If 3 colors failed, try 4
    result = min_conflicts_search(4, max_restarts=100, max_steps_per_restart=10000)
    if result is not None:
        return result

    # Fallback: Return a valid coloring with more colors if needed, or just all distinct
    # All distinct is always valid.
    return list(range(N))
# EVOLVE-BLOCK-END
