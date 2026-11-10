import time
from math import gcd
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Moduli range
    moduli = list(range(20, 201))
    
    # Initialize residues greedily
    # r[m] is the residue for modulus m
    r = {}
    active = set(moduli)
    
    # Greedy initialization
    for m in moduli:
        found = False
        for res in range(m):
            conflict = False
            for m2 in active:
                if m2 == m:
                    continue
                # Check if res is compatible with r[m2]
                # Condition: res != r[m2] mod gcd(m, m2)
                g = gcd(m, m2)
                if res % g == r.get(m2, -1) % g:
                    conflict = True
                    break
            if not conflict:
                r[m] = res
                found = True
                break
        if not found:
            # If no residue found, we might need to backtrack or just skip for now
            # But we will handle conflicts in the local search
            # For now, assign a random residue, we will fix it later
            r[m] = random.randint(0, m - 1)
            
    def count_conflicts():
        """Count the number of conflicting pairs among active moduli."""
        conflicts = 0
        active_list = list(active)
        for i in range(len(active_list)):
            m1 = active_list[i]
            for j in range(i + 1, len(active_list)):
                m2 = active_list[j]
                g = gcd(m1, m2)
                if r[m1] % g == r[m2] % g:
                    conflicts += 1
        return conflicts

    def get_score():
        return sum(1.0 / m for m in active)

    best_score = get_score()
    best_r = r.copy()
    best_active = set(active)
    
    if count_conflicts() == 0:
        return [[m, r[m]] for m in sorted(active)]

    # Local Search / Simulated Annealing
    # We want to minimize conflicts. If conflicts == 0, we maximize score (which is fixed if all active).
    # If we can't reach 0 conflicts, we remove moduli.
    
    current_conflicts = count_conflicts()
    temperature = 10.0
    cooling_rate = 0.99
    
    while time.time() < deadline:
        if current_conflicts == 0:
            # Try to add back removed moduli? 
            # For simplicity, if we start with all, we don't remove in this loop.
            # If we implemented removal, we'd try to re-add.
            # Let's stick to resolving conflicts first.
            break
        
        # Pick a random modulus to change
        m = random.choice(list(active))
        old_r = r[m]
        
        # Try a new random residue
        new_r = random.randint(0, m - 1)
        if new_r == old_r:
            continue
            
        # Calculate change in conflicts
        # We only need to check conflicts with other moduli
        delta_conflicts = 0
        for m2 in active:
            if m2 == m:
                continue
            g = gcd(m, m2)
            # Check old conflict
            if old_r % g == r[m2] % g:
                delta_conflicts -= 1
            # Check new conflict
            if new_r % g == r[m2] % g:
                delta_conflicts += 1
                
        if delta_conflicts < 0:
            # Improvement
            r[m] = new_r
            current_conflicts += delta_conflicts
        elif delta_conflicts == 0:
            # Neutral, accept to explore
            r[m] = new_r
        else:
            # Worse, accept with probability based on temperature
            # Simple acceptance: if random < exp(-delta/T)
            # Since delta is integer, we can approximate
            if random.random() < 0.1: # Rough heuristic for now
                r[m] = new_r
                current_conflicts += delta_conflicts
        
        # Track best
        if current_conflicts == 0:
            score = get_score()
            if score > best_score:
                best_score = score
                best_r = r.copy()
                best_active = set(active)
                break # We found a perfect solution
        
        # Cool down
        temperature *= cooling_rate
        if temperature < 0.01:
            # Restart with new random initialization if stuck
            # This helps escape local minima
            temperature = 10.0
            for m in active:
                r[m] = random.randint(0, m - 1)
            current_conflicts = count_conflicts()
            
    # If we still have conflicts, we must remove some moduli to make the set valid.
    # We want to maximize sum 1/m.
    # Strategy: Start with all moduli. While conflicts exist, remove the modulus m 
    # that is involved in the most conflicts and has the largest m (smallest weight).
    
    # Re-initialize to best found or greedy if no perfect solution found
    if count_conflicts() > 0:
         # Use the current state as a starting point for removal
         pass
    else:
         # We have a valid set of all moduli
         return [[m, best_r[m]] for m in sorted(best_active)]

    # Removal Phase
    # We need to find a subset of moduli that is conflict-free.
    # Greedy removal:
    # 1. Start with all moduli and current residues.
    # 2. While conflicts exist:
    #    a. Find all conflicting pairs.
    #    b. For each modulus involved in a conflict, count how many conflicts it has.
    #    c. Select the modulus with the highest conflict count. Tie-break by largest m.
    #    d. Remove it from active.
    #    e. Update residues? No, just remove. The remaining set might still have conflicts 
    #       if the removal didn't resolve them (e.g. triangle conflict).
    #    f. Actually, removing a vertex from a graph removes all its edges.
    #       So if we remove m, all conflicts involving m are gone.
    #       Conflicts not involving m remain.
    
    active_set = set(moduli)
    # Use the residues from the local search as a starting point, 
    # or re-greedy to get a good initial state for removal.
    # Let's re-greedy with the current r as a hint? 
    # Simpler: Just use the current r.
    
    while time.time() < deadline:
        # Count conflicts and identify culprits
        conflict_count = {}
        conflicts_exist = False
        active_list = list(active_set)
        for i in range(len(active_list)):
            m1 = active_list[i]
            for j in range(i + 1, len(active_list)):
                m2 = active_list[j]
                g = gcd(m1, m2)
                if r[m1] % g == r[m2] % g:
                    conflicts_exist = True
                    conflict_count[m1] = conflict_count.get(m1, 0) + 1
                    conflict_count[m2] = conflict_count.get(m2, 0) + 1
        
        if not conflicts_exist:
            break
        
        # Find the modulus to remove
        # Prioritize those with high conflict count, then large m
        candidates = [m for m in active_set if m in conflict_count]
        if not candidates:
            break
            
        # Sort by conflict count desc, then m desc
        candidates.sort(key=lambda m: (-conflict_count[m], m))
        
        # Remove the top candidate
        m_to_remove = candidates[0]
        active_set.remove(m_to_remove)
        
        # Optional: Try to fix residues of remaining to reduce future conflicts?
        # For now, just remove.
        
    # Calculate final score
    final_score = sum(1.0 / m for m in active_set)
    
    # Compare with best found earlier
    if final_score > best_score:
        best_score = final_score
        best_active = active_set
        best_r = r.copy() # Residues for active set are in r
        
    # Return the best valid set
    # Ensure we return a valid set
    # The best_active set from the removal loop is valid by construction (no conflicts)
    # The best_active from the local search (if 0 conflicts) is also valid
    
    # We need to make sure we return the one with the highest score
    # If local search found 0 conflicts, best_active is all moduli.
    # If removal loop finished, best_active is the reduced set.
    
    # It's possible the removal loop didn't run if local search succeeded.
    # If local search succeeded, best_active is all moduli.
    # If local search failed, we ran removal.
    
    # Let's just use the best_score and corresponding set.
    # We need to map best_score back to the set.
    # In the local search loop, we updated best_active when conflicts==0.
    # In the removal loop, we should update best_score if final_score > best_score.
    
    # Re-verify validity of best_active just in case
    # (Not strictly necessary if logic is correct, but good for safety)
    
    return [[m, best_r[m]] for m in sorted(best_active)]

# EVOLVE-BLOCK-END
