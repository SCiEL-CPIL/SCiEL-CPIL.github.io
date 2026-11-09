import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    fixed = [1, 2, 5, 11]
    best_set = set(fixed)
    best_energy = 0  # 0 conflicts is optimal
    best_size = len(best_set)
    
    # Helper to compute sums and energy (conflicts)
    def compute_energy(s):
        from collections import Counter
        lst = list(s)
        sums = Counter()
        for i in range(len(lst)):
            for j in range(i, len(lst)):
                sums[lst[i] + lst[j]] += 1
        # Energy is sum of max(0, count - 1)
        energy = 0
        for count in sums.values():
            if count > 1:
                energy += count - 1
        return energy, sums

    # Initial setup
    current_set = set(fixed)
    current_energy, current_sums = compute_energy(current_set)
    
    # SA Parameters
    T = 5.0
    cooling = 0.999
    min_T = 0.1
    
    # We want to maximize size, but first ensure validity (energy=0).
    # Objective: Minimize Energy. If Energy is 0, we try to maximize size.
    # Actually, we can just minimize Energy, and if we hit 0, we keep adding if possible.
    # Or, we can minimize a composite cost: Energy * 1000 - Size.
    # Let's stick to minimizing Energy. If Energy is 0, we prefer larger sets.
    
    while time.time() < deadline:
        if T < min_T:
            T = 5.0 # Restart with higher temp if stuck? Or just break?
            # Let's restart with a slightly perturbed state if we hit min T
            # For now, just break and restart the loop with a new greedy init if needed?
            # The current structure is a single SA run. Let's add restarts.
            pass
        
        # Move selection
        r = random.random()
        
        if r < 0.4:
            # Add a random element
            candidates = [x for x in range(1, 285) if x not in current_set]
            if not candidates:
                continue
            x = random.choice(candidates)
            
            # Calculate delta energy for adding x
            # New sums: x + y for y in current_set, and 2*x
            # Check collisions with current_sums
            delta_energy = 0
            new_sums_to_add = []
            for y in current_set:
                s_val = x + y
                if current_sums[s_val] > 0:
                    delta_energy += 1
                new_sums_to_add.append(s_val)
            s_val = 2 * x
            if current_sums[s_val] > 0:
                delta_energy += 1
            new_sums_to_add.append(s_val)
            
            new_energy = current_energy + delta_energy
            # We accept if energy doesn't increase, or with prob exp(-dE/T)
            # But we also want to grow the set.
            # If delta_energy == 0, we accept (grow set).
            # If delta_energy > 0, we might accept if T is high enough.
            
            if delta_energy <= 0 or random.random() < math.exp(-delta_energy / T):
                # Apply move
                current_set.add(x)
                for s_val in new_sums_to_add:
                    current_sums[s_val] += 1
                current_energy = new_energy
                # Update best if energy is 0 and size is larger
                if current_energy == 0 and len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                    if best_size >= 20:
                        return list(best_set)
                        
        elif r < 0.7:
            # Remove a random non-fixed element
            non_fixed = [x for x in current_set if x not in fixed]
            if not non_fixed:
                continue
            x = random.choice(non_fixed)
            
            # Calculate delta energy for removing x
            # Sums to remove: x + y for y in current_set \ {x}, and 2*x
            # Note: x+y where y=x is 2x.
            # We iterate y in current_set. If y==x, sum is 2x.
            # We need to decrement counts.
            # Delta energy:
            # For each sum s_val involving x:
            #   old_count = current_sums[s_val]
            #   new_count = old_count - 1
            #   if old_count >= 2, delta_energy -= 1 (energy decreases)
            
            delta_energy = 0
            sums_to_remove = []
            for y in current_set:
                s_val = x + y
                # We only process each pair once? 
                # The sums are unique per pair (x,y).
                # But we are iterating all y in current_set.
                # Pairs are (x, y) for all y in current_set.
                # This covers all sums involving x.
                if current_sums[s_val] >= 2:
                    delta_energy -= 1
                sums_to_remove.append(s_val)
            
            new_energy = current_energy + delta_energy
            # We generally accept removals if they reduce energy.
            # If they increase energy (removing a "good" element that was causing no conflicts? 
            # Wait, removing an element *cannot* increase energy.
            # Energy is sum of (count-1). Removing an element reduces counts.
            # So delta_energy <= 0 always.
            # So we always accept removals?
            # No, we might want to keep the set large.
            # But if energy > 0, we MUST reduce it.
            # If energy == 0, removing an element keeps energy 0 but reduces size.
            # So if energy == 0, we should NOT remove unless we are trying to swap.
            
            if current_energy > 0:
                # Always accept removal if it helps (which it does)
                current_set.remove(x)
                for s_val in sums_to_remove:
                    current_sums[s_val] -= 1
                current_energy = new_energy
            else:
                # If energy is 0, only remove if we plan to add later?
                # Or accept with low probability to allow restructuring?
                # Let's accept with prob exp(-1/T) to allow fluctuations
                if random.random() < math.exp(-1.0 / T):
                    current_set.remove(x)
                    for s_val in sums_to_remove:
                        current_sums[s_val] -= 1
                    current_energy = new_energy

        else:
            # Swap: Remove x, Add y
            non_fixed = [x for x in current_set if x not in fixed]
            if not non_fixed:
                continue
            x = random.choice(non_fixed)
            candidates = [z for z in range(1, 285) if z not in current_set]
            if not candidates:
                continue
            y = random.choice(candidates)
            
            # 1. Remove x
            delta_energy_rem = 0
            sums_rem = []
            for w in current_set:
                s_val = x + w
                if current_sums[s_val] >= 2:
                    delta_energy_rem -= 1
                sums_rem.append(s_val)
            
            # 2. Add y (to current_set \ {x})
            # We need to check collisions against (current_sums - counts from x)
            # This is tricky to do in O(1) without applying the removal first.
            # Let's apply removal temporarily, calculate add delta, then apply add.
            
            # Apply Removal
            current_set.remove(x)
            for s_val in sums_rem:
                current_sums[s_val] -= 1
            temp_energy = current_energy + delta_energy_rem
            
            # Calculate Add Delta for y
            delta_energy_add = 0
            sums_add = []
            for w in current_set:
                s_val = y + w
                if current_sums[s_val] > 0:
                    delta_energy_add += 1
                sums_add.append(s_val)
            s_val = 2 * y
            if current_sums[s_val] > 0:
                delta_energy_add += 1
            sums_add.append(s_val)
            
            new_energy = temp_energy + delta_energy_add
            delta_E = new_energy - current_energy
            
            if delta_E <= 0 or random.random() < math.exp(-delta_E / T):
                # Accept Swap
                for s_val in sums_add:
                    current_sums[s_val] += 1
                current_set.add(y)
                current_energy = new_energy
                if current_energy == 0 and len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                    if best_size >= 20:
                        return list(best_set)
            else:
                # Reject Swap: Revert Removal
                current_set.add(x)
                for s_val in sums_rem:
                    current_sums[s_val] += 1
                # Energy reverts to current_energy
                # current_energy is unchanged in this branch

        # Cooling
        T *= cooling
        
        # Check time
        if time.time() > deadline:
            break

    # Final Polish: Try to add more elements if energy is 0
    if current_energy == 0:
        cands = [x for x in range(1, 285) if x not in current_set]
        random.shuffle(cands)
        for x in cands:
            if time.time() > deadline:
                break
            # Check if x can be added without conflicts
            valid = True
            for y in current_set:
                if current_sums[x + y] > 0:
                    valid = False
                    break
            if valid and current_sums[2*x] > 0:
                valid = False
            if valid:
                current_set.add(x)
                for y in current_set:
                    current_sums[x + y] += 1
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)

    return list(best_set)
