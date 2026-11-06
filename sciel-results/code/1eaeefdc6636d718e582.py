"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    """
    Use simulated annealing to find a Sidon set containing 0 with all differences 1..30,
    minimizing the largest element.
    """
    random.seed(42)
    np.random.seed(42)
    
    N_TARGETS = 30
    MAX_ELEMENTS = 32
    
    def get_positive_diffs(elements):
        """Get all positive differences from a sorted list of distinct non-negative integers."""
        elems = sorted(set(elements))
        diffs = []
        for i in range(len(elems)):
            for j in range(i + 1, len(elems)):
                diffs.append(elems[j] - elems[i])
        return diffs
    
    def energy(elements, temperature=1.0):
        """
        Composite energy function:
        - Penalty for difference collisions (non-Sidon)
        - Penalty for uncovered targets in 1..30
        - Penalty for maximum element (to minimize diameter)
        - Penalty for having more than MAX_ELEMENTS elements
        """
        elems = sorted(set(int(e) for e in elements if e >= 0))
        
        # Penalty for too many elements
        size_penalty = 0
        if len(elems) > MAX_ELEMENTS:
            size_penalty = (len(elems) - MAX_ELEMENTS) * 10000
        
        # Must contain 0
        zero_penalty = 0 if 0 in elems else 50000
        
        diffs = get_positive_diffs(elems)
        
        # Penalty for duplicate differences (non-Sidon)
        if len(diffs) > 0:
            diff_counts = {}
            for d in diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
            collision_penalty = sum((c - 1) * 100 for c in diff_counts.values() if c > 1)
        else:
            collision_penalty = 0
        
        # Penalty for uncovered targets
        covered = set(diffs)
        missing = [d for d in range(1, N_TARGETS + 1) if d not in covered]
        missing_penalty = len(missing) * 500
        
        # Penalty for max element (scaled to encourage small diameters)
        max_elem = max(elems) if elems else 0
        max_penalty = max_elem * 0.1
        
        total = size_penalty + zero_penalty + collision_penalty + missing_penalty + max_penalty
        return total, max_elem, len(missing)
    
    def try_move(elements, move_type, temperature):
        """Try a random move and return new elements."""
        elems = list(elements)
        
        if move_type == 'add':
            # Add a new element
            if len(elems) >= MAX_ELEMENTS:
                return None
            # Choose a position near existing elements or a random small value
            if random.random() < 0.7:
                # Near an existing element
                base = random.choice(elems)
                offset = random.randint(1, 200)
                if random.random() < 0.5:
                    candidate = base + offset
                else:
                    candidate = base - offset
                if candidate < 0:
                    candidate = base + offset
            else:
                candidate = random.randint(1, 500)
            
            if candidate in elems:
                return None
            new_elems = elems + [candidate]
            return new_elems
        
        elif move_type == 'remove':
            # Remove an element (not 0)
            removable = [e for e in elems if e != 0]
            if not removable:
                return None
            to_remove = random.choice(removable)
            new_elems = [e for e in elems if e != to_remove]
            return new_elems
        
        elif move_type == 'shift':
            # Shift an element (not 0) by a small amount
            shiftable = [e for e in elems if e != 0]
            if not shiftable:
                return None
            idx = random.choice(range(len(shiftable)))
            old_val = shiftable[idx]
            shift = random.randint(-50, 50)
            if shift == 0:
                return None
            new_val = old_val + shift
            if new_val < 0:
                return None
            if new_val in elems:
                return None
            new_elems = elems.copy()
            new_elems[elems.index(old_val)] = new_val
            return new_elems
        
        elif move_type == 'replace':
            # Replace an element (not 0) with a new one
            replaceable = [e for e in elems if e != 0]
            if not replaceable:
                return None
            old_val = random.choice(replaceable)
            if random.random() < 0.7:
                base = random.choice(elems)
                offset = random.randint(1, 300)
                if random.random() < 0.5:
                    candidate = base + offset
                else:
                    candidate = base - offset
                if candidate < 0:
                    candidate = base + offset
            else:
                candidate = random.randint(1, 800)
            
            if candidate in elems:
                return None
            new_elems = [e if e != old_val else candidate for e in elems]
            return new_elems
        
        return None
    
    # Initialize with a reasonable starting set
    # Start with 0 and some spread-out elements
    current = [0]
    # Add elements to cover some initial differences
    for d in range(1, 31):
        if d not in get_positive_diffs(current):
            # Try to add d itself or adjust
            candidate = d
            while candidate in current:
                candidate += 1
            current.append(candidate)
            if len(current) > MAX_ELEMENTS:
                break
    
    # Ensure we don't exceed max elements in initialization
    if len(current) > MAX_ELEMENTS:
        current = current[:MAX_ELEMENTS]
    
    current_energy, current_max, current_missing = energy(current)
    
    best = list(current)
    best_energy, best_max, best_missing = current_energy, current_max, current_missing
    
    # Simulated annealing parameters
    T_start = 1000.0
    T_end = 0.1
    cooling_rate = 0.9995
    max_iterations = 50000
    
    T = T_start
    
    for iteration in range(max_iterations):
        if T < T_end:
            break
        
        # Choose a move type
        move_types = ['add', 'remove', 'shift', 'replace']
        # Weight moves based on current state
        if len(current) < 10:
            weights = [0.5, 0.05, 0.25, 0.2]
        elif len(current) > MAX_ELEMENTS:
            weights = [0.0, 0.5, 0.25, 0.25]
        else:
            weights = [0.2, 0.1, 0.35, 0.35]
        
        move_type = random.choices(move_types, weights=weights, k=1)[0]
        
        new_elems = try_move(current, move_type, T)
        if new_elems is None:
            continue
        
        new_energy, new_max, new_missing = energy(new_elems)
        
        delta_e = new_energy - current_energy
        
        # Accept if better or with probability exp(-delta_e / T)
        if delta_e < 0 or random.random() < np.exp(-delta_e / T):
            current = new_elems
            current_energy, current_max, current_missing = new_energy, new_max, new_missing
            
            # Update best if we have a valid solution
            if current_missing == 0 and len(current) <= MAX_ELEMENTS:
                if current_max < best_max:
                    best = list(current)
                    best_energy, best_max, best_missing = current_energy, current_max, current_missing
        
        T *= cooling_rate
    
    # If we found a valid solution, return it
    if best_missing == 0:
        return sorted(set(int(e) for e in best if e >= 0))
    
    # Fallback: use the greedy approach if SA didn't find a valid solution
    a = [0]
    def diffs(s):
        s = sorted(s)
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    
    for d in range(1, 31):
        if d in diffs(a):
            continue
        found = False
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    
    return a

# EVOLVE-BLOCK-END
