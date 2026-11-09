"""Rewritten program using efficient Max-SAT / simulated annealing with bitmask representation."""
import random
import time

def construct():
    n = 180
    # Precompute forbidden triples
    # Condition: 1/a = 1/b + 1/c for distinct a,b,c in [1, n]
    # This implies a < b and a < c.
    # c = a*b / (b-a)
    
    raw_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            num = a * b
            den = b - a
            if num % den != 0:
                continue
            c = num // den
            if c > n or c <= a or c == b:
                continue
            raw_triples.append((a, b, c))
            
    # Deduplicate triples
    unique_triples = set()
    final_triples = []
    for tri in raw_triples:
        key = frozenset(tri)
        if key not in unique_triples:
            unique_triples.add(key)
            final_triples.append(tuple(key))
            
    triples = final_triples
    num_triples = len(triples)
    
    # Precompute bitmask for each triple
    triple_masks = []
    for tri in triples:
        mask = 0
        for e in tri:
            mask |= (1 << (e - 1))
        triple_masks.append(mask)
        
    # Precompute for each element, the list of (other_mask, triple_idx)
    # other_mask is the bitmask of the other two elements in the triple
    elem_other_masks = {i: [] for i in range(n + 1)}
    for idx, mask in enumerate(triple_masks):
        for e in triples[idx]:
            other_mask = mask ^ (1 << (e - 1))
            elem_other_masks[e].append((other_mask, idx))

    # Simulated Annealing
    # State: bitmask S
    # Energy: Penalty * Violations - Size
    # We want to minimize Energy.
    
    # Initialize with a greedy solution: [80, 180]
    # This range is likely conflict-free or low conflict.
    start_mask = 0
    for i in range(80, 181):
        start_mask |= (1 << (i - 1))
        
    current_mask = start_mask
    current_size = bin(current_mask).count('1')
    best_mask = current_mask
    best_size = current_size
    
    # Calculate initial violations
    violated_flags = [False] * num_triples
    current_violations = 0
    for idx, t_mask in enumerate(triple_masks):
        if (current_mask & t_mask) == t_mask:
            violated_flags[idx] = True
            current_violations += 1
            
    # SA Parameters
    T = 5.0
    T_min = 0.01
    cooling = 0.9995
    max_iters = 200000
    start_time = time.time()
    
    all_elements = list(range(1, n + 1))
    penalty = 1000
    
    for it in range(max_iters):
        if time.time() - start_time > 55:
            break
            
        # Choose random element to flip
        e = random.choice(all_elements)
        bit = 1 << (e - 1)
        
        if current_mask & bit:
            # Remove e
            delta_viol = 0
            for other_mask, idx in elem_other_masks[e]:
                if violated_flags[idx]:
                    # Removing e un-violates this triple
                    delta_viol -= 1
                    violated_flags[idx] = False
            delta_size = -1
            # Delta E = Penalty * delta_viol - delta_size = 1000 * (-1) - (-1) = -999
            # If no violations removed: delta_viol=0, delta_e = 1
            delta_e = penalty * delta_viol - delta_size
            
            if delta_e < 0 or random.random() < (2.718281828 ** (-delta_e / T)):
                current_mask ^= bit
                current_size += delta_size
                current_violations += delta_viol
        else:
            # Add e
            delta_viol = 0
            for other_mask, idx in elem_other_masks[e]:
                if not violated_flags[idx]:
                    # Check if other two are in current_mask
                    if (current_mask & other_mask) == other_mask:
                        # Adding e violates this triple
                        delta_viol += 1
                        violated_flags[idx] = True
            delta_size = 1
            # Delta E = Penalty * delta_viol - delta_size
            # If no new violations: delta_viol=0, delta_e = -1
            # If 1 new violation: delta_viol=1, delta_e = 999
            delta_e = penalty * delta_viol - delta_size
            
            if delta_e < 0 or random.random() < (2.718281828 ** (-delta_e / T)):
                current_mask |= bit
                current_size += delta_size
                current_violations += delta_viol

        # Update best
        if current_violations == 0 and current_size > best_size:
            best_size = current_size
            best_mask = current_mask

        T *= cooling
        if T < T_min:
            T = T_min

    # Result
    result = []
    for i in range(1, n + 1):
        if best_mask & (1 << (i - 1)):
            result.append(i)
            
    if not result:
        result = [180]
        
    return result
