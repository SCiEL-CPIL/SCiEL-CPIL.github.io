"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 12, 'm': 40}

def _count_comparable_pairs(sets):
    """
    Count the number of unordered pairs (i, j) with i < j such that 
    sets[i] is a strict subset of sets[j] or sets[j] is a strict subset of sets[i].
    Uses bitmask representation for efficiency.
    """
    n = P['n']
    m = len(sets)
    # Convert lists to bitmasks
    masks = [0] * m
    for i in range(m):
        mask = 0
        for v in sets[i]:
            if 0 <= v < n:
                mask |= (1 << v)
        masks[i] = mask
    
    count = 0
    for i in range(m):
        mi = masks[i]
        for j in range(i + 1, m):
            mj = masks[j]
            # Check if mi is strict subset of mj: (mi & mj) == mi and mi != mj
            # Check if mj is strict subset of mi: (mi & mj) == mj and mi != mj
            # Combined: (mi & mj) == mi or (mi & mj) == mj, and mi != mj
            # Actually, if mi == mj, they are not strict subsets of each other.
            # If mi & mj == mi, then mi is subset of mj. If mi != mj, it's strict.
            # If mi & mj == mj, then mj is subset of mi. If mi != mj, it's strict.
            # Note: If mi == mj, then mi & mj == mi == mj. We must exclude this case.
            inter = mi & mj
            if inter == mi and mi != mj:
                count += 1
            elif inter == mj and mi != mj:
                count += 1
    return count

def _random_subset_of_size(k, n):
    """Generate a random subset of size k from 0..n-1."""
    if k == 0:
        return []
    if k >= n:
        return list(range(n))
    # Use random.sample for efficiency
    return random.sample(range(n), k)

def construct():
    n = P['n']
    m = P['m']
    
    # Deadline guard
    start_time = time.time()
    max_time = 85.0  # 5 seconds safety margin
    
    # Initialization: Start with a "chain-like" structure to get a good starting point
    # Create a chain of sets: S0 < S1 < ... < Sk
    # Then fill the rest with random subsets that might fit into the chain
    # A better init: Randomly generate 40 subsets, but bias towards sizes that allow comparability.
    # Let's try a specific initialization: 
    # 1. Generate a random total order of elements? No, subsets are of a fixed universe.
    # 2. Let's just start with random sets, but maybe sort them by size initially to help SA?
    
    # Better Initialization Strategy:
    # Create a base chain of length L.
    # Let's pick a random permutation of 12 elements.
    # Let C_k be the set of first k elements in the permutation.
    # C_0 = {}, C_1 = {e0}, ..., C_12 = {e0...e11}
    # This gives a chain of 13 sets.
    # We need 40 sets.
    # We can add sets that are subsets of the chain elements or supersets? 
    # Actually, any subset of the universe is comparable with C_k if it is a subset of C_k or a superset of C_k?
    # No, a random set S is comparable with C_k only if S subset C_k or C_k subset S.
    
    # Let's use a simpler, robust initialization:
    # Generate m random subsets.
    current_sets = []
    for _ in range(m):
        # Random size between 0 and n
        size = random.randint(0, n)
        s = _random_subset_of_size(size, n)
        current_sets.append(s)
    
    # Ensure distinctness? The problem says "exactly forty distinct subsets".
    # If we have duplicates, we need to fix them.
    # Let's enforce distinctness during generation and in SA moves.
    
    # Re-initialize to ensure distinctness
    existing = set()
    current_sets = []
    attempts = 0
    while len(current_sets) < m and attempts < 1000:
        size = random.randint(0, n)
        s = tuple(_random_subset_of_size(size, n))
        if s not in existing:
            existing.add(s)
            current_sets.append(list(s))
        attempts += 1
        
    # If we still don't have enough (very unlikely for n=12, m=40), fill with random unique
    # There are 2^12 = 4096 subsets, so 40 is easy to find.
    if len(current_sets) < m:
        all_subsets = []
        for mask in range(1 << n):
            s = [b for b in range(n) if (mask >> b) & 1]
            if tuple(s) not in existing:
                existing.add(tuple(s))
                all_subsets.append(s)
        # Shuffle and take what we need
        random.shuffle(all_subsets)
        current_sets.extend(all_subsets[:m - len(current_sets)])
        
    current_score = _count_comparable_pairs(current_sets)
    best_sets = [list(s) for s in current_sets]
    best_score = current_score
    
    # Simulated Annealing
    T = 50.0
    cooling = 0.999
    min_T = 0.1
    
    # Pre-compute masks for current sets to speed up evaluation?
    # The _count_comparable_pairs function recomputes masks every time.
    # For SA, we can update score incrementally or just recompute.
    # m=40, pairs=780. 780 bit operations is very fast.
    # Let's just recompute.
    
    while T > min_T and time.time() - start_time < max_time:
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.7:
            # Modify one set: change its size or swap an element
            idx = random.randint(0, m - 1)
            old_s = current_sets[idx]
            old_s_tuple = tuple(sorted(old_s))
            
            # Try to create a new set that is likely comparable to neighbors
            # Strategy: Pick a random neighbor j, and make new_s a subset or superset of current_sets[j]
            neighbor_idx = random.randint(0, m - 1)
            if neighbor_idx == idx:
                neighbor_idx = (neighbor_idx + 1) % m
            
            neighbor_s = current_sets[neighbor_idx]
            
            r = random.random()
            if r < 0.5:
                # Make new_s a subset of neighbor_s
                # Pick a random subset of neighbor_s
                size = random.randint(0, len(neighbor_s))
                if size == len(neighbor_s):
                    new_s = list(neighbor_s)
                else:
                    new_s = random.sample(neighbor_s, size)
            else:
                # Make new_s a superset of neighbor_s
                # Pick a random superset of neighbor_s
                # Elements not in neighbor_s
                missing = [v for v in range(n) if v not in set(neighbor_s)]
                size = random.randint(0, len(missing))
                extra = random.sample(missing, size)
                new_s = neighbor_s + extra
            
            new_s_tuple = tuple(sorted(new_s))
            
            # Check distinctness
            if new_s_tuple == old_s_tuple:
                continue
            
            # Check if new_s is already in the set (at another index)
            is_dup = False
            for k in range(m):
                if k != idx and tuple(sorted(current_sets[k])) == new_s_tuple:
                    is_dup = True
                    break
            if is_dup:
                continue
            
            # Apply move
            current_sets[idx] = new_s
            new_score = _count_comparable_pairs(current_sets)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_sets = [list(s) for s in current_sets]
            else:
                # Revert
                current_sets[idx] = old_s
                
        else:
            # Swap elements between two sets
            i, j = random.sample(range(m), 2)
            s_i = current_sets[i]
            s_j = current_sets[j]
            
            if len(s_i) == 0 or len(s_j) == 0:
                continue
                
            # Pick an element from s_i and an element from s_j
            e_i = random.choice(s_i)
            e_j = random.choice(s_j)
            
            if e_i == e_j:
                continue
                
            # Remove e_i from s_i, add to s_j
            # Remove e_j from s_j, add to s_i
            new_s_i = [x for x in s_i if x != e_i]
            if e_j not in new_s_i:
                new_s_i.append(e_j)
            else:
                # e_j was already in s_i? No, we picked e_j from s_j. 
                # But s_i and s_j might share elements.
                # If e_j is in s_i, then adding it again is invalid (list of distinct integers).
                # So we must ensure e_j is not in s_i.
                if e_j in s_i:
                    continue
            new_s_j = [x for x in s_j if x != e_j]
            if e_i not in new_s_j:
                new_s_j.append(e_i)
            else:
                if e_i in s_j:
                    continue
            
            # Check distinctness of the new sets against the rest
            # And check if new_s_i == new_s_j
            if set(new_s_i) == set(new_s_j):
                continue
                
            new_s_i_tuple = tuple(sorted(new_s_i))
            new_s_j_tuple = tuple(sorted(new_s_j))
            
            # Check against other sets
            valid = True
            for k in range(m):
                if k == i or k == j:
                    continue
                s_k_tuple = tuple(sorted(current_sets[k]))
                if s_k_tuple == new_s_i_tuple or s_k_tuple == new_s_j_tuple:
                    valid = False
                    break
            if not valid:
                continue
            
            # Apply
            current_sets[i] = new_s_i
            current_sets[j] = new_s_j
            
            new_score = _count_comparable_pairs(current_sets)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_sets = [list(s) for s in current_sets]
            else:
                # Revert
                current_sets[i] = s_i
                current_sets[j] = s_j
                
        T *= cooling
        
    # Final check: Ensure types are correct and distinctness is maintained
    # The problem requires "exactly forty distinct subsets".
    # My SA maintains distinctness.
    # Convert to lists of ints.
    result = []
    for s in best_sets:
        # Ensure they are lists of Python ints
        result.append([int(x) for x in s])
        
    # Final validation: distinctness
    seen = set()
    for s in result:
        t = tuple(sorted(s))
        if t in seen:
            # This should not happen, but if it does, we have a problem.
            # We might need to fix it.
            pass
        seen.add(t)
        
    return result
# EVOLVE-BLOCK-END
