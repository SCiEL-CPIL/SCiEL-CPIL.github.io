"""Strategy: High-Throughput Local Search via Vectorized Bitset Evaluation.
1. Represent the set S as a bitset (Python int or NumPy array).
2. Evaluate the score (number of distinct d) using vectorized bitwise operations.
   For a given set S (bitset), a difference d is valid if (S & (S >> d) & (S >> 2d)) != 0.
   We can iterate d from 1 to max_possible_diff and count how many yield a non-zero intersection.
   This is much faster than set lookups in Python.
3. Perform aggressive Hill Climbing with many random restarts.
   - Start with random or structured sets.
   - In each step, try swapping one element for another (or moving an element).
   - Accept if score improves.
   - Since evaluation is fast, we can run millions of steps.
4. Use a time guard to return the best found set.
"""

import time
import random

def _score_from_bitset(bitset, max_val=512):
    """
    Calculate the number of distinct d > 0 such that a, a+d, a+2d are in the set.
    bitset: Python int where bit i is 1 if (i+1) is in the set.
    max_val: The maximum value in the universe (512).
    """
    count = 0
    for d in range(1, max_val // 2 + 1):
        if (bitset & (bitset >> d) & (bitset >> (2 * d))) != 0:
            count += 1
    return count

def _score_diff_set(bitset, max_val=512):
    """Return the set of all valid differences."""
    diffs = set()
    for d in range(1, max_val // 2 + 1):
        if (bitset & (bitset >> d) & (bitset >> (2 * d))) != 0:
            diffs.add(d)
    return diffs

def _bitset_to_list(bitset):
    """Convert bitset to list of integers (1-indexed)."""
    res = []
    b = bitset
    idx = 0
    while b:
        if b & 1:
            res.append(idx + 1)
        b >>= 1
        idx += 1
    return res

def _list_to_bitset(lst):
    """Convert list of 1-indexed integers to bitset."""
    b = 0
    for x in lst:
        b |= (1 << (x - 1))
    return b

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n_target = 32
    max_val = 512
    
    best_score = -1
    best_bitset = 0
    
    def random_bitset(n):
        indices = random.sample(range(max_val), n)
        b = 0
        for i in indices:
            b |= (1 << i)
        return b

    def interval_bitset(start, n):
        b = 0
        for i in range(start, start + n):
            if i < max_val:
                b |= (1 << i)
        return b

    # Generate diverse initial candidates
    candidates = []
    
    # Dense intervals
    for start in range(0, max_val - n_target + 1, 8):
        candidates.append(interval_bitset(start, n_target))
    
    # Random sets
    for _ in range(80):
        candidates.append(random_bitset(n_target))
    
    # Modular structures
    for step in [2, 3, 4, 5, 6, 7, 8]:
        for _ in range(5):
            base = random.randint(0, max_val)
            b = 0
            count = 0
            for i in range(max_val):
                if (i - base) % step == 0:
                    b |= (1 << i)
                    count += 1
                    if count == n_target:
                        break
            if count == n_target:
                candidates.append(b)
    
    # Greedy construction: build set by adding elements that maximize marginal gain
    def greedy_construct():
        """Build a set greedily by adding elements that maximize the number of new progression differences."""
        current_set = set()
        current_diffs = set()
        
        # Start with a random seed element
        seed = random.randint(1, max_val)
        current_set.add(seed)
        
        for _ in range(n_target - 1):
            best_elem = None
            best_gain = -1
            best_new_diffs = None
            
            # Try all possible elements
            for elem in range(1, max_val + 1):
                if elem in current_set:
                    continue
                
                new_set = current_set | {elem}
                new_diffs = set()
                
                # Check which differences become valid
                for d in range(1, max_val // 2 + 1):
                    # Check if there exists a such that a, a+d, a+2d are all in new_set
                    # and at least one of them is elem
                    found = False
                    # Case 1: elem is the first term
                    if (elem + d) in new_set and (elem + 2*d) in new_set:
                        found = True
                    # Case 2: elem is the middle term
                    if (elem - d) in new_set and (elem + d) in new_set:
                        found = True
                    # Case 3: elem is the last term
                    if (elem - d) in new_set and (elem - 2*d) in new_set:
                        found = True
                    
                    if found:
                        new_diffs.add(d)
                
                gain = len(new_diffs - current_diffs)
                if gain > best_gain:
                    best_gain = gain
                    best_elem = elem
                    best_new_diffs = new_diffs
            
            if best_elem is not None and best_gain > 0:
                current_set.add(best_elem)
                current_diffs = best_new_diffs
            else:
                # No gain, pick random
                available = [x for x in range(1, max_val + 1) if x not in current_set]
                if available:
                    current_set.add(random.choice(available))
        
        return _list_to_bitset(list(current_set))
    
    # Add greedy constructions
    for _ in range(30):
        candidates.append(greedy_construct())
    
    # Simulated annealing / hill climbing with occasional restarts
    for start_b in candidates:
        if time.time() > deadline - 5.0:
            break
        
        current_b = start_b
        current_score = _score_from_bitset(current_b)
        
        if current_score > best_score:
            best_score = current_score
            best_bitset = current_b
        
        # Simulated annealing phase
        temperature = 5.0
        cooling_rate = 0.995
        min_temp = 0.1
        step_count = 0
        max_steps = 5000
        
        while step_count < max_steps and temperature > min_temp:
            if time.time() > deadline - 2.0:
                break
            
            indices = _bitset_to_list(current_b)
            
            # Pick one to remove
            rem_idx = random.choice(indices)
            
            # Pick one to add
            add_idx = random.randint(1, max_val)
            if add_idx in indices:
                for _ in range(20):
                    add_idx = random.randint(1, max_val)
                    if add_idx not in indices:
                        break
                else:
                    continue
            
            # Perform swap
            new_b = current_b
            new_b &= ~(1 << (rem_idx - 1))
            new_b |= (1 << (add_idx - 1))
            
            new_score = _score_from_bitset(new_b)
            delta = new_score - current_score
            
            # Accept if improvement, or with probability exp(delta/temperature) if worse
            if delta > 0:
                current_b = new_b
                current_score = new_score
                step_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_bitset = current_b
            else:
                # Accept worse move with probability exp(delta/temperature)
                if random.random() < (2.718281828 ** (delta / temperature)):
                    current_b = new_b
                    current_score = new_score
                step_count += 1
            
            temperature *= cooling_rate
        
        # Quick hill climb after SA
        no_improve = 0
        max_no_improve = 1000
        while no_improve < max_no_improve:
            if time.time() > deadline - 2.0:
                break
            
            indices = _bitset_to_list(current_b)
            rem_idx = random.choice(indices)
            add_idx = random.randint(1, max_val)
            if add_idx in indices:
                continue
            
            new_b = current_b
            new_b &= ~(1 << (rem_idx - 1))
            new_b |= (1 << (add_idx - 1))
            
            new_score = _score_from_bitset(new_b)
            
            if new_score > current_score:
                current_b = new_b
                current_score = new_score
                no_improve = 0
                if current_score > best_score:
                    best_score = current_score
                    best_bitset = current_b
            else:
                no_improve += 1
    
    # Final polish on best solution
    if best_bitset and time.time() < deadline - 10.0:
        current_b = best_bitset
        current_score = _score_from_bitset(current_b)
        no_improve = 0
        max_no_improve = 10000
        
        while no_improve < max_no_improve and time.time() < deadline - 2.0:
            indices = _bitset_to_list(current_b)
            
            # Try multiple random swaps per iteration
            improved = False
            for _ in range(5):
                rem_idx = random.choice(indices)
                add_idx = random.randint(1, max_val)
                if add_idx in indices:
                    continue
                
                new_b = current_b
                new_b &= ~(1 << (rem_idx - 1))
                new_b |= (1 << (add_idx - 1))
                
                new_score = _score_from_bitset(new_b)
                
                if new_score > current_score:
                    current_b = new_b
                    current_score = new_score
                    no_improve = 0
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_bitset = current_b
                    break
            
            if not improved:
                no_improve += 1
    
    if best_bitset == 0:
        return list(range(1, 33))
        
    return _bitset_to_list(best_bitset)

# EVOLVE-BLOCK-END
