import time
import random
import numpy as np

# EVOLVE-BLOCK-START
_MAX = 512
_N = 32
_DEADLINE = 115.0  # Safety margin before 150s kill

def _score_bitmask(mask):
    """
    Count distinct d > 0 such that a, a+d, a+2d are in the set represented by mask.
    mask is a Python integer where bit i is 1 if i is in the set.
    """
    count = 0
    # Max d such that 2*d + 1 <= 512 is 255.
    # If a=1, a+2d <= 512 => 2d <= 511 => d <= 255.
    for d in range(1, 256):
        if (mask & (mask << d) & (mask << (2 * d))) != 0:
            count += 1
    return count

def _greedy_init(rng):
    """
    Build a set greedily by adding the element that maximizes the score gain.
    Returns a set to maintain type consistency for SA.
    """
    current_elements = set()
    current_mask = 0
    current_score = 0
    
    for _ in range(_N):
        best_gain = -1
        best_elem = None
        for x in range(1, _MAX + 1):
            if x in current_elements:
                continue
            new_mask = current_mask | (1 << x)
            new_score = _score_bitmask(new_mask)
            gain = new_score - current_score
            if gain > best_gain or (gain == best_gain and rng.random() < 0.5):
                best_gain = gain
                best_elem = x
        
        if best_elem is not None:
            current_elements.add(best_elem)
            current_mask |= (1 << best_elem)
            current_score += best_gain
            
    return current_elements

def _local_search(elements, rng, time_limit=2.0):
    """
    Perform local search by trying all single swaps.
    Elements can be a list or set.
    """
    current_elements = set(elements)
    current_mask = 0
    for x in current_elements:
        current_mask |= (1 << x)
    current_score = _score_bitmask(current_mask)
    
    start_time = time.time()
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for rem in list(current_elements):
            for add in range(1, _MAX + 1):
                if add in current_elements:
                    continue
                new_mask = (current_mask ^ (1 << rem)) | (1 << add)
                new_score = _score_bitmask(new_mask)
                if new_score > current_score:
                    current_elements.remove(rem)
                    current_elements.add(add)
                    current_mask = new_mask
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
                
    return sorted(current_elements)

def _local_search_2swap_sampled(elements, rng, time_limit=1.0):
    """
    Perform local search with sampled 2-swaps.
    """
    current_elements = set(elements)
    current_mask = 0
    for x in current_elements:
        current_mask |= (1 << x)
    current_score = _score_bitmask(current_mask)
    
    start_time = time.time()
    all_elements = list(range(1, _MAX + 1))
    max_iters = 2000
    
    for _ in range(max_iters):
        if time.time() - start_time >= time_limit:
            break
            
        # Sample 2 to remove
        rem1, rem2 = rng.sample(list(current_elements), 2)
        
        # Sample 2 to add
        while True:
            add1, add2 = rng.sample(all_elements, 2)
            if add1 not in current_elements and add2 not in current_elements:
                break
                
        new_mask = current_mask
        new_mask ^= (1 << rem1)
        new_mask ^= (1 << rem2)
        new_mask |= (1 << add1)
        new_mask |= (1 << add2)
        
        new_score = _score_bitmask(new_mask)
        if new_score > current_score:
            current_elements.remove(rem1)
            current_elements.remove(rem2)
            current_elements.add(add1)
            current_elements.add(add2)
            current_mask = new_mask
            current_score = new_score

    return sorted(current_elements)

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    best_elements = None
    best_score = -1
    
    # Run multiple restarts with more time per restart
    num_restarts = 6
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        rng = random.Random(42 + restart)
        
        # 1. Greedy Initialization (returns a set now)
        current_elements = _greedy_init(rng)
        current_mask = 0
        for x in current_elements:
            current_mask |= (1 << x)
        current_score = _score_bitmask(current_mask)
        
        if current_score > best_score:
            best_score = current_score
            best_elements = list(current_elements)
            
        # 2. Simulated Annealing
        time_per_restart = (_DEADLINE - (time.time() - start_time)) / (num_restarts - restart)
        sa_start = time.time()
        sa_deadline = sa_start + min(12.0, time_per_restart * 0.75)
        
        T = 1.0
        cooling = 0.999
        min_T = 0.001
        
        # Precompute list of all possible elements for fast sampling
        all_elements = list(range(1, _MAX + 1))
        
        while time.time() < sa_deadline:
            if T < min_T:
                T = min_T
                
            # Decide move type: 1-swap or 2-swap
            if rng.random() < 0.25:
                # 2-swap
                rem1, rem2 = rng.sample(list(current_elements), 2)
                
                while True:
                    add_indices = rng.sample(all_elements, 2)
                    add1, add2 = add_indices
                    if add1 not in current_elements and add2 not in current_elements and add1 != add2:
                        break
                        
                new_mask = current_mask
                new_mask ^= (1 << rem1)
                new_mask ^= (1 << rem2)
                new_mask |= (1 << add1)
                new_mask |= (1 << add2)
                
                new_score = _score_bitmask(new_mask)
                delta = new_score - current_score
                
                if delta > 0 or rng.random() < np.exp(delta / T):
                    current_elements.remove(rem1)
                    current_elements.remove(rem2)
                    current_elements.add(add1)
                    current_elements.add(add2)
                    current_mask = new_mask
                    current_score = new_score
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_elements = list(current_elements)
            else:
                # 1-swap
                rem = rng.choice(list(current_elements))
                while True:
                    add = rng.choice(all_elements)
                    if add not in current_elements:
                        break
                        
                new_mask = (current_mask ^ (1 << rem)) | (1 << add)
                new_score = _score_bitmask(new_mask)
                delta = new_score - current_score
                
                if delta > 0 or rng.random() < np.exp(delta / T):
                    current_elements.remove(rem)
                    current_elements.add(add)
                    current_mask = new_mask
                    current_score = new_score
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_elements = list(current_elements)
                        
            T *= cooling

        # 3. Local Search Polish - 1-swap
        if time.time() < deadline:
            remaining = deadline - time.time()
            if remaining > 0.5:
                polished_elements = _local_search(best_elements, rng, time_limit=min(1.5, remaining * 0.3))
                polished_mask = 0
                for x in polished_elements:
                    polished_mask |= (1 << x)
                polished_score = _score_bitmask(polished_mask)
                
                if polished_score > best_score:
                    best_score = polished_score
                    best_elements = polished_elements

        # 4. Final 2-swap polish if time remains
        if time.time() < deadline:
            remaining = deadline - time.time()
            if remaining > 0.5:
                polished2_elements = _local_search_2swap_sampled(best_elements, rng, time_limit=min(1.0, remaining * 0.5))
                polished2_mask = 0
                for x in polished2_elements:
                    polished2_mask |= (1 << x)
                polished2_score = _score_bitmask(polished2_mask)
                
                if polished2_score > best_score:
                    best_score = polished2_score
                    best_elements = polished2_elements

    if best_elements is None:
        # Fallback
        best_elements = list(range(1, _N + 1))
        
    return sorted(best_elements)
# EVOLVE-BLOCK-END
