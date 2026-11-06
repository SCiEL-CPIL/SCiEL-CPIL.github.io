import time
import random
import numpy as np

_MAX = 512
_N = 32
_DEADLINE = 145.0  # Safety margin before 150s kill

def _score_bitmask(mask):
    """
    Count distinct d > 0 such that a, a+d, a+2d are in the set represented by mask.
    mask is a Python integer where bit i is 1 if i is in the set.
    """
    count = 0
    # Max d such that 2*d + 1 <= 512 is 255.
    # If a=1, a+2d <= 512 => 2d <= 511 => d <= 255.
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def _greedy_init(rng):
    """
    Build a set greedily by adding the element that maximizes the score gain.
    Returns a list and mask to maintain type consistency for SA.
    """
    current_elements = []
    current_mask = 0
    current_score = 0
    
    for _ in range(_N):
        best_gain = -1
        best_elem = None
        for x in range(1, _MAX + 1):
            if (current_mask >> x) & 1:
                continue
            new_mask = current_mask | (1 << x)
            new_score = _score_bitmask(new_mask)
            gain = new_score - current_score
            if gain > best_gain or (gain == best_gain and rng.random() < 0.5):
                best_gain = gain
                best_elem = x
        
        if best_elem is not None:
            current_elements.append(best_elem)
            current_mask |= (1 << best_elem)
            current_score += best_gain
            
    return current_elements, current_mask, current_score

def _local_search(elements, mask, rng, time_limit=2.0):
    """
    Perform local search by trying all single swaps.
    Uses list and mask for efficiency.
    """
    current_elements = list(elements)
    current_mask = mask
    current_score = _score_bitmask(current_mask)
    
    start_time = time.time()
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i, rem in enumerate(current_elements):
            rem_bit = 1 << rem
            for add in range(1, _MAX + 1):
                if (current_mask >> add) & 1:
                    continue
                new_mask = (current_mask ^ rem_bit) | (1 << add)
                new_score = _score_bitmask(new_mask)
                if new_score > current_score:
                    current_elements[i] = add
                    current_mask = new_mask
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
                
    return current_elements, current_mask, current_score

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    best_elements = None
    best_score = -1
    best_mask = 0
    
    # Run multiple restarts
    num_restarts = 12 # Increased slightly, but will be limited by time anyway
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        rng = random.Random(42 + restart)
        
        # 1. Greedy Initialization
        current_elements, current_mask, current_score = _greedy_init(rng)
        
        if current_score > best_score:
            best_score = current_score
            best_elements = list(current_elements)
            best_mask = current_mask
            
        # 2. Simulated Annealing
        time_per_restart = (_DEADLINE - (time.time() - start_time)) / max(1, (num_restarts - restart))
        sa_start = time.time()
        sa_deadline = sa_start + min(10.0, time_per_restart * 0.8)  # Cap SA time per restart
        
        T = 1.0
        cooling = 0.999
        min_T = 0.001
        
        # Precompute list of all possible elements for fast sampling
        all_elements = list(range(1, _MAX + 1))
        
        while time.time() < sa_deadline:
            if T < min_T:
                T = min_T
                
            # Decide move type: 1-swap or 2-swap
            if rng.random() < 0.3:
                # 2-swap
                # Pick 2 to remove from the set
                idx1, idx2 = rng.sample(range(_N), 2)
                rem1 = current_elements[idx1]
                rem2 = current_elements[idx2]
                rem1_bit = 1 << rem1
                rem2_bit = 1 << rem2
                
                # Pick 2 to add
                while True:
                    add1 = rng.choice(all_elements)
                    add2 = rng.choice(all_elements)
                    if add1 != add2 and not ((current_mask >> add1) & 1) and not ((current_mask >> add2) & 1):
                        break
                        
                # Apply
                new_mask = current_mask
                new_mask ^= rem1_bit
                new_mask ^= rem2_bit
                new_mask |= (1 << add1)
                new_mask |= (1 << add2)
                
                new_score = _score_bitmask(new_mask)
                delta = new_score - current_score
                
                if delta > 0 or rng.random() < np.exp(delta / T):
                    # Accept
                    current_elements[idx1] = add1
                    current_elements[idx2] = add2
                    current_mask = new_mask
                    current_score = new_score
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_elements = list(current_elements)
                        best_mask = current_mask
            else:
                # 1-swap
                rem_idx = rng.randrange(_N)
                rem = current_elements[rem_idx]
                rem_bit = 1 << rem
                
                while True:
                    add = rng.choice(all_elements)
                    if not ((current_mask >> add) & 1):
                        break
                        
                new_mask = (current_mask ^ rem_bit) | (1 << add)
                new_score = _score_bitmask(new_mask)
                delta = new_score - current_score
                
                if delta > 0 or rng.random() < np.exp(delta / T):
                    current_elements[rem_idx] = add
                    current_mask = new_mask
                    current_score = new_score
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_elements = list(current_elements)
                        best_mask = current_mask
                        
            T *= cooling

        # 3. Local Search Polish
        if time.time() < deadline - 2.0:
            polished_elements, polished_mask, polished_score = _local_search(best_elements, best_mask, rng, time_limit=min(3.0, deadline - time.time() - 1.0))
            
            if polished_score > best_score:
                best_score = polished_score
                best_elements = polished_elements
                best_mask = polished_mask

    if best_elements is None:
        # Fallback
        best_elements = list(range(1, _N + 1))
        
    return sorted(best_elements)
