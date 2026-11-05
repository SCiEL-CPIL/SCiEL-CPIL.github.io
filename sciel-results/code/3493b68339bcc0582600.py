import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _compute_diffs(s):
    """Compute the set of all d > 0 such that a, a+d, a+2d are all in s."""
    s_set = set(s)
    diffs = set()
    elements = sorted(s_set)
    n = len(elements)
    for i in range(n):
        a = elements[i]
        for j in range(i + 1, n):
            b = elements[j]
            d = b - a
            if a + 2 * d in s_set:
                diffs.add(d)
    return diffs

def _score(s):
    """Calculate the number of distinct d > 0 such that a, a+d, a+2d are in s."""
    return len(_compute_diffs(s))

def _generate_initial_set():
    """
    Generate an initial set of 32 elements using multiple strategies.
    Returns the best scoring initial set.
    """
    strategies = []
    
    # Strategy 1: Random
    strategies.append(set(random.sample(range(1, 513), 32)))
    
    # Strategy 2: Arithmetic progression based
    for _ in range(5):
        d = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 25, 30, 40, 50])
        max_start = 512 - 31 * d
        if max_start >= 1:
            a = random.randint(1, max_start)
            s = set(a + i * d for i in range(32))
            if all(1 <= x <= 512 for x in s):
                strategies.append(s)
    
    # Strategy 3: Two interleaved APs
    for _ in range(3):
        d1 = random.choice([2, 3, 4, 5, 6, 8, 10])
        d2 = random.choice([2, 3, 4, 5, 6, 8, 10, 12])
        s = set()
        a1 = random.randint(1, 512 - 15 * d1)
        for i in range(16):
            s.add(a1 + i * d1)
        a2 = random.randint(1, 512 - 15 * d2)
        for i in range(16):
            val = a2 + i * d2
            if val not in s:
                s.add(val)
        # Fill remaining if needed
        while len(s) < 32:
            val = random.randint(1, 512)
            if val not in s:
                s.add(val)
        if len(s) == 32:
            strategies.append(s)
    
    # Strategy 4: Clustered blocks
    for _ in range(3):
        s = set()
        while len(s) < 32:
            base = random.randint(1, 480)
            for k in range(32):
                if base + k <= 512 and len(s) < 32:
                    s.add(base + k)
        strategies.append(s)
    
    # Strategy 5: Weighted random - prefer elements that create many 3-APs
    s = set()
    while len(s) < 32:
        # Pick from a random "region"
        region_start = random.randint(1, 481)
        region_end = min(region_start + 30, 512)
        elem = random.randint(region_start, region_end)
        if elem not in s:
            s.add(elem)
    strategies.append(s)
    
    # Strategy 6: Dense random with local structure
    for _ in range(3):
        s = set()
        # Pick 4-6 anchor points
        anchors = random.sample(range(1, 513), random.randint(4, 6))
        for a in anchors:
            for offset in random.sample(range(-15, 16), min(6, 31)):
                val = a + offset
                if 1 <= val <= 512 and val not in s:
                    s.add(val)
                    if len(s) >= 32:
                        break
            if len(s) >= 32:
                break
        while len(s) < 32:
            val = random.randint(1, 512)
            if val not in s:
                s.add(val)
        strategies.append(s)
    
    # Strategy 7: Multi-scale APs - mix of small and large step sizes
    for _ in range(3):
        s = set()
        # First part: dense cluster
        base1 = random.randint(1, 200)
        for i in range(16):
            s.add(base1 + i)
        # Second part: sparse AP
        d2 = random.choice([8, 12, 16, 20, 24])
        base2 = random.randint(250, 512 - 15 * d2)
        for i in range(16):
            val = base2 + i * d2
            if val not in s:
                s.add(val)
        while len(s) < 32:
            val = random.randint(1, 512)
            if val not in s:
                s.add(val)
        if len(s) == 32:
            strategies.append(s)
    
    # Pick the best initial set
    best = None
    best_sc = -1
    for s in strategies:
        if len(s) != 32:
            continue
        sc = len(_compute_diffs(s))
        if sc > best_sc:
            best_sc = sc
            best = s
    
    return best if best else set(random.sample(range(1, 513), 32))

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    best_set = None
    best_score = 0
    
    # Multiple restarts with different initializations
    num_starts = 15
    
    for start_iter in range(num_starts):
        if time.time() > deadline - 5:
            break
        
        # Generate initial set using diverse strategies
        current_set = _generate_initial_set()
        current_diffs = _compute_diffs(current_set)
        current_score = len(current_diffs)
        
        if current_score > best_score:
            best_score = current_score
            best_set = set(current_set)
        
        # Simulated Annealing / Local Search
        T = 50.0
        cooling = 0.997
        min_T = 0.5
        stuck_count = 0
        max_stuck = 3000
        
        while time.time() < deadline - 2:
            # Generate a neighbor: pick an element to remove and one to add
            # Bias toward removing elements that are less "useful"
            elem_out = random.choice(list(current_set))
            elem_in = random.randint(1, 512)
            
            if elem_in in current_set:
                continue
            
            new_set = set(current_set)
            new_set.remove(elem_out)
            new_set.add(elem_in)
            
            new_diffs = _compute_diffs(new_set)
            new_score = len(new_diffs)
            delta = new_score - current_score
            
            if delta > 0:
                current_set = new_set
                current_score = new_score
                current_diffs = new_diffs
                stuck_count = 0
                
                if new_score > best_score:
                    best_score = new_score
                    best_set = set(current_set)
            else:
                # Accept with probability exp(delta/T)
                if delta < -10:  # Skip very bad moves
                    stuck_count += 1
                elif random.random() < np.exp(delta / T):
                    current_set = new_set
                    current_score = new_score
                    current_diffs = new_diffs
                else:
                    stuck_count += 1
            
            T *= cooling
            if T < min_T:
                T = min_T
            
            if stuck_count > max_stuck:
                # Restart local search with new random perturbation
                for _ in range(5):
                    elem_out = random.choice(list(current_set))
                    elem_in = random.randint(1, 512)
                    if elem_in not in current_set:
                        current_set = set(current_set)
                        current_set.remove(elem_out)
                        current_set.add(elem_in)
                        current_diffs = _compute_diffs(current_set)
                        current_score = len(current_diffs)
                stuck_count = 0
                T = max(T, 30.0)
        
        # Final check
        if current_score > best_score:
            best_score = current_score
            best_set = set(current_set)
    
    # Final polish: try to improve best_set with a focused search
    if best_set is not None and time.time() < deadline - 1:
        current_set = set(best_set)
        current_diffs = _compute_diffs(current_set)
        current_score = len(current_diffs)
        
        # Focused local search on the best solution
        T = 10.0
        cooling = 0.999
        min_T = 0.1
        stuck_count = 0
        
        while time.time() < deadline - 0.5:
            elem_out = random.choice(list(current_set))
            elem_in = random.randint(1, 512)
            
            if elem_in in current_set:
                continue
            
            new_set = set(current_set)
            new_set.remove(elem_out)
            new_set.add(elem_in)
            
            new_diffs = _compute_diffs(new_set)
            new_score = len(new_diffs)
            delta = new_score - current_score
            
            if delta > 0:
                current_set = new_set
                current_score = new_score
                current_diffs = new_diffs
                stuck_count = 0
                
                if new_score > best_score:
                    best_score = new_score
                    best_set = set(current_set)
            else:
                if random.random() < np.exp(delta / T):
                    current_set = new_set
                    current_score = new_score
                    current_diffs = new_diffs
                else:
                    stuck_count += 1
            
            T *= cooling
            if T < min_T:
                T = min_T
            
            if stuck_count > 2000:
                break
    
    return list(best_set) if best_set else list(range(1, 33))

# EVOLVE-BLOCK-END
