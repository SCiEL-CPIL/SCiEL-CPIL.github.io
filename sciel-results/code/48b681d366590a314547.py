import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _compute_potential():
    """
    Precompute for each difference d (1..255), the number of (a, a+d, a+2d)
    triples that fit in 1..512. This tells us how "common" each difference is
    and helps guide search.
    """
    pot = {}
    for d in range(1, 256):
        # a ranges from 1 to 512 - 2d
        count = max(0, 512 - 2 * d)
        pot[d] = count
    return pot

def _fast_score(s):
    """Count distinct d > 0 with a, a+d, a+2d in s. O(n^2) but with small constant."""
    s = set(s)
    diffs = set()
    elements = sorted(s)
    for i in range(len(elements)):
        a = elements[i]
        for j in range(i + 1, len(elements)):
            b = elements[j]
            d = b - a
            c = b + d
            if c in s:
                diffs.add(d)
    return len(diffs)

def _score_with_diffs(s):
    """Return (score, set_of_diffs) for a set."""
    s = set(s)
    diffs = set()
    elements = sorted(s)
    for i in range(len(elements)):
        a = elements[i]
        for j in range(i + 1, len(elements)):
            b = elements[j]
            d = b - a
            c = b + d
            if c in s:
                diffs.add(d)
    return len(diffs), diffs

def _delta_score(s, remove_elem, add_elem):
    """
    Compute change in score when removing remove_elem and adding add_elem from set s.
    Returns (new_score_delta, new_diffs_set) where new_score_delta is the change
    in number of distinct differences.
    Actually, let's just recompute efficiently.
    """
    pass

def _try_move(s, remove_elem, add_elem, current_diffs):
    """
    Try removing remove_elem and adding add_elem.
    Return (new_score, new_diffs) or None if add_elem already in s.
    """
    if add_elem in s:
        return None
    
    new_s = set(s)
    new_s.remove(remove_elem)
    new_s.add(add_elem)
    
    new_score, new_diffs = _score_with_diffs(new_s)
    return new_score, new_diffs

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    potential = _compute_potential()
    
    # Sort differences by potential (how many 3-APs they can form)
    # Higher potential = more "useful" differences
    diff_list = sorted(potential.keys(), key=lambda d: -potential[d])
    
    best_set = None
    best_score = 0
    
    num_starts = 20  # Multiple random starts
    
    for start_iter in range(num_starts):
        if time.time() > deadline - 5:
            break
        
        # Generate initial set using weighted approach
        # Bias toward elements that create many 3-APs
        current_set = _generate_initial_set(potential)
        current_score, current_diffs = _score_with_diffs(current_set)
        
        if current_score > best_score:
            best_score = current_score
            best_set = set(current_set)
        
        # Local search / simulated annealing
        T = 50.0
        cooling = 0.997
        min_T = 0.5
        stuck_count = 0
        max_stuck = 5000
        
        while time.time() < deadline - 2:
            # Generate a neighbor
            elem_out = random.choice(list(current_set))
            elem_in = random.randint(1, 512)
            
            if elem_in in current_set:
                continue
            
            result = _try_move(current_set, elem_out, elem_in, current_diffs)
            if result is None:
                continue
            
            new_score, new_diffs = result
            delta = new_score - current_score
            
            if delta > 0:
                current_set = set(current_set)
                current_set.remove(elem_out)
                current_set.add(elem_in)
                current_score = new_score
                current_diffs = new_diffs
                stuck_count = 0
                
                if new_score > best_score:
                    best_score = new_score
                    best_set = set(current_set)
            else:
                # Accept with probability exp(delta/T)
                if random.random() < np.exp(delta / T):
                    current_set = set(current_set)
                    current_set.remove(elem_out)
                    current_set.add(elem_in)
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
                        current_score, current_diffs = _score_with_diffs(current_set)
                stuck_count = 0
                T = max(T, 30.0)
        
        # Final check
        if current_score > best_score:
            best_score = current_score
            best_set = set(current_set)
    
    return list(best_set) if best_set else list(range(1, 33))

def _generate_initial_set(potential):
    """
    Generate an initial set of 32 elements biased toward creating many 3-APs.
    Strategy: start with a structured set, then perturb.
    """
    # Try a few structured initializations
    strategies = []
    
    # Strategy 1: Random
    strategies.append(set(random.sample(range(1, 513), 32)))
    
    # Strategy 2: Clustered (consecutive blocks)
    for _ in range(3):
        s = set()
        while len(s) < 32:
            base = random.randint(1, 480)
            for k in range(32):
                if base + k <= 512:
                    s.add(base + k)
                if len(s) >= 32:
                    break
        strategies.append(s)
    
    # Strategy 3: Arithmetic progression based
    for _ in range(3):
        d = random.choice([1, 2, 3, 4, 5, 6, 7, 8])
        a = random.randint(1, 512 - 31 * d)
        s = set(a + i * d for i in range(32))
        if all(1 <= x <= 512 for x in s):
            strategies.append(s)
    
    # Strategy 4: Weighted random - prefer elements in dense regions
    s = set()
    while len(s) < 32:
        # Pick from a random "region"
        region_start = random.randint(1, 481)
        region_end = min(region_start + 30, 512)
        elem = random.randint(region_start, region_end)
        if elem not in s:
            s.add(elem)
    strategies.append(s)
    
    # Pick the best initial set
    best = None
    best_sc = -1
    for s in strategies:
        if len(s) != 32:
            continue
        sc, _ = _score_with_diffs(s)
        if sc > best_sc:
            best_sc = sc
            best = s
    
    return best if best else set(random.sample(range(1, 513), 32))

# EVOLVE-BLOCK-END
