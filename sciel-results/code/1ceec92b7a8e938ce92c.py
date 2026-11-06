import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _precompute_aps():
    """
    Precompute all 3-term arithmetic progressions in 1..512.
    Returns:
    ap_by_elem: list of lists. ap_by_elem[x] contains tuples (d, e1, e2) 
                such that {x, e1, e2} is a 3-AP with difference d.
    """
    ap_by_elem = [[] for _ in range(513)]
    # d ranges from 1 to 255
    for d in range(1, 256):
        # a ranges such that a+2d <= 512
        max_a = 512 - 2 * d
        for a in range(1, max_a + 1):
            b = a + d
            c = a + 2 * d
            # Add to each element's list
            # For element a: AP is (a, b, c), diff d
            ap_by_elem[a].append((d, b, c))
            # For element b: AP is (a, b, c), diff d
            ap_by_elem[b].append((d, a, c))
            # For element c: AP is (a, b, c), diff d
            ap_by_elem[c].append((d, a, b))
    return ap_by_elem

# Global precomputed data
AP_BY_ELEM = _precompute_aps()

def _init_state(S):
    """
    Initialize state for a given set S (list of 32 ints).
    Returns:
    S_set: set of elements
    d_count: array of counts for each d
    score: number of d with count > 0
    """
    S_set = set(S)
    d_count = np.zeros(256, dtype=np.int32)
    
    for x in S:
        for d, e1, e2 in AP_BY_ELEM[x]:
            if e1 in S_set and e2 in S_set:
                d_count[d] += 1
    
    d_count //= 3
    score = int(np.sum(d_count > 0))
    return S_set, d_count, score

def _score_set_bitset(S):
    """
    Fast scoring using bitsets.
    Returns the number of distinct d > 0 such that a 3-AP exists in S.
    """
    S_set = set(S)
    mask = 0
    for x in S:
        mask |= (1 << x)
    
    d_count = np.zeros(256, dtype=np.int32)
    for x in S:
        for d, e1, e2 in AP_BY_ELEM[x]:
            if e1 in S_set and e2 in S_set:
                d_count[d] += 1
    
    d_count //= 3
    return int(np.sum(d_count > 0))

def _delta_score_add(S_set, d_count, score, x):
    """
    Calculate delta score if we add element x to S_set.
    Assumes x not in S_set.
    """
    delta = 0
    for d, e1, e2 in AP_BY_ELEM[x]:
        if e1 in S_set and e2 in S_set:
            if d_count[d] == 0:
                delta += 1
    return delta

def _delta_score_remove(S_set, d_count, score, x):
    """
    Calculate delta score if we remove element x from S_set.
    Assumes x in S_set.
    """
    delta = 0
    for d, e1, e2 in AP_BY_ELEM[x]:
        if e1 in S_set and e2 in S_set:
            if d_count[d] == 1:
                delta -= 1
    return delta

def _apply_add(S_set, d_count, score, x):
    """Apply adding x to state. Returns new score."""
    for d, e1, e2 in AP_BY_ELEM[x]:
        if e1 in S_set and e2 in S_set:
            if d_count[d] == 0:
                score += 1
            d_count[d] += 1
    S_set.add(x)
    return score

def _apply_remove(S_set, d_count, score, x):
    """Apply removing x from state. Returns new score."""
    for d, e1, e2 in AP_BY_ELEM[x]:
        if e1 in S_set and e2 in S_set:
            d_count[d] -= 1
            if d_count[d] == 0:
                score -= 1
    S_set.remove(x)
    return score

def _greedy_max_coverage_init(rng):
    """
    Greedy maximum coverage heuristic to initialize a set.
    Directly targets the objective: maximize number of distinct d covered.
    """
    S = []
    S_set = set()
    
    # For each candidate element, track which d's it can contribute to
    # with the current set S
    
    # Start with a random seed element
    S.append(rng.randint(1, 512))
    S_set.add(S[0])
    
    while len(S) < 32:
        best_elem = None
        best_gain = -1
        
        for cand in range(1, 513):
            if cand in S_set:
                continue
            
            # Calculate how many new d's this candidate would cover
            gain = 0
            for d, e1, e2 in AP_BY_ELEM[cand]:
                if e1 in S_set and e2 in S_set:
                    # Check if this d is already covered by existing elements
                    # We need to check if there's already an active AP with this d
                    # For speed, we'll just count potential new d's
                    gain += 1
            
            # Use a heuristic: prefer candidates that create many new APs
            # But we need to be more precise about distinct d's
            # For now, use the raw count as a proxy
            if gain > best_gain:
                best_gain = gain
                best_elem = cand
        
        if best_elem is None:
            # Fallback to random
            available = [x for x in range(1, 513) if x not in S_set]
            if available:
                best_elem = rng.choice(available)
            else:
                break
        
        S.append(best_elem)
        S_set.add(best_elem)
    
    return sorted(S)

def _structured_init(rng, strategy=0):
    """
    Generate diverse structured initializations.
    strategy 0: random
    strategy 1: 3-cluster
    strategy 2: geometric spread
    """
    if strategy == 0:
        return sorted(rng.sample(range(1, 513), 32))
    elif strategy == 1:
        # 3-cluster: three dense clusters
        centers = [rng.randint(1, 100), rng.randint(150, 350), rng.randint(400, 512)]
        S = set()
        for c in centers:
            for offset in range(-10, 11):
                val = c + offset
                if 1 <= val <= 512 and val not in S:
                    S.add(val)
        # Fill remaining randomly
        while len(S) < 32:
            S.add(rng.randint(1, 512))
        return sorted(list(S)[:32])
    elif strategy == 2:
        # Geometric spread
        S = set()
        for i in range(32):
            val = int(1 + i * (511.0 / 31.0))
            S.add(val)
        # Add some random perturbations
        for _ in range(16):
            S.add(rng.randint(1, 512))
        return sorted(list(S)[:32])
    else:
        return sorted(rng.sample(range(1, 513), 32))

def _simulated_annealing(S_init, time_budget, rng):
    S = list(S_init)
    S_set, d_count, current_score = _init_state(S)
    
    best_S = list(S)
    best_score = current_score
    
    start_time = time.time()
    T = 5.0
    cooling = 0.999
    
    while time.time() - start_time < time_budget:
        T *= cooling
        if T < 0.01:
            break
        
        i = rng.randint(0, 31)
        old_elem = S[i]
        
        new_elem = rng.randint(1, 512)
        if new_elem == old_elem:
            continue
        if new_elem in S_set:
            continue
            
        # Apply Remove
        _apply_remove(S_set, d_count, current_score, old_elem)
        # Apply Add
        new_score = _apply_add(S_set, d_count, current_score, new_elem)
        
        delta = new_score - current_score
        
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_score = new_score
            S[i] = new_elem
            if current_score > best_score:
                best_score = current_score
                best_S = list(S)
        else:
            # Revert
            _apply_remove(S_set, d_count, current_score, new_elem)
            _apply_add(S_set, d_count, current_score, old_elem)
    
    return best_S, best_score

def _greedy_polish(S_init, time_limit=5.0):
    """
    Greedy local search: try swapping each element with every other candidate.
    Use delta scoring for speed.
    """
    S = list(S_init)
    S_set, d_count, current_score = _init_state(S)
    
    start_time = time.time()
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(32):
            old_val = S[i]
            best_cand = None
            best_score = current_score
            
            # Try removing old_val
            _apply_remove(S_set, d_count, current_score, old_val)
            score_without_old = current_score
            
            # Try adding each candidate
            for cand in range(1, 513):
                if cand in S_set:
                    continue
                
                # Apply add
                new_score = _apply_add(S_set, d_count, score_without_old, cand)
                
                if new_score > best_score:
                    best_score = new_score
                    best_cand = cand
                    # Revert this add before trying next cand
                    _apply_remove(S_set, d_count, new_score, cand)
                else:
                    # Revert add
                    _apply_remove(S_set, d_count, new_score, cand)
            
            # Restore old_val
            current_score = _apply_add(S_set, d_count, score_without_old, old_val)
            
            if best_cand is not None:
                S[i] = best_cand
                # Apply the best move: remove old_val, add best_cand
                _apply_remove(S_set, d_count, current_score, old_val)
                current_score = _apply_add(S_set, d_count, current_score, best_cand)
                improved = True
    
    return S, current_score

def _greedy_polish_fast(S_init, time_limit=3.0):
    """
    Faster greedy polish using bitset-based scoring for candidate evaluation.
    """
    S = list(S_init)
    S_set, d_count, current_score = _init_state(S)
    
    start_time = time.time()
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(32):
            old_val = S[i]
            best_cand = None
            best_score = current_score
            
            # Try removing old_val
            _apply_remove(S_set, d_count, current_score, old_val)
            score_without_old = current_score
            
            # Try adding each candidate - use quick scoring
            for cand in range(1, 513):
                if cand in S_set:
                    continue
                
                # Quick delta calculation
                delta = 0
                for d, e1, e2 in AP_BY_ELEM[cand]:
                    if e1 in S_set and e2 in S_set:
                        if d_count[d] == 0:
                            delta += 1
                
                new_score = score_without_old + delta
                
                if new_score > best_score:
                    best_score = new_score
                    best_cand = cand
            
            # Restore old_val
            current_score = _apply_add(S_set, d_count, score_without_old, old_val)
            
            if best_cand is not None:
                S[i] = best_cand
                _apply_remove(S_set, d_count, current_score, old_val)
                current_score = _apply_add(S_set, d_count, current_score, best_cand)
                improved = True
    
    return S, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    rng = random.Random(42)
    
    best_S = list(range(1, 33))
    best_score = _init_state(best_S)[2]
    
    # Try multiple restarts with diverse initializations
    num_restarts = 12
    per_restart_time = 8.0
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Use diverse initialization strategies
        if restart == 0:
            S_init = list(range(1, 33))
        elif restart == 1:
            S_init = _greedy_max_coverage_init(rng)
        else:
            strategy = (restart - 2) % 3
            S_init = _structured_init(rng, strategy)
        
        remaining = min(per_restart_time, deadline - time.time() - 5.0)
        if remaining < 1.0:
            break
        
        S, score = _simulated_annealing(S_init, remaining, rng)
        
        # Greedy polish
        if time.time() < deadline - 3.0:
            S, score = _greedy_polish_fast(S, time_limit=2.0)
        
        if score > best_score:
            best_score = score
            best_S = S
    
    # Final polish
    if time.time() < deadline - 1.0:
        best_S, best_score = _greedy_polish(best_S, time_limit=5.0)
    
    return sorted(best_S)

# EVOLVE-BLOCK-END
