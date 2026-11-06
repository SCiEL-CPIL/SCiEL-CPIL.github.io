"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
import time as _time

def _score_bitset(mask, n=512):
    """Score a bitmask: count distinct d>0 such that a, a+d, a+2d all in set."""
    # mask is a Python int bitmask of size n
    # For each d from 1 to n//2, check if there exists a such that
    # bits a, a+d, a+2d are all set
    score = 0
    for d in range(1, n // 2 + 1):
        # Check if there exists a such that mask has bits at a, a+d, a+2d
        # a ranges from 0 to n-1-2d
        # We can check: (mask & (mask >> d) & (mask >> (2*d))) != 0
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            score += 1
    return score

def _bits_to_list(mask, n=512):
    """Convert bitmask to list of 1-indexed integers."""
    result = []
    for i in range(n):
        if (mask >> i) & 1:
            result.append(i + 1)
    return result

def _list_to_bits(lst, n=512):
    """Convert list of 1-indexed integers to bitmask."""
    mask = 0
    for x in lst:
        mask |= (1 << (x - 1))
    return mask

def _random_init(rng):
    """Generate a random 32-element subset of 1..512."""
    elems = rng.sample(range(1, 513), 32)
    return elems

def _greedy_polish(lst, rng, max_iters=2000):
    """Greedy local search: try single swaps to improve score."""
    current = list(lst)
    current_mask = _list_to_bits(current)
    current_score = _score_bitset(current_mask)
    best = list(current)
    best_score = current_score
    best_mask = current_mask
    
    n = 512
    all_vals = set(range(1, n + 1))
    in_set = set(current)
    
    for it in range(max_iters):
        improved = False
        # Try swapping one element out and one in
        out_candidates = list(current)
        rng.shuffle(out_candidates)
        for out_val in out_candidates:
            if (out_val - 1) < 0 or (out_val - 1) >= n:
                continue
            # Remove out_val
            new_mask = best_mask & ~(1 << (out_val - 1))
            # Try adding a random in_val
            in_candidates = list(all_vals - in_set)
            rng.shuffle(in_candidates)
            for in_val in in_candidates[:20]:  # limit for speed
                if (in_val - 1) < 0 or (in_val - 1) >= n:
                    continue
                trial_mask = new_mask | (1 << (in_val - 1))
                trial_score = _score_bitset(trial_mask)
                if trial_score > best_score:
                    best_score = trial_score
                    best_mask = trial_mask
                    best = list(current)
                    best.remove(out_val)
                    best.append(in_val)
                    in_set = set(best)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return best, best_score

def _simulated_annealing(lst, rng, deadline):
    """SA with bitset scoring."""
    current = list(lst)
    current_mask = _list_to_bits(current)
    current_score = _score_bitset(current_mask)
    best = list(current)
    best_score = current_score
    best_mask = current_mask
    
    n = 512
    all_vals = set(range(1, n + 1))
    
    T = 5.0
    T_min = 0.01
    cooling = 0.999
    
    while T > T_min:
        if _time.time() > deadline:
            break
        
        # Generate neighbor: swap one element
        in_set = set(current)
        out_candidates = list(current)
        rng.shuffle(out_candidates)
        out_val = out_candidates[0]
        
        in_candidates = list(all_vals - in_set)
        rng.shuffle(in_candidates)
        in_val = in_candidates[0]
        
        # Compute new mask
        new_mask = current_mask & ~(1 << (out_val - 1))
        new_mask = new_mask | (1 << (in_val - 1))
        new_score = _score_bitset(new_mask)
        
        delta = new_score - current_score
        if delta >= 0 or rng.random() < np.exp(delta / T):
            current_mask = new_mask
            current_score = new_score
            current = list(current)
            current.remove(out_val)
            current.append(in_val)
            
            if current_score > best_score:
                best_score = current_score
                best = list(current)
                best_mask = current_mask
        
        T *= cooling
    
    return best, best_score

def construct():
    """Construct 32 distinct integers in 1..512 maximizing 3-AP differences."""
    start_time = _time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = random.Random(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # Candidate 1: sequential
    candidates.append(list(range(1, 33)))
    
    # Candidate 2: powers of 2 and nearby
    pow2 = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    extra = rng.sample([x for x in range(1, 513) if x not in pow2], 23)
    candidates.append(pow2 + extra)
    
    # Candidate 3: clustered
    clusters = []
    base = 1
    for i in range(8):
        c = rng.sample(range(base, base + 64), 4)
        clusters.extend(c)
        base += 64
    candidates.append(clusters)
    
    # Candidate 4: modular
    mod_cand = []
    for i in range(32):
        val = (i * 17) % 512 + 1
        if val not in mod_cand:
            mod_cand.append(val)
    while len(mod_cand) < 32:
        v = rng.randint(1, 512)
        if v not in mod_cand:
            mod_cand.append(v)
    candidates.append(mod_cand)
    
    # Candidate 5-8: random
    for _ in range(4):
        candidates.append(_random_init(rng))
    
    # Score all candidates
    best_candidate = None
    best_score = -1
    for cand in candidates:
        if len(cand) != 32 or len(set(cand)) != 32:
            continue
        if any(x < 1 or x > 512 for x in cand):
            continue
        mask = _list_to_bits(cand)
        sc = _score_bitset(mask)
        if sc > best_score:
            best_score = sc
            best_candidate = list(cand)
    
    if best_candidate is None:
        return list(range(1, 33))
    
    # Run SA from best candidate
    if _time.time() < deadline - 5:
        sa_result, sa_score = _simulated_annealing(best_candidate, rng, deadline)
        if sa_score > best_score:
            best_candidate = sa_result
            best_score = sa_score
    
    # Greedy polish
    if _time.time() < deadline - 2:
        polished, pol_score = _greedy_polish(best_candidate, rng, max_iters=5000)
        if pol_score > best_score:
            best_candidate = polished
            best_score = pol_score
    
    # Final validation
    result = sorted(best_candidate)
    assert len(result) == 32
    assert len(set(result)) == 32
    assert all(1 <= x <= 512 for x in result)
    
    return result
# EVOLVE-BLOCK-END
