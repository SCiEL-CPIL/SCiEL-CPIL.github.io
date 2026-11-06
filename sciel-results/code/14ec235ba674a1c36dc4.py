# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s
    
    n = 256
    # Precompute incompatibility: lcm(a,b) <= 256 means incompatible
    # lcm(a,b) = a*b // gcd(a,b)
    # So a*b <= 256 * gcd(a,b)
    
    # Build incompatibility bitmask for each number (indices 0..254 for values 2..256)
    # Use a list of bitsets (as integers) for fast compatibility checking
    incomp = [0] * (n + 1)  # 1-indexed, incomp[i] has bit j set if i and j+2 are incompatible
    
    for a in range(2, n + 1):
        ia = a - 2
        for b in range(a + 1, n + 1):
            g = gcd(a, b)
            lcm_val = a * b // g
            if lcm_val <= 256:
                ib = b - 2
                incomp[a] |= (1 << ib)
                incomp[b] |= (1 << ia)
    
    if time.time() > deadline:
        return [256]
    
    def score_of(s):
        return sum(1.0 / x for x in s)
    
    def is_compatible(x, s):
        ix = x - 2
        for y in s:
            iy = y - 2
            if (incomp[x] >> iy) & 1:
                return False
        return True
    
    # Strategy 1: Greedy by descending reciprocal (i.e., ascending value)
    def greedy_by_value():
        selected = set()
        for x in range(2, n + 1):
            if is_compatible(x, selected):
                selected.add(x)
        return selected
    
    # Strategy 2: Greedy by descending value (larger numbers first, more likely to be "unique")
    def greedy_by_value_desc():
        selected = set()
        for x in range(n, 1, -1):
            if is_compatible(x, selected):
                selected.add(x)
        return selected
    
    # Strategy 3: Randomized greedy
    import random
    def randomized_greedy(rng):
        candidates = list(range(2, n + 1))
        rng.shuffle(candidates)
        selected = set()
        for x in candidates:
            if is_compatible(x, selected):
                selected.add(x)
        return selected
    
    # Local search
    def local_search(s, max_iters=200):
        s = set(s)
        cur_score = score_of(s)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            
            # Try adding elements
            for x in range(2, n + 1):
                if x in s:
                    continue
                if is_compatible(x, s):
                    s.add(x)
                    cur_score += 1.0 / x
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swapping: remove one, add one or more
            for a in list(s):
                trial = s - {a}
                trial_score = cur_score - 1.0 / a
                
                # Try adding single elements
                for b in range(2, n + 1):
                    if b in trial:
                        continue
                    if is_compatible(b, trial):
                        new_score = trial_score + 1.0 / b
                        if new_score > cur_score:
                            s = trial | {b}
                            cur_score = new_score
                            improved = True
                            break
                
                if improved:
                    break
                
                # Try adding two elements (replace one with two)
                if not improved:
                    for b in range(2, n + 1):
                        if b in trial or b == a:
                            continue
                        if not is_compatible(b, trial):
                            continue
                        for c in range(b + 1, n + 1):
                            if c in trial or c == a:
                                continue
                            if not is_compatible(c, trial) or not is_compatible(b, {c} | trial):
                                continue
                            # Check b and c compatible with each other and with trial
                            ib, ic = b - 2, c - 2
                            if (incomp[b] >> ic) & 1:
                                continue
                            new_score = trial_score + 1.0 / b + 1.0 / c
                            if new_score > cur_score:
                                s = trial | {b, c}
                                cur_score = new_score
                                improved = True
                                break
                        if improved:
                            break
                
                if improved:
                    break
            
            if not improved:
                break
        
        return s, cur_score
    
    # Generate multiple initial solutions
    best_set = set()
    best_score = 0.0
    
    # Greedy ascending
    s1 = greedy_by_value()
    s1, sc1 = local_search(s1, 50)
    if sc1 > best_score:
        best_set, best_score = s1, sc1
    
    if time.time() > deadline:
        return list(best_set)
    
    # Greedy descending
    s2 = greedy_by_value_desc()
    s2, sc2 = local_search(s2, 50)
    if sc2 > best_score:
        best_set, best_score = s2, sc2
    
    if time.time() > deadline:
        return list(best_set)
    
    # Multiple randomized starts
    rng = random.Random(42)
    for i in range(50):
        if time.time() > deadline:
            break
        s = randomized_greedy(rng)
        s, sc = local_search(s, 30)
        if sc > best_score:
            best_set, best_score = s, sc
    
    if time.time() > deadline:
        return list(best_set)
    
    # Final intensive local search on best
    best_set, best_score = local_search(best_set, 500)
    
    # Additional polish: try removing elements that allow many additions
    for _ in range(20):
        if time.time() > deadline:
            break
        improved = False
        for a in list(best_set):
            trial = best_set - {a}
            trial_score = best_score - 1.0 / a
            
            # Find all compatible additions to trial
            additions = []
            for b in range(2, n + 1):
                if b in trial:
                    continue
                if is_compatible(b, trial):
                    additions.append(b)
            
            # Greedily add best additions
            new_s = set(trial)
            new_score = trial_score
            for b in sorted(additions, key=lambda x: 1.0/x, reverse=True):
                if is_compatible(b, new_s):
                    new_s.add(b)
                    new_score += 1.0 / b
            
            if new_score > best_score:
                best_set = new_s
                best_score = new_score
                improved = True
                break
        
        if not improved:
            break
    
    return list(best_set)
# EVOLVE-BLOCK-END
