"""Find a 32-element set in 1..512 maximizing distinct 3-AP differences via difference-targeting."""
import time
import random
import math

def _score(s):
    """Count distinct d > 0 such that a, a+d, a+2d all in set s."""
    sset = set(s)
    diffs = set()
    for a in s:
        for b in s:
            if b == a:
                continue
            d = b - a
            if d <= 0:
                continue
            if a + 2 * d in sset:
                diffs.add(d)
    return len(diffs)

def _score_fast(s):
    """Faster scoring using sorted list and early termination."""
    sset = set(s)
    n = len(s)
    diffs = set()
    for i in range(n):
        a = s[i]
        for j in range(n):
            if i == j:
                continue
            b = s[j]
            d = b - a
            if d <= 0:
                continue
            if d > 512:
                continue
            if a + 2 * d in sset:
                diffs.add(d)
    return len(diffs)

def _greedy_for_differences(target_diffs, size=32, rng=None):
    """
    Greedily build a set S of `size` elements to realize as many target_diffs as possible.
    For each d in target_diffs, we want some a such that a, a+d, a+2d all in S.
    Strategy: start with some base points, then for each d, try to add points to complete a 3-AP.
    """
    if rng is None:
        rng = random.Random()
    
    S = set()
    # Start with a few spread-out points
    base_points = [1, 512, 256, 128, 384]
    for p in base_points:
        if len(S) < size:
            S.add(p)
    
    # Sort target_diffs by frequency/priority (smaller diffs first as they're easier)
    sorted_diffs = sorted(target_diffs, key=lambda d: d)
    
    # Multiple passes to fill in
    for _pass in range(20):
        if len(S) >= size:
            break
        for d in sorted_diffs:
            if len(S) >= size:
                break
            # Try to find or create a 3-AP with difference d
            # For each a in S, check if a+d and a+2d are in S
            # If not, try to add missing elements
            found = False
            for a in list(S):
                b = a + d
                c = a + 2 * d
                if c > 512:
                    continue
                missing = []
                if b not in S:
                    missing.append(b)
                if c not in S:
                    missing.append(c)
                if len(missing) <= 1:
                    # Can complete by adding at most 1 element
                    for m in missing:
                        if 1 <= m <= 512 and m not in S and len(S) < size:
                            S.add(m)
                            found = True
                            break
                    if found:
                        break
            if not found:
                # Try starting a new 3-AP
                a = rng.randint(1, max(1, 512 - 2 * d))
                b = a + d
                c = a + 2 * d
                to_add = [x for x in [a, b, c] if x not in S and 1 <= x <= 512]
                if to_add:
                    for x in to_add:
                        if len(S) < size:
                            S.add(x)
                        else:
                            break
    
    # Fill remaining slots with random elements
    while len(S) < size:
        x = rng.randint(1, 512)
        S.add(x)
    
    return list(S)[:size]

def _diverse_initializations(deadline, rng):
    """Generate multiple diverse initial sets."""
    inits = []
    
    # 1. Dense small numbers
    inits.append(list(range(1, 33)))
    
    # 2. Spread out
    inits.append([int(i * 512 / 31) + 1 for i in range(32)])
    
    # 3. 5-smooth numbers
    smooth5 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48, 50, 54, 60, 64, 72, 75, 80, 81, 90]
    if len(smooth5) >= 32:
        inits.append(list(smooth5[:32]))
    
    # 4. Geometric progression base
    geo = []
    for i in range(32):
        x = int(1 * (512 / 1) ** (i / 31))
        if x > 512:
            x = 512
        geo.append(x)
    # Deduplicate
    seen = set()
    geo_unique = []
    for x in geo:
        if x not in seen:
            seen.add(x)
            geo_unique.append(x)
    while len(geo_unique) < 32:
        x = rng.randint(1, 512)
        if x not in seen:
            seen.add(x)
            geo_unique.append(x)
    inits.append(geo_unique[:32])
    
    # 5. Difference-targeted: target small differences 1-30
    target_d = list(range(1, 31))
    s = _greedy_for_differences(target_d, 32, rng)
    inits.append(s)
    
    # 6. Difference-targeted: target differences 1-15 and 32-40
    target_d2 = list(range(1, 16)) + list(range(32, 41))
    s = _greedy_for_differences(target_d2, 32, rng)
    inits.append(s)
    
    # 7. Mixed scales: multiples of different bases
    mixed = []
    for base in [1, 2, 3, 4, 5]:
        for mult in range(1, 7):
            x = base * mult
            if x <= 512 and x not in mixed:
                mixed.append(x)
    while len(mixed) < 32:
        x = rng.randint(1, 512)
        if x not in mixed:
            mixed.append(x)
    inits.append(mixed[:32])
    
    # 8. Random
    inits.append(rng.sample(range(1, 513), 32))
    
    # 9. Two-scale: small dense + large sparse
    two_scale = list(range(1, 21))  # 1-20 dense
    for i in range(21, 32):
        x = 256 + (i - 20) * 16
        if x <= 512 and x not in two_scale:
            two_scale.append(x)
    while len(two_scale) < 32:
        x = rng.randint(256, 512)
        if x not in two_scale:
            two_scale.append(x)
    inits.append(two_scale[:32])
    
    return inits

def _local_search(s, deadline, rng, max_iter=3000):
    """Local search with 1-element replacement and 2-element swap moves."""
    best = list(s)
    best_score = _score_fast(best)
    n = len(best)
    best_set = set(best)
    
    stall_count = 0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Move type 1: replace one element
        for i in range(n):
            removed = best[i]
            best_set.remove(removed)
            
            # Generate candidates: random + neighbors + difference-aware
            candidates = set()
            # Random
            for _ in range(15):
                c = rng.randint(1, 512)
                candidates.add(c)
            # Neighbors of removed
            for delta in range(-15, 16):
                c = removed + delta
                if 1 <= c <= 512:
                    candidates.add(c)
            # Difference-aware: for each other element b, try c such that 
            # (removed, b, c) or (c, removed, b) or (b, c, removed) forms 3-AP
            for b in best_set:
                # c = 2*b - removed (so removed, b, c)
                c = 2 * b - removed
                if 1 <= c <= 512:
                    candidates.add(c)
                # c = 2*removed - b (so c, removed, b)
                c = 2 * removed - b
                if 1 <= c <= 512:
                    candidates.add(c)
                # c such that b, removed, c: c = 2*removed - b (same as above)
                # c such that removed, c, b: c = (removed + b) / 2
                if (removed + b) % 2 == 0:
                    c = (removed + b) // 2
                    if 1 <= c <= 512:
                        candidates.add(c)
            
            for c in candidates:
                if c in best_set:
                    continue
                best_set.add(c)
                new_s = list(best)
                new_s[i] = c
                ns = _score_fast(new_s)
                if ns > best_score:
                    best_score = ns
                    best = new_s
                    improved = True
                    break
                best_set.remove(c)
            
            if improved:
                break
            best_set.add(removed)
        
        if improved:
            best_set = set(best)
            stall_count = 0
            continue
        
        # Move type 2: swap two elements (remove one, add two new ones, remove one more)
        # Actually, let's do: remove element i, add new element, then remove element j, add new element
        # Simpler: try removing 2 elements and adding 2 new ones
        for i in range(n):
            for j in range(i + 1, n):
                r1 = best[i]
                r2 = best[j]
                best_set.remove(r1)
                best_set.remove(r2)
                
                # Try random pairs of new elements
                found = False
                for _ in range(8):
                    c1 = rng.randint(1, 512)
                    c2 = rng.randint(1, 512)
                    if c1 in best_set or c2 in best_set or c1 == c2:
                        continue
                    best_set.add(c1)
                    best_set.add(c2)
                    new_s = list(best)
                    new_s[i] = c1
                    new_s[j] = c2
                    ns = _score_fast(new_s)
                    if ns > best_score:
                        best_score = ns
                        best = new_s
                        improved = True
                        found = True
                        break
                    best_set.remove(c1)
                    best_set.remove(c2)
                
                if found:
                    break
                best_set.add(r1)
                best_set.add(r2)
            
            if improved:
                break
        
        if improved:
            best_set = set(best)
            stall_count = 0
        else:
            stall_count += 1
            if stall_count > 50:
                break
    
    return best, best_score

def _simulated_annealing(s, deadline, rng, max_iter=5000):
    """Simulated annealing with 1-element and 2-element moves."""
    best = list(s)
    best_score = _score_fast(best)
    current = list(best)
    current_score = best_score
    n = len(current)
    current_set = set(current)
    
    T = 5.0
    T_min = 0.05
    cooling = 0.9995
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        if T < T_min:
            break
        
        # Choose move type
        if rng.random() < 0.7:
            # 1-element move
            i = rng.randint(0, n - 1)
            removed = current[i]
            current_set.remove(removed)
            
            new_elem = rng.randint(1, 512)
            attempts = 0
            while new_elem in current_set and attempts < 50:
                new_elem = rng.randint(1, 512)
                attempts += 1
            
            if new_elem in current_set:
                current_set.add(removed)
                continue
            
            current_set.add(new_elem)
            new_s = list(current)
            new_s[i] = new_elem
            new_score = _score_fast(new_s)
            
            delta = new_score - current_score
            if delta > 0 or rng.random() < math.exp(delta / T):
                current = new_s
                current_score = new_score
                current_set = set(current)
                if current_score > best_score:
                    best_score = current_score
                    best = list(current)
            else:
                current_set.remove(new_elem)
                current_set.add(removed)
        else:
            # 2-element move
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                continue
            r1 = current[i]
            r2 = current[j]
            current_set.remove(r1)
            current_set.remove(r2)
            
            c1 = rng.randint(1, 512)
            c2 = rng.randint(1, 512)
            if c1 in current_set or c2 in current_set or c1 == c2:
                current_set.add(r1)
                current_set.add(r2)
                continue
            
            current_set.add(c1)
            current_set.add(c2)
            new_s = list(current)
            new_s[i] = c1
            new_s[j] = c2
            new_score = _score_fast(new_s)
            
            delta = new_score - current_score
            if delta > 0 or rng.random() < math.exp(delta / T):
                current = new_s
                current_score = new_score
                current_set = set(current)
                if current_score > best_score:
                    best_score = current_score
                    best = list(current)
            else:
                current_set.remove(c1)
                current_set.remove(c2)
                current_set.add(r1)
                current_set.add(r2)
        
        T *= cooling
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    rng = random.Random(42)
    
    # Generate diverse initializations
    inits = _diverse_initializations(deadline, rng)
    
    best = None
    best_score = -1
    
    # Score all initializations
    for s in inits:
        if len(set(s)) != 32:
            continue
        if not all(1 <= x <= 512 for x in s):
            continue
        sc = _score_fast(s)
        if sc > best_score:
            best_score = sc
            best = list(s)
    
    # Run local search from best
    if time.time() < deadline - 10:
        s, sc = _local_search(best, deadline - 10, rng, max_iter=2000)
        if sc > best_score:
            best_score = sc
            best = s
    
    # Run simulated annealing
    if time.time() < deadline - 5:
        s, sc = _simulated_annealing(best, deadline - 5, rng, max_iter=3000)
        if sc > best_score:
            best_score = sc
            best = s
    
    # More local search rounds with different random seeds
    for round_num in range(5):
        if time.time() > deadline - 3:
            break
        rng2 = random.Random(42 + round_num * 100)
        s, sc = _local_search(best, deadline - 2, rng2, max_iter=1500)
        if sc > best_score:
            best_score = sc
            best = s
    
    # Final: try difference-targeted approaches with optimization
    if time.time() < deadline - 2:
        for target_start in range(1, 10):
            if time.time() > deadline - 1:
                break
            target_d = list(range(target_start, target_start + 25))
            s = _greedy_for_differences(target_d, 32, rng)
            sc = _score_fast(s)
            if sc > best_score - 3:
                s2, sc2 = _local_search(s, deadline - 1, rng, max_iter=500)
                if sc2 > best_score:
                    best_score = sc2
                    best = s2
    
    # Final validation
    assert len(best) == 32
    assert len(set(best)) == 32
    assert all(1 <= x <= 512 for x in best)
    
    return best

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
