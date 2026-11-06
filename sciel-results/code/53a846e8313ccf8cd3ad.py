"""Combined program: bitset-based fast scoring + simulated annealing + local search polish."""
import time
import random

def _count_progression_diffs_bitset(S):
    """
    Count distinct d > 0 such that exists a with a, a+d, a+2d in S.
    Uses a boolean presence array for O(1) lookups.
    S is an iterable of integers in 1..512.
    """
    # Build presence array (index 0 unused, 1..512)
    present = [False] * 513
    for x in S:
        if 1 <= x <= 512:
            present[x] = True
    
    diffs = set()
    S_list = list(S)
    n = len(S_list)
    
    # For each pair (a, b) with a < b, check if a+2*(b-a) is present
    # But we want distinct d = b - a.
    # To avoid O(n^2) with high constant, we iterate over possible d values?
    # No, d can be up to 511.
    # Better: iterate over all a in S, and all d in 1..255 (since a+2d <= 512 implies d <= 255 if a>=1? No, a can be small).
    # Max d is such that a + 2d <= 512. Since min a=1, max d = (512-1)/2 = 255.5 -> 255.
    # Wait, if a=1, d=255, a+2d = 511. If a=2, d=255, a+2d=512.
    # So d is in 1..255.
    # We can iterate d from 1 to 255, and check if there exists a such that a, a+d, a+2d are in S.
    # This is O(255 * 512) = ~130k operations, which is very fast.
    
    for d in range(1, 256):
        found = False
        # a can range from 1 to 512 - 2d
        max_a = 512 - 2 * d
        if max_a < 1:
            continue
        for a in range(1, max_a + 1):
            if present[a] and present[a + d] and present[a + 2 * d]:
                diffs.add(d)
                found = True
                break
        # Early termination for this d if found
    
    return len(diffs)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    random.seed(42)
    N = 32
    MAXV = 512
    
    # Initial set: random sample
    current = set(random.sample(range(1, MAXV + 1), N))
    current_score = _count_progression_diffs_bitset(current)
    best = set(current)
    best_score = current_score
    
    # Simulated Annealing
    T = 50.0
    T_min = 0.1
    cooling = 0.999
    
    # Pre-compute list for random choice efficiency
    all_values = list(range(1, MAXV + 1))
    
    while time.time() < deadline - 20.0 and T > T_min:
        # Generate a move: pick a random element to remove, pick a random new element to add
        elem = random.choice(list(current))
        new_elem = random.choice(all_values)
        
        if new_elem == elem or new_elem in current:
            continue
            
        # Apply move
        current.remove(elem)
        current.add(new_elem)
        
        new_score = _count_progression_diffs_bitset(current)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < pow(2.718281828, delta / max(T, 1e-9)):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best = set(current)
        else:
            # Reject move
            current.remove(new_elem)
            current.add(elem)
            
        T *= cooling
        
        # Optional: small probability of a larger jump (swap two elements)
        if random.random() < 0.1:
            # Swap two random elements in current with two random new elements
            e1, e2 = random.sample(list(current), 2)
            n1, n2 = random.sample(all_values, 2)
            if n1 not in current and n2 not in current and n1 != n2:
                current.discard(e1)
                current.discard(e2)
                current.add(n1)
                current.add(n2)
                new_score = _count_progression_diffs_bitset(current)
                delta = new_score - current_score
                if delta > 0 or random.random() < pow(2.718281828, delta / max(T, 1e-9)):
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best = set(current)
                else:
                    current.remove(n1)
                    current.remove(n2)
                    current.add(e1)
                    current.add(e2)

    # Local Search Polish on best
    # Try to improve by swapping one element with another
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        blist = list(best)
        # Try each element in best
        for elem in blist:
            if time.time() > deadline - 2.0:
                break
            # Try replacing elem with a random candidate
            # To make it more systematic, we can try candidates near elem or random
            # But random is fine if we do many iterations
            for _ in range(10):  # Try 10 random candidates per element
                if time.time() > deadline - 2.0:
                    break
                cand = random.choice(all_values)
                if cand in best or cand == elem:
                    continue
                test = set(best)
                test.remove(elem)
                test.add(cand)
                sc = _count_progression_diffs_bitset(test)
                if sc > best_score:
                    best = test
                    best_score = sc
                    improved = True
                    break  # Restart loop with new best
    
    return list(best)
