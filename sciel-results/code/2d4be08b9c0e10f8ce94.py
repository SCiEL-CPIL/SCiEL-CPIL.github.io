import itertools
import time
import random
import sys

def is_sidon_4(a, b, c, d):
    """Check if 4 sorted elements form a Sidon set (all pair sums including diagonals unique)."""
    # Pair sums: a+a, a+b, a+c, a+d, b+b, b+c, b+d, c+c, c+d, d+d
    sums = (2*a, a+b, a+c, a+d, 2*b, b+c, b+d, 2*c, c+d, 2*d)
    return len(set(sums)) == 10

def max_sidon_size(arr):
    """Return the size of the largest Sidon subset of arr (12 elements)."""
    n = len(arr)
    # Precompute sums and bitmasks for 4-sets and 5-sets
    # A 4-set is Sidon if all 10 sums are unique.
    # We can check this quickly.
    
    # Check for 5-element Sidon subset
    # There are C(12,5) = 792 combinations.
    # For each, check 15 sums.
    # 792 * 15 is small.
    for combo in itertools.combinations(arr, 5):
        sums = set()
        ok = True
        for i in range(5):
            for j in range(i, 5):
                s = combo[i] + combo[j]
                if s in sums:
                    ok = False
                    break
                sums.add(s)
            if not ok:
                break
        if ok:
            return 5
    
    # Check for 4-element Sidon subset
    # There are C(12,4) = 495 combinations.
    for combo in itertools.combinations(arr, 4):
        if is_sidon_4(combo[0], combo[1], combo[2], combo[3]):
            return 4
            
    return 3

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin for 90s limit
    
    best_set = None
    best_val = 12
    
    # Strategy: Search for a set with max Sidon size = 3.
    # This means NO 4-element subset is Sidon.
    
    # 1. Structured Candidates: "Dense" sets
    # Sets with many close pairs tend to have many sum collisions.
    # Example: {0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23}
    # Let's generate sets with small gaps.
    
    candidates = set()
    
    # Pattern: Blocks of 3 or 4 with small gaps
    for gap in range(1, 10):
        for start in range(0, 129 - 3*gap):
            s = set()
            pos = start
            # Try 4 blocks of 3
            for _ in range(4):
                s.add(pos)
                s.add(pos+1)
                s.add(pos+2)
                pos += gap + 3
            if len(s) == 12 and max(s) < 129:
                candidates.add(tuple(sorted(s)))
                
            # Try 3 blocks of 4
            pos = start
            for _ in range(3):
                s.add(pos)
                s.add(pos+1)
                s.add(pos+2)
                s.add(pos+3)
                pos += gap + 4
            if len(s) == 12 and max(s) < 129:
                candidates.add(tuple(sorted(s)))

            # Try 6 blocks of 2
            pos = start
            for _ in range(6):
                s.add(pos)
                s.add(pos+1)
                pos += gap + 2
            if len(s) == 12 and max(s) < 129:
                candidates.add(tuple(sorted(s)))
    
    # Pattern: Arithmetic Progressions (often bad, but include for completeness)
    for d in range(1, 12):
        for start in range(0, 129 - 11*d):
            s = tuple(range(start, start + 12*d, d))
            candidates.add(s)
            
    # Pattern: "Random" but clustered
    # Generate sets by picking 6 pairs (x, x+1) or (x, x+2)
    random.seed(42)
    for _ in range(500):
        s = set()
        while len(s) < 12:
            base = random.randint(0, 120)
            diff = random.choice([1, 1, 2, 3])
            s.add(base)
            if base + diff < 129 and base + diff not in s:
                s.add(base + diff)
            else:
                s.add(base + diff + 1) # ensure distinct
        if len(s) == 12 and max(s) < 129:
            candidates.add(tuple(sorted(s)))
            
    # Evaluate structured candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        val = max_sidon_size(list(cand))
        if val < best_val:
            best_val = val
            best_set = list(cand)
            if best_val == 3:
                return best_set
    
    # 2. Random Search
    # With the faster max_sidon_size, we can do more random samples.
    for _ in range(20000):
        if time.time() > deadline:
            break
        s = tuple(sorted(random.sample(range(129), 12)))
        val = max_sidon_size(list(s))
        if val < best_val:
            best_val = val
            best_set = list(s)
            if best_val == 3:
                return best_set
    
    # 3. Local Search
    # If we haven't found 3, try to improve the best found.
    if best_set is not None and best_val >= 4:
        current = best_set[:]
        current_val = best_val
        
        # Helper to get all 4-Sidon subsets in a set
        def get_sidon_4s(arr):
            sidon_4s = []
            for combo in itertools.combinations(arr, 4):
                if is_sidon_4(combo[0], combo[1], combo[2], combo[3]):
                    sidon_4s.append(combo)
            return sidon_4s
            
        for iteration in range(100000):
            if time.time() > deadline:
                break
            if current_val == 3:
                return current
                
            # Identify a "bad" 4-subset (Sidon) to break
            sidon_4s = get_sidon_4s(current)
            if not sidon_4s:
                return current # Should be 3
            
            # Pick a random Sidon 4-subset
            target_combo = random.choice(sidon_4s)
            
            # Try to replace one of the elements in target_combo with a new value
            # that breaks the Sidon property for this combo (and hopefully others)
            # Or just replace a random element in the set.
            
            idx = random.randint(0, 11)
            old_val = current[idx]
            
            # Try values close to old_val or random
            for _ in range(20):
                new_val = random.randint(0, 128)
                if new_val in current:
                    continue
                
                trial = current[:]
                trial[idx] = new_val
                trial_val = max_sidon_size(trial)
                
                if trial_val < current_val:
                    current = trial
                    current_val = trial_val
                    break
                elif trial_val == current_val and random.random() < 0.05:
                    current = trial
                    current_val = trial_val
                    break
    
    if best_set is not None:
        return best_set
    return list(range(12))
