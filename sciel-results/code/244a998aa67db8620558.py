import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    random.seed(42)
    
    n = 12
    m = 40
    all_vertices = set(range(n))
    
    # Helper: count comparable pairs
    def count_comparable(family):
        count = 0
        for i in range(m):
            si = set(family[i])
            for j in range(i+1, m):
                sj = set(family[j])
                if si < sj or sj < si:
                    count += 1
        return count
    
    # Helper: check if family is valid (distinct sets, all subsets of vertices)
    def is_valid(family):
        seen = set()
        for s in family:
            if len(s) != len(set(s)):
                return False
            if any(v < 0 or v >= n for v in s):
                return False
            if len(s) > n:
                return False
            key = frozenset(s)
            if key in seen:
                return False
            seen.add(key)
        return len(family) == m
    
    # Generate initial family: pyramid structure
    # Use many small sets and many large sets
    def generate_initial():
        family = []
        # Try different size distributions
        # Strategy: use sizes that allow many subset relations
        # e.g., 10 sets of size 2, 10 sets of size 3, 10 sets of size 8, 10 sets of size 9
        # Or: 5 of size 1, 5 of size 2, 5 of size 3, 5 of size 4, 5 of size 8, 5 of size 9, 5 of size 10, 5 of size 11
        
        # Let's try: sizes [1,2,3,4,5,6,7,8,9,10,11] with some repeated
        # Better: use a balanced approach
        # Start with: 10 sets of size 3, 10 sets of size 4, 10 sets of size 8, 10 sets of size 9
        sizes = [3]*10 + [4]*10 + [8]*10 + [9]*10
        
        # Generate random sets of each size
        for size in sizes:
            s = set(random.sample(range(n), size))
            family.append(list(s))
        
        # Ensure distinctness
        seen = set()
        for i in range(len(family)):
            key = frozenset(family[i])
            if key in seen:
                # Regenerate
                attempts = 0
                while attempts < 100:
                    new_s = set(random.sample(range(n), len(family[i])))
                    if frozenset(new_s) not in seen:
                        family[i] = list(new_s)
                        break
                    attempts += 1
            seen.add(frozenset(family[i]))
        
        return family
    
    # Local search: try replacing or swapping sets to improve score
    def local_search(family, time_limit):
        family = [list(s) for s in family]
        current_score = count_comparable(family)
        
        while time.time() < time_limit:
            improved = False
            for i in range(m):
                if time.time() >= time_limit:
                    break
                # Try replacing family[i] with a random set of various sizes
                for size in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
                    if time.time() >= time_limit:
                        break
                    new_s = set(random.sample(range(n), size))
                    # Check distinctness
                    if any(frozenset(new_s) == frozenset(family[j]) for j in range(m) if j != i):
                        continue
                    
                    old_s = family[i]
                    family[i] = list(new_s)
                    new_score = count_comparable(family)
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                        break
                    else:
                        family[i] = old_s
            
            # Also try swapping two sets
            if not improved:
                i, j = random.sample(range(m), 2)
                family[i], family[j] = family[j], family[i]
                new_score = count_comparable(family)
                if new_score > current_score:
                    current_score = new_score
                else:
                    family[i], family[j] = family[j], family[i]
            
            if not improved:
                # Try more aggressive: replace with subset/superset of existing
                if time.time() >= time_limit:
                    break
                i = random.randint(0, m-1)
                base = set(family[i])
                # Try a random subset or superset
                if random.random() < 0.5 and len(base) > 1:
                    k = random.randint(1, len(base)-1)
                    new_s = set(random.sample(list(base), k))
                else:
                    # Try superset: add random elements
                    available = all_vertices - base
                    if available:
                        k = random.randint(1, min(len(available), n - len(base)))
                        new_s = base | set(random.sample(list(available), k))
                    else:
                        continue
                
                if any(frozenset(new_s) == frozenset(family[j]) for j in range(m) if j != i):
                    continue
                
                old_s = family[i]
                family[i] = list(new_s)
                new_score = count_comparable(family)
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                else:
                    family[i] = old_s
        
        return family, current_score
    
    # Run multiple restarts
    best_family = None
    best_score = -1
    
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        family = generate_initial()
        if not is_valid(family):
            continue
        
        time_limit = deadline - 3  # leave margin
        family, score = local_search(family, time_limit)
        
        if score > best_score:
            best_score = score
            best_family = family
    
    if best_family is None:
        # Fallback: return initial
        family = generate_initial()
        best_family = family
    
    # Ensure proper types
    result = []
    for s in best_family:
        result.append([int(v) for v in s])
    
    return result
