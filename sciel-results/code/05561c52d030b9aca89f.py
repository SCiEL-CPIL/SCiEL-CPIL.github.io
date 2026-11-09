import time
import random

def _is_sidon(set_a):
    """Check if all positive differences are distinct."""
    n = len(set_a)
    seen = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = set_a[j] - set_a[i]
            if d in seen:
                return False
            seen.add(d)
    return True

def _has_all_diffs(set_a, max_d=30):
    """Check if all d in 1..max_d occur as differences."""
    diffs = set()
    n = len(set_a)
    for i in range(n):
        for j in range(i + 1, n):
            d = set_a[j] - set_a[i]
            diffs.add(d)
    return all(d in diffs for d in range(1, max_d + 1))

def _score(set_a):
    """Score: minimize max element. Return float('inf') if invalid."""
    if not _is_sidon(set_a):
        return float('inf')
    if not _has_all_diffs(set_a):
        return float('inf')
    return max(set_a)

def _local_search(start_set, max_iters=2000, seed=0):
    """Local search: try replacing elements to improve max element."""
    rng = random.Random(seed)
    best = list(start_set)
    best_score = _score(best)
    n = len(best)
    
    for it in range(max_iters):
        improved = False
        # Pick a random element to replace
        idx = rng.randint(0, n - 1)
        if best[idx] == 0:
            continue # Cannot replace 0
            
        old_val = best[idx]
        # Try to replace with a smaller value
        max_val = max(best)
        # Search range: 1 to max_val - 1
        # We want to reduce the max, so focus on the largest elements
        # Or just random replacement within bounds
        
        for _ in range(50):
            new_val = rng.randint(1, max_val)
            if new_val in best:
                continue
            trial = best[:]
            trial[idx] = new_val
            trial.sort()
            if _is_sidon(trial) and _has_all_diffs(trial):
                new_score = max(trial)
                if new_score < best_score:
                    best = trial
                    best_score = new_score
                    improved = True
                    break
        
        if not improved:
            # If no improvement, try to swap two elements
            i, j = rng.sample(range(n), 2)
            if best[i] == 0 or best[j] == 0:
                continue
            trial = best[:]
            trial[i], trial[j] = trial[j], trial[i]
            trial.sort()
            if _is_sidon(trial) and _has_all_diffs(trial):
                new_score = max(trial)
                if new_score < best_score:
                    best = trial
                    best_score = new_score
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_set = None
    best_score = float('inf')
    
    # Try different sizes n
    for n in range(9, 13):
        if time.time() > deadline:
            break
            
        # Run multiple random restarts for this size
        for restart in range(200):
            if time.time() > deadline:
                break
                
            # Generate a random set of size n containing 0
            # Elements should be small to have a chance of small max
            # Max element likely < 100
            max_bound = 100
            a = {0}
            while len(a) < n:
                a.add(random.randint(1, max_bound))
            a = sorted(a)
            
            # Local search
            candidate = _local_search(a, max_iters=1000, seed=restart)
            s = _score(candidate)
            
            if s < best_score:
                best_score = s
                best_set = candidate
                
            # Early termination if we found a very good solution
            # Theoretical lower bound for max element?
            # If max element is ~30-40, it's good.
            if best_score <= 35:
                return best_set

    # If no valid set found, return a known valid set (fallback)
    # A known Sidon set covering 1..30:
    # Let's try to construct one quickly if search fails
    if best_set is None or best_score == float('inf'):
        # Fallback: Greedy construction
        a = [0]
        for d in range(1, 31):
            if d not in [y - x for i, x in enumerate(a) for y in a[i+1:]]:
                # Add element to create difference d
                # This is not guaranteed to be Sidon, but let's try a known one
                pass
        
        # Hardcoded known solution if search fails completely
        # Example: {0, 1, 4, 14, 16} is too small.
        # Let's use a larger set.
        # {0, 1, 4, 10, 12, 17, 21, 25, 28} ?
        # Let's just return a large valid set to be safe, though search should work.
        # A valid Sidon set: 0, 1, 4, 9, 16, 25, 36, 49, 64 (squares)
        # Differences: 1, 3, 5, 7, 9, ...
        # Squares are Sidon.
        # Does it cover 1..30?
        # Diffs of {0,1,4,9,16,25,36,49,64}:
        # 1, 3, 5, 7, 9, 11, 13, 15
        # 4-1=3, 9-4=5, 16-9=7, 25-16=9, 36-25=11, 49-36=13, 64-49=15
        # 4, 8, 12, 16, 20, 24, 28
        # 9, 15, 21, 27
        # 16, 24, 32
        # 25, 35
        # 36
        # 49
        # 64
        # Missing: 2, 6, 10, 14, 18, 22, 26, 30
        # So squares don't work.
        
        # Let's rely on the search.
        # If search fails, return a large set that is definitely Sidon and covers 1..30.
        # We can construct one by adding elements greedily.
        a = [0]
        used_diffs = set()
        for d in range(1, 31):
            if d in used_diffs:
                continue
            found = False
            # Try to add x such that x - y = d for some y in a, or y - x = d
            for y in a:
                x = y + d
                if x in a:
                    continue
                # Check if adding x breaks Sidon
                new_a = sorted(a + [x])
                if _is_sidon(new_a):
                    a = new_a
                    # Update used_diffs
                    for i in range(len(a)):
                        for j in range(i+1, len(a)):
                            used_diffs.add(a[j] - a[i])
                    found = True
                    break
            if not found:
                # Try x = d - y
                for y in a:
                    x = d - y
                    if x < 0 or x in a:
                        continue
                    new_a = sorted(a + [x])
                    if _is_sidon(new_a):
                        a = new_a
                        for i in range(len(a)):
                            for j in range(i+1, len(a)):
                                used_diffs.add(a[j] - a[i])
                        found = True
                        break
            if not found:
                # Add large element to create d
                # x = max(a) + d + 1
                x = max(a) + d + 1
                new_a = sorted(a + [x])
                while not _is_sidon(new_a):
                    x += 1
                    new_a = sorted(a + [x])
                a = new_a
                for i in range(len(a)):
                    for j in range(i+1, len(a)):
                        used_diffs.add(a[j] - a[i])
        
        best_set = a
        best_score = max(a)

    return best_set

# EVOLVE-BLOCK-END
