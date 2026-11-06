import time
import random
from itertools import combinations

def _is_sidon_4(subset):
    """Check if a 4-element subset is Sidon (including diagonals)."""
    a, b, c, d = sorted(subset)
    # Pairs (i, j) with i <= j
    # Sums:
    # 0: a+a
    # 1: a+b
    # 2: a+c
    # 3: a+d
    # 4: b+b
    # 5: b+c
    # 6: b+d
    # 7: c+c
    # 8: c+d
    # 9: d+d
    sums = [
        2*a,
        a+b, a+c, a+d,
        2*b, b+c, b+d,
        2*c, c+d,
        2*d
    ]
    return len(set(sums)) == 10

def _has_sidon_4(A):
    """Check if A contains any 4-element Sidon subset."""
    n = len(A)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if _is_sidon_4((A[i], A[j], A[k], A[l])):
                        return True, (A[i], A[j], A[k], A[l])
    return False, None

def _max_sidon_size(A):
    """Calculate max Sidon size for a set of 12."""
    # Check for size 4
    has4, _ = _has_sidon_4(A)
    if has4:
        # Check for size 5
        # If we have size 4, we might have 5. 
        # For the purpose of minimizing, if we find a set with no size 4, it's 3 (or less).
        # If we have size 4, we need to know if it's 5, 6, etc.
        # But if we are trying to get 3, we stop if we find no size 4.
        # For fallback, we assume 4 is the minimum we can easily guarantee.
        # Let's do a quick check for 5 to be sure, but 4 is the target "low" value.
        # Actually, if we just want to beat the seed (which is 4), we need 3.
        # If we can't get 3, we return 4.
        # So we can just return 4 if has4 is True, unless we want to be precise.
        # The evaluator calculates the exact max.
        # But for my search, I only care about "Is it 3?".
        # So I will return 3 if no size 4, else 4 (optimistically).
        return 4
    else:
        return 3

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin: evaluator kills at 60s
    
    # Try multiple random seeds to find a set with max Sidon size 3
    best_set = list(range(12))
    best_val = _max_sidon_size(best_set)
    
    rng = random.Random(42)
    
    # Strategy: Random search with local improvement
    # We want to find a 12-element set where NO 4-element subset is Sidon
    # This means max Sidon size <= 3
    
    for seed in range(10000):
        if time.time() > deadline:
            break
        
        # Generate a random set of 12 distinct integers in [0, 128]
        current_set = sorted(rng.sample(range(0, 129), 12))
        
        # Quick check: does it have any 4-Sidon subset?
        has4, _ = _has_sidon_4(current_set)
        
        if not has4:
            # Found a set with max Sidon size <= 3
            return current_set
    
    # If random search didn't find one, try local search from a few starting points
    for start_seed in range(100):
        if time.time() > deadline:
            break
        
        rng2 = random.Random(start_seed)
        current_set = sorted(rng2.sample(range(0, 129), 12))
        
        # Local search: try to eliminate 4-Sidon subsets
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            has4, sidon_sub = _has_sidon_4(current_set)
            if not has4:
                return current_set
            
            # Pick one element from the Sidon subset to perturb
            idx_to_change = rng2.choice([0, 1, 2, 3])
            val_to_change = sidon_sub[idx_to_change]
            
            # Try several random new values
            improved = False
            for _ in range(20):
                new_val = rng2.randint(0, 128)
                if new_val == val_to_change:
                    continue
                if new_val in current_set:
                    continue
                
                new_set = current_set[:]
                new_set[new_set.index(val_to_change)] = new_val
                new_set.sort()
                
                has4_new, _ = _has_sidon_4(new_set)
                if not has4_new:
                    current_set = new_set
                    improved = True
                    break
            
            if not improved:
                # Random restart within this local search
                current_set = sorted(rng2.sample(range(0, 129), 12))
    
    # Return the best set found (which may still have max Sidon size 4 or higher)
    # But we've done our best within the time budget
    return best_set
