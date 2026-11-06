import itertools
import time
import random

def _is_sidon(subset):
    """Check if a subset is Sidon (all unordered pair sums unique)."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(candidate):
    """Find the size of the largest Sidon subset."""
    n = len(candidate)
    # Check from largest possible size downwards
    for k in range(n, 0, -1):
        for subset in itertools.combinations(candidate, k):
            if _is_sidon(subset):
                return k
    return 1

def _find_sidon_subset(candidate, k):
    """Find one Sidon subset of size k."""
    for subset in itertools.combinations(candidate, k):
        if _is_sidon(subset):
            return subset
    return None

def construct():
    start_time = time.time()
    timeout = 55  # Safety margin
    
    # Start with a known set
    candidate = list(range(12))
    best_candidate = candidate.copy()
    best_score = _max_sidon_size(best_candidate)
    
    # If we already have size 3, we are likely done (min possible for 12 ints is likely 3)
    # But let's try to see if we can get 2? Unlikely.
    
    # Local search
    # Strategy: Find a bad Sidon subset, break it by swapping an element.
    
    while time.time() - start_time < timeout:
        current_score = _max_sidon_size(candidate)
        if current_score < best_score:
            best_score = current_score
            best_candidate = candidate.copy()
            
        if best_score <= 3:
            # 3 is likely the minimum. 2 is very hard/impossible for 12 distinct ints.
            # We can break early if we are confident.
            # But let's run a bit longer to ensure stability or try to find 2?
            # Actually, if we find 3, it's a huge improvement over 4.
            break
        
        # Find a Sidon subset of size current_score
        bad_subset = _find_sidon_subset(candidate, current_score)
        if not bad_subset:
            break
            
        # Try to break this subset
        improved = False
        # Pick a random element from the bad subset to replace
        idx_to_replace = random.choice([i for i, x in enumerate(candidate) if x in bad_subset])
        old_val = candidate[idx_to_replace]
        
        # Try random new values
        for _ in range(50):
            new_val = random.randint(0, 128)
            if new_val in candidate:
                continue
            
            candidate[idx_to_replace] = new_val
            new_score = _max_sidon_size(candidate)
            
            if new_score < current_score:
                improved = True
                break
            else:
                candidate[idx_to_replace] = old_val
        
        if not improved:
            # If no simple swap worked, try a random perturbation
            # or accept current and try different subsets
            # To avoid getting stuck, occasionally do a random restart or larger move
            if random.random() < 0.1:
                # Random restart with some structure
                candidate = list(range(12))
                # Perturb slightly
                for _ in range(3):
                    i = random.randint(0, 11)
                    candidate[i] = random.randint(0, 128)
                    # Ensure distinctness
                    while len(set(candidate)) < 12:
                        i = random.randint(0, 11)
                        candidate[i] = random.randint(0, 128)
            
    return best_candidate
