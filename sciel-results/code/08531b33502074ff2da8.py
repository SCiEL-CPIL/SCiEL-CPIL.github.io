# EVOLVE-BLOCK-START
import random
import time
import itertools

def _is_sidon(subset):
    """Check if a subset has all unordered pair sums unique INCLUDING diagonals.
    Diagonals means a+a for all a in subset, plus a+b for a<b.
    So we need all sums i+j for i,j in subset (with i<=j) to be unique.
    """
    n = len(subset)
    if n <= 1:
        return True
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(ambient):
    """Find the largest Sidon subset of ambient set."""
    n = len(ambient)
    # Try from largest possible size down
    for k in range(n, 0, -1):
        # Check all combinations of size k
        for combo in itertools.combinations(ambient, k):
            if _is_sidon(list(combo)):
                return k
    return 1

def construct():
    start_time = time.time()
    deadline = start_time + 70  # Safety margin before 90s kill
    
    # Start with the current design
    best_set = None
    best_size = None
    
    # Initial candidate from current program
    s = []
    for k in range(6):
        s.append(10*k)
        s.append(10*k + 1)
    best_set = s
    best_size = _max_sidon_size(s)
    
    # Try to improve by searching for sets with smaller max Sidon size
    # Strategy: use structured patterns that create many additive relations
    
    # Pattern 1: Arithmetic progressions and near-APs
    # Pattern 2: Clusters that create many 3-term APs
    # Pattern 3: Random search with local optimization
    
    random.seed(42)
    
    # Try various structured patterns
    patterns = []
    
    # AP-based patterns
    for step in range(1, 11):
        if 11 * step > 128:
            continue
        ap = list(range(0, 12 * step, step))
        if len(ap) == 12 and max(ap) <= 128:
            patterns.append(ap)
    
    # Pair patterns with varying offsets
    for base in range(0, 20):
        for offset in range(1, 5):
            for gap in range(5, 15):
                s = []
                for k in range(6):
                    s.append(base + gap * k)
                    s.append(base + gap * k + offset)
                if len(set(s)) == 12 and all(0 <= x <= 128 for x in s):
                    patterns.append(s)
    
    # Evaluate patterns
    for pat in patterns:
        if time.time() > deadline:
            break
        sz = _max_sidon_size(pat)
        if sz < best_size:
            best_size = sz
            best_set = pat
    
    # Local search: perturb the best set
    if best_size > 2:  # If we haven't found size 2, try to improve
        current = list(best_set)
        for iteration in range(5000):
            if time.time() > deadline:
                break
            
            # Random perturbation
            idx = random.randint(0, 11)
            old_val = current[idx]
            new_val = random.randint(0, 128)
            
            if new_val == old_val or new_val in current:
                continue
            
            current[idx] = new_val
            sz = _max_sidon_size(current)
            
            if sz < best_size:
                best_size = sz
                best_set = list(current)
            else:
                current[idx] = old_val
    
    # Final verification
    if len(set(best_set)) != 12:
        return list(range(12))
    if any(x < 0 or x > 128 for x in best_set):
        return list(range(12))
    
    return best_set
# EVOLVE-BLOCK-END
