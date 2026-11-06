"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def _is_sidon_fast(subset):
    """
    Fast check if a subset has all unordered pair sums unique (including diagonals).
    Uses early termination.
    """
    sums = set()
    m = len(subset)
    for i in range(m):
        for j in range(i, m):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _exact_max_sidon_size(A):
    """
    Exhaustively computes the largest subset with all unordered pair sums unique.
    N is small (12), so this is fast. Uses early termination in the outer loop.
    """
    A_sorted = sorted(A)
    n = len(A_sorted)
    
    # Check subsets from largest size to smallest
    for k in range(n, 0, -1):
        for comb in combinations(A_sorted, k):
            if _is_sidon_fast(comb):
                return k
    return 0

def _find_max_sidon_subset(A):
    """
    Finds one maximal Sidon subset of the largest size.
    Returns the subset as a list.
    """
    A_sorted = sorted(A)
    n = len(A_sorted)
    for k in range(n, 0, -1):
        for comb in combinations(A_sorted, k):
            if _is_sidon_fast(comb):
                return list(comb)
    return []

def _evaluate_set(A):
    return _exact_max_sidon_size(A)

def _generate_candidates():
    """Generate diverse candidate sets likely to have small max Sidon subsets."""
    candidates = []
    
    # APs with small steps (dense arithmetic progressions)
    for d in range(1, 15):
        if 11 * d <= 128:
            candidates.append(list(range(0, 12 * d, d)))
    
    # Dense clusters at various offsets
    for offset in range(0, 100, 5):
        base = offset
        if base + 11 <= 128:
            candidates.append(list(range(base, base + 12)))
    
    # Two-cluster sets
    for gap in range(3, 20):
        for cluster_size in range(4, 9):
            if cluster_size * 2 == 12:
                c1 = list(range(0, cluster_size))
                c2 = list(range(gap, gap + cluster_size))
                if max(c2) <= 128:
                    candidates.append(sorted(c1 + c2))
    
    # Three-cluster sets
    for g1 in range(2, 10):
        for g2 in range(g1 + 2, 15):
            c1 = list(range(0, 4))
            c2 = list(range(g1, g1 + 4))
            c3 = list(range(g1 + g2, g1 + g2 + 4))
            if max(c3) <= 128:
                candidates.append(sorted(c1 + c2 + c3))
    
    # Structured dense sets with patterns
    candidates.append(sorted([0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23]))
    candidates.append(sorted([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]))
    candidates.append(sorted([0, 1, 2, 3, 4, 5, 20, 21, 22, 23, 24, 25]))
    candidates.append(sorted([0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16]))
    candidates.append(sorted([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]))
    candidates.append(sorted([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))
    
    # Random dense sets (small range) - multiple seeds for diversity
    for seed in range(42, 102):
        random.seed(seed)
        candidates.append(sorted(random.sample(range(25), 12)))
        
    # Random medium density
    for seed in range(102, 162):
        random.seed(seed)
        candidates.append(sorted(random.sample(range(50), 12)))
    
    # Random with bias toward small values
    for seed in range(162, 222):
        random.seed(seed)
        # Sample from a distribution biased toward small numbers
        # Ensure we get 12 distinct integers by retrying or using set-based sampling
        while True:
            vals = set()
            while len(vals) < 12:
                if random.random() < 0.7:
                    vals.add(random.randint(0, 15))
                else:
                    vals.add(random.randint(0, 40))
            if max(vals) <= 128:
                candidates.append(sorted(vals))
                break
    
    # Random geometric-like spacing
    for seed in range(222, 282):
        random.seed(seed)
        vals = [0]
        for i in range(1, 12):
            step = random.randint(1, 5)
            vals.append(vals[-1] + step)
        if max(vals) <= 128:
            candidates.append(sorted(vals))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    return unique_candidates

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin (90s limit, leave buffer)
    
    candidates = _generate_candidates()
    
    best_set = None
    best_val = 12
    current_set = None
    current_val = 12
    
    # Evaluate initial candidates
    for c in candidates:
        if time.time() > deadline:
            break
        val = _evaluate_set(c)
        if val < best_val:
            best_val = val
            best_set = c[:]
            current_set = c[:]
            current_val = val
        if best_val <= 3:
            break
            
    if best_set is None:
        best_set = list(range(12))
        current_set = best_set[:]
        current_val = _evaluate_set(current_set)
        best_val = current_val

    # Local Search
    if best_val > 3:
        random.seed(123)
        # Pre-compute all valid elements not in current set? 
        # It's easier to just pick random from 0-128 and check membership.
        
        while time.time() < deadline:
            if current_val <= 3:
                break
            
            # Find a maximal Sidon subset to target
            max_sub = _find_max_sidon_subset(current_set)
            if not max_sub:
                break
            
            # Try to replace one element from max_sub with a new random element
            # This is expected to break the Sidon property of max_sub
            success = False
            for _ in range(10): # Try a few replacements per iteration
                if time.time() > deadline:
                    break
                
                idx_to_replace = random.choice([current_set.index(x) for x in max_sub])
                new_elem = random.randint(0, 128)
                
                if new_elem in current_set:
                    continue
                
                old_elem = current_set[idx_to_replace]
                current_set[idx_to_replace] = new_elem
                current_set.sort()
                
                new_val = _evaluate_set(current_set)
                
                if new_val < current_val:
                    current_val = new_val
                    if current_val < best_val:
                        best_val = current_val
                        best_set = current_set[:]
                    success = True
                    # Restart the max_sub search for the next iteration
                    break 
                elif new_val == current_val:
                    # Accept with some probability to escape local minima
                    if random.random() < 0.5:
                        success = True
                        break
                    else:
                        # Revert
                        current_set[idx_to_replace] = old_elem
                        current_set.sort()
                else:
                    # Revert
                    current_set[idx_to_replace] = old_elem
                    current_set.sort()
            
            if success:
                continue
                
            # If no improvement or acceptance, occasionally restart or random kick
            if random.random() < 0.1:
                # Random kick: replace a random element with a random new one
                idx = random.randint(0, 11)
                old_elem = current_set[idx]
                new_elem = random.randint(0, 128)
                if new_elem not in current_set:
                    current_set[idx] = new_elem
                    current_set.sort()
                    new_val = _evaluate_set(current_set)
                    if new_val <= current_val:
                        current_val = new_val
                        if current_val < best_val:
                            best_val = current_val
                            best_set = current_set[:]
                    else:
                        current_set[idx] = old_elem
                        current_set.sort()
            else:
                # Full restart
                current_set = sorted(random.sample(range(129), 12))
                current_val = _evaluate_set(current_set)
                if current_val < best_val:
                    best_val = current_val
                    best_set = current_set[:]
                    
    # Final safety check: ensure distinctness and range
    if len(set(best_set)) != 12 or min(best_set) < 0 or max(best_set) > 128:
        # Fallback to a known valid set (e.g., 0-11)
        best_set = list(range(12))
        
    return best_set
# EVOLVE-BLOCK-END
