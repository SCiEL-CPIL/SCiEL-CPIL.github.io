"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import Counter
from itertools import combinations

def _is_sidon(vals):
    """Check if vals has all unordered pair sums unique (including diagonals)."""
    n = len(vals)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            s = vals[i] + vals[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _max_sidon_subset_size_fast(subset, deadline=None):
    """
    Find largest subset with all unordered pair sums unique (including diagonals).
    Uses backtracking with pruning for speed.
    """
    n = len(subset)
    subset = list(subset)
    
    # Try to build largest Sidon subset via backtracking
    # Start from full set, try to remove elements to fix collisions
    # Better: try all subsets from largest to smallest, but with early termination
    
    # First, quick check: is the full set Sidon?
    if _is_sidon(subset):
        return n
    
    # Try sizes from n-1 down
    for size in range(n - 1, 0, -1):
        if deadline and time.time() > deadline:
            return 0
        for combo in combinations(range(n), size):
            if deadline and time.time() > deadline:
                return 0
            vals = [subset[i] for i in combo]
            if _is_sidon(vals):
                return size
    return 1

def _count_duplicate_sums(subset):
    """Count how many unordered pair sums (including diagonals) are duplicated."""
    n = len(subset)
    sums = [subset[i] + subset[j] for i in range(n) for j in range(i, n)]
    c = Counter(sums)
    return sum(k * (k - 1) // 2 for k in c.values())

def _generate_structured_candidate(seed=None):
    """
    Generate a candidate with high collision density.
    Strategy: create pairs that share common sums.
    """
    if seed is not None:
        random.seed(seed)
    
    # Create 6 pairs that share common sum values
    # This maximizes collisions
    candidates = []
    
    # Strategy 1: Pairs around a central sum
    central_sum = random.randint(40, 90)
    pairs = []
    used = set()
    for _ in range(6):
        a = random.randint(0, 128)
        b = central_sum - a
        if 0 <= b <= 128 and a not in used and b not in used and a != b:
            pairs.append((a, b))
            used.add(a)
            used.add(b)
    
    if len(pairs) == 6:
        vals = []
        for a, b in pairs:
            vals.extend([a, b])
        if len(set(vals)) == 12:
            candidates.append(vals)
    
    # Strategy 2: Arithmetic progression with collisions
    # e.g., 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22
    # Many pairs sum to same values
    start = random.randint(0, 20)
    step = random.randint(1, 5)
    vals = [start + i * step for i in range(12)]
    if all(0 <= v <= 128 for v in vals) and len(set(vals)) == 12:
        candidates.append(vals)
    
    # Strategy 3: Two interleaved progressions
    start1 = random.randint(0, 10)
    start2 = random.randint(10, 20)
    step = random.randint(2, 6)
    vals = []
    for i in range(6):
        vals.append(start1 + i * step)
        vals.append(start2 + i * step)
    if all(0 <= v <= 128 for v in vals) and len(set(vals)) == 12:
        candidates.append(vals)
    
    # Strategy 4: Random with bias toward collisions
    vals = random.sample(range(129), 12)
    candidates.append(vals)
    
    return random.choice(candidates) if candidates else random.sample(range(129), 12)

def _local_search_maximize_collisions(candidate, deadline, max_iter=5000):
    """
    Local search to maximize duplicate sums.
    Uses simulated annealing for better exploration.
    """
    candidate = list(candidate)
    n = len(candidate)
    current_dups = _count_duplicate_sums(candidate)
    best_candidate = list(candidate)
    best_dups = current_dups
    
    temp = 10.0
    cooling = 0.999
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Choose a random element to replace
        i = random.randint(0, n - 1)
        old_val = candidate[i]
        new_val = random.randint(0, 128)
        
        if new_val in candidate and new_val != old_val:
            continue
        
        candidate[i] = new_val
        new_dups = _count_duplicate_sums(candidate)
        
        delta = new_dups - current_dups
        
        # Accept if better, or with probability exp(delta/temp) if worse
        if delta > 0 or random.random() < pow(2.718281828, delta / temp):
            current_dups = new_dups
            if new_dups > best_dups:
                best_dups = new_dups
                best_candidate = list(candidate)
        else:
            candidate[i] = old_val
        
        temp *= cooling
    
    return best_candidate, best_dups

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    # Initial candidates
    best_subset = list(range(12))
    best_score = _max_sidon_subset_size_fast(best_subset, deadline)
    
    random.seed(42)
    
    # Try structured candidates first
    for seed in range(20):
        if time.time() > deadline - 10:
            break
        
        candidate = _generate_structured_candidate(seed)
        
        # Improve via local search
        time_left = deadline - time.time()
        max_iter = min(2000, int(time_left * 100))
        improved, dups = _local_search_maximize_collisions(candidate, deadline, max_iter=max_iter)
        
        # Evaluate
        if time.time() > deadline - 2:
            break
        
        score = _max_sidon_subset_size_fast(improved, deadline)
        if score < best_score:
            best_score = score
            best_subset = improved
            if best_score <= 4:
                break
    
    # If still not good, do more random restarts
    if best_score > 4 and time.time() < deadline - 5:
        for restart in range(30):
            if time.time() > deadline - 3:
                break
            
            candidate = random.sample(range(129), 12)
            time_left = deadline - time.time()
            max_iter = min(1000, int(time_left * 50))
            improved, dups = _local_search_maximize_collisions(candidate, deadline, max_iter=max_iter)
            
            if time.time() > deadline - 2:
                break
            
            score = _max_sidon_subset_size_fast(improved, deadline)
            if score < best_score:
                best_score = score
                best_subset = improved
                if best_score <= 4:
                    break
    
    return sorted(best_subset)
# EVOLVE-BLOCK-END
