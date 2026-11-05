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

def _max_sidon_size_exact(subset):
    """
    Find the largest Sidon subset size for a given 12-element set.
    Uses backtracking with pruning.
    """
    subset = sorted(subset)
    n = len(subset)
    
    if _is_sidon(subset):
        return n
    
    # Try sizes from n-1 down to 1
    for size in range(n - 1, 0, -1):
        for combo in combinations(range(n), size):
            vals = [subset[i] for i in combo]
            if _is_sidon(vals):
                return size
    return 1

def _evaluate_candidate(vals):
    """Evaluate a candidate by finding its exact max Sidon subset size."""
    return _max_sidon_size_exact(vals)

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    best_size = 12
    best_candidate = list(range(12))
    
    # First, evaluate the baseline
    baseline_size = _evaluate_candidate(best_candidate)
    best_size = baseline_size
    
    # Generate structured candidates with high collision density
    candidates = []
    
    # Strategy 1: Arithmetic progressions
    for start in range(0, 120):
        for step in range(1, 12):
            vals = [start + i * step for i in range(12)]
            if max(vals) <= 128 and len(set(vals)) == 12:
                candidates.append(vals)
    
    # Strategy 2: Pairs summing to common values
    for S in range(2, 250):
        # Try to find 6 distinct pairs (a,b) with a+b=S, all in [0,128]
        pairs = []
        used = set()
        for a in range(max(0, S-128), min(128, S//2)+1):
            b = S - a
            if 0 <= b <= 128 and a not in used and b not in used and a < b:
                pairs.append((a, b))
                used.add(a)
                used.add(b)
            if len(pairs) == 6:
                break
        if len(pairs) == 6:
            vals = []
            for a, b in pairs:
                vals.extend([a, b])
            if len(set(vals)) == 12:
                candidates.append(sorted(vals))
    
    # Strategy 3: Two interleaved progressions
    for start1 in range(0, 30):
        for start2 in range(30, 60):
            for step in range(2, 10):
                vals = []
                for i in range(6):
                    vals.append(start1 + i * step)
                    vals.append(start2 + i * step)
                if all(0 <= v <= 128 for v in vals) and len(set(vals)) == 12:
                    candidates.append(sorted(vals))
    
    # Strategy 4: Random candidates with collision bias
    random.seed(42)
    for _ in range(200):
        # Create pairs that share sums
        S = random.randint(20, 200)
        vals = []
        used = set()
        for _ in range(6):
            a = random.randint(max(0, S-128), min(128, S))
            b = S - a
            if 0 <= b <= 128 and a not in used and b not in used and a != b:
                vals.extend([a, b])
                used.add(a)
                used.add(b)
        if len(set(vals)) == 12:
            candidates.append(sorted(vals))
    
    # Deduplicate candidates
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    # Evaluate candidates, prioritizing those with high collision counts
    def collision_score(vals):
        sums = [vals[i] + vals[j] for i in range(len(vals)) for j in range(i, len(vals))]
        c = Counter(sums)
        return sum(k * (k - 1) // 2 for k in c.values())
    
    # Sort by collision score (descending) to evaluate high-collision ones first
    unique_candidates.sort(key=collision_score, reverse=True)
    
    for vals in unique_candidates:
        if time.time() > deadline:
            break
        size = _evaluate_candidate(vals)
        if size < best_size:
            best_size = size
            best_candidate = vals
            if best_size <= 4:  # Can't do better than 4 for 12 elements
                break
    
    return best_candidate
# EVOLVE-BLOCK-END
