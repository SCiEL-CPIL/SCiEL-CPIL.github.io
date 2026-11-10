import random
import itertools
import time
from collections import defaultdict

def _is_sidon(subset):
    """Check if a list of distinct ints is a Sidon set (pair sums unique, incl. diagonals)."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_subset_size(A):
    """Exhaustively find the largest Sidon subset of A (A has 12 elements)."""
    n = len(A)
    # Try from largest down
    for k in range(n, 1, -1):
        for subset in itertools.combinations(A, k):
            if _is_sidon(list(subset)):
                return k
    return 2

def _collision_score(A):
    """
    Calculate a score that is higher when there are more sum collisions.
    We use the sum of squares of the counts of each sum.
    This penalizes unique sums (count 1, score 1) less than collisions (count 2, score 4).
    Higher score means more 'structured' collisions, which typically leads to smaller max Sidon subsets.
    """
    sum_count = defaultdict(int)
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            s = A[i] + A[j]
            sum_count[s] += 1
    score = 0
    for c in sum_count.values():
        score += c * c
    return score

def _local_search_max_collisions(A, deadline, rng):
    """
    Local search maximizing the collision score via single-element swaps.
    """
    n = len(A)
    A = list(A)
    current_score = _collision_score(A)
    best_A = list(A)
    best_score = current_score
    used = set(A)
    universe = list(range(129))

    # Simulated annealing parameters for MAXIMIZATION
    T = 10.0
    T_min = 0.1
    cooling = 0.995

    step = 0
    while time.time() < deadline:
        step += 1
        i = rng.randrange(n)
        old_val = A[i]
        
        # Sample new value
        # To be faster, we can precompute available or just check membership
        # For n=12, checking membership in a set of 129 is fast.
        new_val = rng.choice(universe)
        if new_val in used:
            continue
        
        # Apply
        A[i] = new_val
        used.remove(old_val)
        used.add(new_val)
        new_score = _collision_score(A)

        # Accept?
        delta = new_score - current_score
        # We want to maximize score, so delta > 0 is good.
        # Accept if delta > 0 or with probability exp(delta / T)
        if delta >= 0 or (T > T_min and rng.random() < pow(2.718281828, delta / max(T, 1e-9))):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_A = list(A)
        else:
            # Revert
            A[i] = old_val
            used.remove(new_val)
            used.add(old_val)

        T *= cooling

    return best_A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    rng = random.Random(12345)

    best_A = None
    best_exact_size = 12  # Start with worst case (largest possible)
    
    # Initial candidates
    # 1. Arithmetic Progression (dense collisions)
    # 2. Random sets
    # 3. Sets with small gaps
    
    candidates = []
    
    # APs
    for step in range(1, 11):
        # Start at 0
        ap = list(range(0, 129, step))
        if len(ap) >= 12:
            # Take first 12
            candidates.append(ap[:12])
            # Take middle 12?
            mid = len(ap) // 2
            candidates.append(ap[mid-6:mid+6])
            
    # Random sets
    for _ in range(20):
        candidates.append(rng.sample(range(129), 12))
        
    # Small range sets (0-11, 0-15, etc.)
    for max_val in range(11, 30):
        candidates.append(rng.sample(range(max_val + 1), 12))

    # Deduplicate candidates
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
            
    # Evaluate initial candidates
    for c in unique_candidates:
        if time.time() > deadline - 10:
            break
        # Quick check: if it's already an AP, it's likely good.
        # Let's just run local search on a subset of them
        pass

    # Run local search on the most promising initial candidates
    # We'll pick a few diverse ones
    search_starts = unique_candidates[:10]
    
    for start in search_starts:
        if time.time() > deadline - 5:
            break
            
        candidate, _ = _local_search_max_collisions(start, deadline - 5, rng)
        
        # Evaluate exact max Sidon subset size
        exact_size = _max_sidon_subset_size(candidate)
        
        if exact_size < best_exact_size:
            best_exact_size = exact_size
            best_A = candidate
            
            if best_exact_size <= 3:
                # Can't do much better than 3 (theoretical lower bound for 12 elements in [0,128] might be 3 or 4)
                # If we get 3, we are very good.
                break

    # If we didn't find a very small one, try a few more random restarts
    if best_exact_size > 4:
        for _ in range(5):
            if time.time() > deadline - 2:
                break
            start = rng.sample(range(129), 12)
            candidate, _ = _local_search_max_collisions(start, deadline - 2, rng)
            exact_size = _max_sidon_subset_size(candidate)
            if exact_size < best_exact_size:
                best_exact_size = exact_size
                best_A = candidate

    # Fallback
    if best_A is None:
        best_A = list(range(12))

    return best_A

if __name__ == "__main__":
    result = construct()
    print("Candidate:", result)
    print("Max Sidon subset size:", _max_sidon_subset_size(result))
    print("Collision Score:", _collision_score(result))
