import time
import random
import sys

START_TIME = time.time()
DEADLINE = 85.0

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

def validate(A):
    if not A:
        return True
    if len(set(A)) != len(A):
        return False
    for x in A:
        if not isinstance(x, int) or x < 1 or x > 256:
            return False
    n = len(A)
    S = [set() for _ in range(n + 1)]
    S[0].add(0)
    for i in range(n):
        x = A[i]
        for j in range(i, -1, -1):
            if not S[j]:
                continue
            for s in S[j]:
                S[j+1].add(s + x)
    for j in range(n + 1):
        for k in range(j + 1, n + 1):
            if S[j] & S[k]:
                return False
    return True

def build_sumsets(A):
    """Build cardinality-indexed subset sum sets for set A."""
    n = len(A)
    S = [set() for _ in range(n + 1)]
    S[0].add(0)
    for i in range(n):
        x = A[i]
        for j in range(i, -1, -1):
            if S[j]:
                for s in S[j]:
                    S[j+1].add(s + x)
    return S

def check_validity(S):
    """Check if cardinality-indexed sumsets have no cross-cardinality collisions."""
    n = len(S) - 1
    for j in range(n + 1):
        for k in range(j + 1, n + 1):
            if S[j] & S[k]:
                return False
    return True

def can_add_incremental(S, k, x):
    """Check if element x can be added to a set of size k with sumsets S.
    S has length k+1 (cardinalities 0..k).
    Returns True if adding x doesn't create cross-cardinality collisions.
    """
    # New sums at cardinality k+1: {s + x for s in S[k]}
    new_sums = set()
    for s in S[k]:
        new_sums.add(s + x)
    
    # Check collision with all existing cardinalities 0..k
    for j in range(k + 1):
        if new_sums & S[j]:
            return False
    return True

def add_element_incremental(S, k, x):
    """Add element x to sumsets S (size k), returning new S (size k+1)."""
    new_S = [set(s) for s in S]
    new_S.append(set())  # cardinality k+1
    for j in range(k + 1):
        for s in S[j]:
            new_S[j+1].add(s + x)
    return new_S

def backtrack_search(chosen, S, k, target_size, time_budget, rng):
    """Backtracking with forward checking to extend chosen set to target_size.
    Returns the extended list or None if infeasible within time.
    """
    start = time.time()
    all_values = set(range(1, 257))
    
    def _bt(chosen, S, k):
        if k >= target_size:
            return list(chosen)
        if is_done() or (time.time() - start) > time_budget:
            return None
        
        candidates = list(all_values - set(chosen))
        rng.shuffle(candidates)
        
        for x in candidates:
            if is_done() or (time.time() - start) > time_budget:
                return None
            
            if not can_add_incremental(S, k, x):
                continue
            
            new_S = add_element_incremental(S, k, x)
            new_chosen = chosen + [x]
            
            result = _bt(new_chosen, new_S, k + 1)
            if result is not None:
                return result
        
        return None
    
    return _bt(chosen, S, k)

def construct():
    # Start with a known valid base
    base_solution = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    best = list(base_solution)
    best_size = len(best)
    
    rng = random.Random(42)
    
    # Strategy 1: Try to build from scratch with backtracking
    time_left = DEADLINE - elapsed()
    if time_left > 5:
        bt_budget = min(time_left * 0.5, 30.0)
        # Try building a set of size up to 60
        for target in [60, 50, 40, 30, 20, 15, 12, 10]:
            if is_done():
                break
            result = backtrack_search([], [set([0])], 0, target, bt_budget, rng)
            if result is not None and len(result) > best_size:
                best = result
                best_size = len(result)
                break
    
    # Strategy 2: Extend the base solution using backtracking
    time_left = DEADLINE - elapsed()
    if time_left > 5 and best_size < 60:
        base_S = build_sumsets(best)
        bt_budget = min(time_left * 0.3, 20.0)
        result = backtrack_search(best, base_S, best_size, 60, bt_budget, rng)
        if result is not None and len(result) > best_size:
            best = result
            best_size = len(result)
    
    # Strategy 3: Try different initial seeds with backtracking
    time_left = DEADLINE - elapsed()
    if time_left > 5 and best_size < 60:
        for seed_attempt in range(5):
            if is_done() or time_left < 3:
                break
            # Create a small random seed
            seed_vals = rng.sample(range(1, 257), min(3, 256))
            seed_S = [set([0])]
            valid_seed = True
            for x in seed_vals:
                if not can_add_incremental(seed_S, len(seed_S)-1, x):
                    valid_seed = False
                    break
                seed_S = add_element_incremental(seed_S, len(seed_S)-1, x)
            
            if not valid_seed:
                continue
            
            bt_budget = min((DEADLINE - elapsed()) * 0.15, 10.0)
            result = backtrack_search(seed_vals, seed_S, len(seed_vals), 60, bt_budget, rng)
            if result is not None and len(result) > best_size:
                best = result
                best_size = len(result)
    
    # Final validation
    if validate(best):
        return best
    
    return base_solution

if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}: {res}")
