"""Find maximum-size reciprocal-subsum-free subset of [1,24] via branch-and-bound."""
import time
from fractions import Fraction

# Precompute reciprocals as Fractions for exact arithmetic
N = 24
recips = [Fraction(1, i) for i in range(1, N + 1)]

# We want to find a subset S of {1,...,24} such that for every x in S,
# 1/x != sum of 1/y for any nonempty subset T of (S \ {x}).
# Equivalently, 1/x is not in the set of all subset-sums of {1/y : y in S, y != x}.

def _check_valid(S):
    """Check if subset S (list of ints) is reciprocal-subsum-free."""
    S_set = set(S)
    for x in S:
        # Compute all subset sums of reciprocals of S \ {x}
        others = [recips[y - 1] for y in S if y != x]
        # Use a set to track all achievable subset sums
        sums = {Fraction(0)}
        for r in others:
            new_sums = set()
            for s in sums:
                ns = s + r
                new_sums.add(ns)
            sums = sums | new_sums
        # Check if 1/x is in sums (excluding the empty subset sum 0, but 1/x != 0 always)
        if recips[x - 1] in sums:
            return False
    return True


def _is_conflict(x, S_without_x):
    """Check if adding x to S_without_x creates a conflict.
    Conflict if 1/x = sum of subset of reciprocals of S_without_x,
    OR if any y in S_without_x has 1/y = sum of subset of reciprocals of (S_without_x \ {y}) ∪ {1/x}.
    
    For efficiency in branch-and-bound, we check both directions.
    """
    # Check 1/x against subset sums of S_without_x
    others = [recips[y - 1] for y in S_without_x]
    sums = {Fraction(0)}
    for r in others:
        new_sums = set()
        for s in sums:
            ns = s + r
            new_sums.add(ns)
        sums = sums | new_sums
    if recips[x - 1] in sums:
        return True
    
    # Check each y in S_without_x: does 1/y equal sum of subset of (S_without_x \ {y}) ∪ {1/x}?
    for y in S_without_x:
        # Subset sums of (S_without_x \ {y}) ∪ {x}
        others_y = [recips[z - 1] for z in S_without_x if z != y]
        others_y.append(recips[x - 1])
        sums_y = {Fraction(0)}
        for r in others_y:
            new_sums = set()
            for s in sums_y:
                ns = s + r
                new_sums.add(ns)
            sums_y = sums_y | new_sums
        if recips[y - 1] in sums_y:
            return True
    return False


def _subset_sums_cache(elements):
    """Compute all subset sums of a list of Fractions."""
    sums = {Fraction(0)}
    for r in elements:
        new_sums = set()
        for s in sums:
            ns = s + r
            new_sums.add(ns)
        sums = sums | new_sums
    return sums


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # We'll use branch-and-bound to find the maximum-size valid subset.
    # Strategy: try to include as many elements as possible.
    # Order elements by some heuristic (e.g., larger elements first since they have smaller reciprocals,
    # making them less likely to be formed by sums of others).
    
    # Actually, let's think about this differently. The constraint is:
    # For x in S: 1/x is NOT a subset sum of {1/y : y in S, y != x}
    
    # Let's use a greedy + local search approach with exact verification,
    # then try to improve via branch-and-bound on the remaining elements.
    
    # First, let's try a systematic branch-and-bound.
    # We process elements in order and decide include/exclude.
    # Upper bound: current_size + number of remaining elements.
    
    best_size = 0
    best_set = None
    
    # Try all subsets in order of decreasing size? That's too expensive for n=24.
    # Instead, use branch-and-bound with pruning.
    
    elements = list(range(1, N + 1))
    
    # Precompute: for each element x, which other elements' reciprocals could sum to 1/x?
    # This helps with pruning.
    
    def branch_and_bound(idx, current, remaining_count):
        nonlocal best_size, best_set, deadline
        
        if time.time() > deadline:
            return
        
        # Upper bound
        if len(current) + remaining_count <= best_size:
            return
        
        if idx == N:
            if len(current) > best_size:
                best_size = len(current)
                best_set = list(current)
            return
        
        # Option 1: Include elements[idx]
        x = elements[idx]
        # Check if adding x is valid with current
        if not _is_conflict(x, current):
            current.append(x)
            branch_and_bound(idx + 1, current, remaining_count - 1)
            current.pop()
        
        # Option 2: Exclude elements[idx]
        branch_and_bound(idx + 1, current, remaining_count - 1)
    
    # This naive B&B will be too slow for n=24 in worst case.
    # Let's use a smarter approach: start with a good greedy solution, then try to add elements.
    
    # Greedy: add elements in order of decreasing value (larger numbers first)
    # since larger numbers have smaller reciprocals, less likely to be subset sums.
    greedy_order = sorted(range(1, N + 1), reverse=True)
    
    current = []
    for x in greedy_order:
        if not _is_conflict(x, current):
            current.append(x)
    
    if current:
        best_size = len(current)
        best_set = list(current)
    
    # Now try to improve by swapping elements
    # Try removing one element and adding one or more others
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_set = set(best_set)
        # Try removing each element
        for remove_elem in list(best_set):
            trial = [x for x in best_set if x != remove_elem]
            trial_set = set(trial)
            # Try adding each element not in trial
            for add_elem in range(1, N + 1):
                if add_elem in trial_set:
                    continue
                # Check if we can add add_elem
                if not _is_conflict(add_elem, trial):
                    # Also need to check that no existing element in trial conflicts with add_elem
                    # _is_conflict already checks both directions
                    new_size = len(trial) + 1
                    if new_size > best_size:
                        best_size = new_size
                        best_set = trial + [add_elem]
                        improved = True
                        break
            if improved:
                break
    
    # Try a more systematic search: branch-and-bound with better ordering
    # Sort by "difficulty" - elements that are hard to include (small reciprocals that are easy to form)
    # Actually, let's try all subsets of size best_size+1, best_size+2, etc.
    # But that's combinatorially expensive.
    
    # Let's try a randomized local search to find better solutions
    import random
    random.seed(42)
    
    for _ in range(5000):
        if time.time() > deadline:
            break
        
        # Start from best_set and try random modifications
        trial = list(best_set)
        random.shuffle(trial)
        
        # Try to add random elements
        for _inner in range(20):
            if time.time() > deadline:
                break
            candidates = [x for x in range(1, N + 1) if x not in trial]
            if not candidates:
                break
            x = random.choice(candidates)
            if not _is_conflict(x, trial):
                trial.append(x)
        
        if len(trial) > best_size:
            best_size = len(trial)
            best_set = list(trial)
    
    # Final verification
    if best_set and _check_valid(best_set):
        return list(best_set)
    
    # Fallback: return the seed solution
    return list(range(13, 25))


# Quick test
if __name__ == "__main__":
    result = construct()
    print(f"Result: {result}, size: {len(result)}")
    print(f"Valid: {_check_valid(result)}")
