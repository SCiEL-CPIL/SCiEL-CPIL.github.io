"""Find maximum cardinality dissociated reciprocal subset of [1,22].

Key insight: A set S is dissociated (all subset sums distinct) iff no two distinct
subsets have the same sum. This is equivalent to: for every element v in the set,
v is not in the span of the other elements (over the integers, with non-negative
coefficients bounded by 1).

More precisely: when adding element with weight w to a set with subset-sums S,
we need w not to equal any difference of two elements in S, i.e., no s1 - s2 = w
for s1, s2 in S with s1 != s2. Equivalently, (S + w) ∩ S = ∅.

Strategy: Use a Maximum Independent Set (MIS) formulation on a conflict graph.
Two elements i, j are in conflict if there exist subsets A ⊆ {i,j} and B ⊆ {i,j}
with A ≠ B but sum(A) = sum(B). For two elements, this means 1/i = 1/j (impossible
for distinct i,j), so no direct pairwise conflicts. However, conflicts can be
higher-order: a set is NOT dissociated if some non-trivial integer combination
with coefficients in {-1,0,1} sums to zero.

Better approach: Direct backtracking with strong pruning using the subset-sum set.
We use bitset representation for speed and try to find the largest valid set.

The maximum size for [1..22] is known to be around 9-10. We'll use a systematic
branch-and-bound search with:
1. Upper bound: remaining elements + current size
2. Pruning: if we can't beat best, stop
3. Ordering: try elements that are "hardest" to add first (larger weights)
4. Multiple restarts with different orderings
"""
import time
import random

_LCM = 232792560  # LCM(1..22)

def _weight(x):
    return _LCM // x

def _can_add(sums, v):
    """Check if weight v can be added to existing subset-sums without collision."""
    for s in sums:
        if (s + v) in sums:
            return False
    return True

def _build_sums(subset):
    """Build set of all subset sums for given subset (as integer weights)."""
    sums = {0}
    for x in subset:
        v = _weight(x)
        sums = sums | {s + v for s in sums}
    return sums

def _is_dissociated(subset):
    """Check if all reciprocal subset sums are distinct."""
    sums = {0}
    for x in subset:
        v = _weight(x)
        new_sums = {s + v for s in sums}
        if new_sums & sums:
            return False
        sums |= new_sums
    return True

def _branch_and_bound(remaining, current, sums, best, deadline, best_size):
    """Branch and bound: try to build a dissociated set larger than best_size.
    
    remaining: list of candidate elements not yet considered
    current: list of elements in current solution
    sums: set of subset sums for current
    best: list of best solution found so far
    deadline: time limit
    best_size: size of best solution
    
    Returns (best, best_size) - updated best solution.
    """
    # Upper bound: current size + all remaining
    if len(current) + len(remaining) <= best_size:
        return best, best_size
    
    if time.time() > deadline:
        return best, best_size
    
    if not remaining:
        if len(current) > best_size:
            return current[:], len(current)
        return best, best_size
    
    # Try each remaining element
    for i in range(len(remaining)):
        if time.time() > deadline:
            return best, best_size
        
        x = remaining[i]
        v = _weight(x)
        
        if _can_add(sums, v):
            new_sums = sums | {s + v for s in sums}
            new_remaining = remaining[i+1:]
            new_current = current + [x]
            
            best, best_size = _branch_and_bound(
                new_remaining, new_current, new_sums, best, deadline, best_size
            )
        
        # Pruning: even if we take all remaining after i, can we beat best?
        if len(current) + (len(remaining) - i) <= best_size:
            break
    
    return best, best_size

def _greedy_with_backtrack(order, target_size, deadline):
    """Greedy build following order, then backtrack to try to reach target_size."""
    current = []
    sums = {0}
    
    for x in order:
        if time.time() > deadline:
            break
        v = _weight(x)
        if _can_add(sums, v):
            current.append(x)
            sums = sums | {s + v for s in sums}
    
    return current, sums

def _local_search_add(subset, sums, universe, deadline):
    """Try to add elements from universe to subset."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in universe:
            if x in subset:
                continue
            v = _weight(x)
            if _can_add(sums, v):
                subset.append(x)
                sums = sums | {s + v for s in sums}
                improved = True
                break
    return subset, sums

def _swap_search(subset, universe, deadline):
    """Try swap moves to improve the set. Returns best subset found."""
    best = subset[:]
    best_size = len(subset)
    
    # Try all single swaps
    for i in range(len(subset)):
        if time.time() > deadline:
            break
        for x in universe:
            if x in subset:
                continue
            new_subset = subset[:i] + subset[i+1:] + [x]
            if _is_dissociated(new_subset):
                new_size = len(new_subset)
                if new_size > best_size:
                    best = new_subset
                    best_size = new_size
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    universe = list(range(1, 23))
    
    # Start with a known good seed of size 9
    best_subset = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_subset)
    
    # Verify seed
    if not _is_dissociated(best_subset):
        # Find a valid seed
        for sz in range(9, 0, -1):
            import itertools
            found = False
            for combo in itertools.combinations(range(1, 23), sz):
                if _is_dissociated(list(combo)):
                    best_subset = list(combo)
                    best_size = sz
                    found = True
                    break
            if found:
                break
    
    # Strategy 1: Systematic branch-and-bound for larger sets
    # Try target sizes from best_size+1 upward
    for target in range(best_size + 1, min(best_size + 4, 14)):
        if time.time() > deadline:
            break
        
        # Try different orderings
        orderings = [
            universe[:],
            sorted(universe, reverse=True),
            sorted(universe, key=lambda x: _weight(x)),
            sorted(universe, key=lambda x: -_weight(x)),
        ]
        
        for ordering in orderings:
            if time.time() > deadline:
                break
            
            # Branch and bound
            result, result_size = _branch_and_bound(
                ordering, [], {0}, best_subset, deadline, best_size
            )
            
            if result_size > best_size:
                best_subset = result
                best_size = result_size
        
        if best_size >= target:
            break
    
    # Strategy 2: Greedy with multiple random orderings + local search
    random.seed(123)
    
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        shuffled = universe[:]
        random.shuffle(shuffled)
        
        # Various biases
        if attempt % 5 == 0:
            shuffled.sort(key=lambda x: -x)  # Descending
        elif attempt % 5 == 1:
            shuffled.sort(key=lambda x: x)   # Ascending
        elif attempt % 5 == 2:
            shuffled.sort(key=lambda x: _weight(x))  # By weight ascending
        elif attempt % 5 == 3:
            shuffled.sort(key=lambda x: -_weight(x))  # By weight descending
        # else: already shuffled
        
        current, cur_sums = _greedy_with_backtrack(shuffled, 12, deadline)
        
        # Local search: add more
        current, cur_sums = _local_search_add(current, cur_sums, universe, deadline)
        
        if len(current) > best_size:
            best_subset = current[:]
            best_size = len(current)
    
    # Strategy 3: Simulated annealing-style search
    if time.time() < deadline:
        temperature = 1.0
        cooling = 0.98
        current_subset = best_subset[:]
        current_size = best_size
        
        for step in range(3000):
            if time.time() > deadline:
                break
            
            if not current_subset:
                break
            
            move_type = random.random()
            
            if move_type < 0.3:
                # Add
                available = [x for x in universe if x not in current_subset]
                if not available:
                    continue
                in_elem = random.choice(available)
                new_subset = current_subset + [in_elem]
            elif move_type < 0.5:
                # Remove
                if len(current_subset) <= 1:
                    continue
                out_idx = random.randint(0, len(current_subset) - 1)
                new_subset = current_subset[:out_idx] + current_subset[out_idx+1:]
            else:
                # Swap
                if not current_subset:
                    continue
                out_idx = random.randint(0, len(current_subset) - 1)
                out_elem = current_subset[out_idx]
                available = [x for x in universe if x not in current_subset]
                if not available:
                    continue
                in_elem = random.choice(available)
                new_subset = current_subset[:out_idx] + current_subset[out_idx+1:] + [in_elem]
            
            if _is_dissociated(new_subset):
                new_size = len(new_subset)
                if new_size >= current_size:
                    current_subset = new_subset
                    current_size = new_size
                else:
                    if random.random() < (1.0 / (temperature * 10)):
                        current_subset = new_subset
                        current_size = new_size
            
            if current_size > best_size:
                best_subset = current_subset[:]
                best_size = current_size
            
            temperature *= cooling
        
        # Final local search
        if time.time() < deadline:
            best_sums = _build_sums(best_subset)
            best_subset, _ = _local_search_add(best_subset, best_sums, universe, deadline)
            best_size = len(best_subset)
    
    return best_subset
