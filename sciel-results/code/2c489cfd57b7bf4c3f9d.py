"""Improved program: branch-and-bound with cardinality-indexed subset-sum DP."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict
from itertools import combinations

DEADLINE_SECONDS = 58.0  # safety margin before 60s kill


def _build_dp_state(S):
    """
    Build DP state for set S:
      sums_by_k[k] = set of sums achievable with exactly k elements
    """
    n = len(S)
    sums_by_k = [set() for _ in range(n + 1)]
    sums_by_k[0].add(0)
    for x in S:
        for k in range(n, 0, -1):
            if sums_by_k[k - 1]:
                for s in sums_by_k[k - 1]:
                    sums_by_k[k].add(s + x)
    return sums_by_k


def _check_add_valid_fast(sums_by_k, candidate, n):
    """
    Fast check: can we add candidate to a set of size n?
    Returns True if valid, False otherwise.
    """
    # New sums at cardinality k (1..n+1): {s + candidate for s in sums_by_k[k-1]}
    # These must not collide with existing sums at different cardinalities
    
    # Build a map: sum -> set of cardinalities where it appears
    sum_to_ks = defaultdict(set)
    for k in range(n + 1):
        for s in sums_by_k[k]:
            sum_to_ks[s].add(k)
    
    for k in range(1, n + 2):
        src = sums_by_k[k - 1]
        if not src:
            continue
        for s in src:
            new_s = s + candidate
            if new_s in sum_to_ks:
                if k not in sum_to_ks[new_s]:
                    return False
    return True


def _count_addable(sums_by_k, candidates, n, deadline):
    """
    Count how many candidates from the list can be added to the current set.
    Used as a heuristic for branching priority.
    """
    count = 0
    for x in candidates:
        if time.time() > deadline:
            break
        if _check_add_valid_fast(sums_by_k, x, n):
            count += 1
    return count


def _branch_and_bound(start_set, remaining_candidates, target_size, deadline, best_size):
    """
    Branch-and-bound: try to extend start_set to reach target_size.
    Returns the best set found (size >= current best if possible).
    """
    current = start_set[:]
    n = len(current)
    
    if n >= target_size:
        return current
    
    if not remaining_candidates:
        return current
    
    sums_by_k = _build_dp_state(current)
    
    # Sort candidates by how "compatible" they are (fewer collisions first)
    # Heuristic: prefer smaller values (they create smaller sums, less likely to collide)
    # and values that don't create many new sums
    scored_candidates = []
    for x in remaining_candidates:
        if time.time() > deadline:
            break
        if _check_add_valid_fast(sums_by_k, x, n):
            # Score: prefer values that create fewer new sums
            new_sum_count = 0
            for k in range(1, n + 2):
                src = sums_by_k[k - 1]
                if src:
                    new_sum_count += len({s + x for s in src})
            scored_candidates.append((new_sum_count, x))
    
    if not scored_candidates:
        return current
    
    scored_candidates.sort()
    
    # Try candidates in order of increasing "disruption"
    for _, x in scored_candidates[:30]:  # Limit to top 30 to keep fast
        if time.time() > deadline:
            break
        
        new_current = current + [x]
        new_n = n + 1
        new_remaining = [y for y in remaining_candidates if y != x]
        
        # Recursively try to extend
        result = _branch_and_bound(new_current, new_remaining, target_size, deadline, best_size)
        
        if len(result) > best_size:
            best_size = len(result)
            if len(result) >= target_size:
                return result
    
    return current


def _construct_greedy_with_lookahead(candidates, deadline, max_lookahead=3):
    """
    Greedy construction with lookahead: at each step, try adding each candidate
    and see how many more can be added afterwards. Pick the best.
    """
    current = []
    sums_by_k = [set()]
    sums_by_k[0].add(0)
    
    remaining = list(candidates)
    
    while remaining and time.time() < deadline and len(current) < 60:
        n = len(current)
        
        # Find all valid candidates
        valid_candidates = []
        for x in remaining:
            if time.time() > deadline:
                break
            if _check_add_valid_fast(sums_by_k, x, n):
                valid_candidates.append(x)
        
        if not valid_candidates:
            break
        
        if len(valid_candidates) == 1:
            x = valid_candidates[0]
            current.append(x)
            sums_by_k = _build_dp_state(current)
            remaining.remove(x)
            continue
        
        # Lookahead: for each valid candidate, see how many more can be added
        best_x = None
        best_score = -1
        
        for x in valid_candidates[:15]:  # Limit to top 15 for speed
            if time.time() > deadline:
                break
            
            # Build DP state after adding x
            test_set = current + [x]
            test_sums_by_k = _build_dp_state(test_set)
            test_n = n + 1
            
            # Count how many more can be added
            further_count = 0
            for y in remaining:
                if y == x:
                    continue
                if time.time() > deadline:
                    break
                if _check_add_valid_fast(test_sums_by_k, y, test_n):
                    further_count += 1
            
            if further_count > best_score:
                best_score = further_count
                best_x = x
        
        if best_x is None:
            break
        
        current.append(best_x)
        sums_by_k = _build_dp_state(current)
        remaining.remove(best_x)
    
    return current


def _construct_diverse_sets(deadline):
    """
    Construct multiple diverse sets using different strategies.
    Returns the best set found.
    """
    best_set = []
    best_size = 0
    
    # Strategy 1: Greedy with lookahead on ascending order
    candidates = list(range(1, 257))
    current = _construct_greedy_with_lookahead(candidates, deadline)
    if len(current) > best_size:
        best_size = len(current)
        best_set = current[:]
    
    # Strategy 2: Greedy with lookahead on descending order
    candidates = list(range(256, 0, -1))
    current = _construct_greedy_with_lookahead(candidates, deadline)
    if len(current) > best_size:
        best_size = len(current)
        best_set = current[:]
    
    # Strategy 3: Branch-and-bound starting from small seeds
    for seed_size in range(3, 8):
        if time.time() > deadline:
            break
        
        # Try several random seeds
        for trial in range(5):
            if time.time() > deadline:
                break
            
            random.seed(trial * 100 + seed_size)
            seed = random.sample(range(1, 257), seed_size)
            remaining = [x for x in range(1, 257) if x not in set(seed)]
            
            result = _branch_and_bound(seed, remaining, 20, deadline, best_size)
            if len(result) > best_size:
                best_size = len(result)
                best_set = result[:]
    
    # Strategy 4: Random restarts with greedy lookahead
    random.seed(42)
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        current = _construct_greedy_with_lookahead(candidates, deadline)
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
    
    return best_set


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE_SECONDS
    
    best_set = _construct_diverse_sets(deadline)
    
    # Final: ensure all elements are Python int
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
