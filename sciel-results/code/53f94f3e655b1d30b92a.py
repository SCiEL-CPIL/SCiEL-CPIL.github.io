"""B2[2]-set construction for {1..1000} with time-budgeted local search."""
import time
import random
import sys

N = 1000
TIME_LIMIT = 135  # seconds; leaves margin before the 150 s kill

def _count_sums(selected):
    """Return a dict mapping each sum to its count, for the given selected set."""
    counts = {}
    # Use a sorted list for efficiency
    s = sorted(selected)
    for i in range(len(s)):
        a = s[i]
        # a + a = 2a
        sa = 2 * a
        counts[sa] = counts.get(sa, 0) + 1
        for j in range(i + 1, len(s)):
            b = s[j]
            if b > N:
                break
            # We only care about sums <= 2*N (max possible sum from two elements in 1..N)
            # but actually sums can go up to 2000. That's fine.
            s2 = a + b
            counts[s2] = counts.get(s2, 0) + 1
    return counts

def _is_valid(counts):
    """Check if all sums have at most 2 representations."""
    for c in counts.values():
        if c > 2:
            return False
    return True

def _greedy_build(starting_set=None):
    """
    Greedy construction: try to add elements in increasing order.
    If starting_set is provided, start from it and continue.
    """
    if starting_set is None:
        chosen = []
        counts = {}
    else:
        chosen = sorted(list(starting_set))
        counts = _count_sums(chosen)

    # Determine which values are already chosen
    chosen_set = set(chosen)

    # Try to add remaining values in increasing order
    for x in range(1, N + 1):
        if x in chosen_set:
            continue
        # Check if adding x keeps validity
        # New sums: x + y for y in chosen (y >= x to avoid double-counting? No, we need all)
        # Actually, representations are a+b with a<=b from the set.
        # When we add x, new representations are:
        #   x + y where y is in chosen and y >= x  (these are new)
        #   2*x  (new)
        #   But also x + y where y < x? No, because those would have been counted when y was added and x was not yet in the set.
        #   Wait, the representation a+b with a<=b: if we add x, the new pairs involving x are:
        #     (x, x) -> 2x
        #     (x, y) where y in chosen and y > x -> x+y
        #     (y, x) where y in chosen and y < x -> these are the same as (x, y) but with a=y, b=x. Since a<=b requires y<=x, and y<x, these ARE new representations too!
        #   So actually: for each y in chosen, the pair (min(x,y), max(x,y)) gives sum x+y.
        #   So new sums are: for each y in chosen, sum = x+y (with the constraint that we count each unordered pair once).
        #   Plus 2x.
        
        trial = dict(counts)
        # Add 2x
        trial[2*x] = trial.get(2*x, 0) + 1
        # Add x + y for each y in chosen
        for y in chosen:
            s2 = x + y
            trial[s2] = trial.get(s2, 0) + 1
        
        # Check validity
        valid = True
        for s2, c in trial.items():
            if c > 2:
                valid = False
                break
        
        if valid:
            chosen.append(x)
            chosen_set.add(x)
            counts = trial

    return chosen

def _try_local_search(initial_set, time_limit):
    """
    Try to improve the set via local search: swap out an element for another.
    Returns the best set found.
    """
    best = sorted(initial_set)
    best_counts = _count_sums(best)
    best_size = len(best)
    chosen_set = set(best)
    start_time = time.time()
    
    # Candidate pool: all numbers not in best
    candidates = [x for x in range(1, N + 1) if x not in chosen_set]
    random.shuffle(candidates)
    
    # For each candidate, we can try swapping it in for some element in best
    # We'll iterate through swaps and accept if it improves or maintains size while enabling future improvements
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for new_elem in candidates:
            if new_elem in chosen_set:
                continue
            # Try swapping new_elem in for each element in best
            for i in range(len(best)):
                old_elem = best[i]
                # Build the trial set
                trial_set = list(best)
                trial_set[i] = new_elem
                trial_set = sorted(trial_set)
                
                # Quick check: compute new counts
                # Remove old_elem's contributions, add new_elem's
                # This is complex; let's just recompute for simplicity if set is small enough
                # But recomputing is O(n^2) which for n~600 is 360000 ops, too slow for many iterations
                
                # Better: incremental update
                # First, let's compute the effect of removing old_elem and adding new_elem
                # We need to know which sums are affected
                # Sums involving old_elem: old_elem + y for y in best (y != old_elem), and 2*old_elem
                # Sums involving new_elem: new_elem + y for y in trial_set (y != new_elem), and 2*new_elem
                
                # Let's just do a fast validity check by computing only the changed parts
                # Actually, let's precompute the contribution of each element to the sum counts
                
                # For efficiency, let's use a different approach:
                # Precompute for the current best set, the count of each sum
                # When we remove old_elem, we subtract its contributions
                # When we add new_elem, we add its contributions
                
                # But we need to be careful: sums involving both old_elem and other elements
                # This is getting complex. Let's just recompute if the set is not too large.
                
                if len(best) < 400:
                    trial_counts = _count_sums(trial_set)
                    if _is_valid(trial_counts):
                        # Accept the swap
                        best = trial_set
                        best_counts = trial_counts
                        chosen_set = set(best)
                        candidates.remove(new_elem)
                        if len(best) > best_size:
                            best_size = len(best)
                            improved = True
                            # Also, some previously rejected candidates might now be acceptable
                            # But for simplicity, we'll just continue
                        break
                else:
                    # For larger sets, use a faster check
                    # Compute which sums would exceed 2
                    # We need to check: for each sum s, the new count must be <= 2
                    # The change is: remove old_elem's contribution, add new_elem's contribution
                    
                    # Let's compute the delta
                    # For each y in best where y != old_elem:
                    #   sum old_elem + y loses 1
                    #   sum new_elem + y gains 1 (if y != new_elem)
                    #   2*old_elem loses 1
                    #   2*new_elem gains 1
                    #   Also, if new_elem is in best (it's not, by construction), skip
                    #   If old_elem == new_elem, skip (they're different)
                    
                    # But we also need to consider: sums that involved old_elem and another element z
                    # Those sums lose 1 when old_elem is removed
                    # Sums that involve new_elem and z gain 1 when new_elem is added
                    
                    # Let's just check if any sum would go above 2
                    # Start with a copy of best_counts
                    trial = dict(best_counts)
                    
                    # Subtract old_elem's contributions
                    for y in best:
                        if y == old_elem:
                            continue
                        s2 = old_elem + y
                        if s2 in trial:
                            trial[s2] -= 1
                            if trial[s2] < 0:
                                trial[s2] = 0  # Should not happen
                    s2 = 2 * old_elem
                    if s2 in trial:
                        trial[s2] -= 1
                        if trial[s2] < 0:
                            trial[s2] = 0
                    
                    # Add new_elem's contributions
                    for y in best:
                        if y == old_elem:
                            # y is being replaced, so skip
                            continue
                        if y == new_elem:
                            # new_elem is not in best yet, so this shouldn't happen
                            continue
                        s2 = new_elem + y
                        trial[s2] = trial.get(s2, 0) + 1
                    s2 = 2 * new_elem
                    trial[s2] = trial.get(s2, 0) + 1
                    
                    # Check validity
                    valid = True
                    for c in trial.values():
                        if c > 2:
                            valid = False
                            break
                    
                    if valid:
                        best = sorted(best)
                        best[i] = new_elem
                        best.sort()
                        best_counts = trial
                        chosen_set = set(best)
                        if new_elem in candidates:
                            candidates.remove(new_elem)
                        improved = True
                        break
                
                if (time.time() - start_time) > time_limit:
                    break
            if (time.time() - start_time) > time_limit:
                break
    
    return best

def construct():
    start_time = time.time()
    
    # Step 1: Greedy construction
    greedy_result = _greedy_build()
    current_best = greedy_result
    current_size = len(current_best)
    
    if (time.time() - start_time) > TIME_LIMIT - 30:
        return current_best
    
    # Step 2: Local search to improve
    remaining_time = TIME_LIMIT - (time.time() - start_time) - 5
    if remaining_time > 10:
        improved = _try_local_search(current_best, remaining_time)
        if len(improved) > current_size:
            current_best = improved
            current_size = len(current_best)
    
    # Step 3: Try to add more elements if time remains
    if (time.time() - start_time) < TIME_LIMIT - 10:
        extended = _greedy_build(current_best)
        if len(extended) > current_size:
            current_best = extended
            current_size = len(current_best)
    
    return current_best
