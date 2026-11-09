"""B2[2]-set construction for {1..1000} combining greedy, swap local search, and conflict-aware filling."""
import time
import random

N = 1000
TIME_LIMIT = 115  # seconds; leaves margin before the 150 s kill

def _count_sums(selected):
    """Return a dict mapping each sum to its count, for the given selected set."""
    counts = {}
    s = sorted(selected)
    for i in range(len(s)):
        a = s[i]
        sa = 2 * a
        counts[sa] = counts.get(sa, 0) + 1
        for j in range(i + 1, len(s)):
            b = s[j]
            s2 = a + b
            counts[s2] = counts.get(s2, 0) + 1
    return counts

def _is_valid(counts):
    """Check if all sums have at most 2 representations."""
    for c in counts.values():
        if c > 2:
            return False
    return True

def _greedy_build(starting_set=None, order=None):
    """
    Greedy construction: try to add elements in given order (default increasing).
    If starting_set is provided, start from it and continue with remaining values.
    """
    if starting_set is None:
        chosen = []
        counts = {}
    else:
        chosen = sorted(list(starting_set))
        counts = _count_sums(chosen)

    chosen_set = set(chosen)

    if order is None:
        order = list(range(1, N + 1))

    for x in order:
        if x in chosen_set:
            continue
        # Check if adding x keeps validity
        trial = dict(counts)
        trial[2 * x] = trial.get(2 * x, 0) + 1
        valid = True
        if trial[2 * x] > 2:
            valid = False
        else:
            for y in chosen:
                s2 = x + y
                trial[s2] = trial.get(s2, 0) + 1
                if trial[s2] > 2:
                    valid = False
                    break
        
        if valid:
            chosen.append(x)
            chosen_set.add(x)
            counts = trial

    return chosen

def _try_local_search(initial_set, time_limit, seed=42):
    """
    Try to improve the set via local search: swap out an element for another.
    Returns the best set found.
    """
    best = sorted(initial_set)
    best_counts = _count_sums(best)
    best_size = len(best)
    chosen_set = set(best)
    start_time = time.time()
    rng = random.Random(seed)
    
    # Candidate pool: all numbers not in best
    candidates = [x for x in range(1, N + 1) if x not in chosen_set]
    rng.shuffle(candidates)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Shuffle candidates occasionally to escape local patterns
        if rng.random() < 0.1:
            rng.shuffle(candidates)
            
        for new_elem in candidates:
            if new_elem in chosen_set:
                continue
            # Try swapping new_elem in for each element in best
            for i in range(len(best)):
                old_elem = best[i]
                
                # Quick check: compute new counts incrementally
                trial = dict(best_counts)
                
                # Subtract old_elem's contributions
                for y in best:
                    if y == old_elem:
                        continue
                    s2 = old_elem + y
                    if s2 in trial:
                        trial[s2] -= 1
                        if trial[s2] < 0:
                            trial[s2] = 0
                s2 = 2 * old_elem
                if s2 in trial:
                    trial[s2] -= 1
                    if trial[s2] < 0:
                        trial[s2] = 0
                
                # Add new_elem's contributions
                for y in best:
                    if y == old_elem:
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
                    # Accept the swap
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

def _fill_remaining(best_set, counts, time_limit):
    """
    Try to add remaining elements in descending order (conflict-aware filling).
    Larger numbers tend to have fewer conflicts with small numbers already present.
    """
    chosen = sorted(best_set)
    chosen_set = set(chosen)
    start_time = time.time()
    
    # Try to add remaining values in descending order
    for x in range(N, 0, -1):
        if (time.time() - start_time) > time_limit:
            break
        if x in chosen_set:
            continue
        
        # Check if adding x keeps validity
        trial = dict(counts)
        trial[2 * x] = trial.get(2 * x, 0) + 1
        valid = True
        if trial[2 * x] > 2:
            valid = False
        else:
            for y in chosen:
                s2 = x + y
                trial[s2] = trial.get(s2, 0) + 1
                if trial[s2] > 2:
                    valid = False
                    break
        
        if valid:
            chosen.append(x)
            chosen_set.add(x)
            counts = trial
    
    return chosen, counts

def construct():
    start_time = time.time()
    
    # Step 1: Multiple greedy constructions with different orderings
    best = []
    best_size = 0
    
    # Try increasing order
    greedy1 = _greedy_build(order=list(range(1, N + 1)))
    if len(greedy1) > best_size:
        best = greedy1
        best_size = len(greedy1)
    
    # Try decreasing order
    if (time.time() - start_time) < TIME_LIMIT - 30:
        greedy2 = _greedy_build(order=list(range(N, 0, -1)))
        if len(greedy2) > best_size:
            best = greedy2
            best_size = len(greedy2)
    
    # Try random orderings
    rng = random.Random(123)
    for _ in range(5):
        if (time.time() - start_time) > TIME_LIMIT - 40:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        greedy_rand = _greedy_build(order=order)
        if len(greedy_rand) > best_size:
            best = greedy_rand
            best_size = len(greedy_rand)
    
    if (time.time() - start_time) > TIME_LIMIT - 20:
        return best
    
    # Step 2: Local search to improve via swaps
    remaining_time = TIME_LIMIT - (time.time() - start_time) - 5
    if remaining_time > 15:
        improved = _try_local_search(best, remaining_time, seed=42)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    if (time.time() - start_time) > TIME_LIMIT - 10:
        return best
    
    # Step 3: Fill remaining with conflict-aware descending pass
    counts = _count_sums(best)
    remaining_time = TIME_LIMIT - (time.time() - start_time) - 3
    if remaining_time > 5:
        filled, counts = _fill_remaining(best, counts, remaining_time)
        if len(filled) > best_size:
            best = filled
            best_size = len(best)
    
    # Step 4: One more local search pass if time remains
    if (time.time() - start_time) < TIME_LIMIT - 5:
        remaining_time = TIME_LIMIT - (time.time() - start_time) - 2
        if remaining_time > 5:
            improved2 = _try_local_search(best, remaining_time, seed=99)
            if len(improved2) > best_size:
                best = improved2
                best_size = len(best)
    
    return best
