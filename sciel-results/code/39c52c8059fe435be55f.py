"""Rewritten program for B2[2] set construction on {1,...,1000}."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 1000
MAX_SUM = 2 * N  # 2000


def _count_sums(chosen):
    """Return an array of length MAX_SUM+1 with counts of pairwise sums a+b (a<=b)."""
    counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
    arr = np.array(chosen, dtype=np.int32)
    n = len(arr)
    if n == 0:
        return counts
    i_idx, j_idx = np.triu_indices(n, 0)
    sums = arr[i_idx] + arr[j_idx]
    np.add.at(counts, sums, 1)
    return counts


def _can_add_np(chosen_arr, counts, x):
    """Vectorized check if x can be added. chosen_arr is numpy array of chosen elements."""
    if len(chosen_arr) == 0:
        return True
    # All sums x + y for y in chosen_arr
    sums = x + chosen_arr
    # Filter to valid range
    valid_mask = sums <= MAX_SUM
    if not valid_mask.any():
        return True
    # Check if any of these sums already have count >= 2
    if (counts[sums[valid_mask]] >= 2).any():
        return False
    # Check 2*x
    if 2 * x <= MAX_SUM and counts[2 * x] >= 2:
        return False
    return True


def _can_add_many_np(chosen_arr, counts, candidates):
    """Vectorized check for multiple candidates. Returns boolean array of valid candidates."""
    if len(chosen_arr) == 0:
        return np.ones(len(candidates), dtype=bool)
    
    n_cand = len(candidates)
    n_chosen = len(chosen_arr)
    
    # Create matrix of all sums: (n_cand, n_chosen)
    # sums[i, j] = candidates[i] + chosen_arr[j]
    sums = candidates[:, None] + chosen_arr[None, :]
    
    # Mask out sums > MAX_SUM
    valid_sums = sums <= MAX_SUM
    
    # For each candidate, check if any of its valid sums have count >= 2
    # We need to check counts at the sum positions
    # Create a masked version where invalid sums get a sentinel value
    # Instead, let's check per-candidate using vectorized ops
    
    valid = np.ones(n_cand, dtype=bool)
    
    # For each candidate i, check all valid sums
    # This is tricky to fully vectorize, so let's do it in chunks
    for i in range(n_cand):
        s_i = sums[i]  # all sums for candidate i
        v_i = valid_sums[i]  # which are valid
        if v_i.any():
            if (counts[s_i[v_i]] >= 2).any():
                valid[i] = False
                continue
        # Check 2*x
        if 2 * candidates[i] <= MAX_SUM and counts[2 * candidates[i]] >= 2:
            valid[i] = False
    
    return valid


def _add_element(chosen, chosen_set, counts, x):
    """Add x to chosen, updating counts in-place."""
    chosen.append(x)
    chosen_set.add(x)
    for y in chosen[:-1]:
        s = x + y
        counts[s] += 1
    counts[2 * x] += 1


def _remove_element(chosen, chosen_set, counts, x):
    """Remove x from chosen, updating counts in-place."""
    chosen.remove(x)
    chosen_set.discard(x)
    for y in chosen:
        s = x + y
        counts[s] -= 1
    counts[2 * x] -= 1


def _greedy_asc():
    """Simple ascending greedy."""
    counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
    chosen = []
    chosen_set = set()
    for x in range(1, N + 1):
        if _can_add_np(np.array(chosen, dtype=np.int32), counts, x):
            _add_element(chosen, chosen_set, counts, x)
    return chosen


def _greedy_desc():
    """Simple descending greedy."""
    counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
    chosen = []
    chosen_set = set()
    for x in range(N, 0, -1):
        if _can_add_np(np.array(chosen, dtype=np.int32), counts, x):
            _add_element(chosen, chosen_set, counts, x)
    return chosen


def _greedy_random(seed=0):
    """Greedy with randomized order."""
    random.seed(seed)
    counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
    chosen = []
    chosen_set = set()
    order = list(range(1, N + 1))
    random.shuffle(order)
    for x in order:
        if _can_add_np(np.array(chosen, dtype=np.int32), counts, x):
            _add_element(chosen, chosen_set, counts, x)
    return chosen


def _greedy_density_aware(seed=0):
    """
    Density-aware greedy: at each step, pick the candidate that creates the fewest
    'near-conflict' sums (sums that would go from count 1 to 2).
    """
    random.seed(seed)
    counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
    chosen = []
    chosen_set = set()
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)

    while candidates:
        best_x = None
        best_cost = float('inf')
        best_idx = -1

        # Evaluate a sample of candidates
        sample_size = min(len(candidates), 150)
        sample_indices = random.sample(range(len(candidates)), sample_size)

        for idx in sample_indices:
            x = candidates[idx]
            # Cost: number of sums that would reach count 2 (i.e., currently at 1)
            cost = 0
            for y in chosen_set:
                s = x + y
                if s <= MAX_SUM and counts[s] == 1:
                    cost += 1
            if 2 * x <= MAX_SUM and counts[2 * x] == 1:
                cost += 1

            # Check validity
            if not _can_add_np(np.array(chosen, dtype=np.int32), counts, x):
                continue

            if cost < best_cost:
                best_cost = cost
                best_x = x
                best_idx = idx

            if cost == 0:
                break

        # If no zero-cost found in sample, scan all
        if best_cost > 0:
            for idx in range(len(candidates)):
                x = candidates[idx]
                if not _can_add_np(np.array(chosen, dtype=np.int32), counts, x):
                    continue
                cost = 0
                for y in chosen_set:
                    s = x + y
                    if s <= MAX_SUM and counts[s] == 1:
                        cost += 1
                if 2 * x <= MAX_SUM and counts[2 * x] == 1:
                    cost += 1
                if cost < best_cost:
                    best_cost = cost
                    best_x = x
                    best_idx = idx
                if cost == 0:
                    break

        if best_x is None:
            break

        _add_element(chosen, chosen_set, counts, best_x)
        candidates.pop(best_idx)

    return chosen


def _local_search_insert(chosen, deadline=None, max_rounds=50):
    """
    Local search with INSERT moves: try to add any valid element to grow the set.
    Also does swaps to free capacity.
    """
    if deadline is None:
        deadline = time.time() + 30

    chosen = sorted(chosen)
    counts = _count_sums(chosen)
    chosen_set = set(chosen)
    n0 = len(chosen)

    for rnd in range(max_rounds):
        if time.time() > deadline:
            break

        # Phase 1: Try to insert any valid element
        inserted = False
        all_candidates = np.array([x for x in range(1, N + 1) if x not in chosen_set], dtype=np.int32)
        if len(all_candidates) > 0:
            # Shuffle candidates
            idx_perm = np.random.permutation(len(all_candidates))
            all_candidates = all_candidates[idx_perm]
            
            # Check validity in batches
            batch_size = 50
            for start in range(0, len(all_candidates), batch_size):
                if time.time() > deadline:
                    break
                batch = all_candidates[start:start + batch_size]
                valid_mask = _can_add_many_np(np.array(chosen, dtype=np.int32), counts, batch)
                for j in range(len(batch)):
                    if valid_mask[j]:
                        x = int(batch[j])
                        _add_element(chosen, chosen_set, counts, x)
                        inserted = True
                        # Continue checking remaining in batch (counts changed, so re-check)
                        # Actually, after adding, we should break and restart
                        break
                if inserted:
                    break
        
        if not inserted:
            # Phase 2: Try swaps to free capacity
            swapped = False
            for i in range(len(chosen)):
                if time.time() > deadline:
                    break
                x_remove = chosen[i]
                
                # Temporarily remove
                temp_counts = counts.copy()
                for y in chosen:
                    if y == x_remove:
                        continue
                    s = x_remove + y
                    temp_counts[s] -= 1
                temp_counts[2 * x_remove] -= 1
                
                temp_chosen_arr = np.array([y for y in chosen if y != x_remove], dtype=np.int32)
                temp_chosen_set = chosen_set - {x_remove}
                
                # Find best addition
                best_add = None
                best_score = float('inf')
                
                candidates = [x for x in range(1, N + 1) if x not in temp_chosen_set]
                random.shuffle(candidates)
                sample = candidates[:60]
                
                for x_add in sample:
                    # Check validity
                    valid = True
                    for y in temp_chosen_set:
                        s = x_add + y
                        if s <= MAX_SUM and temp_counts[s] >= 2:
                            valid = False
                            break
                    if valid and 2 * x_add <= MAX_SUM and temp_counts[2 * x_add] >= 2:
                        valid = False
                    
                    if not valid:
                        continue
                    
                    # Score: prefer additions that create fewer sums at count 1
                    score = 0
                    for y in temp_chosen_set:
                        s = x_add + y
                        if s <= MAX_SUM and temp_counts[s] == 1:
                            score += 1
                    if 2 * x_add <= MAX_SUM and temp_counts[2 * x_add] == 1:
                        score += 1
                    
                    if score < best_score:
                        best_score = score
                        best_add = x_add
                
                if best_add is not None:
                    # Perform the swap
                    _remove_element(chosen, chosen_set, counts, x_remove)
                    _add_element(chosen, chosen_set, counts, best_add)
                    chosen = sorted(chosen)
                    counts = _count_sums(chosen)
                    swapped = True
                    break
            
            if not swapped and len(chosen) <= n0:
                break  # No improvement possible

    return chosen


def _saturating_greedy(seed=0):
    """
    Saturating greedy: repeatedly scan all remaining candidates and add any valid one.
    Continue until no more can be added.
    """
    random.seed(seed)
    counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
    chosen = []
    chosen_set = set()
    remaining = set(range(1, N + 1))

    progress = True
    while progress and remaining:
        progress = False
        # Shuffle for variety
        order = list(remaining)
        random.shuffle(order)
        for x in order:
            if x not in remaining:
                continue
            if _can_add_np(np.array(chosen, dtype=np.int32), counts, x):
                _add_element(chosen, chosen_set, counts, x)
                remaining.discard(x)
                progress = True

    return chosen


def _conflict_graph_greedy(seed=0):
    """
    Model as maximum independent set on conflict graph.
    Two elements conflict if adding both would create a sum with >2 representations.
    Use greedy coloring-like heuristic with randomized ordering.
    """
    random.seed(seed)
    counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
    chosen = []
    chosen_set = set()

    # Randomized order
    order = list(range(1, N + 1))
    random.shuffle(order)

    for x in order:
        if _can_add_np(np.array(chosen, dtype=np.int32), counts, x):
            _add_element(chosen, chosen_set, counts, x)

    return chosen


def _polish(chosen, deadline=None):
    """
    Final polish: try to add any remaining valid elements, then do local search with inserts.
    """
    if deadline is None:
        deadline = time.time() + 30

    chosen = sorted(chosen)
    counts = _count_sums(chosen)
    chosen_set = set(chosen)

    # Try adding any missing valid elements (quick pass)
    for x in range(1, N + 1):
        if x in chosen_set:
            continue
        if _can_add_np(np.array(chosen, dtype=np.int32), counts, x):
            _add_element(chosen, chosen_set, counts, x)

    # Local search with insert moves
    if time.time() < deadline - 5:
        chosen = _local_search_insert(chosen, deadline=deadline, max_rounds=30)

    return chosen


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    best_set = []
    best_size = 0

    # Strategy 1: Ascending greedy
    set1 = _greedy_asc()
    if len(set1) > best_size:
        best_size = len(set1)
        best_set = set1

    if time.time() > deadline - 100:
        return _polish(best_set, deadline)

    # Strategy 2: Descending greedy
    set2 = _greedy_desc()
    if len(set2) > best_size:
        best_size = len(set2)
        best_set = set2

    if time.time() > deadline - 90:
        return _polish(best_set, deadline)

    # Strategy 3: Multiple random-order greedies
    for trial in range(10):
        if time.time() > deadline - 80:
            break
        s = _greedy_random(seed=trial * 7 + 1)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    if time.time() > deadline - 70:
        return _polish(best_set, deadline)

    # Strategy 4: Density-aware greedy
    for trial in range(5):
        if time.time() > deadline - 60:
            break
        s = _greedy_density_aware(seed=trial * 13 + 3)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    if time.time() > deadline - 50:
        return _polish(best_set, deadline)

    # Strategy 5: Saturating greedy
    for trial in range(3):
        if time.time() > deadline - 40:
            break
        s = _saturating_greedy(seed=trial * 17 + 5)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    if time.time() > deadline - 30:
        return _polish(best_set, deadline)

    # Strategy 6: Conflict graph greedy
    for trial in range(3):
        if time.time() > deadline - 20:
            break
        s = _conflict_graph_greedy(seed=trial * 23 + 7)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Final polish on the best set with insert-based local search
    if time.time() < deadline - 5:
        polished = _polish(best_set, deadline)
        if len(polished) > len(best_set):
            best_set = polished

    return best_set
# EVOLVE-BLOCK-END
