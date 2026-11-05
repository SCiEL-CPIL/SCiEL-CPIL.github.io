import time
import random
import heapq
import numpy as np

DEADLINE_SECONDS = 145.0
START_TIME = time.time()

def _time_left():
    return DEADLINE_SECONDS - (time.time() - START_TIME)

def _build_counts(chosen, cap=1000):
    """Build sum-count array from a list of chosen integers."""
    counts = np.zeros(2 * cap + 1, dtype=np.int32)
    if not chosen:
        return counts
    c = np.asarray(chosen, dtype=np.int64)
    n = len(c)
    ii, jj = np.triu_indices(n)
    sums = c[ii] + c[jj]
    np.add.at(counts, sums, 1)
    return counts

def _pq_greedy(cap=1000, seed=42, time_fn=None, stop_fn=None):
    """
    Priority-queue based greedy construction.
    At each step, we evaluate a batch of candidates.
    We score them by "future conflict cost":
      - Feasibility: All sums x+y (y in S, y<=x) and 2x must have count < 2.
      - Cost: Sum of (counts[s] + 1)^2 for all new sums? Or just count of sums that become 2?
      Let's use: cost = sum( (counts[s] + 1)^2 ) for s in new_sums.
      This penalizes adding to already saturated sums (which is forbidden anyway) and
      prefers adding to sums with count 0 (new) over count 1 (becomes 2).
      Actually, since count 2 is forbidden, we only care about count 0 -> 1 and 1 -> 2.
      Let's define cost as:
        100 * (number of sums that become 2) + (number of sums that become 1)
      This strongly prefers moves that don't saturate sums.
    """
    rng = random.Random(seed)
    pool = list(range(1, cap + 1))
    rng.shuffle(pool)
    
    chosen = []
    chosen_np = np.empty(0, dtype=np.int64)
    counts = np.zeros(2 * cap + 1, dtype=np.int32)
    
    # We will use a heap to store (cost, x) for candidates we've evaluated.
    # But since the state (counts) changes, previously evaluated costs are stale.
    # So we will re-evaluate a batch each time, or keep a sliding window.
    # Given N=1000, re-evaluating 100 candidates per step is 100 * 1000 ops = 100k ops.
    # 1000 steps * 100k = 100M ops. Might be too slow in Python.
    # Let's use NumPy vectorization for the scoring.
    
    while pool:
        if time_fn is not None and time_fn() < 10.0:
            break
        if stop_fn is not None and stop_fn():
            break
        
        # Take a batch of candidates
        # As the set grows, we might want to look further ahead?
        # Let's stick to a fixed batch size for simplicity, maybe 50-100.
        batch_size = min(100, len(pool))
        batch = pool[:batch_size]
        
        if chosen_np.size == 0:
            x = batch[0]
            chosen.append(x)
            pool = pool[1:]
            chosen_np = np.array([x], dtype=np.int64)
            counts[2 * x] += 1
            continue
        
        best_x = None
        best_cost = float('inf')
        best_new_sums = None
        
        # Vectorized scoring for the batch
        # batch is a list of ints
        batch_np = np.array(batch, dtype=np.int64)
        
        # We need to check feasibility for each x in batch.
        # Sums for x are: x + chosen_np (where chosen_np <= x? No, a<=b is just a set of pairs.
        # The constraint is on the multiset of sums.
        # When adding x, the new sums are {x + y | y in chosen} U {2x}.
        # We must ensure counts[x+y] < 2 for all y in chosen, and counts[2x] < 2.
        
        # To vectorize this check efficiently:
        # Create a matrix of sums? That's batch_size x n_chosen.
        # 100 x 500 = 50k elements. This is fine.
        
        # sums_matrix[i, j] = batch[i] + chosen_np[j]
        sums_matrix = batch_np[:, None] + chosen_np[None, :]
        
        # Get counts for these sums
        # counts is 1D. We need counts[sums_matrix].
        current_counts_matrix = counts[sums_matrix]
        
        # Also check 2x
        double_sums = 2 * batch_np
        double_counts = counts[double_sums]
        
        # Feasibility:
        # 1. All elements in current_counts_matrix must be < 2.
        # 2. double_counts must be < 2.
        
        feasible_mask = np.all(current_counts_matrix < 2, axis=1) & (double_counts < 2)
        
        # If no feasible candidates in batch, discard them?
        # Or keep them? If we discard them, we lose them forever.
        # In greedy, we usually just move on. But maybe a later addition frees up space?
        # No, adding elements only increases counts. So if x is infeasible now, it will be infeasible later.
        # So we can safely discard infeasible candidates from the pool permanently?
        # Wait, if we discard them, we might miss them if we removed some other element?
        # But we don't remove elements in pure greedy.
        # So yes, infeasible candidates can be removed from the pool.
        
        if not np.any(feasible_mask):
            pool = pool[batch_size:]
            continue
            
        # Score feasible candidates
        # Cost:
        # new_counts_matrix = current_counts_matrix + 1
        # tight_mask = (new_counts_matrix == 2)
        # num_tight = tight_mask.sum(axis=1)
        # num_new = (current_counts_matrix == 0).sum(axis=1)
        # cost = 100 * num_tight + num_new
        
        new_counts_matrix = current_counts_matrix + 1
        num_tight = np.sum(new_counts_matrix == 2, axis=1)
        # We also want to minimize the number of sums that are "almost" tight?
        # Let's stick to num_tight.
        
        # Add cost for 2x becoming tight
        double_new_counts = double_counts + 1
        tight_double = (double_new_counts == 2).astype(np.int32)
        
        cost = 100 * (num_tight + tight_double)
        
        # Among feasible, pick min cost. Tie-break: random? or smaller x?
        # Let's add a small random noise to break ties to encourage diversity.
        noise = rng.random(len(batch)) * 0.1
        cost = cost + noise
        
        # Mask out infeasible candidates with inf
        cost[~feasible_mask] = float('inf')
        
        min_idx = np.argmin(cost)
        
        if cost[min_idx] == float('inf'):
            pool = pool[batch_size:]
            continue
            
        best_x = batch[min_idx]
        best_new_sums = np.concatenate((chosen_np + best_x, np.array([2 * best_x], dtype=np.int64)))
        
        # Add best_x
        chosen.append(best_x)
        pool = pool[batch_size:]
        chosen_np = np.concatenate((chosen_np, np.array([best_x], dtype=np.int64)))
        counts[best_new_sums] += 1
        
    return chosen

def _local_improve(chosen, cap=1000, time_fn=None, stop_fn=None):
    """
    Local search:
    1. Try to add any remaining element.
    2. Try remove-one-add-two.
    """
    if not chosen:
        return chosen
    
    current = set(chosen)
    counts = _build_counts(list(current), cap)
    best_size = len(current)
    best_set = set(current)
    
    rng = random.Random(42)
    
    # Pre-compute list of candidates
    candidates = [x for x in range(1, cap + 1) if x not in current]
    
    # Sort candidates? Maybe by value?
    # Let's just shuffle them.
    rng.shuffle(candidates)
    
    no_improve = 0
    max_no_improve = 2000
    
    # To speed up validity checks, we can keep a numpy array of current set members
    # But for "remove one add two", the set changes.
    # Let's use a list for the current set for fast iteration in validity checks.
    current_list = list(current)
    rng.shuffle(current_list)
    
    while True:
        if time_fn is not None and time_fn() < 10.0:
            break
        if stop_fn is not None and stop_fn():
            break
        
        if no_improve > max_no_improve:
            break
            
        improved = False
        
        # 1. Try to add a candidate
        if candidates:
            # Try a few random candidates
            for _ in range(20):
                if not candidates:
                    break
                idx = rng.randrange(len(candidates))
                x = candidates[idx]
                
                # Check validity
                valid = True
                if counts[2 * x] >= 2:
                    valid = False
                else:
                    # Check x + y for y in current
                    # We can vectorize this if current_list is numpy
                    # But current_list is small (<= 500).
                    # Let's just loop.
                    for y in current_list:
                        if counts[x + y] >= 2:
                            valid = False
                            break
                
                if valid:
                    # Add x
                    current.add(x)
                    current_list.append(x)
                    candidates.pop(idx)
                    
                    # Update counts
                    for y in current_list[:-1]:
                        counts[x + y] += 1
                    counts[2 * x] += 1
                    
                    if len(current) > best_size:
                        best_size = len(current)
                        best_set = set(current)
                    
                    improved = True
                    no_improve = 0
                    break
        
        if improved:
            continue
            
        # 2. Try remove-one-add-two
        if len(current_list) < 2:
            break
            
        # Pick a random element to remove
        # We can bias towards elements that are "expensive" (create many tight sums)?
        # For now, random.
        rem_idx = rng.randrange(len(current_list))
        rem = current_list[rem_idx]
        
        # Tentatively remove rem
        # Save old counts
        old_counts = counts.copy()
        
        # Remove rem from current
        current.remove(rem)
        current_list.pop(rem_idx)
        
        # Decrement counts
        for y in current_list:
            counts[rem + y] -= 1
        counts[2 * rem] -= 1
        
        # Try to add two elements
        added = []
        # We need to pick from candidates (which excludes rem? No, rem was in current, so it's not in candidates)
        # But we should also consider rem itself? No, we removed it.
        # We want to add x1, x2 such that x1 != x2, x1, x2 not in current.
        
        # To speed this up, we can try a few candidates.
        # Let's sample a subset of candidates.
        # The number of candidates is ~500.
        # We want to find x1, x2.
        
        # Strategy:
        # 1. Find all valid x1.
        # 2. For each x1, find valid x2 given x1 is added.
        
        # This might be slow if done naively.
        # Let's just try random pairs?
        # Or iterate through candidates for x1, and for each valid x1, iterate for x2.
        
        # Optimization:
        # If we add x1, the counts change.
        # So we need to check validity of x2 against (current U {x1}).
        
        # Let's limit the search for x1 and x2 to speed things up.
        # Try top 50 candidates for x1.
        
        found_pair = False
        candidates_copy = candidates[:]
        rng.shuffle(candidates_copy)
        
        for x1 in candidates_copy[:50]:
            # Check if x1 is valid against current
            valid1 = True
            if counts[2 * x1] >= 2:
                valid1 = False
            else:
                for y in current_list:
                    if counts[x1 + y] >= 2:
                        valid1 = False
                        break
            
            if not valid1:
                continue
            
            # Tentatively add x1
            current.add(x1)
            current_list.append(x1)
            for y in current_list[:-1]:
                counts[x1 + y] += 1
            counts[2 * x1] += 1
            
            # Now find x2
            # x2 must be valid against (current U {x1})
            # And x2 != x1
            # And x2 not in current (which now includes x1)
            
            for x2 in candidates_copy[:50]:
                if x2 == x1:
                    continue
                
                valid2 = True
                if counts[2 * x2] >= 2:
                    valid2 = False
                else:
                    for y in current_list:
                        if counts[x2 + y] >= 2:
                            valid2 = False
                            break
                
                if valid2:
                    # Add x2
                    current.add(x2)
                    current_list.append(x2)
                    for y in current_list[:-1]:
                        counts[x2 + y] += 1
                    counts[2 * x2] += 1
                    
                    # Update candidates list (remove x1, x2)
                    if x1 in candidates:
                        candidates.remove(x1)
                    if x2 in candidates:
                        candidates.remove(x2)
                    
                    added = [x1, x2]
                    found_pair = True
                    break
            
            if found_pair:
                break
            
            if not found_pair:
                # Rollback x1
                current.remove(x1)
                # Remove x1 from current_list (it's at the end)
                current_list.pop()
                for y in current_list:
                    counts[x1 + y] -= 1
                counts[2 * x1] -= 1
        
        if found_pair:
            improved = True
            no_improve = 0
            if len(current) > best_size:
                best_size = len(current)
                best_set = set(current)
        else:
            # Rollback rem
            current.add(rem)
            current_list.append(rem)
            for y in current_list[:-1]:
                counts[rem + y] += 1
            counts[2 * rem] += 1
            
            # Restore counts from old_counts to be safe
            counts[:] = old_counts
            
            no_improve += 1
            
    if len(best_set) > len(current):
        return list(best_set)
    return list(current)

def construct():
    global START_TIME, DEADLINE_SECONDS
    START_TIME = time.time()
    DEADLINE_SECONDS = 145.0
    
    cap = 1000

    def time_left():
        return _time_left()

    def stopped():
        return _time_left() < 15.0

    best = []
    best_len = -1

    # Phase 1: Multiple PQ-Greedy runs with different seeds
    # The PQ-Greedy is deterministic given the seed (shuffling).
    # Let's run a few.
    for seed in range(10):
        if _time_left() < 30.0:
            break
        res = _pq_greedy(cap=cap, seed=seed, time_fn=time_left, stop_fn=stopped)
        if len(res) > best_len:
            best_len = len(res)
            best = res

    # Phase 2: Local improvement on the best greedy solution
    if _time_left() > 20.0 and best:
        improved = _local_improve(best, cap=cap, time_fn=time_left, stop_fn=stopped)
        if len(improved) > best_len:
            best_len = len(improved)
            best = improved

    # Phase 3: Try simple structured patterns (quick checks)
    patterns = [
        list(range(1, cap + 1, 2)),  # every 2nd
        [i for i in range(1, cap + 1) if i % 3 != 0],  # skip multiples of 3
        [i for i in range(1, cap + 1) if i % 4 in (1, 2)],  # 2 per 4
    ]
    for pat in patterns:
        if len(pat) > best_len:
            # Validate pattern
            c = np.asarray(pat, dtype=np.int64)
            n = len(c)
            ii, jj = np.triu_indices(n)
            sums = c[ii] + c[jj]
            if sums.size > 0:
                counts = np.bincount(sums, minlength=2 * cap + 1)
                if counts.max() <= 2:
                    best_len = len(pat)
                    best = pat

    # Final validation and return
    if best:
        c = np.asarray(best, dtype=np.int64)
        n = len(c)
        ii, jj = np.triu_indices(n)
        sums = c[ii] + c[jj]
        if sums.size > 0:
            counts = np.bincount(sums, minlength=2 * cap + 1)
            if counts.max() > 2:
                # Fallback to empty or a simple valid set
                return list(range(1, cap + 1, 3))
    
    return best
