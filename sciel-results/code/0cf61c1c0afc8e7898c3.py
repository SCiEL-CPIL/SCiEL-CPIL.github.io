"""Rewritten program for B2[2] set construction in [1, 1000].
Uses incremental count maintenance for fast local search:
  1) Greedy baseline with incremental counts.
  2) Local search (add/remove/swap) with O(n) per-move validity checks.
  3) Deadline guard to ensure we finish well within 120s.
"""
import time
import random
import sys

def _greedy_build():
    """Standard greedy: add x if it doesn't cause any sum to exceed 2 reps.
    Uses incremental count updates for O(n) per candidate."""
    chosen = []
    counts = {}
    for x in range(1, 1001):
        ok = True
        for y in chosen:
            sm = x + y
            if counts.get(sm, 0) >= 2:
                ok = False
                break
        if ok:
            sm = 2 * x
            if counts.get(sm, 0) >= 2:
                ok = False
        if ok:
            # Update counts
            for y in chosen:
                sm = x + y
                counts[sm] = counts.get(sm, 0) + 1
            counts[2 * x] = counts.get(2 * x, 0) + 1
            chosen.append(x)
    return chosen

def _build_counts(set_list):
    """Build sum -> count dict for a set."""
    counts = {}
    n = len(set_list)
    for i in range(n):
        for j in range(i, n):
            sm = set_list[i] + set_list[j]
            counts[sm] = counts.get(sm, 0) + 1
    return counts

def _fast_is_valid(set_list):
    """Faster validity check using counting array."""
    n = len(set_list)
    if n == 0:
        return True
    s = sorted(set_list)
    max_sum = 2 * s[-1]
    counts = [0] * (max_sum + 1)
    for i in range(n):
        for j in range(i, n):
            sm = s[i] + s[j]
            counts[sm] += 1
            if counts[sm] > 2:
                return False
    return True

def _local_search_incremental(initial, deadline, seed=42):
    """Local search maintaining incremental counts for O(n) per-move checks.
    
    State: current list, counts dict, current_set.
    Moves:
      - Add x: check all (x+y) for y in current plus 2x against counts.
      - Remove x: decrement counts for all (x+y) and 2x.
      - Swap (remove a, add b): check after removal.
    """
    rng = random.Random(seed)
    current = list(initial)
    current_set = set(current)
    counts = _build_counts(current)
    best = list(current)
    best_size = len(current)
    
    all_vals = list(range(1, 1001))
    
    while time.time() < deadline:
        op = rng.random()
        
        if op < 0.35:
            # Pure add
            x = rng.choice(all_vals)
            if x in current_set:
                continue
            # Check if adding x violates B2[2]
            ok = True
            for y in current:
                sm = x + y
                if counts.get(sm, 0) >= 2:
                    ok = False
                    break
            if ok:
                sm = 2 * x
                if counts.get(sm, 0) >= 2:
                    ok = False
            if ok:
                # Commit: update counts
                for y in current:
                    sm = x + y
                    counts[sm] = counts.get(sm, 0) + 1
                counts[2 * x] = counts.get(2 * x, 0) + 1
                current.append(x)
                current_set.add(x)
                if len(current) > best_size:
                    best = list(current)
                    best_size = len(current)
        elif op < 0.85:
            # Swap: remove one, add another
            if not current:
                continue
            remove_x = rng.choice(current)
            add_x = rng.choice(all_vals)
            if add_x == remove_x or add_x in current_set:
                continue
            # Tentatively remove remove_x
            # Decrement counts for pairs involving remove_x
            for y in current:
                if y == remove_x:
                    continue
                sm = remove_x + y
                counts[sm] -= 1
            sm = 2 * remove_x
            counts[sm] -= 1
            # Now check if we can add add_x
            ok = True
            for y in current:
                if y == remove_x:
                    continue
                sm = add_x + y
                if counts.get(sm, 0) >= 2:
                    ok = False
                    break
            if ok:
                sm = 2 * add_x
                if counts.get(sm, 0) >= 2:
                    ok = False
            if ok:
                # Commit: remove remove_x, add add_x
                current.remove(remove_x)
                current_set.discard(remove_x)
                for y in current:
                    sm = add_x + y
                    counts[sm] = counts.get(sm, 0) + 1
                counts[2 * add_x] = counts.get(2 * add_x, 0) + 1
                current.append(add_x)
                current_set.add(add_x)
                if len(current) > best_size:
                    best = list(current)
                    best_size = len(current)
            else:
                # Undo removal of remove_x
                for y in current:
                    sm = remove_x + y
                    counts[sm] += 1
                sm = 2 * remove_x
                counts[sm] += 1
                # Need to put remove_x back
                current.append(remove_x)
                current_set.add(remove_x)
        else:
            # Remove one random element (to free up capacity)
            if current:
                remove_x = rng.choice(current)
                for y in current:
                    if y == remove_x:
                        continue
                    sm = remove_x + y
                    counts[sm] -= 1
                sm = 2 * remove_x
                counts[sm] -= 1
                current.remove(remove_x)
                current_set.discard(remove_x)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin: 120s limit, use 110s
    
    # Step 1: Greedy baseline
    greedy_set = _greedy_build()
    best = list(greedy_set)
    best_size = len(best)
    
    # Step 2: Local search with multiple random seeds using incremental counts
    remaining = deadline - time.time()
    if remaining > 5:
        # Use more seeds since each iteration is O(n) instead of O(n^2)
        seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
        n_seeds = len(seeds)
        chunk = max(1.0, (remaining - 2) / n_seeds)
        for seed in seeds:
            if time.time() >= deadline - 2:
                break
            result = _local_search_incremental(best, time.time() + chunk, seed)
            if len(result) > best_size:
                best = result
                best_size = len(result)
    
    # Final validation
    if not _fast_is_valid(best):
        # Fallback to greedy if something went wrong
        best = _greedy_build()
    
    return best
