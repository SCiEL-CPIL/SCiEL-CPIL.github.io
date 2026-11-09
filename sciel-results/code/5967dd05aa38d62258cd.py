"""Rewritten program: greedy + local search (swap moves) for B2[2] sets in 1..1000."""
import time
import random
import sys

def _count_sums(chosen):
    """Return dict: sum -> count of representations a+b with a<=b from chosen."""
    counts = {}
    for i, a in enumerate(chosen):
        for j in range(i, len(chosen)):
            s = a + chosen[j]
            counts[s] = counts.get(s, 0) + 1
    return counts

def _is_valid(chosen):
    """Check if chosen is a valid B2[2] set (every sum has at most 2 reps)."""
    counts = _count_sums(chosen)
    return all(c <= 2 for c in counts.values())

def _greedy_build():
    """Greedy: add x if it keeps the B2[2] constraint."""
    chosen = []
    counts = {}
    for x in range(1, 1001):
        new_sums = []
        for y in chosen:
            new_sums.append(x + y)
        new_sums.append(2 * x)
        trial = dict(counts)
        ok = True
        for s in new_sums:
            trial[s] = trial.get(s, 0) + 1
            if trial[s] > 2:
                ok = False
                break
        if ok:
            chosen.append(x)
            counts = trial
    return chosen

def _try_add(chosen, counts, x):
    """Try to add x to chosen; return (success, new_counts)."""
    new_sums = []
    for y in chosen:
        new_sums.append(x + y)
    new_sums.append(2 * x)
    trial = dict(counts)
    for s in new_sums:
        trial[s] = trial.get(s, 0) + 1
        if trial[s] > 2:
            return False, None
    return True, trial

def _local_search(chosen, counts, deadline, seed=42):
    """Local search with swap moves to improve the set size."""
    rng = random.Random(seed)
    chosen_set = set(chosen)
    best_size = len(chosen)
    best_chosen = list(chosen)
    best_counts = dict(counts)
    
    # Precompute: for each x not in chosen, how many sums would exceed limit
    # We'll use a scoring function: number of "blocked" additions
    def score(chosen_list, cnt):
        # Count how many x in 1..1000 not in chosen could be added
        cs = set(chosen_list)
        addable = 0
        for x in range(1, 1001):
            if x in cs:
                continue
            ok = True
            for y in chosen_list:
                s = x + y
                if cnt.get(s, 0) >= 2:
                    ok = False
                    break
            if ok:
                s2 = 2 * x
                if cnt.get(s2, 0) >= 2:
                    ok = False
            if ok:
                addable += 1
        return addable
    
    current_score = score(chosen, counts)
    
    while time.time() < deadline:
        # Try a random swap: remove one element, add another
        if not chosen:
            break
        idx = rng.randrange(len(chosen))
        removed = chosen[idx]
        # Find a candidate to add
        candidates = [x for x in range(1, 1001) if x not in chosen_set and x != removed]
        if not candidates:
            break
        # Try a few random candidates
        for _ in range(5):
            add_candidate = rng.choice(candidates)
            # Remove 'removed'
            new_chosen = [x for i, x in enumerate(chosen) if i != idx]
            new_set = set(new_chosen)
            # Recompute counts without 'removed'
            new_counts = {}
            for i, a in enumerate(new_chosen):
                for j in range(i, len(new_chosen)):
                    s = a + new_chosen[j]
                    new_counts[s] = new_counts.get(s, 0) + 1
            # Try to add add_candidate
            ok = True
            for y in new_chosen:
                s = add_candidate + y
                if new_counts.get(s, 0) >= 2:
                    ok = False
                    break
            if ok:
                s2 = 2 * add_candidate
                if new_counts.get(s2, 0) >= 2:
                    ok = False
            if ok:
                new_chosen.append(add_candidate)
                new_counts[2 * add_candidate] = new_counts.get(2 * add_candidate, 0) + 1
                for y in new_chosen:
                    if y != add_candidate:
                        s = add_candidate + y
                        new_counts[s] = new_counts.get(s, 0) + 1
                # Check validity
                if all(c <= 2 for c in new_counts.values()):
                    if len(new_chosen) > best_size:
                        best_size = len(new_chosen)
                        best_chosen = list(new_chosen)
                        best_counts = dict(new_counts)
                    # Accept the swap (hill climbing)
                    chosen = new_chosen
                    chosen_set = set(chosen)
                    counts = new_counts
                    current_score = score(chosen, counts)
                    break
    
    return best_chosen, best_counts, best_size

def construct():
    deadline = time.time() + 110  # 10s safety margin
    
    # Phase 1: Greedy construction
    chosen = _greedy_build()
    counts = _count_sums(chosen)
    best_size = len(chosen)
    best_chosen = list(chosen)
    best_counts = dict(counts)
    
    print(f"Greedy size: {best_size}", file=sys.stderr)
    
    # Phase 2: Local search with swap moves
    if time.time() < deadline:
        # Multiple restarts with different seeds
        for seed in range(20):
            if time.time() >= deadline:
                break
            # Start from greedy result
            c, cnt, sz = _local_search(list(best_chosen), dict(best_counts), deadline, seed=seed)
            if sz > best_size:
                best_size = sz
                best_chosen = c
                best_counts = cnt
                print(f"Improved to {best_size} at seed {seed}", file=sys.stderr)
    
    # Phase 3: Try to add remaining elements greedily from the best found
    if time.time() < deadline:
        chosen_set = set(best_chosen)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in range(1, 1001):
                if x in chosen_set:
                    continue
                ok, new_counts = _try_add(best_chosen, best_counts, x)
                if ok:
                    best_chosen.append(x)
                    chosen_set.add(x)
                    best_counts = new_counts
                    best_size += 1
                    improved = True
                    break
    
    print(f"Final size: {best_size}", file=sys.stderr)
    return best_chosen
