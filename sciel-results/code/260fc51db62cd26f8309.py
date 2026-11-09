"""Search for a size-efficient 16-wire sorting network (fewer than 120 comparators)."""
import time
import numpy as np
import random

N = 16
ALL_INPUTS = 1 << N  # 65536
TIME_LIMIT = 85.0    # safety margin under 90s kill
_start = time.time()


def _elapsed():
    return time.time() - _start


def _apply_comparators(inputs, comps):
    """Apply a list of (i,j) comparators to a (65536, 16) uint8 matrix.
    Returns the matrix after all comparators (min->i, max->j)."""
    x = inputs
    for i, j in comps:
        a = x[:, i]
        b = x[:, j]
        mn = np.minimum(a, b)
        mx = np.maximum(a, b)
        x[:, i] = mn
        x[:, j] = mx
    return x


def _is_sorted(x):
    """True if every row is non-decreasing (sorted)."""
    return bool(np.all(x[:, :-1] <= x[:, 1:]))


def _verify(comps):
    """Return True if the comparator list sorts all 65536 Boolean inputs."""
    # Build all Boolean inputs as a (65536, 16) uint8 matrix
    bits = np.arange(ALL_INPUTS, dtype=np.uint32)[:, None]
    shifts = (np.arange(N - 1, -1, -1, dtype=np.uint32))[None, :]
    x = ((bits >> shifts) & 1).astype(np.uint8)
    x = _apply_comparators(x, comps)
    return _is_sorted(x)


def _baseline_network():
    """Odd-even transposition network: 120 comparators."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]


def _all_pairs():
    return [(i, j) for i in range(N) for j in range(i + 1, N)]


def _greedy_deletion(cur):
    """Perform greedy deletion until no more comparators can be removed."""
    changed = True
    while changed and _elapsed() < TIME_LIMIT:
        changed = False
        for idx in range(len(cur) - 1, -1, -1):
            if _elapsed() > TIME_LIMIT:
                return cur
            cand = cur[:idx] + cur[idx + 1:]
            if _verify(cand):
                cur = cand
                changed = True
    return cur


def construct():
    best = _baseline_network()
    best_len = len(best)
    
    # Initialize with baseline
    cur = list(best)
    
    # Phase 1: Greedy Deletion
    cur = _greedy_deletion(cur)
    if len(cur) < best_len:
        best = cur
        best_len = len(best)
        
    # Phase 2: Randomized Local Search with Kicks
    rng = random.Random(42)
    all_pairs = _all_pairs()
    pair_set = set(all_pairs)
    
    # We will run a loop that tries to improve the current best
    # If stuck, we perform a "kick" by removing a few comparators
    # and then greedily restoring validity while trying to keep size small.
    
    cur = list(best)
    stuck_count = 0
    
    while _elapsed() < TIME_LIMIT:
        # Try simple deletions
        improved = False
        for idx in range(len(cur) - 1, -1, -1):
            if _elapsed() > TIME_LIMIT:
                break
            cand = cur[:idx] + cur[idx + 1:]
            if _verify(cand):
                cur = cand
                improved = True
                if len(cur) < best_len:
                    best = list(cur)
                    best_len = len(cur)
        
        if improved:
            stuck_count = 0
            continue
            
        # If no simple deletion worked, try a kick
        stuck_count += 1
        if stuck_count > 5: # Try kick if stuck for a few iterations
            # Remove 3 random comparators
            kick_size = 3
            if len(cur) > kick_size:
                remove_indices = rng.sample(range(len(cur)), kick_size)
                remove_indices.sort(reverse=True)
                for idx in remove_indices:
                    cur.pop(idx)
                
                # Now cur is likely invalid. Try to restore it.
                # We want to restore it with as few additions as possible.
                # Greedy restoration:
                # We have a set of "candidates" to add.
                # Start with the ones we removed, plus some random others.
                
                removed_comps = [best[i] for i in remove_indices] # Wait, we popped from cur, but best is the reference.
                # Actually, we should just try to add comparators back.
                
                # Strategy:
                # 1. Try to add back the removed ones in random order?
                # 2. Or try to add ANY comparator that helps.
                
                # Let's try a limited search to restore validity.
                # State: cur (invalid)
                # Goal: Find a superset of cur that is valid and small.
                
                # We will try to add comparators one by one.
                # At each step, pick a random comparator from all_pairs not in cur.
                # If adding it makes it valid, we stop.
                # If not, we keep it and continue.
                
                # To avoid getting too big, we limit the number of additions.
                max_additions = kick_size + 2 
                added = []
                
                for _ in range(max_additions):
                    if _elapsed() > TIME_LIMIT:
                        break
                    # Pick a random comparator to add
                    # Prioritize the ones we removed?
                    candidates = [p for p in all_pairs if p not in cur]
                    if not candidates:
                        break
                    # 50% chance to pick one of the removed ones if available
                    if removed_comps and rng.random() < 0.5:
                        rc = removed_comps[rng.randrange(len(removed_comps))]
                        if rc in candidates:
                            c = rc
                        else:
                            c = rng.choice(candidates)
                    else:
                        c = rng.choice(candidates)
                    
                    cur.append(c)
                    added.append(c)
                    
                    if _verify(cur):
                        # We restored validity
                        # Now try to delete any redundant comparators
                        cur = _greedy_deletion(cur)
                        if len(cur) < best_len:
                            best = list(cur)
                            best_len = len(best)
                        stuck_count = 0
                        break
                else:
                    # Failed to restore validity within limit
                    # Reset to best
                    cur = list(best)
                    stuck_count = 0
            else:
                # Cannot kick if too small
                break
                
    # Final check
    if not _verify(best):
        best = _baseline_network()
        
    return best
