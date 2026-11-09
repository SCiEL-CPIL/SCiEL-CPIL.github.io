"""Improved program: search-based approach for small inclusion-maximal Sidon sets in [1,256]."""
import time
import random
import numpy as np

N = 256
DEADLINE = time.time() + 25  # 25s safety margin (evaluator kills at 30s)

# EVOLVE-BLOCK-START
def _conflicts_valid(selected, sums):
    """Check if every missing element conflicts with the Sidon set (maximality)."""
    sset = set(selected)
    for x in range(1, N + 1):
        if x in sset:
            continue
        # x conflicts if adding it creates a duplicate sum
        # New sums involving x: x+a for a in selected (a<=x or a>x both count as unordered), and 2x
        # Actually unordered pairs: for each a in selected, sum x+a; and 2x
        # x conflicts iff any of these sums is already in `sums`, OR two of the new sums collide
        new_sums = [2 * x]
        for a in selected:
            new_sums.append(x + a)
        # Check collision with existing sums
        if any(s in sums for s in new_sums):
            continue  # x conflicts (good for maximality)
        # Check internal collision among new sums
        if len(set(new_sums)) != len(new_sums):
            continue  # x conflicts
        # x does NOT conflict -> not maximal
        return False
    return True

def _build_sums(selected):
    """Build the set of all unordered pair sums a+b with a<=b."""
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            sums.add(selected[i] + selected[j])
    return sums

def _is_valid_sidon(selected):
    """Verify all unordered sums a+b (a<=b) are distinct."""
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            s = selected[i] + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _greedy_randomize(perm):
    """Greedy: process elements in given order, add if keeps Sidon property."""
    selected = []
    sums = set()
    for x in perm:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected, sums

def _is_maximal(selected, sums):
    """Check inclusion maximality."""
    sset = set(selected)
    for x in range(1, N + 1):
        if x in sset:
            continue
        new_sums = [2 * x] + [x + a for a in selected]
        if any(s in sums for s in new_sums):
            continue
        if len(set(new_sums)) != len(new_sums):
            continue
        return False  # x can be added
    return True

def _try_remove(selected, sums, idx):
    """Try removing element at idx; return new (selected, sums) if still valid Sidon, else None."""
    # Remove selected[idx]
    new_sel = selected[:idx] + selected[idx+1:]
    # Rebuild sums
    new_sums = set()
    for i in range(len(new_sel)):
        for j in range(i, len(new_sel)):
            s = new_sel[i] + new_sel[j]
            if s in new_sums:
                return None  # shouldn't happen if original was valid
            new_sums.add(s)
    return new_sel, new_sums

def _try_remove_fast(selected, sums, idx):
    """Remove element at idx; return (new_sel, new_sums) if still Sidon, else None.

    Incremental: removing an element only ever removes sums, never adds new
    collisions, so validity is guaranteed. Rebuild is O(n^2) but only when the
    removal is actually accepted, which is rare."""
    new_sel = selected[:idx] + selected[idx+1:]
    new_sums = set()
    for i in range(len(new_sel)):
        for j in range(i, len(new_sel)):
            new_sums.add(new_sel[i] + new_sel[j])
    return new_sel, new_sums

def _add_to_set(selected, sums, x):
    """Try adding x; return (new_sel, new_sums) if still Sidon, else None."""
    added = [2 * x] + [x + a for a in selected]
    if len(set(added)) != len(added):
        return None
    if sums.intersection(added):
        return None
    new_sel = selected + [x]
    new_sums = sums.union(added)
    return new_sel, new_sums

def _local_search(selected, sums, rng, time_limit):
    """Local search with simulated annealing: removals, swaps, and add-then-remove
    moves, with stochastic acceptance to escape local optima."""
    best = (len(selected), list(selected), set(sums))
    cur_sel, cur_sums = list(selected), set(sums)
    start = time.time()
    T = 1.0
    T_min = 0.01
    cooling = 0.995

    while time.time() - start < time_limit and time.time() < DEADLINE:
        if T < T_min:
            T = T_min
        # Pick a move
        r = rng.random()
        if r < 0.4:
            # Removal move
            if len(cur_sel) <= 1:
                T *= cooling
                continue
            i = rng.randrange(len(cur_sel))
            new_sel, new_sums = _try_remove_fast(cur_sel, cur_sums, i)
            if new_sel is None:
                T *= cooling
                continue
            new_valid = True
            new_max = _is_maximal(new_sel, new_sums)
            # Score: prefer smaller, valid, maximal
            new_cost = len(new_sel) + (0 if new_max else 100)
            cur_cost = len(cur_sel) + (0 if _is_maximal(cur_sel, cur_sums) else 100)
            delta = new_cost - cur_cost
            if delta <= 0 or rng.random() < np.exp(-delta / max(T, 1e-9)):
                cur_sel, cur_sums = new_sel, new_sums
                if new_max and len(cur_sel) < best[0]:
                    best = (len(cur_sel), list(cur_sel), set(cur_sums))
        elif r < 0.7:
            # Swap: remove one, add another
            if len(cur_sel) <= 1:
                T *= cooling
                continue
            i = rng.randrange(len(cur_sel))
            removed = cur_sel[i]
            new_sel = cur_sel[:i] + cur_sel[i+1:]
            new_sums = set()
            for a in range(len(new_sel)):
                for b in range(a, len(new_sel)):
                    new_sums.add(new_sel[a] + new_sel[b])
            # Pick a candidate to add
            candidates = [x for x in range(1, N+1) if x not in new_sel]
            if not candidates:
                T *= cooling
                continue
            cand = rng.choice(candidates)
            res = _add_to_set(new_sel, new_sums, cand)
            if res is None:
                T *= cooling
                continue
            new_sel2, new_sums2 = res
            new_max = _is_maximal(new_sel2, new_sums2)
            new_cost = len(new_sel2) + (0 if new_max else 100)
            cur_cost = len(cur_sel) + (0 if _is_maximal(cur_sel, cur_sums) else 100)
            delta = new_cost - cur_cost
            if delta <= 0 or rng.random() < np.exp(-delta / max(T, 1e-9)):
                cur_sel, cur_sums = new_sel2, new_sums2
                if new_max and len(cur_sel) < best[0]:
                    best = (len(cur_sel), list(cur_sel), set(cur_sums))
        else:
            # Add-then-remove: add a random element (may break Sidon), then try
            # to remove a different element to restore validity while reducing size
            if len(cur_sel) <= 1:
                T *= cooling
                continue
            candidates = [x for x in range(1, N+1) if x not in cur_sel]
            if not candidates:
                T *= cooling
                continue
            cand = rng.choice(candidates)
            # Check if adding cand keeps Sidon
            added = [2*cand] + [cand+a for a in cur_sel]
            if len(set(added)) != len(added) or cur_sums.intersection(added):
                T *= cooling
                continue
            # Now try to remove a random other element
            i = rng.randrange(len(cur_sel))
            new_sel = cur_sel[:i] + cur_sel[i+1:] + [cand]
            new_sums = set()
            for a in range(len(new_sel)):
                for b in range(a, len(new_sel)):
                    s = new_sel[a] + new_sel[b]
                    new_sums.add(s)
            new_max = _is_maximal(new_sel, new_sums)
            new_cost = len(new_sel) + (0 if new_max else 100)
            cur_cost = len(cur_sel) + (0 if _is_maximal(cur_sel, cur_sums) else 100)
            delta = new_cost - cur_cost
            if delta <= 0 or rng.random() < np.exp(-delta / max(T, 1e-9)):
                cur_sel, cur_sums = new_sel, new_sums
                if new_max and len(cur_sel) < best[0]:
                    best = (len(cur_sel), list(cur_sel), set(cur_sums))
        T *= cooling
    return best

def construct():
    rng = random.Random(42) # Fixed seed for reproducibility in debugging, but random for general? No, random.Random() uses system time. Let's use random.Random()
    rng = random.Random()
    best_size = N + 1
    best_result = None
    
    # Pre-compute nothing, just run fast greedy loops
    # Greedy is O(N^2) approx. 256^2 = 65k ops. Very fast.
    # Maximality check is O(N^2 * |S|). 256^2 * 20 = 1.3M ops. Also fast.
    
    while time.time() < DEADLINE - 1:
        perm = list(range(1, N + 1))
        rng.shuffle(perm)
        
        selected, sums = _greedy_randomize(perm)
        
        # Quick check: if it's larger than best, skip expensive maximality check
        if len(selected) >= best_size:
            continue
        
        # Check maximality
        # _is_maximal is the bottleneck. 
        # We can optimize _is_maximal by checking conflicts more efficiently?
        # For now, just call it.
        
        if _is_maximal(selected, sums):
            best_size = len(selected)
            best_result = list(selected)
            
            # If we found a very small set, we can break or continue to try for smaller
            # Theoretical lower bound for maximal Sidon set in [1, N] is ~sqrt(N) or so?
            # For N=256, sqrt(N)=16. 
            # If we find size 16, it's likely near optimal.
            if best_size <= 16:
                # Do a few more iterations to see if we can beat it
                for _ in range(10):
                    if time.time() > DEADLINE - 1: break
                    perm = list(range(1, N + 1))
                    rng.shuffle(perm)
                    selected, sums = _greedy_randomize(perm)
                    if len(selected) < best_size and _is_maximal(selected, sums):
                        best_size = len(selected)
                        best_result = list(selected)
                break

    if best_result is None:
        # Fallback: deterministic greedy
        selected, sums = [], set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        best_result = selected
        best_size = len(selected)
    
    return best_result
# EVOLVE-BLOCK-END
