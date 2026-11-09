"""Direct construction for 'Large sets with distinct subset products'."""

import time
import itertools

DEADLINE = 85.0  # wall-clock seconds (90s limit, safety margin)
T_START = time.time()

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

# Pre-compute vectors
BIT_ALLOC = {2: 5, 3: 4, 5: 3, 7: 2}
for p in PRIMES[4:]:
    BIT_ALLOC[p] = 2

SHIFTS = {}
shift = 0
for p in PRIMES:
    SHIFTS[p] = shift
    shift += BIT_ALLOC[p]

def get_vector(n):
    v = 0
    for p in PRIMES:
        e = 0
        while n % p == 0:
            e += 1
            n //= p
        if e > 0:
            v |= (e << SHIFTS[p])
    return v

CANDIDATES = [(i, get_vector(i)) for i in range(2, 41)]

def _verify(items):
    """Verify all 2^n subset products are distinct using exponent vectors."""
    n = len(items)
    vecs = [get_vector(x) for x in items]
    sums = {0}
    for v in vecs:
        new_sums = set()
        for s in sums:
            new_sums.add(s)
            new_sums.add(s + v)
        if len(new_sums) != 2 * len(sums):
            return False
        sums = new_sums
    return True

def _can_add(current_sums, vec):
    """Check if vec can be added to current_sums without collision."""
    new_sums = set()
    for s in current_sums:
        new_sums.add(s)
        new_sums.add(s + vec)
    return len(new_sums) == 2 * len(current_sums)

def _get_sums(vecs):
    """Compute all subset sums of exponent vectors."""
    sums = {0}
    for v in vecs:
        new_sums = set()
        for s in sums:
            new_sums.add(s)
            new_sums.add(s + v)
        sums = new_sums
    return sums

def _try_add(current_sums, vec):
    """Try to add vec; return new sums or None if collision."""
    new_sums = set()
    for s in current_sums:
        new_sums.add(s)
        new_sums.add(s + vec)
    if len(new_sums) == 2 * len(current_sums):
        return new_sums
    return None

def _branch_and_bound(candidates, current_vecs, current_sums, best_size, time_limit):
    """Branch and bound to find largest valid subset."""
    best = current_vecs[:]
    best_sums = current_sums
    
    def _search(idx, cur_vecs, cur_sums, cur_size):
        nonlocal best, best_sums
        if time.time() > time_limit:
            return
        if cur_size >= best_size:
            if cur_size > len(best):
                best = cur_vecs[:]
                best_sums = cur_sums
                best_size[0] = cur_size
        if idx == len(candidates):
            return
        
        # Try including candidates[idx]
        val, vec = candidates[idx]
        new_sums = _try_add(cur_sums, vec)
        if new_sums is not None:
            cur_vecs.append(vec)
            _search(idx + 1, cur_vecs, new_sums, cur_size + 1)
            cur_vecs.pop()
        
        # Try excluding candidates[idx]
        # Pruning: if even adding all remaining can't beat best, stop
        remaining = len(candidates) - idx - 1
        if cur_size + 1 + remaining <= best_size[0]:
            return
        _search(idx + 1, cur_vecs, cur_sums, cur_size)
    
    _search(0, current_vecs[:], current_sums, len(current_vecs))
    return best, best_sums

def construct():
    global DEADLINE, T_START
    
    # Greedy initialization: sort by value descending
    candidates_sorted = sorted(CANDIDATES, key=lambda x: x[0], reverse=True)
    
    current_items = []
    current_vecs = []
    current_sums = {0}
    
    for val, vec in candidates_sorted:
        new_sums = _try_add(current_sums, vec)
        if new_sums is not None:
            current_items.append(val)
            current_vecs.append(vec)
            current_sums = new_sums
            
    best_items = current_items[:]
    best_size = len(current_items)
    best_size_ref = [best_size]
    
    # Local Search with branch-and-bound
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        remaining = [c for c in CANDIDATES if c[0] not in current_items]
        
        # Try simple additions first
        for val, vec in remaining:
            new_sums = _try_add(current_sums, vec)
            if new_sums is not None:
                current_items.append(val)
                current_vecs.append(vec)
                current_sums = new_sums
                improved = True
                if len(current_items) > best_size:
                    best_size = len(current_items)
                    best_items = current_items[:]
                remaining = [c for c in CANDIDATES if c[0] not in current_items]
                break
        
        if improved:
            continue
            
        # Try swaps using vector-based check (fast)
        for i in range(len(current_items)):
            val_out, vec_out = current_items[i], current_vecs[i]
            for val_in, vec_in in remaining:
                if val_in == val_out:
                    continue
                # Check if we can swap: remove vec_out, add vec_in
                # First, compute sums without vec_out
                # This is expensive; instead, just try the swap and verify
                new_items = current_items[:]
                new_items.pop(i)
                new_items.append(val_in)
                
                # Fast vector-based verification
                new_vecs = [get_vector(v) for v in new_items]
                sums = {0}
                valid = True
                for v in new_vecs:
                    new_sums = set()
                    for s in sums:
                        new_sums.add(s)
                        new_sums.add(s + v)
                    if len(new_sums) != 2 * len(sums):
                        valid = False
                        break
                    sums = new_sums
                
                if valid:
                    current_items = new_items
                    current_vecs = new_vecs
                    current_sums = sums
                    improved = True
                    if len(current_items) > best_size:
                        best_size = len(current_items)
                        best_items = current_items[:]
                    remaining = [c for c in CANDIDATES if c[0] not in current_items]
                    break
            if improved:
                break
        
        # Try branch-and-bound on remaining candidates
        if not improved and time.time() < DEADLINE - 5:
            # Use B&B to try to find a larger set
            bb_best_vecs, bb_best_sums = _branch_and_bound(
                remaining, current_vecs, current_sums, best_size_ref, DEADLINE - 2
            )
            if len(bb_best_vecs) > best_size:
                # Reconstruct items from vectors
                vec_to_val = {v: val for val, v in CANDIDATES}
                new_items = [vec_to_val[v] for v in bb_best_vecs]
                if _verify(new_items):
                    current_items = new_items
                    current_vecs = bb_best_vecs[:]
                    current_sums = bb_best_sums
                    best_size = len(current_items)
                    best_items = current_items[:]
                    improved = True
    
    # Final verification
    if not _verify(best_items):
        best_items = [int(p) for p in PRIMES]
        
    return [int(x) for x in best_items]
