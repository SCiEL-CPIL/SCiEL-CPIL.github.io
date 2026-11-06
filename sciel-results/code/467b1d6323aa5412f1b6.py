import time
import random
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N_TRIPLES = len(ALL_TRIPLES)  # 560
N_WORDS = (N_TRIPLES + 63) // 64  # 9

# Precompute triple index lookup
_triple_idx = {}
for _i, _t in enumerate(ALL_TRIPLES):
    _triple_idx[_t] = _i

def _tidx(a, b, c):
    """Get index of triple {a,b,c} (all distinct)."""
    if a > b: a, b = b, a
    if b > c: b, c = c, b
    if a > b: a, b = b, a
    return _triple_idx[(a, b, c)]

def _has_presence(presence, idx):
    """Check if triple idx is in the presence bitmask."""
    return (presence[idx >> 6] >> (idx & 63)) & 1

def _check_conflict(a, b, c, presence):
    """Check if adding triple (a,b,c) creates a K(2,2,2)."""
    excluded = {a, b, c}
    
    # Iterate over possible a'
    for a_p in range(V):
        if a_p in excluded:
            continue
        
        # Check {a', b, c}
        if not _has_presence(presence, _tidx(a_p, b, c)):
            continue
            
        # Iterate over possible b'
        for b_p in range(V):
            if b_p in excluded or b_p == a_p:
                continue
                
            # Check {a, b', c}
            if not _has_presence(presence, _tidx(a, b_p, c)):
                continue
            # Check {a', b', c}
            if not _has_presence(presence, _tidx(a_p, b_p, c)):
                continue
            
            # Iterate over possible c'
            for c_p in range(V):
                if c_p in excluded or c_p == a_p or c_p == b_p:
                    continue
                
                # Check {a, b, c'}
                if not _has_presence(presence, _tidx(a, b, c_p)):
                    continue
                # Check {a, b', c'}
                if not _has_presence(presence, _tidx(a, b_p, c_p)):
                    continue
                # Check {a', b, c'}
                if not _has_presence(presence, _tidx(a_p, b, c_p)):
                    continue
                # Check {a', b', c'}
                if not _has_presence(presence, _tidx(a_p, b_p, c_p)):
                    continue
                
                # Conflict found
                return True
                
    return False

def _greedy(order, deadline):
    """Greedy: add triples in order, skipping conflicts."""
    presence = [0] * N_WORDS
    count = 0
    for idx in order:
        if time.time() >= deadline:
            break
        a, b, c = ALL_TRIPLES[idx]
        if _check_conflict(a, b, c, presence):
            continue
        presence[idx >> 6] |= (1 << (idx & 63))
        count += 1
    return count, presence

def _local_search(presence, count, deadline, max_iters=50):
    """Local search with Add and Swap moves."""
    present_set = set()
    for i in range(N_TRIPLES):
        if (presence[i >> 6] >> (i & 63)) & 1:
            present_set.add(i)
    
    absent_list = [i for i in range(N_TRIPLES) if i not in present_set]
    
    improved = True
    iter_count = 0
    while improved and iter_count < max_iters and time.time() < deadline:
        improved = False
        iter_count += 1
        
        # 1. Try to add any absent triple
        random.shuffle(absent_list)
        added = False
        for idx in absent_list:
            if time.time() >= deadline:
                return count, presence
            a, b, c = ALL_TRIPLES[idx]
            if not _check_conflict(a, b, c, presence):
                presence[idx >> 6] |= (1 << (idx & 63))
                present_set.add(idx)
                absent_list.remove(idx)
                count += 1
                improved = True
                added = True
                break # Restart loop to re-evaluate
        
        if added:
            continue
        
        # 2. If no add possible, try swaps (remove 1, add 1)
        # Pick a random absent triple
        if not absent_list:
            break
            
        random.shuffle(absent_list)
        new_idx = absent_list[0]
        a, b, c = ALL_TRIPLES[new_idx]
        
        # Find all triples that conflict with new_idx
        # A triple t_old conflicts if removing it allows new_idx to be added?
        # No, we want to find t_old such that {E - {t_old}} U {new_idx} is valid.
        # This is equivalent to: in E U {new_idx}, the only conflicts involving new_idx are resolved by removing t_old.
        # So, find all K(2,2,2)s that contain new_idx and some triples from E.
        # Let Conflicts be the set of such K(2,2,2)s.
        # We need to hit all these conflicts with a single triple t_old from E.
        # If such t_old exists, we can swap.
        
        # To find candidates for t_old:
        # Iterate through all present triples? Too slow.
        # Instead, identify the "witnesses" of the conflict.
        # If adding new_idx creates a conflict, there exists a' b' c' such that the 7 other triples are in E.
        # Let's find one such witness (a', b', c').
        witness_found = False
        witness_triples = []
        for a_p in range(V):
            if a_p in {a, b, c}: continue
            if not _has_presence(presence, _tidx(a_p, b, c)): continue
            for b_p in range(V):
                if b_p in {a, b, c, a_p}: continue
                if not _has_presence(presence, _tidx(a, b_p, c)): continue
                if not _has_presence(presence, _tidx(a_p, b_p, c)): continue
                for c_p in range(V):
                    if c_p in {a, b, c, a_p, b_p}: continue
                    if not _has_presence(presence, _tidx(a, b, c_p)): continue
                    if not _has_presence(presence, _tidx(a_p, b, c_p)): continue
                    if not _has_presence(presence, _tidx(a, b_p, c_p)): continue
                    if not _has_presence(presence, _tidx(a_p, b_p, c_p)): continue
                    # Found a witness K(2,2,2)
                    # The 7 triples in E are:
                    t1 = _tidx(a_p, b, c)
                    t2 = _tidx(a, b_p, c)
                    t3 = _tidx(a_p, b_p, c)
                    t4 = _tidx(a, b, c_p)
                    t5 = _tidx(a_p, b, c_p)
                    t6 = _tidx(a, b_p, c_p)
                    t7 = _tidx(a_p, b_p, c_p)
                    witness_triples = [t1, t2, t3, t4, t5, t6, t7]
                    witness_found = True
                    break
                if witness_found: break
            if witness_found: break
        
        if not witness_found:
            # This shouldn't happen if _check_conflict returned True, but _check_conflict isn't called here directly
            # We just picked a random absent triple. It might be addable.
            # But we already tried to add all absent triples in step 1.
            # So if we are here, new_idx is NOT addable.
            # So there MUST be a witness.
            pass

        # Now we have a witness set of 7 triples.
        # We can try to remove any of these 7 triples and see if new_idx becomes addable.
        # Note: Removing one might not be enough if there are multiple disjoint witnesses.
        # But let's try.
        swapped = False
        for t_rem in witness_triples:
            # Temporarily remove t_rem
            presence[t_rem >> 6] &= ~(1 << (t_rem & 63))
            present_set.remove(t_rem)
            
            # Check if new_idx is now addable
            if not _check_conflict(a, b, c, presence):
                # Swap successful
                presence[new_idx >> 6] |= (1 << (new_idx & 63))
                present_set.add(new_idx)
                absent_list.remove(new_idx)
                # Count remains same
                swapped = True
                improved = True
                break
            else:
                # Restore t_rem
                presence[t_rem >> 6] |= (1 << (t_rem & 63))
                present_set.add(t_rem)
                
        if not swapped:
            # Try a different absent triple? Or give up on this iteration?
            # We can try a few random absent triples.
            pass

    return count, presence

def construct():
    start = time.time()
    deadline = start + 115  # Safety margin
    
    best_count = 0
    best_presence = [0] * N_WORDS
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        order = list(range(N_TRIPLES))
        random.shuffle(order)
        count, presence = _greedy(order, deadline)
        
        # Local search
        count, presence = _local_search(presence, count, deadline)
        
        if count > best_count:
            best_count = count
            best_presence = presence[:]
            
        if time.time() > deadline - 5:
            break
    
    result = []
    for i in range(N_TRIPLES):
        if (best_presence[i >> 6] >> (i & 63)) & 1:
            result.append(list(ALL_TRIPLES[i]))
            
    return result

random.seed(42)
