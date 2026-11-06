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
    """Check if adding triple (a,b,c) creates a K(2,2,2).
    
    K(2,2,2): 6 distinct vertices a,a',b,b',c,c' with all 8 transversal triples.
    We fix a,b,c as one corner and look for a',b',c' such that all 8 exist.
    """
    for a_p in range(V):
        if a_p == a or a_p == b or a_p == c:
            continue
        # Easy check 1: (a_p, b, c) must be present
        if not _has_presence(presence, _tidx(a_p, b, c)):
            continue
        # Find valid b_p: distinct from a,b,c,a_p; (a,b_p,c) and (a_p,b_p,c) present
        valid_bp = []
        for b_p in range(V):
            if b_p == a or b_p == b or b_p == c or b_p == a_p:
                continue
            if not _has_presence(presence, _tidx(a, b_p, c)):
                continue
            if not _has_presence(presence, _tidx(a_p, b_p, c)):
                continue
            valid_bp.append(b_p)
        if not valid_bp:
            continue
        # Find valid c_p: distinct from a,b,c,a_p; (a,b,c_p) and (a_p,b,c_p) present
        valid_cp = []
        for c_p in range(V):
            if c_p == a or c_p == b or c_p == c or c_p == a_p:
                continue
            if not _has_presence(presence, _tidx(a, b, c_p)):
                continue
            if not _has_presence(presence, _tidx(a_p, b, c_p)):
                continue
            valid_cp.append(c_p)
        if not valid_cp:
            continue
        # Check pairs (b_p, c_p) with b_p != c_p
        for b_p in valid_bp:
            for c_p in valid_cp:
                if b_p == c_p:
                    continue
                if not _has_presence(presence, _tidx(a, b_p, c_p)):
                    continue
                if not _has_presence(presence, _tidx(a_p, b_p, c_p)):
                    continue
                return True
    return False

def _greedy(order, deadline):
    """Greedy: add triples in order, skipping conflicts. Returns (count, presence)."""
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

def _local_search(presence, count, deadline, max_iters=100):
    """Local search: try to add missing triples, remove+add to resolve conflicts.
    Returns (count, presence)."""
    # Build set of present triple indices for quick lookup
    present_set = set()
    for i in range(N_TRIPLES):
        if (presence[i >> 6] >> (i & 63)) & 1:
            present_set.add(i)
    
    # Build list of absent triple indices
    absent_list = [i for i in range(N_TRIPLES) if i not in present_set]
    
    improved = True
    iter_count = 0
    while improved and iter_count < max_iters and time.time() < deadline:
        improved = False
        iter_count += 1
        
        # Phase 1: Try to add any absent triple that doesn't create conflict
        random.shuffle(absent_list)
        added_any = False
        new_absent = []
        for idx in absent_list:
            if time.time() >= deadline:
                return count, presence
            a, b, c = ALL_TRIPLES[idx]
            if not _check_conflict(a, b, c, presence):
                presence[idx >> 6] |= (1 << (idx & 63))
                present_set.add(idx)
                count += 1
                improved = True
                added_any = True
            else:
                new_absent.append(idx)
        if added_any:
            absent_list = new_absent
        
        # Phase 2: Try remove+add: remove one present triple, then try to add
        # multiple absent triples to net gain
        if not absent_list:
            break
        present_list = list(present_set)
        random.shuffle(present_list)
        for rem_idx in present_list:
            if time.time() >= deadline:
                return count, presence
            # Temporarily remove
            presence[rem_idx >> 6] &= ~(1 << (rem_idx & 63))
            present_set.discard(rem_idx)
            
            # Try to add all absent triples that fit
            added = []
            new_absent = []
            for add_idx in absent_list:
                if time.time() >= deadline:
                    break
                a, b, c = ALL_TRIPLES[add_idx]
                if not _check_conflict(a, b, c, presence):
                    presence[add_idx >> 6] |= (1 << (add_idx & 63))
                    present_set.add(add_idx)
                    added.append(add_idx)
                else:
                    new_absent.append(add_idx)
            
            # Net gain = len(added) - 1
            if len(added) > 1:
                # Keep the change
                count += len(added) - 1
                improved = True
                absent_list = new_absent
            else:
                # Revert
                presence[rem_idx >> 6] |= (1 << (rem_idx & 63))
                present_set.add(rem_idx)
                for add_idx in added:
                    presence[add_idx >> 6] &= ~(1 << (add_idx & 63))
                    present_set.discard(add_idx)
                # absent_list stays the same since we reverted
    
    return count, presence

def _perturb(presence, count, n_remove, deadline):
    """Perturb a solution by removing n_remove random triples and re-optimizing.
    Returns (count, presence)."""
    # Build set of present triple indices
    present_list = []
    for i in range(N_TRIPLES):
        if (presence[i >> 6] >> (i & 63)) & 1:
            present_list.append(i)
    
    if not present_list:
        return 0, presence
    
    # Remove n_remove random triples
    n_remove = min(n_remove, len(present_list))
    to_remove = random.sample(present_list, n_remove)
    for idx in to_remove:
        presence[idx >> 6] &= ~(1 << (idx & 63))
    count -= n_remove
    
    # Re-optimize with local search
    count, presence = _local_search(presence, count, deadline, max_iters=50)
    return count, presence

def construct():
    start = time.time()
    deadline = start + 115  # safety margin for 150s wall-clock limit
    
    best_count = 0
    best_presence = [0] * N_WORDS
    
    # Phase 1: Initial greedy + local search to get a good starting point
    order = list(range(N_TRIPLES))
    random.shuffle(order)
    count, presence = _greedy(order, deadline)
    count, presence = _local_search(presence, count, deadline, max_iters=80)
    
    if count > best_count:
        best_count = count
        best_presence = presence[:]
    
    # Phase 2: Perturbation-based restarts from best solution
    perturb_round = 0
    while time.time() < deadline - 10:
        perturb_round += 1
        
        # Determine how many triples to remove (increasing perturbation over time)
        base_remove = 3 + perturb_round * 2
        n_remove = min(base_remove, max(3, best_count // 4))
        
        # Perturb and re-optimize
        pert_count, pert_presence = _perturb(best_presence[:], best_count, n_remove, deadline)
        
        if pert_count > best_count:
            best_count = pert_count
            best_presence = pert_presence[:]
        
        # Occasionally do a fresh random restart if stuck
        if perturb_round % 5 == 0 and time.time() < deadline - 20:
            order = list(range(N_TRIPLES))
            random.shuffle(order)
            count, presence = _greedy(order, deadline)
            count, presence = _local_search(presence, count, deadline, max_iters=40)
            if count > best_count:
                best_count = count
                best_presence = presence[:]
    
    # Final polish: try to add any remaining triples
    if time.time() < deadline - 2:
        final_count, final_presence = _local_search(best_presence[:], best_count, deadline, max_iters=30)
        if final_count > best_count:
            best_count = final_count
            best_presence = final_presence[:]
    
    # Extract triples
    result = []
    for i in range(N_TRIPLES):
        if (best_presence[i >> 6] >> (i & 63)) & 1:
            result.append(list(ALL_TRIPLES[i]))
    return result

# Pre-seed random for reproducibility across runs is NOT desired; we want variety.
# But we do want a reasonable seed for the first call.
random.seed(42)
