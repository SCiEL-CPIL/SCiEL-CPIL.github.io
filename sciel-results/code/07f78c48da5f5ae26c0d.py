import time
import random
import math
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

def _find_witnesses(a, b, c, presence, max_witnesses=10):
    """Find up to max_witnesses K(2,2,2)s that block triple (a,b,c)."""
    witnesses = []
    excluded = {a, b, c}
    
    # Precompute indices for speed
    abc = _tidx(a, b, c)
    
    for a_p in range(V):
        if a_p in excluded:
            continue
        idx_apbc = _tidx(a_p, b, c)
        if not _has_presence(presence, idx_apbc):
            continue
        for b_p in range(V):
            if b_p in excluded or b_p == a_p:
                continue
            idx_abpc = _tidx(a, b_p, c)
            if not _has_presence(presence, idx_abpc):
                continue
            idx_apbpc = _tidx(a_p, b_p, c)
            if not _has_presence(presence, idx_apbpc):
                continue
            for c_p in range(V):
                if c_p in excluded or c_p == a_p or c_p == b_p:
                    continue
                idx_abcp = _tidx(a, b, c_p)
                if not _has_presence(presence, idx_abcp):
                    continue
                idx_apbcp = _tidx(a_p, b, c_p)
                if not _has_presence(presence, idx_apbcp):
                    continue
                idx_abpcp = _tidx(a, b_p, c_p)
                if not _has_presence(presence, idx_abpcp):
                    continue
                idx_apbpcp = _tidx(a_p, b_p, c_p)
                if not _has_presence(presence, idx_apbpcp):
                    continue
                
                # Found a witness
                witness = [
                    idx_apbc,
                    idx_abpc,
                    idx_apbpc,
                    idx_abcp,
                    idx_apbcp,
                    idx_abpcp,
                    idx_apbpcp
                ]
                witnesses.append(witness)
                if len(witnesses) >= max_witnesses:
                    return witnesses
    return witnesses

def _greedy(order, deadline):
    """Greedy: add triples in order, skipping conflicts."""
    presence = [0] * N_WORDS
    count = 0
    for idx in order:
        if time.time() >= deadline:
            break
        a, b, c = ALL_TRIPLES[idx]
        witnesses = _find_witnesses(a, b, c, presence, 1)
        if witnesses:
            continue
        presence[idx >> 6] |= (1 << (idx & 63))
        count += 1
    return count, presence

def _simulated_annealing(presence, count, deadline):
    """Simulated annealing with add, remove, and swap moves."""
    present_set = set()
    for i in range(N_TRIPLES):
        if (presence[i >> 6] >> (i & 63)) & 1:
            present_set.add(i)
    
    absent_list = [i for i in range(N_TRIPLES) if i not in present_set]
    
    # Initial temperature
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.99
    
    best_count = count
    best_presence = presence[:]
    best_present_set = set(present_set)
    
    # Precompute for speed
    triples = ALL_TRIPLES
    
    while T > T_min and time.time() < deadline:
        # Choose a move type with bias toward swaps
        move_type = random.random()
        
        if move_type < 0.3:
            # Add move: try to add a random absent triple
            if not absent_list:
                T *= cooling_rate
                continue
            new_idx = random.choice(absent_list)
            a, b, c = triples[new_idx]
            witnesses = _find_witnesses(a, b, c, presence, 1)
            if not witnesses:
                # Accept
                presence[new_idx >> 6] |= (1 << (new_idx & 63))
                present_set.add(new_idx)
                absent_list.remove(new_idx)
                count += 1
                if count > best_count:
                    best_count = count
                    best_presence = presence[:]
                    best_present_set = set(present_set)
        elif move_type < 0.5:
            # Remove move: remove a random present triple
            if not present_set:
                T *= cooling_rate
                continue
            old_idx = random.choice(list(present_set))
            # Remove
            presence[old_idx >> 6] &= ~(1 << (old_idx & 63))
            present_set.remove(old_idx)
            absent_list.append(old_idx)
            count -= 1
        else:
            # Swap move: remove one, add one (more frequent)
            if not present_set or not absent_list:
                T *= cooling_rate
                continue
            new_idx = random.choice(absent_list)
            a, b, c = triples[new_idx]
            witnesses = _find_witnesses(a, b, c, presence, 15)
            if not witnesses:
                # Just add
                presence[new_idx >> 6] |= (1 << (new_idx & 63))
                present_set.add(new_idx)
                absent_list.remove(new_idx)
                count += 1
                if count > best_count:
                    best_count = count
                    best_presence = presence[:]
                    best_present_set = set(present_set)
            else:
                # Try swapping: remove a candidate from witnesses
                candidates = set()
                for w in witnesses:
                    for t in w:
                        candidates.add(t)
                candidates = [t for t in candidates if t in present_set]
                if candidates:
                    random.shuffle(candidates)
                    for t_rem in candidates[:5]:  # Try up to 5 candidates
                        # Remove t_rem
                        presence[t_rem >> 6] &= ~(1 << (t_rem & 63))
                        present_set.remove(t_rem)
                        # Check if new_idx is now addable
                        remaining_witnesses = _find_witnesses(a, b, c, presence, 1)
                        if not remaining_witnesses:
                            # Swap successful
                            presence[new_idx >> 6] |= (1 << (new_idx & 63))
                            present_set.add(new_idx)
                            absent_list.remove(new_idx)
                            # count stays the same
                            if count > best_count:
                                best_count = count
                                best_presence = presence[:]
                                best_present_set = set(present_set)
                            break
                        else:
                            # Restore t_rem
                            presence[t_rem >> 6] |= (1 << (t_rem & 63))
                            present_set.add(t_rem)
        
        T *= cooling_rate
    
    # Return best solution found during SA
    # Restore best
    for i in range(N_TRIPLES):
        bit = 1 << (i & 63)
        word = i >> 6
        if i in best_present_set:
            presence[word] |= bit
        else:
            presence[word] &= ~bit
    
    return best_count, presence

def _local_search(presence, count, deadline):
    """Local search with Add and Swap moves."""
    present_set = set()
    for i in range(N_TRIPLES):
        if (presence[i >> 6] >> (i & 63)) & 1:
            present_set.add(i)
    
    absent_list = [i for i in range(N_TRIPLES) if i not in present_set]
    
    improved = True
    iter_count = 0
    max_iters = 100
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
            witnesses = _find_witnesses(a, b, c, presence, 1)
            if not witnesses:
                presence[idx >> 6] |= (1 << (idx & 63))
                present_set.add(idx)
                absent_list.remove(idx)
                count += 1
                improved = True
                added = True
                break
        
        if added:
            continue
        
        # 2. If no add possible, try swaps (remove 1, add 1)
        if not absent_list:
            break
            
        random.shuffle(absent_list)
        swapped = False
        for new_idx in absent_list:
            if time.time() >= deadline:
                return count, presence
            a, b, c = ALL_TRIPLES[new_idx]
            
            witnesses = _find_witnesses(a, b, c, presence, 20)
            if not witnesses:
                presence[new_idx >> 6] |= (1 << (new_idx & 63))
                present_set.add(new_idx)
                absent_list.remove(new_idx)
                count += 1
                improved = True
                swapped = True
                break

            candidates = set()
            for w in witnesses:
                for t in w:
                    candidates.add(t)
            candidates = list(candidates)
            random.shuffle(candidates)
            
            for t_rem in candidates:
                if time.time() >= deadline:
                    return count, presence
                if t_rem not in present_set:
                    continue
                
                presence[t_rem >> 6] &= ~(1 << (t_rem & 63))
                present_set.remove(t_rem)
                
                remaining_witnesses = _find_witnesses(a, b, c, presence, 1)
                
                if not remaining_witnesses:
                    presence[new_idx >> 6] |= (1 << (new_idx & 63))
                    present_set.add(new_idx)
                    absent_list.remove(new_idx)
                    swapped = True
                    improved = True
                    break
                else:
                    presence[t_rem >> 6] |= (1 << (t_rem & 63))
                    present_set.add(t_rem)
                    
            if swapped:
                break
        
        if not swapped:
            break

    return count, presence

def _perturb_and_improve(presence, count, deadline):
    """Perturb current solution and try to improve."""
    present_set = set()
    for i in range(N_TRIPLES):
        if (presence[i >> 6] >> (i & 63)) & 1:
            present_set.add(i)
    
    # Remove 2-3 random triples to perturb
    num_remove = random.randint(2, 3)
    to_remove = random.sample(list(present_set), min(num_remove, len(present_set)))
    for idx in to_remove:
        presence[idx >> 6] &= ~(1 << (idx & 63))
        present_set.remove(idx)
        count -= 1
    
    # Now try to improve with local search
    count, presence = _local_search(presence, count, deadline)
    return count, presence

def construct():
    start = time.time()
    deadline = start + 115
    
    best_count = 0
    best_presence = [0] * N_WORDS
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        if attempts == 1:
            # First attempt: fresh greedy
            order = list(range(N_TRIPLES))
            random.shuffle(order)
            count, presence = _greedy(order, deadline)
        else:
            # Subsequent attempts: perturb best solution
            if best_count > 0:
                count, presence = _perturb_and_improve(best_presence[:], best_count, deadline)
            else:
                order = list(range(N_TRIPLES))
                random.shuffle(order)
                count, presence = _greedy(order, deadline)
        
        # Local search first
        count, presence = _local_search(presence, count, deadline)
        
        # Then simulated annealing if time permits
        if time.time() < deadline - 15:
            count, presence = _simulated_annealing(presence, count, deadline)
        
        # Final local search polish
        if time.time() < deadline - 5:
            count, presence = _local_search(presence, count, deadline)
        
        if count > best_count:
            best_count = count
            best_presence = presence[:]
        
        # Stop if we're running out of time
        if time.time() > deadline - 8:
            break
    
    result = []
    for i in range(N_TRIPLES):
        if (best_presence[i >> 6] >> (i & 63)) & 1:
            result.append(list(ALL_TRIPLES[i]))
            
    return result

random.seed(42)
