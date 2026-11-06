"""Optimized 4-AP-free set construction using bit-parallel checks and local search."""
import time
import random
import numpy as np

N = 512

def _build_conflict_lists():
    """For each element x (1-indexed), precompute list of triples (a,b,c) such that
    if a, b, c are all in the set, then adding x creates a 4-AP."""
    conflicts = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > N:
                break
            conflicts[a].append((b, c, e))
            conflicts[b].append((a, c, e))
            conflicts[c].append((a, b, e))
            conflicts[e].append((a, b, c))
    return conflicts

def _can_add(x, bitset, conflicts):
    """Check if x can be added given current set in bitset."""
    for (a, b, c) in conflicts[x]:
        if (bitset[(a-1)//64] >> ((a-1)%64)) & 1 and \
           (bitset[(b-1)//64] >> ((b-1)%64)) & 1 and \
           (bitset[(c-1)//64] >> ((c-1)%64)) & 1:
            return False
    return True

def _greedy_randomized(conflicts, seed, time_limit=0.8):
    rng = random.Random(seed)
    bitset = np.zeros(N // 64 + 1, dtype=np.uint64)
    count = 0
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    
    start = time.time()
    for x in candidates:
        if time.time() - start > time_limit:
            break
        if _can_add(x, bitset, conflicts):
            bitset[(x-1)//64] |= (1 << ((x-1)%64))
            count += 1
    return bitset, count

def _local_search(bitset, conflicts, time_limit=5.0):
    """Local search: add elements, then try 1-swap moves."""
    start = time.time()
    count = int(np.sum([(bitset[i] != 0) for i in range(len(bitset))]))  # rough
    # Better: count bits
    count = 0
    for i in range(len(bitset)):
        count += bin(bitset[i]).count('1')
    
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        
        # Phase 1: try adding any element
        for x in range(1, N + 1):
            if time.time() - start > time_limit:
                break
            if (bitset[(x-1)//64] >> ((x-1)%64)) & 1:
                continue
            if _can_add(x, bitset, conflicts):
                bitset[(x-1)//64] |= (1 << ((x-1)%64))
                count += 1
                improved = True
        
        # Phase 2: try swaps
        if time.time() - start > time_limit:
            break
        for x_rm in range(1, N + 1):
            if time.time() - start > time_limit:
                break
            if not ((bitset[(x_rm-1)//64] >> ((x_rm-1)%64)) & 1):
                continue
            # Remove x_rm
            bitset[(x_rm-1)//64] &= ~(1 << ((x_rm-1)%64))
            count -= 1
            
            # Try to add something
            added = False
            for x_add in range(1, N + 1):
                if (bitset[(x_add-1)//64] >> ((x_add-1)%64)) & 1:
                    continue
                if _can_add(x_add, bitset, conflicts):
                    bitset[(x_add-1)//64] |= (1 << ((x_add-1)%64))
                    count += 1
                    added = True
                    improved = True
                    break
            
            if not added:
                # Restore
                bitset[(x_rm-1)//64] |= (1 << ((x_rm-1)%64))
                count += 1
    
    return bitset, count

def _bitset_to_sorted_list(bitset):
    result = []
    for i in range(len(bitset)):
        word = bitset[i]
        while word:
            bit = word & (-word)
            pos = i * 64 + bit.bit_length() - 1
            result.append(pos + 1)
            word ^= bit
    return result

# Precompute
_conflicts = None

def construct():
    global _conflicts
    deadline = time.time() + 55.0
    
    if _conflicts is None:
        _conflicts = _build_conflict_lists()
    conflicts = _conflicts
    
    best_count = 0
    best_bitset = None
    
    num_starts = 0
    while time.time() < deadline - 8.0:
        seed = num_starts
        bitset, count = _greedy_randomized(conflicts, seed, time_limit=0.6)
        num_starts += 1
        
        if count > best_count:
            best_count = count
            best_bitset = bitset.copy()
        
        # Local search on this candidate
        if time.time() > deadline - 12.0:
            break
        ls_time = min(3.0, deadline - 10.0 - time.time())
        if ls_time > 0.3:
            bitset, count = _local_search(bitset, conflicts, time_limit=ls_time)
        
        if count > best_count:
            best_count = count
            best_bitset = bitset.copy()
    
    # Final local search on best
    if best_bitset is not None and time.time() < deadline - 2.0:
        final_time = deadline - 3.0 - time.time()
        if final_time > 0.5:
            best_bitset, best_count = _local_search(best_bitset, conflicts, time_limit=final_time)
    
    if best_bitset is None:
        return [1]
    
    return _bitset_to_sorted_list(best_bitset)
