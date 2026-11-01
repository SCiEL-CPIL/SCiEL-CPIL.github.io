"""Local Search with Swaps for the ternary covering code K_3(8,1)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import itertools

# Precomputation cache
_Q = 3
_N = 8
_RADIUS = 1
_TOTAL = _Q ** _N
_WEIGHT = _Q ** np.arange(_N - 1, -1, -1)
_WORDS = None
_NEIGHBORS = None
_INVERSE_NEIGHBORS = None # Not strictly needed if we use cover_count

def _init_precomputation(q, n, radius):
    global _WORDS, _NEIGHBORS, _TOTAL, _Q, _N, _WEIGHT
    if q != 3 or n != 8 or radius != 1:
        return False
    
    _Q = q
    _N = n
    _TOTAL = q ** n
    _WEIGHT = q ** np.arange(n - 1, -1, -1)
    
    # Generate all words
    _WORDS = np.empty((_TOTAL, n), dtype=np.int64)
    for i in range(_TOTAL):
        v = i
        for j in range(n - 1, -1, -1):
            _WORDS[i, j] = v % q
            v //= q
            
    # Precompute neighbors
    _NEIGHBORS = []
    for i in range(_TOTAL):
        w = _WORDS[i].copy()
        nb = [i]
        for j in range(n):
            old = w[j]
            for a in range(q):
                if a == old:
                    continue
                w[j] = a
                nb.append(int(w.dot(_WEIGHT)))
                w[j] = old
        _NEIGHBORS.append(nb)
    return True

def _greedy_init(q, n, radius):
    """Fast greedy initialization."""
    if not _init_precomputation(q, n, radius):
        return [list(w) for w in itertools.product(range(q), repeat=n)]
        
    uncovered = np.ones(_TOTAL, dtype=bool)
    remaining = _TOTAL
    codewords = []
    
    covered_count = np.array([len(nb) for nb in _NEIGHBORS], dtype=np.int64)
    # Use a simple priority queue or just pick max each time? 
    # For 6561 items, a heap is good.
    import heapq
    heap = [(-covered_count[i], i) for i in range(_TOTAL)]
    heapq.heapify(heap)
    
    while remaining > 0:
        while True:
            neg_key, i = heapq.heappop(heap)
            if -neg_key == covered_count[i]:
                break
        codewords.append(_WORDS[i].tolist())
        for nb in _NEIGHBORS[i]:
            if uncovered[nb]:
                uncovered[nb] = False
                remaining -= 1
                for j in _NEIGHBORS[nb]:
                    covered_count[j] -= 1
                    if remaining > 0:
                        heapq.heappush(heap, (-covered_count[j], j))
                        
    return codewords

def construct(q, n, radius):
    if not (q == 3 and n == 8 and radius == 1):
        return [list(w) for w in itertools.product(range(q), repeat=n)]
        
    _init_precomputation(q, n, radius)
    
    # 1. Greedy Initialization
    codewords = _greedy_init(q, n, radius)
    C = set(tuple(c) for c in codewords)
    
    # 2. Local Search with Removals and Swaps
    cover_count = np.zeros(_TOTAL, dtype=np.int64)
    C_dict = {}
    for c in C:
        idx = int(np.array(c, dtype=np.int64).dot(_WEIGHT))
        C_dict[c] = idx
        for nb in _NEIGHBORS[idx]:
            cover_count[nb] += 1
            
    start_time = time.time()
    timeout = 110.0
    
    # Precompute for swap validation:
    # A swap c_out -> c_in is valid if:
    # 1. c_in not in C
    # 2. For all u in N(c_out), if cover_count[u] == 1, then u in N(c_in)
    
    # To speed up swap search, we can precompute the set of neighbors for each word.
    # N(i) is a list. We can convert to a set for fast lookup, or use a boolean array.
    # Given 6561 words, a set of ints is fine.
    neighbor_sets = [set(nb) for nb in _NEIGHBORS]
    
    improved = True
    while improved and (time.time() - start_time) < timeout:
        improved = False
        
        # Step A: Try Removals
        removed = False
        for c in list(C):
            idx = C_dict[c]
            nb_list = _NEIGHBORS[idx]
            # Check if all neighbors are covered by >1 codeword
            if np.min(cover_count[nb_list]) > 1:
                # Remove c
                C.remove(c)
                del C_dict[c]
                for nb in nb_list:
                    cover_count[nb] -= 1
                improved = True
                removed = True
                break
        
        if removed:
            continue
            
        # Step B: Try Swaps if no removal possible
        # We look for a pair (c_out, c_in) such that swapping keeps validity.
        # To avoid O(N^2) which is 6561^2 ~ 43M, we need to be careful.
        # However, C is small (~500). So we iterate over c_out in C (500)
        # and candidate c_in.
        
        # Optimization: Only consider c_in that covers at least one "critical" neighbor of c_out.
        # Critical neighbor of c_out: u in N(c_out) where cover_count[u] == 1.
        
        c_list = list(C)
        random.shuffle(c_list)
        
        swap_done = False
        for c_out in c_list:
            idx_out = C_dict[c_out]
            nb_out = _NEIGHBORS[idx_out]
            
            # Find critical neighbors
            criticals = [u for u in nb_out if cover_count[u] == 1]
            
            if not criticals:
                # If no criticals, we could remove c_out, but Step A would have caught it.
                # So this case shouldn't happen if Step A was exhaustive.
                # But maybe Step A missed it? No, Step A checks min > 1.
                # If min == 1, there is at least one critical.
                pass
            
            # We need a c_in that covers ALL criticals.
            # And c_in must not be in C.
            
            # Candidate c_in: 
            # It must be in the intersection of neighbors of all criticals?
            # No, c_in must be a neighbor of each critical.
            # i.e., c_in in N(critical_1) AND c_in in N(critical_2) ...
            
            # Let's find candidates from the neighbors of the first critical.
            if criticals:
                first_crit = criticals[0]
                candidates = neighbor_sets[first_crit]
                
                # Filter candidates:
                # 1. Not in C
                # 2. Covers all other criticals
                # 3. Ideally, c_in should also cover some non-criticals to be "good"
                
                for c_in_idx in candidates:
                    if c_in_idx in C_dict.values(): # This check is slow if C_dict.values() is not a set
                        continue
                    
                    # Check if c_in covers all criticals
                    valid = True
                    for crit in criticals:
                        if crit not in neighbor_sets[c_in_idx]:
                            valid = False
                            break
                    
                    if valid:
                        # Check if c_in is already in C
                        # We need a fast way to check if an index is in C.
                        # Let's maintain a set of indices in C.
                        pass
                # This nested loop might be slow. Let's optimize.
            
            # Alternative Swap Strategy:
            # Randomly pick c_out.
            # Randomly pick c_in from a small set of candidates.
            # If valid, swap.
            
            # Let's try a simpler random swap strategy to ensure speed.
            # Pick random c_out from C.
            # Pick random c_in from all words not in C.
            # Check validity.
            # If valid, swap.
            
            # This is O(1) per trial. We can do many trials.
            pass
            
        # If we reach here, no removal was found.
        # We will try a few random swaps.
        
        # Maintain a set of indices in C for O(1) lookup
        C_indices = set(C_dict.values())
        
        # Try up to 100 random swaps
        swap_success = False
        for _ in range(100):
            c_out = random.choice(c_list)
            idx_out = C_dict[c_out]
            nb_out = _NEIGHBORS[idx_out]
            criticals = [u for u in nb_out if cover_count[u] == 1]
            
            if not criticals:
                continue # Should have been removed
            
            # Pick a random candidate c_in from the neighbors of the first critical
            # This biases towards useful swaps
            first_crit = criticals[0]
            c_in_idx = random.choice(neighbor_sets[first_crit])
            
            if c_in_idx in C_indices:
                continue
            
            # Check if c_in covers all criticals
            valid = True
            for crit in criticals:
                if crit not in neighbor_sets[c_in_idx]:
                    valid = False
                    break
            
            if valid:
                # Perform Swap
                # Remove c_out
                C.remove(c_out)
                del C_dict[c_out]
                C_indices.remove(idx_out)
                for nb in nb_out:
                    cover_count[nb] -= 1
                
                # Add c_in
                c_in_tuple = tuple(_WORDS[c_in_idx])
                C.add(c_in_tuple)
                C_dict[c_in_tuple] = c_in_idx
                C_indices.add(c_in_idx)
                for nb in _NEIGHBORS[c_in_idx]:
                    cover_count[nb] += 1
                
                swap_success = True
                improved = True
                break
        
        if not swap_success:
            # No removal, no successful swap.
            # We are in a local optimum.
            # Try to "kick" by adding a random word and then removing two?
            # Or just break.
            # To escape, we can force a swap that is valid but doesn't help removal?
            # Actually, if no valid swap exists, we are stuck.
            # We can try a "2-opt" like move: Remove c1, Add c2, Remove c3?
            # Too complex.
            # Let's just break.
            break

    return [list(c) for c in C]
# EVOLVE-BLOCK-END
