import itertools
import numpy as np
import random
import time
import sys

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for non-target parameters
        # Simple greedy set cover
        all_words = list(itertools.product(range(q), repeat=n))
        N = len(all_words)
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        neighbors = []
        for i, w in enumerate(all_words):
            neigh_set = set()
            neigh_set.add(i)
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_w_list = list(w)
                    new_w_list[pos] = val
                    new_w = tuple(new_w_list)
                    neigh_set.add(word_to_idx[new_w])
            neighbors.append(list(neigh_set))
        
        uncovered = np.ones(N, dtype=bool)
        C = []
        start_time = time.time()
        time_limit = 110.0
        
        while np.any(uncovered):
            best_idx = -1
            best_count = -1
            for i in range(N):
                cnt = np.sum(uncovered[neighbors[i]])
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            if best_count == 0: break
            C.append(best_idx)
            uncovered[neighbors[best_idx]] = False
            if time.time() - start_time > time_limit: break
            
        # Reduction
        coverage_counts = np.zeros(N, dtype=int)
        for c in C:
            coverage_counts[neighbors[c]] += 1
        changed = True
        while changed:
            changed = False
            new_C = []
            for c in C:
                if np.min(coverage_counts[neighbors[c]]) >= 2:
                    coverage_counts[neighbors[c]] -= 1
                    changed = True
                else:
                    new_C.append(c)
            C = new_C
        return [list(all_words[i]) for i in C]

    # Specific optimization for q=3, n=8, radius=1
    N = 6561
    all_words = list(itertools.product(range(3), repeat=8))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors as lists of indices and as bitsets
    # Bitsets are Python integers. 6561 bits is ~820 bytes.
    neighbor_indices = []
    neighbor_bits = []
    
    for i, w in enumerate(all_words):
        neigh_set = set()
        neigh_set.add(i)
        for pos in range(8):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                neigh_set.add(word_to_idx[new_w])
        
        idx_list = list(neigh_set)
        neighbor_indices.append(np.array(idx_list, dtype=np.int32))
        
        bit = 0
        for idx in idx_list:
            bit |= (1 << idx)
        neighbor_bits.append(bit)
        
    full_mask = (1 << N) - 1
    
    start_time = time.time()
    time_limit = 115.0
    
    best_C = None
    best_size = float('inf')
    
    # We will perform several randomized greedy restarts
    # The bitset greedy is fast, so we can afford more restarts.
    max_restarts = 10
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit - 20:
            break
            
        # 1. Randomized Greedy Initialization
        uncovered_bits = full_mask
        C = []
        
        # Randomize the order of candidates to break ties and introduce diversity
        # Instead of shuffling the whole list every time (expensive), 
        # we can just add a small random noise to the gain or shuffle indices.
        # Shuffling 6561 items is fast.
        candidates_order = list(range(N))
        random.shuffle(candidates_order)
        
        while uncovered_bits:
            best_idx = -1
            best_gain = -1
            
            # To speed up, we could use a heap, but scanning 6561 bits is fast.
            # However, doing this 500 times * 10 restarts = 5000 scans.
            # 6561 * 5000 = 32.5 million bit_count ops.
            # Python int bit_count is very fast (C level).
            # Let's estimate: ~1 ns per bit_count? No, probably ~10-50 ns.
            # 32.5M * 50ns = 1.6 seconds. Acceptable.
            
            for i in candidates_order:
                gain = (neighbor_bits[i] & uncovered_bits).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                # Early break if we found a perfect cover? Unlikely.
                # If gain is 0, no point continuing? No, others might have gain.
                
            if best_gain == 0:
                break
                
            C.append(best_idx)
            uncovered_bits &= ~neighbor_bits[best_idx]
            
        # 2. Local Search on the result
        C, counts = reduce_and_local_search(C, neighbor_indices, N, time_limit - (time.time() - start_time))
        
        if len(C) < best_size:
            best_size = len(C)
            best_C = C.copy()
            
        # Check time
        if time.time() - start_time > time_limit - 5:
            break
            
    # Final reduction on the best candidate
    if best_C:
        best_C, _ = reduce_code(best_C, neighbor_indices, N)
        return [list(all_words[i]) for i in best_C]
    else:
        return []

def reduce_code(C, neighbor_indices, N):
    """
    Reduces code by removing redundant words.
    Returns new_code and new_counts.
    """
    if not C:
        return C, np.zeros(N, dtype=np.int16)
        
    # Build initial counts
    counts = np.zeros(N, dtype=np.int16)
    for c in C:
        counts[neighbor_indices[c]] += 1
        
    current_code = list(C)
    changed = True
    while changed:
        changed = False
        next_code = []
        for c in current_code:
            # If min count in neighbors is >= 2, we can remove c
            if np.min(counts[neighbor_indices[c]]) >= 2:
                counts[neighbor_indices[c]] -= 1
                changed = True
            else:
                next_code.append(c)
        current_code = next_code
        
    return current_code, counts

def reduce_and_local_search(C, neighbor_indices, N, time_budget):
    """
    Applies local search to minimize code size.
    """
    C, counts = reduce_code(C, neighbor_indices, N)
    
    if not C:
        return C, counts
    
    start_time = time.time()
    
    # Set to O(1) lookup
    C_set = set(C)
    
    # Local Search Loop
    while time.time() - start_time < time_budget:
        old_size = len(C)
        
        # 1. Try to reduce
        C, counts = reduce_code(C, counts)
        C_set = set(C)
        
        if len(C) < old_size:
            continue # Restart loop as structure changed significantly
        
        # 2. Try swaps
        if not C:
            break
            
        # Pick a random codeword to try and remove/swap
        c_idx = random.randint(0, len(C) - 1)
        c = C[c_idx]
        
        # Find unique points for c
        # unique points are those in neighbors[c] with count == 1
        neigh_c = neighbor_indices[c]
        unique_mask = counts[neigh_c] == 1
        unique_points = neigh_c[unique_mask]
        
        # If no unique points, it should have been reduced, but maybe counts changed?
        if len(unique_points) == 0:
            # Remove it
            C.remove(c)
            C_set.remove(c)
            counts[neigh_c] -= 1
            continue
            
        # If too many unique points, we can't cover them with 1 word
        # (Max radius is 1, so a word covers 1+8*2=17 points. 
        # If |unique| > 17, impossible. 
        # Actually, if |unique| is large, finding a single replacement is hard/impossible.)
        if len(unique_points) > 5:
            continue # Skip for efficiency, try another c
        
        # Find candidates c_new that cover all unique_points
        if len(unique_points) == 1:
            u0 = unique_points[0]
            candidates = neighbor_indices[u0]
        elif len(unique_points) == 2:
            u0, u1 = unique_points[0], unique_points[1]
            # Intersection of neighbor sets
            set0 = set(neighbor_indices[u0])
            set1 = set(neighbor_indices[u1])
            candidates = np.array(list(set0 & set1), dtype=np.int32)
        else:
            # Intersection of 3+ sets is more expensive, but rare in good codes
            # We can just sample from the first set and check
            u0 = unique_points[0]
            candidates = neighbor_indices[u0]
            # Filter manually
            valid_candidates = []
            for cand in candidates:
                if cand in C_set:
                    continue
                # Check if it covers all other unique points
                neigh_cand = neighbor_indices[cand]
                # Simple check: is u1 in neigh_cand? etc.
                # This is slow, so we only do this if |unique| is small (<=3)
                if len(unique_points) <= 3:
                    ok = True
                    for u in unique_points[1:]:
                        if u not in set(neigh_cand): # This check is O(1) if we precompute neighbor sets as sets? No, they are arrays.
                             # Actually, checking membership in a numpy array is O(N).
                             # Let's just skip if > 2 unique points to keep it fast.
                             ok = False
                             break
                    if ok:
                         valid_candidates.append(cand)
                else:
                    continue # Too complex, skip
            
            candidates = np.array(valid_candidates, dtype=np.int32) if valid_candidates else np.array([], dtype=np.int32)

        if len(candidates) == 0:
            continue
            
        # Filter out current C
        candidates = [cand for cand in candidates if cand not in C_set]
        if not candidates:
            continue
            
        # Pick a candidate
        # Heuristic: pick one that covers the most "redundant" points (count > 1)
        # to enable further reductions.
        best_cand = None
        best_score = -1
        # Sample a few if many
        if len(candidates) > 10:
            sample = random.sample(candidates, 10)
        else:
            sample = candidates
            
        for cand in sample:
            # Score: number of points in neighbors[cand] that are already covered > 1 time
            # This encourages overlap.
            score = np.sum(counts[neighbor_indices[cand]] > 1)
            if score > best_score:
                best_score = score
                best_cand = cand
                
        if best_cand is None:
            continue
            
        # Perform Swap
        # Remove c, add best_cand
        C.remove(c)
        C_set.remove(c)
        counts[neighbor_indices[c]] -= 1
        
        C.append(best_cand)
        C_set.add(best_cand)
        counts[neighbor_indices[best_cand]] += 1
        
        # Check validity (should be valid)
        if np.any(counts < 1):
            # Revert
            counts[neighbor_indices[best_cand]] -= 1
            C.remove(best_cand)
            C_set.remove(best_cand)
            
            counts[neighbor_indices[c]] += 1
            C.append(c)
            C_set.add(c)
            continue
            
        # Try to reduce again immediately
        # (This is done at the top of the while loop, so we just continue)
        
    return C, counts
