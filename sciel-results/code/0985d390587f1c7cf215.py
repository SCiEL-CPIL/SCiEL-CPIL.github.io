"""New Approach: Bitset-based Local Search for ternary covering codes K_3(7,1)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _precompute_bitsets(q, n, radius):
    """Precompute neighbor bitsets for all words."""
    N = q ** n
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other cases, though task is specific
        # For general case, we might need a different strategy or just standard lists
        # Here we assume the task constraints q=3, n=7, radius=1
        pass
        
    bitsets = [0] * N
    
    # Precompute powers of 3 for index conversion
    # Index = d0 + d1*3 + ... + d6*3^6
    
    for idx in range(N):
        word = [0] * 7
        tmp = idx
        for i in range(7):
            word[i] = tmp % 3
            tmp //= 3
            
        mask = 1 << idx
        
        # Self
        bitsets[idx] |= mask
        
        # Neighbors
        for pos in range(7):
            old = word[pos]
            # Two alternative values
            for v in range(3):
                if v != old:
                    new_word = word.copy()
                    new_word[pos] = v
                    # Calculate index
                    new_idx = new_word[0] + new_word[1]*3 + new_word[2]*9 + new_word[3]*27 + new_word[4]*81 + new_word[5]*243 + new_word[6]*729
                    bitsets[idx] |= (1 << new_idx)
                    
    return bitsets

def construct(q, n, radius):
    start_time = time.time()
    time_limit = 110.0
    N = q ** n
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback to a simple greedy for other parameters if needed, 
        # but the task specifies q=3, n=7, radius=1.
        # We implement a basic greedy here just to be safe.
        nbrs = []
        for idx in range(N):
            word = [0]*n
            tmp = idx
            for i in range(n):
                word[i] = tmp % q
                tmp //= q
            nbrs.append([idx])
            for pos in range(n):
                old = word[pos]
                for v in range(q):
                    if v != old:
                        nw = word.copy()
                        nw[pos] = v
                        ni = 0
                        mul = 1
                        for i in range(n):
                            ni += nw[i] * mul
                            mul *= q
                        nbrs.append(ni)
        # Basic Greedy
        covered = [0]*N
        selected = []
        while True:
            best_idx = -1
            best_gain = -1
            for i in range(N):
                gain = sum(1 for v in nbrs[i] if covered[v] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_gain == 0:
                break
            selected.append(best_idx)
            for v in nbrs[best_idx]:
                covered[v] += 1
            if time.time() - start_time > time_limit - 5:
                break
        res = []
        for idx in selected:
            word = [0]*n
            tmp = idx
            for i in range(n):
                word[i] = tmp % q
                tmp //= q
            res.append(word)
        return res

    bitsets = _precompute_bitsets(q, n, radius)
    
    # Helper to check if a set of indices covers all N points
    # Union of their bitsets == (1 << N) - 1
    full_mask = (1 << N) - 1
    
    def is_valid(selected_indices):
        union = 0
        for idx in selected_indices:
            union |= bitsets[idx]
            if union == full_mask:
                return True
        return union == full_mask

    def try_remove_redundant(selected, cover_counts):
        """Remove any redundant codewords. Returns new selected list."""
        changed = True
        while changed:
            changed = False
            for i in range(len(selected)):
                w = selected[i]
                # Check if any point is covered ONLY by w
                # Points covered by w: bitsets[w]
                # Points covered by others: full_union - bitsets[w]? 
                # Easier: count how many times each point in bitsets[w] is covered.
                # We maintain cover_counts.
                # If any point in bitsets[w] has cover_count == 1, w is critical.
                # If all points in bitsets[w] have cover_count > 1, w is redundant.
                
                # Iterate bits in bitsets[w]
                b = bitsets[w]
                critical = False
                while b:
                    # Get lowest set bit
                    lsb = b & -b
                    idx = lsb.bit_length() - 1
                    if cover_counts[idx] == 1:
                        critical = True
                        break
                    b ^= lsb
                
                if not critical:
                    # Remove w
                    # Update cover_counts
                    b = bitsets[w]
                    while b:
                        lsb = b & -b
                        idx = lsb.bit_length() - 1
                        cover_counts[idx] -= 1
                        b ^= lsb
                    selected.pop(i)
                    changed = True
                    break # Restart loop
        return selected

    def run_local_search(selected, time_limit, start_time):
        """Perform 2-1 swaps and redundancy removal."""
        selected = list(selected)
        
        # Initialize cover counts
        cover_counts = np.zeros(N, dtype=np.int32)
        for w in selected:
            b = bitsets[w]
            while b:
                lsb = b & -b
                idx = lsb.bit_length() - 1
                cover_counts[idx] += 1
                b ^= lsb
        
        # 1. Remove redundant
        selected = try_remove_redundant(selected, cover_counts)
        
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            
            # 2. Try 2-1 Swaps
            # We want to find u, v in selected and w not in selected
            # such that removing u, v and adding w is valid.
            
            # Optimization: Iterate u, v. 
            # To make it faster, we can limit the search or use heuristics.
            # Given N=2187 and K~200, K^2 = 40000.
            # For each pair, we need to find if there is a w covering the critical set.
            
            # Precompute set of selected for O(1) lookup
            selected_set = set(selected)
            
            # To speed up finding w, we can precompute a mapping from "critical set signature" to candidate w?
            # Too complex.
            
            # Instead, iterate w (not in selected) and see if it can replace any pair?
            # No, that's K * (N-K) which is similar.
            
            # Let's iterate pairs (u, v) but break early if time is tight.
            # We can also randomize the order of pairs to find improvements faster.
            
            pairs = []
            for i in range(len(selected)):
                for j in range(i+1, len(selected)):
                    pairs.append((i, j))
            
            random.shuffle(pairs)
            
            for i, j in pairs:
                if time.time() - start_time > time_limit:
                    break
                
                u = selected[i]
                v = selected[j]
                
                # Calculate critical set C: points that will be uncovered if u, v are removed.
                # A point p is in C if:
                # 1. p is in N(u) U N(v)
                # 2. AND ( (p in N(u) and p not in N(v) and cover_counts[p] == 1)
                #    OR (p in N(v) and p not in N(u) and cover_counts[p] == 1)
                #    OR (p in N(u) and p in N(v) and cover_counts[p] == 2) )
                
                # Bitset operations:
                # mask_u = bitsets[u]
                # mask_v = bitsets[v]
                # union_uv = mask_u | mask_v
                # inter_uv = mask_u & mask_v
                # only_u = mask_u & ~mask_v
                # only_v = mask_v & ~mask_u
                
                # We need to check cover_counts for these specific points.
                # This requires iterating bits.
                
                mask_u = bitsets[u]
                mask_v = bitsets[v]
                
                critical_mask = 0
                
                # Points only in u
                only_u = mask_u & ~mask_v
                b = only_u
                while b:
                    lsb = b & -b
                    idx = lsb.bit_length() - 1
                    if cover_counts[idx] == 1:
                        critical_mask |= lsb
                    b ^= lsb
                    
                # Points only in v
                only_v = mask_v & ~mask_u
                b = only_v
                while b:
                    lsb = b & -b
                    idx = lsb.bit_length() - 1
                    if cover_counts[idx] == 1:
                        critical_mask |= lsb
                    b ^= lsb
                    
                # Points in both
                inter_uv = mask_u & mask_v
                b = inter_uv
                while b:
                    lsb = b & -b
                    idx = lsb.bit_length() - 1
                    if cover_counts[idx] == 2:
                        critical_mask |= lsb
                    b ^= lsb
                
                if critical_mask == 0:
                    # One of them is redundant? Or both?
                    # If critical_mask is 0, removing both leaves no uncovered points?
                    # This implies that all points in N(u) U N(v) are covered by other nodes too.
                    # We can remove both.
                    # Update cover counts
                    b = mask_u
                    while b:
                        lsb = b & -b
                        idx = lsb.bit_length() - 1
                        cover_counts[idx] -= 1
                        b ^= lsb
                    b = mask_v
                    while b:
                        lsb = b & -b
                        idx = lsb.bit_length() - 1
                        cover_counts[idx] -= 1
                        b ^= lsb
                        
                    selected.pop(j) # Remove v (higher index)
                    selected.pop(i) # Remove u
                    selected_set.discard(u)
                    selected_set.discard(v)
                    improved = True
                    break # Restart pair search with new list
                
                # Check if there exists w not in selected such that critical_mask is subset of bitsets[w]
                # Iterate all w not in selected
                # To speed up, we can check a few random ones or all.
                # N is 2187. Checking 2000 candidates per pair is 40000 * 2000 = 80M checks.
                # Each check is (bitsets[w] & critical_mask) == critical_mask.
                # This is very fast in Python.
                
                found_w = -1
                # To avoid checking all 2000 every time, we can maybe pre-sort?
                # For now, iterate all.
                for w in range(N):
                    if w in selected_set:
                        continue
                    if (bitsets[w] & critical_mask) == critical_mask:
                        found_w = w
                        break
                
                if found_w != -1:
                    # Perform swap: remove u, v; add found_w
                    # Update cover counts
                    b = mask_u
                    while b:
                        lsb = b & -b
                        idx = lsb.bit_length() - 1
                        cover_counts[idx] -= 1
                        b ^= lsb
                    b = mask_v
                    while b:
                        lsb = b & -b
                        idx = lsb.bit_length() - 1
                        cover_counts[idx] -= 1
                        b ^= lsb
                    b = bitsets[found_w]
                    while b:
                        lsb = b & -b
                        idx = lsb.bit_length() - 1
                        cover_counts[idx] += 1
                        b ^= lsb
                        
                    selected.pop(j)
                    selected.pop(i)
                    selected_set.discard(u)
                    selected_set.discard(v)
                    selected.append(found_w)
                    selected_set.add(found_w)
                    improved = True
                    break # Restart pair search
            
            # After swaps, try to remove redundant again
            selected = try_remove_redundant(selected, cover_counts)
            
        return selected

    # Main Loop: Multi-start
    best_code = None
    best_size = float('inf')
    
    # Initial Greedy
    # We use a simple greedy to start
    covered_count = np.zeros(N, dtype=np.int32)
    selected = []
    
    # To speed up greedy, we can just pick random or highest gain.
    # Let's do a few random starts.
    
    for run in range(5):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Randomized Greedy
        rng = np.random.RandomState(run * 12345 + 1)
        covered_count = np.zeros(N, dtype=np.int32)
        selected = []
        
        # Precompute neighbor lists for greedy gain calculation
        # We already have bitsets, but lists are easier for gain.
        # Let's reuse the bitsets to compute gain?
        # gain[w] = number of uncovered points in N(w)
        # Uncovered points mask = ~(union of selected bitsets)
        # We can maintain an "uncovered_mask".
        # Initially full_mask.
        # When adding w, uncovered_mask &= ~bitsets[w]
        # gain[w] = bit_count(uncovered_mask & bitsets[w])
        
        uncovered_mask = full_mask
        selected_set = set()
        
        while uncovered_mask != 0 and time.time() - start_time < time_limit - 10:
            # Find w with max gain
            # Iterate all w not in selected?
            # N=2187. 
            # We can just iterate all w.
            best_w = -1
            best_gain = -1
            # Randomize tie-breaking
            candidates = []
            for w in range(N):
                if w in selected_set:
                    continue
                # gain = number of bits set in (uncovered_mask & bitsets[w])
                # Python int bit_count is fast
                gain = (uncovered_mask & bitsets[w]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_w = w
                    candidates = [w]
                elif gain == best_gain and gain > 0:
                    candidates.append(w)
            
            if best_gain == 0:
                break
            
            # Pick randomly from candidates
            w = rng.choice(candidates)
            selected.append(w)
            selected_set.add(w)
            uncovered_mask &= ~bitsets[w]
            
        # Local Search
        if time.time() - start_time < time_limit - 2:
            selected = run_local_search(selected, time_limit - 2, start_time)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_code = selected.copy()
            
    # Convert indices to words
    res = []
    for idx in best_code:
        word = [0] * n
        tmp = idx
        for i in range(n):
            word[i] = tmp % q
            tmp //= q
        res.append(word)
        
    return res
# EVOLVE-BLOCK-END
