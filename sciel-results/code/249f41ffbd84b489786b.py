import time
import itertools
import numpy as np
import random

def construct(q, n, radius):
    """
    Build a covering code of length n over alphabet size q with covering radius.
    For q=3, n=8, radius=1, use an optimized greedy + local search approach.
    """
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1()
    
    # General fallback: greedy set cover
    return _greedy_general(q, n, radius)

def _solve_ternary_8_1():
    """Optimized solver for K_3(8,1)."""
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    q = 3
    n = 8
    total = q ** n  # 6561
    
    # Precompute all words
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    
    # Map word -> index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors (list of lists of indices)
    # And masks (list of Python ints for bit-parallel operations)
    neighbors = []
    masks = []
    for i in range(total):
        w = all_words[i]
        nbrs = set()
        nbrs.add(i)
        for pos in range(n):
            orig_val = w[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                nbrs.add(word_to_idx[tuple(new_w_list)])
        neighbors.append(sorted(nbrs))
        
        # Create bitmask
        mask = 0
        for idx in neighbors[-1]:
            mask |= (1 << idx)
        masks.append(mask)
        
    # 1. Greedy Set Cover using Bitmasks
    uncovered = (1 << total) - 1  # All bits set
    selected = []
    
    while time.time() < deadline:
        best_idx = -1
        best_count = 0
        
        # Find candidate covering most uncovered points
        # We iterate through all candidates. 6561 is small enough.
        # Optimization: We can skip if we already have a count of 17 (max possible)
        for i in range(total):
            # Count how many bits are set in (uncovered & masks[i])
            cnt = (uncovered & masks[i]).bit_count()
            if cnt > best_count:
                best_count = cnt
                best_idx = i
                if best_count == 17: # Max possible, can't do better
                    break
        
        if best_count == 0:
            break
            
        selected.append(best_idx)
        uncovered &= ~masks[best_idx]
        
        if uncovered == 0:
            break
            
    if not (uncovered == 0):
        # Fallback if greedy didn't finish in time (unlikely)
        # Just return what we have? No, must be valid.
        # But greedy is very fast, should finish in <1s.
        pass

    # 2. Reduction Phase
    # Convert selected to a set for fast lookup
    code_set = set(selected)
    
    # Compute coverage counts
    # counts[w] = number of codewords covering w
    counts = np.zeros(total, dtype=np.int32)
    for c in code_set:
        for w in neighbors[c]:
            counts[w] += 1
            
    # Iteratively remove redundant codewords
    removed = True
    while removed and time.time() < deadline:
        removed = False
        # Iterate over a copy of the list to allow modification
        for c in list(code_set):
            # Check if all neighbors of c are covered by > 1 codeword
            # i.e., min(counts[w] for w in neighbors[c]) > 1
            is_redundant = True
            for w in neighbors[c]:
                if counts[w] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove c
                for w in neighbors[c]:
                    counts[w] -= 1
                code_set.remove(c)
                removed = True
                
    # 3. Local Search (Swaps)
    # Goal: Find a codeword c and a new position c' not in code_set
    # such that moving c to c' allows removing another codeword (or just improves structure)
    # Simple heuristic:
    # For each c in code_set:
    #   Find unique points U_c (where counts[w] == 1 and w in neighbors[c])
    #   If U_c is empty, c is redundant (should have been removed, but maybe counts changed?)
    #   If |U_c| <= 3:
    #      Find candidates c' in intersection of neighbors[w] for w in U_c
    #      Filter out c' in code_set
    #      Pick a candidate c'
    #      Try swap:
    #         Decrement counts for neighbors[c]
    #         Increment counts for neighbors[c']
    #         Update code_set
    #         Run reduction
    #         If size decreased, keep. Else revert.
        
    best_size = len(code_set)
    best_code_set = set(code_set)
    best_counts = counts.copy()
    
    # To avoid getting stuck, we can randomize the order of codewords we examine
    # and maybe do multiple passes.
    
    while time.time() < deadline:
        improved = False
        # Shuffle codewords to explore different local minima
        current_code_list = list(code_set)
        random.shuffle(current_code_list)
        
        for c in current_code_list:
            if time.time() > deadline:
                break
                
            # Find unique points for c
            unique_points = []
            for w in neighbors[c]:
                if counts[w] == 1:
                    unique_points.append(w)
            
            if not unique_points:
                # Should be redundant, remove it
                for w in neighbors[c]:
                    counts[w] -= 1
                code_set.remove(c)
                improved = True
                break
                
            # If too many unique points, a single codeword can't cover them all
            # (Max overlap of two distance-1 balls is limited, but generally >3 is hard)
            if len(unique_points) > 3:
                continue
                
            # Find candidates c' that cover all unique_points
            # Intersection of neighbors[w] for w in unique_points
            # Start with neighbors of first unique point
            candidates = set(neighbors[unique_points[0]])
            for w in unique_points[1:]:
                candidates &= set(neighbors[w])
                if not candidates:
                    break
            
            if not candidates:
                continue
                
            # Remove existing codewords
            candidates -= code_set
            
            if not candidates:
                continue
                
            # Pick a candidate
            # Heuristic: Pick one that covers the most "almost unique" points?
            # Or just random.
            c_prime = random.choice(list(candidates))
            
            # Try the swap
            # Save state
            old_counts = counts.copy()
            old_code_set = set(code_set)
            old_size = len(code_set)
            
            # Apply swap
            for w in neighbors[c]:
                counts[w] -= 1
            for w in neighbors[c_prime]:
                counts[w] += 1
            
            code_set.remove(c)
            code_set.add(c_prime)
            
            # Run reduction
            removed = True
            while removed:
                removed = False
                for cw in list(code_set):
                    is_redundant = True
                    for w in neighbors[cw]:
                        if counts[w] == 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        for w in neighbors[cw]:
                            counts[w] -= 1
                        code_set.remove(cw)
                        removed = True
                        
            new_size = len(code_set)
            
            if new_size < old_size:
                # Keep the improvement
                improved = True
                if new_size < best_size:
                    best_size = new_size
                    best_code_set = set(code_set)
                    best_counts = counts.copy()
                break # Restart loop with new code_set
                
            else:
                # Revert
                counts = old_counts
                code_set = old_code_set
                
        if not improved:
            # No improvement in this pass. 
            # We could try more aggressive moves (2-for-1 swaps) but that's complex.
            # Or we can just break if we've done a few passes without improvement?
            # Given time, let's just keep shuffling and trying until deadline.
            # However, if we are stuck, shuffling the same set might not help much.
            # Let's add a small random perturbation: randomly move a codeword to a random valid spot?
            # No, that might break coverage.
            # Let's just break if we've made no progress in a full pass?
            # Actually, the local search landscape might have many plateaus.
            # Let's just let it run until deadline, as the shuffle changes the order of checks.
            pass

    # Restore best found
    code_set = best_code_set
    
    # Final result
    codewords = [list(all_words[c]) for c in code_set]
    codewords.sort()
    
    return codewords

def _greedy_general(q, n, radius):
    """General greedy set cover for other parameters."""
    start_time = time.time()
    deadline = start_time + 110.0
    
    total = q ** n
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    
    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * q + d
        return idx
        
    def get_neighbors(w):
        yield tuple(w)
        if radius >= 1:
            for i in range(n):
                for v in range(q):
                    if v != w[i]:
                        new = list(w)
                        new[i] = v
                        yield tuple(new)
                        
    covered = [False] * total
    codewords = []
    
    neighbors_list = []
    for i in range(total):
        nbrs = set()
        for cand in get_neighbors(all_words[i]):
            nbrs.add(word_to_idx(cand))
        neighbors_list.append(nbrs)
        
    uncovered_count = [len(neighbors_list[i]) for i in range(total)]
    
    while time.time() < deadline:
        best_idx = -1
        best_count = 0
        for i in range(total):
            if uncovered_count[i] > best_count:
                best_count = uncovered_count[i]
                best_idx = i
            if best_count == 0:
                break
                
        if best_idx == -1 or best_count == 0:
            break
            
        codewords.append(list(all_words[best_idx]))
        
        for j in neighbors_list[best_idx]:
            if not covered[j]:
                covered[j] = True
                
        new_uncovered_count = [0] * total
        for i in range(total):
            cnt = 0
            for j in neighbors_list[i]:
                if not covered[j]:
                    cnt += 1
            new_uncovered_count[i] = cnt
        uncovered_count = new_uncovered_count
        
        if all(covered):
            break
            
    return codewords
