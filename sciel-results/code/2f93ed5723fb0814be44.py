"""Ternary covering code K_3(8,1) via fast neighbor generation + vectorized local search."""
import time
import random
import numpy as np

def _generate_neighbors(q, n, radius):
    """
    Efficiently generate neighbor lists for all words in {0..q-1}^n.
    Returns a list of arrays, where neighbors[i] is an array of indices within distance radius.
    """
    total = q ** n
    neighbors = [None] * total
    
    # Precompute words as numpy arrays of shape (total, n) for potential use, 
    # but we can generate neighbors on the fly for each index.
    # Actually, generating neighbors for each index is O(total * n * (q-1)) which is fast.
    
    for i in range(total):
        # Convert i to word
        word = np.zeros(n, dtype=np.int8)
        t = i
        for k in range(n - 1, -1, -1):
            word[k] = t % q
            t //= q
        
        nbs = [i]
        if radius >= 1:
            # Generate all words at distance 1
            for pos in range(n):
                orig_val = word[pos]
                for v in range(q):
                    if v == orig_val:
                        continue
                    new_word = word.copy()
                    new_word[pos] = v
                    # Convert back to index
                    idx = 0
                    for val in new_word:
                        idx = idx * q + val
                    nbs.append(idx)
        if radius >= 2:
             # For this task radius is 1, but keeping generic logic is good.
             # However, for q=3, n=8, radius=1 is the specific case.
             # If radius > 1, we'd need to expand further.
             # Given the task is specifically K_3(8,1), we can optimize for radius=1.
             pass
        
        neighbors[i] = np.array(nbs, dtype=np.int32)
        
    return neighbors

def _greedy_init(q, n, radius, neighbors, max_time=10.0):
    """
    Fast greedy set cover.
    """
    start = time.time()
    total = q ** n
    covered = np.zeros(total, dtype=bool)
    code = []
    
    # Precompute inverse neighbors? Or just use the fact that dist(i,j)<=r implies dist(j,i)<=r.
    # For each uncovered word, find candidates that cover it.
    
    # To speed up, we maintain a list of uncovered words.
    uncovered = list(range(total))
    random.shuffle(uncovered)
    
    # We will iterate. To find the best candidate, we can pick a random uncovered word u,
    # look at its neighbors (candidates c such that dist(c,u)<=r), and pick the one
    # that covers the most other uncovered words.
    
    # Optimization: Instead of checking all candidates for all u, we can just pick
    # the candidate from the first uncovered word's sphere that covers the most.
    # This is a standard greedy heuristic.
    
    # Better: Maintain a count of how many uncovered words each candidate covers?
    # That's expensive to update.
    
    # Let's use a simple iterative approach:
    # While uncovered:
    #   Pick a random uncovered word u.
    #   Candidates = neighbors[u]
    #   For each c in Candidates:
    #       count how many words in neighbors[c] are uncovered.
    #   Pick c with max count.
    #   Mark neighbors[c] as covered.
    
    # To make it faster, we can limit the search space for 'best c'.
    
    while uncovered:
        if time.time() - start > max_time:
            break
            
        # Pick a random uncovered word
        u_idx = random.randint(0, len(uncovered) - 1)
        u = uncovered[u_idx]
        
        best_c = -1
        best_count = -1
        best_covered_mask = None
        
        # Candidates are neighbors of u
        candidates = neighbors[u]
        
        # We can't check all candidates if it's too slow. 
        # But neighbors[u] has size 1 + 8*2 = 17. So it's very small!
        # Checking 17 candidates is fast.
        
        for c in candidates:
            # Count uncovered in neighbors[c]
            nbs_c = neighbors[c]
            # Use numpy to check how many are in uncovered set?
            # We don't have a fast "is in uncovered set" for numpy array without a mask.
            # Let's maintain a boolean array `is_uncovered`.
            
            # Wait, I didn't define is_uncovered. Let's use a boolean array.
            # Re-initializing logic to use boolean array for speed.
            pass
        
        # Let's rewrite the loop with boolean arrays for speed.
        # This inner loop is O(|neighbors[u]| * |neighbors[c]|) = 17 * 17 = 289 ops per step.
        # Total steps ~ 400-500. Total ops ~ 100k. Very fast.
        
        # Actually, let's just implement it cleanly.
        
        # I need `is_uncovered` array.
        # Let's restart the function logic with proper state.
        
        # The previous snippet was just thinking. Here is the real implementation.
        
        # We need to break out of the dummy loop above.
        break
        
    # Real Implementation:
    covered_arr = np.zeros(total, dtype=bool)
    code = []
    
    # To avoid O(N) scan for uncovered, we can just iterate until no progress or time limit.
    # But we need to pick u.
    
    # A simpler, effective greedy:
    # 1. Start with empty code.
    # 2. While there are uncovered words:
    #    a. Pick a random uncovered word u.
    #    b. Among all candidates c that cover u (i.e., c in neighbors[u]), pick the one 
    #       that covers the most *other* uncovered words.
    #    c. Add c to code.
    
    # To check "covers most other uncovered", we need to know which words are uncovered.
    # We can maintain a list of uncovered words and a set for O(1) lookup?
    # Or a boolean array and just iterate neighbors[c] to count trues.
    
    uncovered_set = set(range(total))
    
    while uncovered_set:
        if time.time() - start > max_time:
            break
            
        # Pick random uncovered
        u = random.choice(list(uncovered_set))
        
        best_c = -1
        best_count = 0
        
        for c in neighbors[u]:
            # Count how many neighbors of c are in uncovered_set
            # neighbors[c] is a numpy array.
            # We can't easily count intersection with set in numpy.
            # Let's convert neighbors[c] to list? No, small size (17).
            cnt = 0
            for nb in neighbors[c]:
                if nb in uncovered_set:
                    cnt += 1
            
            if cnt > best_count:
                best_count = cnt
                best_c = c
                
        if best_c == -1:
            # Should not happen if u is uncovered and radius>=0
            best_c = u
            
        # Add best_c
        code.append(best_c)
        
        # Update uncovered_set
        for nb in neighbors[best_c]:
            uncovered_set.discard(nb)
            
        # Also update covered_arr if needed (not strictly necessary if we use set)
        
    return code

def _local_search(code, q, n, radius, neighbors, max_time=100.0):
    """
    Local search to minimize code size.
    1. Removal: If a codeword's neighbors are all covered by others, remove it.
    2. Swap: Replace a codeword with another that allows removal of more or maintains coverage with fewer.
    """
    start = time.time()
    total = q ** n
    code = list(code)
    
    # Build coverage count array
    cover_count = np.zeros(total, dtype=np.int32)
    for c in code:
        for nb in neighbors[c]:
            cover_count[nb] += 1
            
    # Helper to check if code is valid (all cover_count > 0)
    def is_valid():
        return np.all(cover_count > 0)
        
    # Helper to try removing codeword at index i
    def try_remove(i):
        nonlocal cover_count, code
        c = code[i]
        # Check if any neighbor has cover_count == 1
        # If so, removing c would make it 0 -> invalid.
        if np.any(cover_count[neighbors[c]] == 1):
            return False
        
        # Safe to remove
        for nb in neighbors[c]:
            cover_count[nb] -= 1
        code.pop(i)
        return True

    # Helper to try swapping codeword i with new_c
    def try_swap(i, new_c):
        nonlocal cover_count, code
        old_c = code[i]
        if old_c == new_c:
            return False
            
        # Calculate new cover counts
        # Decrease old_c neighbors, increase new_c neighbors
        # Check if any word drops to 0
        
        # Words affected: union of neighbors[old_c] and neighbors[new_c]
        # We can simulate on a copy or check conditions.
        
        # Condition: For all w in neighbors[old_c], if cover_count[w] == 1, 
        # then w must be in neighbors[new_c] (so it gets +1 back).
        
        set_new = set(neighbors[new_c])
        for w in neighbors[old_c]:
            if cover_count[w] == 1:
                if w not in set_new:
                    return False
                    
        # If valid, perform swap
        for w in neighbors[old_c]:
            cover_count[w] -= 1
        for w in neighbors[new_c]:
            cover_count[w] += 1
        code[i] = new_c
        return True

    no_improve = 0
    max_no_improve = 20
    
    while time.time() - start < max_time:
        improved = False
        
        # Phase 1: Removals
        # Iterate backwards to avoid index issues
        for i in range(len(code) - 1, -1, -1):
            if time.time() - start > max_time:
                break
            if try_remove(i):
                improved = True
                no_improve = 0
                # Restart removal phase? Or continue?
                # Continuing is fine, but indices shift. 
                # Since we popped i, next i-1 is safe.
                # But we might miss new removals. Let's just break and restart loop.
                break
        
        if improved:
            continue
            
        # Phase 2: Swaps
        # Try random swaps
        swaps_tried = 0
        max_swaps = 500
        while swaps_tried < max_swaps:
            if time.time() - start > max_time:
                break
            swaps_tried += 1
            
            if not code:
                break
                
            i = random.randint(0, len(code) - 1)
            # Generate a random new codeword
            # To be smarter, we could pick from neighbors of current codeword?
            # Or just random.
            new_c = random.randint(0, total - 1)
            
            if try_swap(i, new_c):
                improved = True
                no_improve = 0
                break
                
        if not improved:
            no_improve += 1
            if no_improve > max_no_improve:
                break
                
    return code

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # 1. Precompute neighbors
    # This is O(N * n * q) which is 6561 * 8 * 3 ~ 150k operations. Very fast.
    neighbors = _generate_neighbors(q, n, radius)
    
    # 2. Greedy Init
    # Use a tight time limit for init to leave time for local search.
    # Init usually converges fast.
    code = _greedy_init(q, n, radius, neighbors, max_time=5.0)
    
    # 3. Local Search
    remaining = deadline - time.time()
    if remaining > 5.0:
        code = _local_search(code, q, n, radius, neighbors, max_time=remaining - 5.0)
        
    # 4. Convert to list of lists
    result = []
    for c in code:
        w = []
        t = c
        for _ in range(n):
            w.append(t % q)
            t //= q
        w.reverse()
        result.append(w)
        
    return result
