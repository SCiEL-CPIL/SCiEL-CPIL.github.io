import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Returns a list of codewords (each a list of ints in 0..q-1).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    if n == 0:
        return [[]]

    # Specific optimization for the known hard case: q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        return _solve_q3_n6_r1(deadline)

    # Fallback for other cases
    # Use a simple greedy or direct sum if applicable
    if q == 3 and radius == 1 and n >= 4:
        # Tetracode direct sum
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return code
    
    # Generic greedy for small n
    return _greedy_covering(q, n, radius, deadline)

def _solve_q3_n6_r1(deadline):
    q = 3
    n = 6
    radius = 1
    
    # Generate all words and map to indices
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    idx_to_word = {i: w for i, w in enumerate(all_words)}
    N = q ** n  # 729
    
    # Precompute neighbors for each word index
    # neighbors[i] is a list of indices covered by word i (distance <= radius)
    neighbors = [None] * N
    for i, w in enumerate(all_words):
        neigh = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                neigh.append(word_to_idx[tuple(new_w)])
        neighbors[i] = neigh
        
    # 1. Greedy Initialization
    uncovered = set(range(N))
    code_indices = []
    
    # Optimization: Pre-compute neighbors as sets for faster intersection?
    # Or just iterate. 729 * 13 is small.
    # To speed up greedy, we can maintain a count of how many uncovered words each candidate covers.
    # But updating this after each selection is O(N * 13) which is ~9000 ops. 729 selections -> ~6M ops. Fast.
    
    # Let's use a simpler approach: Pick best, update uncovered.
    # To break ties and potentially improve, we can use randomization in tie-breaking or restarts.
    # Let's try deterministic greedy first, then local search.
    
    while uncovered and time.time() < deadline:
        best_idx = -1
        best_count = -1
        
        # We can restrict candidates to words that cover at least one uncovered word.
        # Actually, any word is a candidate, but we only care about intersection with uncovered.
        
        # Optimization: Instead of checking all N words every time, 
        # we can just check all N. 729 is small.
        for idx in range(N):
            # Count how many of neighbors[idx] are in uncovered
            # Using set intersection is fast in Python
            cnt = len(neighbors[idx] & uncovered) if isinstance(neighbors[idx], set) else sum(1 for x in neighbors[idx] if x in uncovered)
            if cnt > best_count:
                best_count = cnt
                best_idx = idx
        
        if best_count == 0:
            break
            
        code_indices.append(best_idx)
        uncovered.difference_update(neighbors[best_idx])
        
    # Convert neighbors to lists for local search (faster iteration if not using set ops heavily)
    # Actually, we used sets in greedy? No, neighbors were lists. 
    # Let's ensure neighbors are lists for the removal phase, but we used `&` which implies set? 
    # Ah, in the greedy loop above, I wrote `neighbors[idx] & uncovered`. 
    # `neighbors[idx]` is a list. `list & set` is not valid. 
    # I must convert neighbors to sets for the greedy phase or use a loop.
    
    # Let's re-do the greedy part properly in the function logic below.
    # I will rewrite the function content to be correct.
    
    # --- RE-IMPLEMENTATION OF _solve_q3_n6_r1 ---
    
    # Precompute neighbors as sets for fast intersection in Greedy
    neighbors_set = [set(neigh) for neigh in neighbors]
    
    uncovered = set(range(N))
    code_indices = []
    
    while uncovered and time.time() < deadline:
        best_idx = -1
        best_count = -1
        
        # Find word covering most uncovered
        for idx in range(N):
            cnt = len(neighbors_set[idx] & uncovered)
            if cnt > best_count:
                best_count = cnt
                best_idx = idx
        
        if best_count == 0:
            break
            
        code_indices.append(best_idx)
        uncovered.difference_update(neighbors_set[best_idx])
        
    # 2. Local Search
    # Initialize coverage counts
    counts = np.zeros(N, dtype=np.int32)
    for idx in code_indices:
        for n_idx in neighbors[idx]:
            counts[n_idx] += 1
            
    active_indices = list(code_indices)
    best_size = len(active_indices)
    best_code = list(active_indices)
    
    # Helper to check if index can be removed
    def can_remove(idx):
        for n_idx in neighbors[idx]:
            if counts[n_idx] == 1:
                return False
        return True

    # Removal Loop
    def remove_redundant():
        removed_any = False
        # Iterate multiple times to catch cascading redundancies
        # Or just shuffle and check once per pass.
        while time.time() < deadline:
            # Shuffle to avoid bias
            random.shuffle(active_indices)
            pass_removed = False
            for idx in list(active_indices):
                if can_remove(idx):
                    for n_idx in neighbors[idx]:
                        counts[n_idx] -= 1
                    active_indices.remove(idx)
                    pass_removed = True
            if not pass_removed:
                break
        return pass_removed

    # Run initial removal
    remove_redundant()
    
    if len(active_indices) < best_size:
        best_size = len(active_indices)
        best_code = list(active_indices)
        
    # Perturbation Loop
    # Try adding a random word, then removing redundancies.
    # If size decreases, keep. Else revert.
    
    # To escape local minima, we can also try "Swap" strategies, but Add-Remove is good.
    
    while time.time() < deadline:
        # Pick a random word to add
        random_idx = random.randint(0, N - 1)
        
        if random_idx in active_indices:
            continue
            
        # Save state
        old_size = len(active_indices)
        old_counts = counts.copy()
        old_active = list(active_indices)
        
        # Add random_idx
        active_indices.append(random_idx)
        for n_idx in neighbors[random_idx]:
            counts[n_idx] += 1
            
        # Try to remove redundant words
        # We might be able to remove more than 1 word
        remove_redundant()
        
        if len(active_indices) < best_size:
            best_size = len(active_indices)
            best_code = list(active_indices)
            
        if len(active_indices) > best_size:
            # Revert
            counts[:] = old_counts
            active_indices = old_active
            
        # If we didn't improve, we revert implicitly by not updating best_code,
        # but we need to revert the current state to continue searching from a valid base?
        # Actually, if we don't revert, we are exploring a larger code.
        # Usually, we want to stay at the best found so far to keep the search tight.
        # So we revert if we didn't improve.
        if len(active_indices) >= best_size:
             counts[:] = old_counts
             active_indices = old_active
             
    return [list(idx_to_word[i]) for i in best_code]

def _greedy_covering(q, n, radius, deadline):
    """
    Greedy construction of a covering code.
    """
    if n > 8:
        # Fallback: Direct sum if possible, else random
        if q == 3:
             tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
             if n >= 4:
                 return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
             else:
                 pass
        
        if n <= 8:
            all_words = list(itertools.product(range(q), repeat=n))
            uncovered = set(all_words)
            code = []
            
            def get_neighbors(c):
                neighbors = {c}
                for i in range(n):
                    orig = c[i]
                    for val in range(q):
                        if val == orig: continue
                        lst = list(c)
                        lst[i] = val
                        neighbors.add(tuple(lst))
                return neighbors

            while uncovered and time.time() < deadline:
                best_word = None
                best_count = -1
                
                if n <= 6:
                    # Cache neighbors for speed
                    neighbors_map = {w: get_neighbors(w) for w in all_words}
                    for w in all_words:
                        cnt = sum(1 for x in neighbors_map[w] if x in uncovered)
                        if cnt > best_count:
                            best_count = cnt
                            best_word = w
                else:
                    best_word = random.choice(all_words)
                    
                if best_word is None or best_count == 0:
                    break
                    
                code.append(list(best_word))
                uncovered.difference_update(get_neighbors(best_word))
                
            return code
        else:
            if q == 3 and n >= 4:
                 tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
                 return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
            else:
                 return [[0]*n]
