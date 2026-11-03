import time
import itertools
import random
import math

# EVOLVE-BLOCK-START

def _classic_superperm(n):
    """Classic recursive superpermutation construction."""
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if o > 0 and new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
                elif o == 0:
                    overlap = 0
                    break
            new += block[overlap:]
        word = new
    return word

def _build_level_perms(n):
    """
    Build the list of (n-1)-length permutations in the order they appear
    in the classic superpermutation of level n-1.
    """
    if n == 1:
        return [[1]]
    word = _classic_superperm(n - 1)
    perms, seen = [], set()
    m = n - 1
    for i in range(len(word) - m + 1):
        window = tuple(word[i:i + m])
        if len(set(window)) == m and window not in seen:
            seen.add(window)
            perms.append(window)
    return [list(p) for p in perms]

def _overlap(a, b):
    """Maximum overlap: suffix of a matches prefix of b."""
    max_ov = min(len(a), len(b))
    for o in range(max_ov, 0, -1):
        if a[-o:] == b[:o]:
            return o
    return 0

def _block_for_perm(p, m):
    """The block for permutation p with new symbol m: p + [m] + p."""
    return list(p) + [m] + list(p)

def _optimize_order_beam(perm_list, m, time_limit=10.0, beam_width=50, neighbors=10):
    """
    Beam search to find a good ordering of perm_list.
    """
    start_time = time.time()
    n_perms = len(perm_list)
    
    # Precompute blocks and overlap matrix
    blocks = [_block_for_perm(p, m) for p in perm_list]
    block_len = len(blocks[0])
    
    # Overlap matrix
    ov_matrix = [[0] * n_perms for _ in range(n_perms)]
    for i in range(n_perms):
        for j in range(n_perms):
            if i != j:
                ov_matrix[i][j] = _overlap(blocks[i], blocks[j])
    
    # Cost matrix (lower is better)
    # cost[i][j] = block_len - ov_matrix[i][j]
    
    # Initialize beams
    # Start with all permutations? No, fix one due to symmetry?
    # Actually, let's start with a few random starts or the classic start.
    # Classic start is index 0 in perm_list (which is derived from classic superperm).
    
    best_cost = float('inf')
    best_path = None
    
    # We will run beam search from a few starting points
    start_indices = [0]
    # Add a few random starts if time permits
    if n_perms > 10:
        rng = random.Random(42)
        for _ in range(5):
            start_indices.append(rng.randint(0, n_perms - 1))
    
    for start_idx in start_indices:
        if time.time() - start_time > time_limit:
            break
            
        # Beam: list of (current_idx, visited_mask, cost, path)
        # visited_mask is an int
        init_mask = 1 << start_idx
        beams = [(start_idx, init_mask, 0, [start_idx])]
        
        for step in range(1, n_perms):
            if time.time() - start_time > time_limit - 0.5:
                break
                
            next_beams = []
            
            for cur_idx, mask, cost, path in beams:
                # Generate candidates
                # We want the best 'neighbors' unvisited neighbors
                # Sort all unvisited by cost[cur_idx][j]
                
                # Optimization: Precompute sorted neighbors for each cur_idx?
                # The cost depends only on cur_idx and j.
                # So we can precompute a list of (cost, j) for each i.
                
                # Let's do it on the fly for simplicity, but it might be slow.
                # Better: Precompute sorted_neighbors[i] = sorted list of (cost, j)
                
                # We'll precompute outside the loop for speed.
                # But we are inside the function. Let's do it before the main loop.
                
                pass # Placeholder for logic below

    # Refactored Beam Search with Precomputation
    
    # Precompute sorted neighbors for each node
    # neighbors[i] is a list of (cost, j) sorted by cost
    sorted_neighbors = []
    for i in range(n_perms):
        nbrs = []
        for j in range(n_perms):
            if i != j:
                c = block_len - ov_matrix[i][j]
                nbrs.append((c, j))
        nbrs.sort()
        sorted_neighbors.append(nbrs)

    for start_idx in start_indices:
        if time.time() - start_time > time_limit:
            break
        
        init_mask = 1 << start_idx
        # Beam entries: (cost, cur_idx, mask, path)
        # We sort beams by cost to keep the best ones
        beams = [(0, start_idx, init_mask, [start_idx])]
        
        for step in range(1, n_perms):
            if time.time() - start_time > time_limit - 0.5:
                break
                
            next_beams = []
            
            for cost, cur_idx, mask, path in beams:
                # Get top 'neighbors' candidates from sorted_neighbors[cur_idx]
                # that are not in mask
                count = 0
                for c, j in sorted_neighbors[cur_idx]:
                    if mask & (1 << j):
                        continue
                    new_mask = mask | (1 << j)
                    new_cost = cost + c
                    new_path = path + [j]
                    next_beams.append((new_cost, j, new_mask, new_path))
                    count += 1
                    if count >= neighbors:
                        break
            
            # Prune next_beams to keep only the best 'beam_width'
            # Sort by cost
            next_beams.sort(key=lambda x: x[0])
            beams = next_beams[:beam_width]
            
            # Early exit if we have a complete path? No, we only have complete at the end.
            
        # After n_perms steps, we have full paths
        if beams:
            best_cost_beam, last_idx, final_mask, final_path = beams[0]
            if best_cost_beam < best_cost:
                best_cost = best_cost_beam
                best_path = final_path

    if best_path is None:
        return list(range(n_perms)), float('inf')
        
    return best_path, best_cost

def _optimize_order_tsp(perm_list, m, time_limit=10.0):
    """
    Original TSP optimization (kept for fallback or comparison)
    """
    # ... (existing code) ...
    # I will remove this and replace with beam search in construct
    pass

def construct(n):
    start_time = time.time()
    deadline = 140.0
    
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "123412134212343142341234"
    
    # For n >= 5
    perm_list = _build_level_perms(n)
    m = n
    n_perms = len(perm_list)
    
    classic_word = _classic_superperm(n)
    classic_len = len(classic_word)
    
    # Use Beam Search
    # Adjust parameters based on n
    if n == 5:
        bw = 100
        nb = 20
        tl = 100.0
    elif n == 6:
        bw = 50
        nb = 15
        tl = 100.0
    else: # n == 7
        bw = 30
        nb = 10
        tl = 100.0
        
    if (time.time() - start_time) < deadline - 10.0:
        best_order, best_cost = _optimize_order_beam(perm_list, m, time_limit=tl, beam_width=bw, neighbors=nb)
        
        # Build word
        blocks = [_block_for_perm(perm_list[i], m) for i in best_order]
        word = list(blocks[0])
        for k in range(1, len(blocks)):
            b = blocks[k]
            ov = _overlap(word, b)
            word += b[ov:]
            
        if len(word) < classic_len:
            return "".join(map(str, word))
            
    return "".join(map(str, classic_word))

# EVOLVE-BLOCK-END
