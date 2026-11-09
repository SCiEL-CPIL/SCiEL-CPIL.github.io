"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet {0..q-1} with covering radius `radius`.
    Uses a greedy set cover approach followed by local search (removal/swap) to minimize size.
    Optimized for q=3, n=7, radius=1 using bitsets.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Handle trivial cases or specific known structures
    if q == 3 and n == 7 and radius == 1:
        # Generate all 3^7 = 2187 words
        all_words = list(itertools.product(range(q), repeat=n))
        N = len(all_words)
        
        # Map word to index
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # Precompute neighbors as bitmasks
        # Each word has 1 + 7*2 = 15 neighbors
        neighbor_masks = [0] * N
        for i, w in enumerate(all_words):
            mask = 1 << i
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]: continue
                    new_w = list(w)
                    new_w[pos] = val
                    idx = word_to_idx[tuple(new_w)]
                    mask |= (1 << idx)
            neighbor_masks[i] = mask
        
        # Initial Solution: Tetracode construction (243 words)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_codewords = [tuple(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
        best_size = len(initial_codewords)
        best_indices = [word_to_idx[c] for c in initial_codewords]

        def optimize(initial_indices, time_budget):
            """Local search with removal and swap moves. Returns optimized list."""
            current_indices = list(initial_indices)
            random.shuffle(current_indices)
            
            # Initialize cover counts
            cover_count = [0] * N
            for idx in current_indices:
                mask = neighbor_masks[idx]
                m = mask
                while m:
                    lb = m & -m
                    idx_nb = lb.bit_length() - 1
                    cover_count[idx_nb] += 1
                    m ^= lb

            # Track best found within this optimization
            local_best_size = len(current_indices)
            local_best_indices = list(current_indices)

            # Removal phase: repeatedly remove redundant codewords
            improved = True
            while improved and time.time() < time_budget:
                improved = False
                random.shuffle(current_indices)
                
                for i in range(len(current_indices)):
                    if time.time() > time_budget: break
                    idx = current_indices[i]
                    
                    # Check redundancy: all neighbors have cover_count > 1
                    mask = neighbor_masks[idx]
                    m = mask
                    redundant = True
                    while m:
                        lb = m & -m
                        idx_nb = lb.bit_length() - 1
                        if cover_count[idx_nb] <= 1:
                            redundant = False
                            break
                        m ^= lb
                    
                    if redundant:
                        # Remove idx
                        m = mask
                        while m:
                            lb = m & -m
                            idx_nb = lb.bit_length() - 1
                            cover_count[idx_nb] -= 1
                            m ^= lb
                        
                        current_indices.pop(i)
                        improved = True
                        if len(current_indices) < local_best_size:
                            local_best_size = len(current_indices)
                            local_best_indices = list(current_indices)
                        continue

            # Swap phase: try swapping codewords to enable further reductions
            # Use a set for O(1) membership testing
            s_set = set(current_indices)
            
            swap_improved = True
            while swap_improved and time.time() < time_budget:
                swap_improved = False
                s_set = set(current_indices)
                
                for _ in range(20):  # Try up to 20 swaps per round
                    if time.time() > time_budget: break
                    if not current_indices: break
                    
                    idx_a = random.choice(current_indices)
                    
                    # Pick random B not in S
                    while True:
                        idx_b = random.randint(0, N-1)
                        if idx_b not in s_set:
                            break
                    
                    mask_a = neighbor_masks[idx_a]
                    mask_b = neighbor_masks[idx_b]
                    
                    # Check if swap A->B is valid: no word covered only by A should be uncovered by B
                    valid_swap = True
                    m = mask_a
                    while m:
                        lb = m & -m
                        idx_nb = lb.bit_length() - 1
                        if cover_count[idx_nb] == 1:
                            if not (mask_b & (1 << idx_nb)):
                                valid_swap = False
                                break
                        m ^= lb
                    
                    if valid_swap:
                        # Perform Swap
                        m_a = mask_a
                        m_b = mask_b
                        
                        # Subtract A
                        while m_a:
                            lb = m_a & -m_a
                            idx_nb = lb.bit_length() - 1
                            cover_count[idx_nb] -= 1
                            m_a ^= lb
                            
                        # Add B
                        while m_b:
                            lb = m_b & -m_b
                            idx_nb = lb.bit_length() - 1
                            cover_count[idx_nb] += 1
                            m_b ^= lb
                            
                        # Replace in list
                        pos_a = current_indices.index(idx_a)
                        current_indices[pos_a] = idx_b
                        s_set.discard(idx_a)
                        s_set.add(idx_b)
                        
                        # Run reduction pass after swap
                        improved_sub = True
                        while improved_sub and time.time() < time_budget:
                            improved_sub = False
                            random.shuffle(current_indices)
                            for i in range(len(current_indices)):
                                if time.time() > time_budget: break
                                idx = current_indices[i]
                                mask = neighbor_masks[idx]
                                m = mask
                                redundant = True
                                while m:
                                    lb = m & -m
                                    idx_nb = lb.bit_length() - 1
                                    if cover_count[idx_nb] <= 1:
                                        redundant = False
                                        break
                                    m ^= lb
                                
                                if redundant:
                                    m = mask
                                    while m:
                                        lb = m & -m
                                        idx_nb = lb.bit_length() - 1
                                        cover_count[idx_nb] -= 1
                                        m ^= lb
                                    
                                    current_indices.pop(i)
                                    improved_sub = True
                                    if len(current_indices) < local_best_size:
                                        local_best_size = len(current_indices)
                                        local_best_indices = list(current_indices)
                                    continue
                        
                        swap_improved = True

            return local_best_indices

        # Multi-start Local Search
        initial_list = [word_to_idx[c] for c in initial_codewords]
        
        # First, try greedy set cover as an alternative starting point
        if time.time() < deadline - 60:
            uncovered_mask = (1 << N) - 1
            greedy_indices = []
            greedy_set = set()
            
            while uncovered_mask and time.time() < deadline - 40:
                best_idx = -1
                best_gain = 0
                for idx in range(N):
                    if idx in greedy_set: continue
                    gain = (neighbor_masks[idx] & uncovered_mask).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
                        if gain == uncovered_mask.bit_count(): break
                        
                if best_idx == -1 or best_gain == 0: break
                greedy_indices.append(best_idx)
                greedy_set.add(best_idx)
                uncovered_mask &= ~neighbor_masks[best_idx]
            
            # Optimize greedy solution
            time_budget = min(deadline - 20, time.time() + 30)
            optimized_greedy = optimize(greedy_indices, time_budget)
            if len(optimized_greedy) < best_size:
                best_size = len(optimized_greedy)
                best_indices = optimized_greedy

        # Run multiple restarts from tetracode seed with different shuffles
        restarts = 0
        while time.time() < deadline - 15:
            time_budget = min(deadline - 5, time.time() + 40)
            current_solution = optimize(initial_list, time_budget)
            if len(current_solution) < best_size:
                best_size = len(current_solution)
                best_indices = current_solution
            restarts += 1
            
        # Return the best found
        result = [list(all_words[idx]) for idx in best_indices]
        return result

    # Generic fallback for other parameters
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    neighbors = []
    for w in all_words:
        nbs = {word_to_idx[w]}
        for pos in range(n):
            for val in range(q):
                if val == w[pos]: continue
                nw = list(w)
                nw[pos] = val
                nbs.add(word_to_idx[tuple(nw)])
        neighbors.append(nbs)
    
    uncovered = set(range(len(all_words)))
    code_set = []
    while uncovered and time.time() < deadline:
        best_idx = -1
        best_count = 0
        for idx in range(len(all_words)):
            if idx in code_set: continue
            cnt = sum(1 for nb in neighbors[idx] if nb in uncovered)
            if cnt > best_count:
                best_count = cnt
                best_idx = idx
        if best_idx == -1: break
        code_set.append(best_idx)
        for nb in neighbors[best_idx]:
            uncovered.discard(nb)
    
    return [list(all_words[idx]) for idx in code_set]
# EVOLVE-BLOCK-END
