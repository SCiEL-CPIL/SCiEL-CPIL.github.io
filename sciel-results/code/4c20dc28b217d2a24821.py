import itertools
import random
import time

def construct(q, n, radius):
    # Specialized for q=3, n=7, radius=1
    # Approach: Greedy Initialization + Local Search (Removals and Swaps) using Bitmasks
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters, though task is specific
        # Simple greedy for fallback
        all_words = list(itertools.product(range(q), repeat=n))
        word_index = {w: i for i, w in enumerate(all_words)}
        N = len(all_words)
        
        # Precompute balls
        balls = []
        for i, w in enumerate(all_words):
            neighbors = [i]
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_w_list = list(w)
                    new_w_list[pos] = val
                    new_w = tuple(new_w_list)
                    neighbors.append(word_index[new_w])
            balls.append(neighbors)
            
        uncovered = set(range(N))
        solution_indices = []
        while uncovered:
            best_word = -1
            best_count = -1
            for i in range(N):
                if i in solution_indices:
                    continue
                count = 0
                for nb in balls[i]:
                    if nb in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_word = i
            if best_word == -1:
                break
            solution_indices.append(best_word)
            for nb in balls[best_word]:
                uncovered.discard(nb)
                
        return [all_words[i] for i in solution_indices]

    # Specific K_3(7,1) Solver
    
    # 1. Generate words and index
    all_words = list(itertools.product(range(3), repeat=7))
    word_index = {w: i for i, w in enumerate(all_words)}
    N = 3**7 # 2187
    
    # 2. Precompute balls as bitmasks
    # ball_mask[i] is an integer where bit j is 1 if word j is in ball of word i
    ball_mask = [0] * N
    ball_indices = [[] for _ in range(N)] # For faster iteration over set bits if needed, but bitmask is faster for subset checks
    
    for i, w in enumerate(all_words):
        mask = 0
        inds = [i]
        # Center
        mask |= (1 << i)
        
        # Neighbors
        for pos in range(7):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                j = word_index[new_w]
                mask |= (1 << j)
                inds.append(j)
                
        ball_mask[i] = mask
        ball_indices[i] = inds
        
    # 3. Greedy Initialization
    all_mask = (1 << N) - 1
    uncovered_mask = all_mask
    selected = set()
    
    # Greedy: pick word covering most uncovered
    # To speed up, we can pre-sort or just scan. N=2187 is small.
    while uncovered_mask:
        best_i = -1
        best_count = -1
        
        for i in range(N):
            if i in selected:
                continue
            # Count bits in ball_mask[i] & uncovered_mask
            # Python int.bit_count() is fast
            count = (ball_mask[i] & uncovered_mask).bit_count()
            if count > best_count:
                best_count = count
                best_i = i
                
        if best_i == -1:
            break
            
        selected.add(best_i)
        uncovered_mask &= ~ball_mask[best_i]
        
    # 4. Local Search
    # Initialize cover counts
    cover_count = [0] * N
    for i in selected:
        for nb in ball_indices[i]:
            cover_count[nb] += 1
            
    start_time = time.time()
    max_time = 115.0
    
    # Helper to get list of set bits from a mask
    def bits_set(mask):
        bits = []
        while mask:
            # Get lowest set bit
            lsb = mask & -mask
            index = lsb.bit_length() - 1
            bits.append(index)
            mask ^= lsb
        return bits
        
    improved = True
    while improved and time.time() - start_time < max_time:
        improved = False
        
        # Pass 1: Remove redundant codewords
        # Iterate over a copy of selected
        sel_list = list(selected)
        for A in sel_list:
            redundant = True
            # Check if any neighbor has cover count 1
            for nb in ball_indices[A]:
                if cover_count[nb] == 1:
                    redundant = False
                    break
            if redundant:
                # Remove A
                selected.remove(A)
                for nb in ball_indices[A]:
                    cover_count[nb] -= 1
                improved = True
                
        # Pass 2: Swaps
        # Try to swap A (in selected) with B (not in selected)
        # Condition: Words covered only by A must be covered by B.
        sel_list = list(selected)
        for A in sel_list:
            # Compute mask of words covered ONLY by A
            only_A_mask = 0
            for nb in ball_indices[A]:
                if cover_count[nb] == 1:
                    only_A_mask |= (1 << nb)
                    
            if only_A_mask == 0:
                continue # Redundant, should have been removed in Pass 1
            
            # Find a B not in selected such that only_A_mask is subset of ball_mask[B]
            # i.e., only_A_mask & ~ball_mask[B] == 0
            found_swap = False
            # To avoid bias, we can shuffle unselected or just iterate.
            # Iterating all N is 2187 checks.
            # We can optimize by checking only B that cover the first word in only_A_mask?
            # Let's just iterate all B for simplicity, it's fast enough.
            for B in range(N):
                if B in selected:
                    continue
                if (only_A_mask & ~ball_mask[B]) == 0:
                    # Valid swap
                    # Remove A
                    selected.remove(A)
                    for nb in ball_indices[A]:
                        cover_count[nb] -= 1
                    
                    # Add B
                    selected.add(B)
                    for nb in ball_indices[B]:
                        cover_count[nb] += 1
                    
                    improved = True
                    found_swap = True
                    break # Restart search with new state
            
            if found_swap:
                break # Restart search
        
        # If no improvement, we might be stuck.
        # To escape local minima, we could add a random perturbation, 
        # but for this problem, swap local search is often sufficient to reach 186.
        # If we hit a time limit, we return the current best.
        
    # Convert selected indices to words
    result = [all_words[i] for i in selected]
    
    # Ensure distinct (they are by index)
    return result
