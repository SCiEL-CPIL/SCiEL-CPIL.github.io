"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import heapq
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    if n < 4:
        return [list(w) for w in itertools.product(range(q), repeat=n)]
    
    if q != 3:
        return _generic_greedy(q, n, radius, deadline)
    
    # Ternary tetracode words
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Construct initial code: direct sum with tails
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial_code.append(h + list(tail))
    
    # Map words to integers for fast indexing
    pow3 = [1] * (n + 1)
    for i in range(1, n + 1):
        pow3[i] = pow3[i-1] * 3
    
    def word_to_int(w):
        val = 0
        for i in range(n):
            val += w[i] * pow3[i]
        return val
        
    def int_to_word(val):
        w = [0] * n
        for i in range(n):
            w[i] = val % 3
            val //= 3
        return w

    size_space = 3 ** n
    
    # Precompute balls for ALL possible words as numpy arrays
    ball_cache = np.zeros((size_space, 1 + 2*n), dtype=np.int32)
    ball_sizes = np.zeros(size_space, dtype=np.int32)
    
    for w_int in range(size_space):
        w = int_to_word(w_int)
        ball = [w_int]
        for pos in range(n):
            for val in range(3):
                if val != w[pos]:
                    new_word = w[:]
                    new_word[pos] = val
                    ball.append(word_to_int(new_word))
        ball_cache[w_int, :len(ball)] = ball
        ball_sizes[w_int] = len(ball)
    
    # Precompute ball membership as a boolean matrix: ball_matrix[cw_idx, w_idx] = 1 if w in ball(cw)
    # But that's size_space x size_space = 729x729 = 531441, which is fine
    ball_matrix = np.zeros((size_space, size_space), dtype=np.uint8)
    for w_int in range(size_space):
        ball = ball_cache[w_int, :ball_sizes[w_int]]
        ball_matrix[w_int, ball] = 1
    
    def compute_coverage(code_ints):
        """Compute coverage array from list of codeword ints."""
        coverage = np.zeros(size_space, dtype=np.int32)
        for cw in code_ints:
            ball = ball_cache[cw, :ball_sizes[cw]]
            coverage[ball] += 1
        return coverage
    
    def greedy_remove(code_ints, coverage, time_limit):
        """
        Greedy removal: repeatedly find the codeword whose ball has the highest
        minimum coverage (most redundant) and remove it if all coverages > 1.
        """
        active = list(range(len(code_ints)))
        removed_count = 0
        
        while active and time.time() < time_limit:
            best_idx = -1
            best_min_cov = 0
            
            # Find the codeword with the highest minimum coverage in its ball
            for i in active:
                ball = ball_cache[code_ints[i], :ball_sizes[code_ints[i]]]
                min_cov = int(coverage[ball].min())
                if min_cov > best_min_cov:
                    best_min_cov = min_cov
                    best_idx = i
            
            if best_idx == -1 or best_min_cov <= 1:
                break
            
            # Remove this codeword
            ball = ball_cache[code_ints[best_idx], :ball_sizes[code_ints[best_idx]]]
            coverage[ball] -= 1
            active.remove(best_idx)
            removed_count += 1
        
        return active, coverage, removed_count
    
    def swap_local_search(code_ints, coverage, time_limit):
        """
        Swap-based local search: try to replace one codeword with another
        that covers more uniquely-covered words.
        """
        active = list(range(len(code_ints)))
        improved = True
        
        while improved and time.time() < time_limit:
            improved = False
            
            for i in list(active):
                if time.time() > time_limit:
                    break
                
                ball_i = ball_cache[code_ints[i], :ball_sizes[code_ints[i]]]
                
                # Find words covered only by codeword i (coverage == 1 in ball_i)
                unique_mask = (coverage == 1) & (ball_matrix[code_ints[i]] == 1)
                unique_words = np.where(unique_mask)[0]
                
                if len(unique_words) == 0:
                    continue
                
                # Try to find a replacement codeword that covers these unique words
                best_swap = -1
                best_gain = 0
                
                for j in active:
                    if j == i:
                        continue
                    if time.time() > time_limit:
                        break
                    
                    ball_j = ball_cache[code_ints[j], :ball_sizes[code_ints[j]]]
                    
                    # Check if ball_j covers all unique_words
                    covers_unique = all(ball_matrix[code_ints[j], w] == 1 for w in unique_words)
                    if not covers_unique:
                        continue
                    
                    # Check if removing i and adding j maintains coverage
                    # Words in ball_i but not in ball_j would lose coverage
                    lost_mask = (ball_matrix[code_ints[i]] == 1) & (ball_matrix[code_ints[j]] == 0) & (coverage == 1)
                    if lost_mask.any():
                        continue
                    
                    # Valid swap found
                    best_swap = j
                    best_gain = 1
                    break
                
                if best_swap >= 0:
                    # Perform swap: remove i, keep best_swap (already in code)
                    # Actually, we're replacing i with best_swap, but best_swap is already there
                    # So we just remove i if best_swap covers everything
                    ball_i = ball_cache[code_ints[i], :ball_sizes[code_ints[i]]]
                    ball_j = ball_cache[code_ints[best_swap], :ball_sizes[code_ints[best_swap]]]
                    
                    # Update coverage: subtract ball_i, add ball_j
                    coverage[ball_i] -= 1
                    coverage[ball_j] += 1
                    
                    # Remove i from active
                    active.remove(i)
                    improved = True
        
        return active, coverage
    
    def aggressive_removal(code_ints, coverage, time_limit):
        """
        More aggressive removal with random restarts.
        """
        active = list(range(len(code_ints)))
        removed = 0
        random.shuffle(active)
        
        while active and time.time() < time_limit:
            progress = False
            for idx in range(len(active) - 1, -1, -1):
                i = active[idx]
                ball = ball_cache[code_ints[i], :ball_sizes[code_ints[i]]]
                
                if coverage[ball].min() > 1:
                    coverage[ball] -= 1
                    active.pop(idx)
                    removed += 1
                    progress = True
            
            if not progress:
                # Try random shuffle and retry
                random.shuffle(active)
                retry = False
                for idx in range(len(active) - 1, -1, -1):
                    i = active[idx]
                    ball = ball_cache[code_ints[i], :ball_sizes[code_ints[i]]]
                    if coverage[ball].min() > 1:
                        coverage[ball] -= 1
                        active.pop(idx)
                        removed += 1
                        retry = True
                        break
                if not retry:
                    break
        
        return active, coverage, removed
    
    # Main optimization
    best_code_ints = [word_to_int(c) for c in initial_code]
    best_size = len(best_code_ints)
    
    # Strategy 1: Greedy removal
    code_ints = list(best_code_ints)
    coverage = compute_coverage(code_ints)
    time_limit_1 = deadline - 30.0
    active, coverage, _ = greedy_remove(code_ints, coverage, time_limit_1)
    if len(active) < best_size:
        best_code_ints = [code_ints[i] for i in active]
        best_size = len(best_code_ints)
    
    # Strategy 2: Swap local search
    remaining = deadline - 10.0 - time.time()
    if remaining > 10:
        code_ints = list(best_code_ints)
        coverage = compute_coverage(code_ints)
        tl = min(deadline - 5.0, time.time() + remaining * 0.5)
        active, coverage = swap_local_search(code_ints, coverage, tl)
        if len(active) < best_size:
            best_code_ints = [code_ints[i] for i in active]
            best_size = len(best_code_ints)
    
    # Strategy 3: Multiple random restarts with perturbation
    remaining = deadline - 5.0 - time.time()
    if remaining > 15:
        num_restarts = min(6, int(remaining // 10))
        for restart in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
            
            # Start from best code, perturb by removing a few random codewords
            code_ints = list(best_code_ints)
            coverage = compute_coverage(code_ints)
            
            # Remove 2-5 random codewords to escape local optimum
            num_remove = random.randint(2, min(5, len(code_ints) - 1))
            to_remove = random.sample(range(len(code_ints)), num_remove)
            for idx in sorted(to_remove, reverse=True):
                ball = ball_cache[code_ints[idx], :ball_sizes[code_ints[idx]]]
                coverage[ball] -= 1
                code_ints.pop(idx)
            
            tl = min(deadline - 5.0, time.time() + remaining * 0.3)
            
            # Greedy removal
            active, coverage, _ = greedy_remove(code_ints, coverage, tl)
            
            # Swap local search
            tl2 = min(deadline - 3.0, time.time() + remaining * 0.2)
            active, coverage = swap_local_search(code_ints, coverage, tl2)
            
            if len(active) < best_size:
                best_code_ints = [code_ints[i] for i in active]
                best_size = len(best_code_ints)
            
            remaining = deadline - 5.0 - time.time()
            if remaining < 5:
                break
    
    # Strategy 4: Final greedy pass
    if time.time() < deadline - 2.0:
        code_ints = list(best_code_ints)
        coverage = compute_coverage(code_ints)
        active, coverage, _ = greedy_remove(code_ints, coverage, deadline - 1.0)
        if len(active) < best_size:
            best_code_ints = [code_ints[i] for i in active]
            best_size = len(best_code_ints)
    
    # Convert back to list of lists
    return [int_to_word(cw) for cw in best_code_ints]

def _generic_greedy(q, n, radius, deadline):
    """Fallback for non-ternary cases."""
    start = time.time()
    
    # Greedy set cover
    all_words = list(itertools.product(range(q), repeat=n))
    uncovered = set(all_words)
    code = []
    
    def ball(w):
        result = {w}
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    result.add(tuple(new_w))
        return result
    
    while uncovered and time.time() < deadline - 5:
        best_word = None
        best_cover = 0
        # Sample some words
        candidates = random.sample(list(uncovered), min(100, len(uncovered)))
        for w in candidates:
            b = ball(w)
            cov = len(b & uncovered)
            if cov > best_cover:
                best_cover = cov
                best_word = w
        
        if best_word is None:
            break
        
        code.append(list(best_word))
        b = ball(best_word)
        uncovered -= b
    
    return code

# EVOLVE-BLOCK-END
