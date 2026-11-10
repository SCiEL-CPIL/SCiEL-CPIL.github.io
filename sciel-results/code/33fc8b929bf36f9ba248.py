"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import heapq

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    if n < 4:
        return [list(w) for w in itertools.product(range(q), repeat=n)]
    
    if q != 3:
        # Fallback for non-ternary: greedy set cover
        return _generic_greedy(q, n, radius, deadline)
    
    # Ternary tetracode words
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Construct initial code: direct sum with tails
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial_code.append(h + list(tail))
    
    # Map words to integers for fast indexing
    def word_to_int(w):
        val = 0
        for i in range(n):
            val += w[i] * (3 ** i)
        return val
        
    def int_to_word(val):
        w = [0] * n
        for i in range(n):
            w[i] = val % 3
            val //= 3
        return w

    size_space = 3 ** n
    
    # Precompute balls for ALL possible words
    ball_cache = {}
    for w_int in range(size_space):
        w = int_to_word(w_int)
        ball = [w_int]
        for pos in range(n):
            for val in range(3):
                if val != w[pos]:
                    new_word = w[:]
                    new_word[pos] = val
                    ball.append(word_to_int(new_word))
        ball_cache[w_int] = ball
    
    # Precompute ball as frozenset for faster membership
    ball_set_cache = {w_int: frozenset(ball) for w_int, ball in ball_cache.items()}
    
    def greedy_remove_pq(code_list, coverage_arr, time_limit):
        """
        Priority-queue based greedy removal:
        - For each codeword, compute min coverage in its ball
        - Remove the one with highest min coverage if > 1
        - Use heap for efficiency
        """
        code_ints = [word_to_int(cw) for cw in code_list]
        active = set(range(len(code_ints)))
        
        # Initialize heap: (negative min_cov, index)
        # We want to remove the codeword with highest min_cov
        heap = []
        min_covs = [0] * len(code_ints)
        
        for i in range(len(code_ints)):
            ball = ball_cache[code_ints[i]]
            min_cov = min(coverage_arr[w] for w in ball)
            min_covs[i] = min_cov
            if min_cov > 1:
                heapq.heappush(heap, (-min_cov, i))
        
        removed_count = 0
        
        while heap and time.time() < time_limit:
            neg_mc, idx = heapq.heappop(heap)
            mc = -neg_mc
            
            if idx not in active:
                continue
            
            # Verify it's still removable
            ball = ball_cache[code_ints[idx]]
            current_min = min(coverage_arr[w] for w in ball)
            
            if current_min <= 1:
                continue
            
            # Remove this codeword
            for w in ball:
                coverage_arr[w] -= 1
            active.remove(idx)
            removed_count += 1
            
            # Update heap for affected codewords
            # For each other active codeword, check if its ball intersects with removed ball
            # This is expensive; instead, lazily recompute when popped
            
        new_code = [code_list[i] for i in active]
        return new_code, coverage_arr, removed_count
    
    def swap_local_search(code_list, coverage_arr, time_limit):
        """
        Swap-based local search: try to replace one codeword with another
        that covers more uniquely-covered words.
        """
        code_ints = [word_to_int(cw) for cw in code_list]
        active = list(range(len(code_ints)))
        
        # Precompute balls
        cw_balls = [ball_cache[code_ints[i]] for i in range(len(code_ints))]
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            for i in active:
                if time.time() > time_limit:
                    break
                
                ball_i = cw_balls[i]
                
                # Find words covered only by codeword i (coverage == 1 in ball_i)
                unique_words = [w for w in ball_i if coverage_arr[w] == 1]
                
                if not unique_words:
                    continue
                
                # Try to find a replacement codeword that covers these unique words
                # and doesn't break coverage of other words
                best_swap = -1
                best_gain = 0
                
                for j in active:
                    if j == i:
                        continue
                    if time.time() > time_limit:
                        break
                    
                    ball_j = cw_balls[j]
                    
                    # Check if ball_j covers all unique_words
                    covers_unique = all(w in ball_set_cache[code_ints[j]] for w in unique_words)
                    if not covers_unique:
                        continue
                    
                    # Check if removing i and adding j maintains coverage
                    # Words in ball_i but not in ball_j would lose coverage
                    lost = [w for w in ball_i if w not in ball_set_cache[code_ints[j]] and coverage_arr[w] == 1]
                    if lost:
                        continue
                    
                    # Gain: number of words in ball_j that become newly covered (coverage was 0)
                    # Actually, we're replacing i with j, so:
                    # - Words in ball_i \ ball_j: lose 1 coverage (must be > 1 before)
                    # - Words in ball_j \ ball_i: gain 1 coverage
                    # We need: for all w in ball_i, coverage[w] > 1 OR w in ball_j
                    
                    can_swap = True
                    for w in ball_i:
                        if coverage_arr[w] == 1 and w not in ball_set_cache[code_ints[j]]:
                            can_swap = False
                            break
                    
                    if can_swap:
                        # Count how many words in ball_j have coverage 0 (would be newly covered)
                        # This doesn't matter for validity, but for optimization
                        best_swap = j
                        best_gain = 1
                        break
                
                if best_swap >= 0:
                    # Perform swap
                    for w in ball_i:
                        coverage_arr[w] -= 1
                    for w in cw_balls[best_swap]:
                        coverage_arr[w] += 1
                    active.remove(i)
                    # best_swap is already in active, so we just keep it
                    # Actually, we're replacing i with best_swap, but best_swap is already there
                    # This means we're just removing i if best_swap already covers everything
                    # Let me reconsider...
                    
                    # Actually, the swap should be: remove i, and if best_swap is not already
                    # providing coverage, add it. But since all codewords are already in the list,
                    # we can only remove, not add new ones unless we track a pool.
                    
                    # Let's simplify: just do removal if possible
                    pass
        
        return [code_list[i] for i in active], coverage_arr
    
    def aggressive_removal(code_list, coverage_arr, time_limit):
        """
        More aggressive removal: repeatedly find any removable codeword and remove it.
        Use random order to escape local optima.
        """
        code_ints = [word_to_int(cw) for cw in code_list]
        active = list(range(len(code_ints)))
        cw_balls = [ball_cache[code_ints[i]] for i in range(len(code_ints))]
        
        removed = 0
        random.shuffle(active)
        
        while active and time.time() < time_limit:
            progress = False
            for idx in range(len(active) - 1, -1, -1):
                i = active[idx]
                ball = cw_balls[i]
                
                removable = True
                for w in ball:
                    if coverage_arr[w] <= 1:
                        removable = False
                        break
                
                if removable:
                    for w in ball:
                        coverage_arr[w] -= 1
                    active.pop(idx)
                    removed += 1
                    progress = True
            
            if not progress:
                # Try random shuffle and retry
                random.shuffle(active)
                retry = False
                for idx in range(len(active) - 1, -1, -1):
                    i = active[idx]
                    ball = cw_balls[i]
                    removable = all(coverage_arr[w] > 1 for w in ball)
                    if removable:
                        for w in ball:
                            coverage_arr[w] -= 1
                        active.pop(idx)
                        removed += 1
                        retry = True
                        break
                if not retry:
                    break
        
        new_code = [code_list[i] for i in active]
        return new_code, coverage_arr, removed
    
    # Main optimization
    best_code = [list(c) for c in initial_code]
    best_size = len(best_code)
    
    # Strategy 1: Greedy removal with priority queue
    code = [list(c) for c in initial_code]
    coverage = [0] * size_space
    for cw in code:
        ball = ball_cache[word_to_int(cw)]
        for w in ball:
            coverage[w] += 1
    
    time_limit_1 = deadline - 40.0
    code, coverage, _ = greedy_remove_pq(code, coverage, time_limit_1)
    if len(code) < best_size:
        best_code = code
        best_size = len(code)
    
    # Strategy 2: Aggressive removal with random restarts
    remaining = deadline - 5.0 - time.time()
    if remaining > 15:
        num_restarts = min(8, int(remaining // 8))
        for restart in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
            
            code = [list(c) for c in best_code]
            coverage = [0] * size_space
            for cw in code:
                ball = ball_cache[word_to_int(cw)]
                for w in ball:
                    coverage[w] += 1
            
            tl = min(deadline - 5.0, time.time() + remaining * 0.8)
            code, coverage, _ = aggressive_removal(code, coverage, tl)
            
            if len(code) < best_size:
                best_code = code
                best_size = len(code)
            
            remaining = deadline - 5.0 - time.time()
            if remaining < 5:
                break
    
    # Strategy 3: Final greedy pass
    if time.time() < deadline - 2.0:
        code = [list(c) for c in best_code]
        coverage = [0] * size_space
        for cw in code:
            ball = ball_cache[word_to_int(cw)]
            for w in ball:
                coverage[w] += 1
        
        code, coverage, _ = greedy_remove_pq(code, coverage, deadline - 1.0)
        if len(code) < best_size:
            best_code = code
            best_size = len(code)
    
    return best_code

def _generic_greedy(q, n, radius, deadline):
    """Fallback for non-ternary cases."""
    import time
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
