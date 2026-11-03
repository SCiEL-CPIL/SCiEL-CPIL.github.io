# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct(q, n, radius):
    """
    Optimized greedy set-cover algorithm with multiple strategies and random restarts.
    """
    if n <= 0:
        return [[]]
    
    total_words = q ** n
    
    if total_words > 10000:
        return _construct_greedy_simple(q, n, radius)
        
    return _construct_greedy_optimized(q, n, radius)

def _construct_greedy_optimized(q, n, radius):
    """
    Optimized greedy set-cover with precomputed balls, multiple scoring strategies,
    and efficient restarts.
    """
    total_words = q ** n
    
    # Precompute powers of q
    q_powers = [1] * n
    for i in range(1, n):
        q_powers[i] = q_powers[i-1] * q
        
    # Precompute ball masks
    ball_masks = [0] * total_words
    
    if radius == 1:
        for i in range(total_words):
            mask = 1 << i
            for p in range(n):
                step = q_powers[p]
                digit = (i // step) % q
                for v in range(q):
                    if v == digit:
                        continue
                    j = i + (v - digit) * step
                    mask |= 1 << j
            ball_masks[i] = mask
    else:
        # Handle radius > 1 with brute force neighbor generation
        for i in range(total_words):
            mask = 1 << i
            center_digits = [(i // q_powers[p]) % q for p in range(n)]
            # Generate all words within Hamming distance 'radius'
            for delta in range(1, radius + 1):
                # Choose 'delta' positions to change
                for pos_combo in _combinations(range(n), delta):
                    # For each combination, try all possible digit changes
                    for digits_combo in _product(range(q), repeat=delta):
                        new_idx = i
                        for pos, new_digit in zip(pos_combo, digits_combo):
                            step = q_powers[pos]
                            old_digit = (i // step) % q
                            if new_digit != old_digit:
                                new_idx += (new_digit - old_digit) * step
                        mask |= 1 << new_idx
            ball_masks[i] = mask

    all_mask = (1 << total_words) - 1
    best_codewords = None
    best_len = float('inf')
    
    start_time = time.time()
    max_time = 115.0  # Leave 5s for overhead
    
    # Multiple strategies for tie-breaking and selection
    strategies = []
    
    # Strategy 1: Random permutation
    rng1 = np.random.RandomState(42)
    strategies.append(('random', rng1))
    
    # Strategy 2: Multiple random seeds
    for seed in [123, 456, 789, 321, 654]:
        rng = np.random.RandomState(seed)
        strategies.append(('random', rng))
    
    # Strategy 3: Weighted by current coverage (more aggressive)
    strategies.append(('weighted', None))
    
    restart_count = 0
    
    while True:
        strategy_name, rng = strategies[restart_count % len(strategies)]
        
        if strategy_name == 'random':
            perm = rng.permutation(total_words)
            codewords = _greedy_select(ball_masks, perm, all_mask, total_words, n, q, q_powers)
        else:
            # Weighted strategy: prioritize codewords that cover more uncovered words
            codewords = _greedy_select_weighted(ball_masks, all_mask, total_words, n, q, q_powers)
        
        if len(codewords) < best_len:
            best_len = len(codewords)
            best_codewords = codewords
            
            # Early termination if we hit known optimal or near-optimal
            if best_len <= 73:
                break
                
        restart_count += 1
        
        # Check time
        if time.time() - start_time > max_time:
            break
            
        # Safety cap on restarts
        if restart_count > 20000:
            break
            
    return best_codewords

def _greedy_select(ball_masks, perm, all_mask, total_words, n, q, q_powers):
    """Standard greedy selection with given permutation order."""
    uncovered = all_mask
    codewords = []
    
    while uncovered != 0:
        max_score = 0
        best_idx = -1
        
        for idx in perm:
            score = (ball_masks[idx] & uncovered).bit_count()
            if score > max_score:
                max_score = score
                best_idx = idx
                
        if max_score == 0:
            break
            
        # Convert best_idx to word
        temp = best_idx
        cw = [0] * n
        for k in range(n-1, -1, -1):
            cw[k] = temp % q
            temp //= q
        codewords.append(cw)
        
        uncovered &= ~ball_masks[best_idx]
        
    return codewords

def _greedy_select_weighted(ball_masks, all_mask, total_words, n, q, q_powers):
    """Weighted greedy that considers multiple candidates more carefully."""
    uncovered = all_mask
    codewords = []
    
    while uncovered != 0:
        max_score = 0
        best_candidates = []
        
        for idx in range(total_words):
            score = (ball_masks[idx] & uncovered).bit_count()
            if score > max_score:
                max_score = score
                best_candidates = [idx]
            elif score == max_score and score > 0:
                best_candidates.append(idx)
                
        if max_score == 0:
            break
            
        # Pick randomly from best candidates
        best_idx = random.choice(best_candidates)
        
        # Convert best_idx to word
        temp = best_idx
        cw = [0] * n
        for k in range(n-1, -1, -1):
            cw[k] = temp % q
            temp //= q
        codewords.append(cw)
        
        uncovered &= ~ball_masks[best_idx]
        
    return codewords

def _combinations(iterable, r):
    """Generate combinations of iterable taken r at a time."""
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i:] = range(indices[i]+1, indices[i]+1+r-i)
        yield tuple(pool[i] for i in indices)

def _product(*args, repeat=1):
    """Generate Cartesian product (simplified for our use case)."""
    pools = [tuple(pool) for pool in args]
    repeat = int(repeat)
    result = [[]]
    for pool in pools:
        result = [x + [y] for x in result for y in pool]
    for _ in range(repeat - 1):
        result = [x + [y] for x in result for y in pools[0]]
    return result

def _construct_greedy_simple(q, n, radius):
    """Fallback simple greedy for larger spaces."""
    total_words = q ** n
    
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word
    
    ball_masks = [0] * total_words
    for idx in range(total_words):
        mask = 1 << idx
        center_digits = [(idx // (q ** p)) % q for p in range(n)]
        if radius == 1:
            for pos in range(n):
                for val in range(q):
                    if val != center_digits[pos]:
                        step = q ** pos
                        nidx = idx + (val - center_digits[pos]) * step
                        mask |= 1 << nidx
        ball_masks[idx] = mask
        
    uncovered = (1 << total_words) - 1
    codewords = []
    while uncovered != 0:
        best_idx = -1
        best_count = -1
        for idx in range(total_words):
            covered = ball_masks[idx] & uncovered
            count = covered.bit_count()
            if count > best_count:
                best_count = count
                best_idx = idx
        if best_idx == -1:
            break
        codewords.append(idx_to_word(best_idx))
        uncovered &= ~ball_masks[best_idx]
    return codewords
# EVOLVE-BLOCK-END
