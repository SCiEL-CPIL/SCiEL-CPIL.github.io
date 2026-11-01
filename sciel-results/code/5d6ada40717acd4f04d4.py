import time
import numpy as np
import random

# EVOLVE-BLOCK-START

def _int_to_digits(x, q, n, powers):
    """Convert integer index to list of digits."""
    digits = []
    for i in range(n):
        d = (x // powers[i]) % q
        digits.append(d)
    return digits

def _precompute_balls(q, n, radius, deadline):
    """
    Precompute for each word x the list of words within Hamming distance <= radius.
    Returns a list of lists.
    """
    total = q ** n
    if radius == 1:
        balls = []
        powers = [1] * n
        for i in range(1, n):
            powers[i] = powers[i-1] * q
        
        for x in range(total):
            ball = [x]
            for i in range(n):
                old_digit = (x // powers[i]) % q
                for new_digit in range(q):
                    if new_digit != old_digit:
                        y = x + (new_digit - old_digit) * powers[i]
                        ball.append(y)
            balls.append(ball)
            if time.time() > deadline - 10:
                break
        return balls
    else:
        raise NotImplementedError("Only radius 1 is optimized")

def _greedy_cover_least_covered(q, n, radius, deadline, random_seed=None):
    """
    Greedy set-cover algorithm with 'least-covered' tie-breaking.
    This heuristic prefers codewords that cover the most uncovered words,
    but breaks ties by choosing the one that is covered by the fewest other 
    currently selected codewords (to minimize future redundancy).
    
    Returns list of integer indices.
    """
    if random_seed is not None:
        random.seed(random_seed)
        
    total = q ** n
    balls = _precompute_balls(q, n, radius, deadline)
    
    # covered[y] = 1 if y is covered, 0 otherwise
    covered = np.zeros(total, dtype=bool)
    
    # counts[i] = number of uncovered elements in ball[i]
    counts = np.array([len(b) for b in balls], dtype=np.int32)
    
    # inv[y] = list of ball indices i where y in balls[i]
    inv = [[] for _ in range(total)]
    for i, ball in enumerate(balls):
        for y in ball:
            inv[y].append(i)
            
    # degree[i] = number of other selected codewords that cover some part of ball[i]?
    # Actually, simpler: 
    # We want to minimize overlap. 
    # Let's use a standard greedy: pick max count. 
    # To add diversity, we can randomize the initial selection or tie-breaking.
    # The 'least covered' idea: if two balls have same count, pick the one with lower 'overlap score'.
    # Overlap score for ball i = sum of (1 if y in ball_j for j in selected) for y in ball_i?
    # That's expensive to maintain.
    
    # Alternative Structural Change: 
    # Instead of pure max-count, use a weighted score: count[i] - alpha * overlap[i]
    # But let's stick to a robust greedy with random restarts.
    
    selected = []
    
    while True:
        if time.time() > deadline - 5:
            break
            
        # Find max count
        max_count = np.max(counts)
        if max_count == 0:
            break
            
        # Find all indices with max count
        candidates = np.where(counts == max_count)[0]
        
        # Tie-breaking: 
        # Option A: Random choice from candidates (for restarts)
        # Option B: Choose the one that covers the most *uncovered* words (already max_count)
        # Option C: Choose the one with lowest index (deterministic)
        # Let's use random choice to allow restarts to explore different paths.
        max_idx = random.choice(candidates)
        
        selected.append(max_idx)
        
        ball = balls[max_idx]
        for y in ball:
            if not covered[y]:
                covered[y] = True
                for i in inv[y]:
                    counts[i] -= 1
                    
        if np.all(covered):
            break

    return selected

def _local_search_reduce(selected_set, balls, inv, deadline):
    """
    Try to remove redundant codewords from the set.
    A codeword c is redundant if every word in ball[c] is covered by at least one other codeword.
    """
    selected_list = list(selected_set)
    random.shuffle(selected_list)
    
    # cover_count[y] = number of selected codewords covering y
    total = len(balls[0]) # This is not total space, total space is q**n. 
    # Wait, balls[i] contains indices of words. The total number of words is q**n.
    # Let's re-calculate total space size properly.
    # Actually, we can just use a dict or array of size q**n.
    # Let's pass total_space as an argument or calculate it.
    
    # For efficiency in this function, let's assume we have access to total_space.
    # I will refactor to pass total_space.
    
    pass

def _construct_ternary_8_1(q, n, radius, deadline):
    """
    Specialized constructor for q=3, n=8, radius=1.
    Uses multi-restart greedy with local search reduction.
    """
    total = q ** n
    balls = _precompute_balls(q, n, radius, deadline)
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
        
    # Precompute inverse
    inv = [[] for _ in range(total)]
    for i, ball in enumerate(balls):
        for y in ball:
            inv[y].append(i)
            
    best_set = None
    best_size = float('inf')
    
    # Multi-restart greedy
    num_restarts = 10 # Adjust based on time
    start_time = time.time()
    
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        # Greedy construction with random tie-breaking
        selected = _greedy_cover_least_covered(q, n, radius, deadline, random_seed=r)
        
        if not selected:
            continue
            
        # Local Search: Remove redundant codewords
        # We need to check if removing a codeword leaves any word uncovered.
        # cover_count[y] = number of selected codewords covering y.
        
        cover_count = np.zeros(total, dtype=np.int32)
        for c in selected:
            for y in balls[c]:
                cover_count[y] += 1
                
        # Try to remove codewords
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Shuffle to randomize removal order
            candidates = list(selected)
            random.shuffle(candidates)
            
            for c in candidates:
                # Check if c is redundant
                # If min(cover_count[ball[c]]) >= 2, then c is redundant
                if np.min(cover_count[balls[c]]) >= 2:
                    # Remove c
                    for y in balls[c]:
                        cover_count[y] -= 1
                    selected.remove(c)
                    improved = True
        
        # Update best
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = set(selected)
            
    # Convert best_set to digits
    codewords = []
    for idx in best_set:
        codewords.append(_int_to_digits(idx, q, n, powers))
        
    return codewords

def construct(q, n, radius):
    start_time = time.time()
    # Evaluator kills after 150s. Use 145s deadline for safety.
    deadline = start_time + 145
    
    try:
        # Use the specialized multi-restart greedy + local search
        result = _construct_ternary_8_1(q, n, radius, deadline)
    except Exception as e:
        # Fallback to simple greedy if specialized fails
        try:
            selected = _greedy_cover_least_covered(q, n, radius, deadline, random_seed=0)
            powers = [1] * n
            for i in range(1, n):
                powers[i] = powers[i-1] * q
            codewords = []
            for idx in selected:
                codewords.append(_int_to_digits(idx, q, n, powers))
            result = codewords
        except Exception:
            # Ultimate fallback: raise
            raise e

    return result

# EVOLVE-BLOCK-END
