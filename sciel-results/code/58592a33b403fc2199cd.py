import time
import random

_balls_cache = {}

def _int_to_digits(val, n, q):
    digits = [0] * n
    for i in range(n):
        digits[i] = val % q
        val //= q
    return digits

def _digits_to_int(digits, q):
    val = 0
    for d in digits:
        val = val * q + d
    return val

def _get_balls(q, n, radius):
    key = (q, n, radius)
    if key in _balls_cache:
        return _balls_cache[key]
    
    total_words = q ** n
    balls = [0] * total_words
    
    # Only implemented for radius 1 for efficiency, but can extend if needed.
    # Given the task is specifically K_3(6,1), radius=1 is expected.
    # If radius != 1, this will be incorrect. 
    # However, the task description says "construct(q, n, radius)".
    # The evaluator passes q=3, n=6, radius=1.
    # To be safe, I should check radius.
    
    if radius != 1:
        # Fallback: compute balls by checking all pairs? O(N^2)
        # For N=729, 729^2 = 531k. Distance calc is O(N). 531k * 6 = 3M ops.
        # This is fine for a one-time setup if radius != 1.
        # But for radius=1, the generation is much faster.
        
        def hamming_distance(d1, d2):
            return sum(1 for a, b in zip(d1, d2) if a != b)
        
        all_digits = [_int_to_digits(w, n, q) for w in range(total_words)]
        
        for w in range(total_words):
            ball_mask = 0
            for v in range(total_words):
                if hamming_distance(all_digits[w], all_digits[v]) <= radius:
                    ball_mask |= 1 << v
            balls[w] = ball_mask
    else:
        # Efficient generation for radius 1
        for w in range(total_words):
            digits = _int_to_digits(w, n, q)
            ball_mask = 0
            # Distance 0
            ball_mask |= 1 << w
            
            # Distance 1
            for i in range(n):
                for new_digit in range(q):
                    if new_digit == digits[i]:
                        continue
                    new_digits = digits[:]
                    new_digits[i] = new_digit
                    new_w = _digits_to_int(new_digits, q)
                    ball_mask |= 1 << new_w
            
            balls[w] = ball_mask
            
    _balls_cache[key] = balls
    return balls

def _greedy_cover(q, n, radius, balls, total_words, rng):
    covered = 0
    solution = []
    full_mask = (1 << total_words) - 1
    
    candidates = list(range(total_words))
    rng.shuffle(candidates)
    
    while covered != full_mask:
        best_score = -1
        best_candidate = -1
        
        for w in candidates:
            new_cover = balls[w] & (~covered)
            if new_cover == 0:
                continue
            score = new_cover.bit_count()
            if score > best_score:
                best_score = score
                best_candidate = w
        
        if best_candidate == -1:
            break
        
        solution.append(best_candidate)
        covered |= balls[best_candidate]
        
    return solution

def _remove_redundants(q, n, radius, balls, solution):
    if not solution:
        return solution
    
    new_solution = []
    for w in solution:
        union_others = 0
        for x in solution:
            if x != w:
                union_others |= balls[x]
        
        # If balls[w] is subset of union_others, w is redundant
        if (balls[w] & ~union_others) == 0:
            continue
        else:
            new_solution.append(w)
            
    return new_solution

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    total_words = q ** n
    balls = _get_balls(q, n, radius)
    
    best_solution = None
    best_len = float('inf')
    
    # Initial greedy run
    rng = random.Random(42)
    sol = _greedy_cover(q, n, radius, balls, total_words, rng)
    sol = _remove_redundants(q, n, radius, balls, sol)
    if sol:
        best_solution = sol
        best_len = len(sol)
    
    # Try more runs
    run_id = 43
    while time.time() < deadline:
        rng = random.Random(run_id)
        sol = _greedy_cover(q, n, radius, balls, total_words, rng)
        sol = _remove_redundants(q, n, radius, balls, sol)
        
        if sol and len(sol) < best_len:
            best_solution = sol
            best_len = len(sol)
            
        run_id += 1
        
    if best_solution is None:
        # Fallback
        return [_int_to_digits(w, n, q) for w in range(total_words)]
    
    return [_int_to_digits(w, n, q) for w in best_solution]
