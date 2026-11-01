import numpy as np
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0
    
    total_words = q ** n
    
    if total_words == 0:
        return []
        
    # Helper to convert int to base-q digits (MSB first)
    def int_to_digits(num, base, length):
        digits = [0] * length
        for i in range(length):
            digits[length - 1 - i] = num % base
            num //= base
        return digits

    # Helper to convert base-q digits to int
    def digits_to_int(digits, base):
        num = 0
        for d in digits:
            num = num * base + d
        return num

    # Precompute balls for radius 1
    if radius == 1:
        ball_size = 1 + n * (q - 1)
        all_balls = np.empty((total_words, ball_size), dtype=np.int32)
        
        for w in range(total_words):
            digits = int_to_digits(w, q, n)
            ball = [w]
            for i in range(n):
                orig = digits[i]
                for val in range(q):
                    if val == orig:
                        continue
                    digits[i] = val
                    new_w = digits_to_int(digits, q)
                    ball.append(new_w)
                digits[i] = orig
            all_balls[w] = ball
    else:
        # Fallback for other radii: generate on the fly? 
        # For now, assume radius 1 as per task description.
        # If radius > 1, we can't use the precomputed matrix easily without more code.
        # We'll stick to radius 1 optimization.
        pass

    # Greedy Set Cover
    uncovered = np.ones(total_words, dtype=bool)
    code = []
    
    while uncovered.any():
        if time.time() > deadline:
            break
            
        if radius == 1:
            gains = np.sum(uncovered[all_balls], axis=1)
        else:
            # Slow path for other radii
            gains = np.zeros(total_words, dtype=np.int32)
            for w in range(total_words):
                # Generate ball for w
                # This is too slow for general case, but task is radius 1.
                pass 

        best = np.argmax(gains)
        
        if gains[best] == 0:
            break
            
        code.append(best)
        if radius == 1:
            uncovered[all_balls[best]] = False
        else:
            pass

    # Reduction Phase
    if radius == 1:
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in code:
            cover_count[all_balls[c]] += 1
        
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            in_code = [True] * len(code)
            
            for i in range(len(code)):
                if not in_code[i]:
                    continue
                c = code[i]
                min_cov = np.min(cover_count[all_balls[c]])
                if min_cov > 1:
                    cover_count[all_balls[c]] -= 1
                    in_code[i] = False
                    changed = True
            
            if changed:
                code = [c for c, keep in zip(code, in_code) if keep]
                
    # Convert to list of lists
    result = []
    for c in code:
        result.append(int_to_digits(c, q, n))
        
    return result
