import time
import random

def construct():
    n = 256
    t0 = time.time()
    deadline = t0 + 55.0

    def thue_morse(n):
        signs = [1] * n
        for i in range(n):
            if bin(i).count('1') % 2 == 1:
                signs[i] = -1
        return signs

    def evaluate_limited(x, limit):
        # Returns the maximum absolute sum on any AP.
        # If it exceeds 'limit', returns a value > limit early.
        max_disc = 0
        # d=1 is the most critical usually.
        for d in range(1, n):
            for s in range(n):
                curr = 0
                min_p = 0
                max_p = 0
                p = s
                while p < n:
                    curr += x[p]
                    if curr < min_p: min_p = curr
                    if curr > max_p: max_p = curr
                    if max_p - min_p > limit:
                        return limit + 1
                    p += d
                disc = max_p - min_p
                if disc > max_disc:
                    max_disc = disc
                    if max_disc > limit:
                        return max_disc
        return max_disc

    best_signs = thue_morse(n)
    best_disc = evaluate_limited(best_signs, 100) # Start with high limit
    
    if best_disc <= 2:
        return best_signs

    rng = random.Random(42)
    
    while time.time() < deadline:
        # Hill Climbing
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                best_signs[i] *= -1
                new_disc = evaluate_limited(best_signs, best_disc - 1)
                if new_disc < best_disc:
                    best_disc = new_disc
                    improved = True
                    if best_disc <= 2:
                        return best_signs
                else:
                    best_signs[i] *= -1
        
        # If stuck, try a random kick
        if time.time() < deadline:
            # Flip a few random bits
            kicks = rng.randint(3, 7)
            for _ in range(kicks):
                idx = rng.randint(0, n - 1)
                best_signs[idx] *= -1
            
            # Re-evaluate
            current_disc = evaluate_limited(best_signs, 100)
            
            # Try to improve from this kicked state
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(n):
                    best_signs[i] *= -1
                    new_disc = evaluate_limited(best_signs, best_disc - 1)
                    if new_disc < best_disc:
                        best_disc = new_disc
                        improved = True
                        if best_disc <= 2:
                            return best_signs
                    else:
                        best_signs[i] *= -1

    return best_signs
