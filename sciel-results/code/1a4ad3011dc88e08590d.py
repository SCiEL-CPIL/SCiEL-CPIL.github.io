import random
import time

def construct():
    def is_valid(S):
        n = len(S)
        if n < 2:
            return True
        for i, a in enumerate(S):
            # Construct others list efficiently
            others = S[:i] + S[i+1:]
            m = len(others)
            if m < 2:
                continue
            
            dp = [0] * (m + 1)
            dp[0] = 1
            for y in others:
                for k in range(m, 0, -1):
                    dp[k] |= (dp[k-1] << y)
            
            max_sum = sum(others)
            for k in range(2, m + 1):
                target = k * a
                if target > max_sum:
                    break
                if (dp[k] >> target) & 1:
                    return False
        return True

    target_size = 48
    start_time = time.time()
    
    # Phase 1: Random Sampling
    # Try to find a valid set of size 48 by random sampling
    # This is a different approach family (Random Search vs Greedy Construction)
    while time.time() - start_time < 40:
        S = random.sample(range(1, 513), target_size)
        if is_valid(S):
            return S

    # Phase 2: Fallback to Greedy if Random Sampling fails
    # This ensures we return *something* valid, even if not size 48
    best_set = []
    remaining_time = 55 - (time.time() - start_time)
    
    # Try a few greedy restarts
    max_restarts = max(1, int(remaining_time / 0.1)) # Adjust based on speed
    for _ in range(max_restarts):
        if time.time() - start_time > 55:
            break
        order = random.sample(range(1, 513), 512)
        selected = []
        for x in order:
            candidate = selected + [x]
            if is_valid(candidate):
                selected = candidate
        if len(selected) > len(best_set):
            best_set = selected
            if len(best_set) == 48:
                return best_set

    # If still empty or small, try a deterministic large-number greedy
    if len(best_set) < 48:
        order = list(range(512, 0, -1))
        selected = []
        for x in order:
            candidate = selected + [x]
            if is_valid(candidate):
                selected = candidate
        if len(selected) > len(best_set):
            best_set = selected

    return best_set
