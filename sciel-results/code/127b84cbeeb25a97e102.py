import time
import random

def construct():
    n = 128
    start_time = time.time()
    timeout = 85.0

    def is_valid_assignment(colors, i):
        for d in range(1, (i // 3) + 1):
            idx0 = i - 3 * d
            idx1 = i - 2 * d
            idx2 = i - 1 * d
            idx3 = i
            if len({colors[idx0], colors[idx1], colors[idx2], colors[idx3]}) < 3:
                return False
        return True

    def backtrack(k, current_index, colors):
        if current_index == n:
            return True
        
        # Random order
        perm = list(range(k))
        random.shuffle(perm)
        
        for c in perm:
            colors[current_index] = c
            if is_valid_assignment(colors, current_index):
                if backtrack(k, current_index + 1, colors):
                    return True
            colors[current_index] = -1
        
        return False

    for k in range(3, 7):
        k_start = time.time()
        k_limit = 15.0
        
        found = False
        while not found and (time.time() - k_start) < k_limit and (time.time() - start_time) < timeout:
            colors = [-1] * n
            if backtrack(k, 0, colors):
                # Verify
                valid = True
                for i in range(3, n):
                    if not is_valid_assignment(colors, i):
                        valid = False
                        break
                if valid:
                    found = True
                    return colors
        
    # Fallback
    return [0] * n
