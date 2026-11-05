import time

def construct():
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    best_start = 1
    best_len = 0
    
    start_time = time.time()
    deadline = start_time + 50.0
    
    L = 1
    for R in range(1, N + 1):
        if time.time() > deadline:
            break
        
        inv_R = inv[R]
        if inv_R >= L and inv_R <= R:
            L = inv_R + 1
        
        current_len = R - L + 1
        if current_len > best_len:
            best_len = current_len
            best_start = L
    
    # The two-pointer algorithm is O(N) and runs in microseconds for N=1008.
    # Verify the result is valid, and if not, fall back to a brute-force search.
    # This ensures correctness in case the two-pointer logic has an edge case.
    
    def is_valid_interval(start, length):
        if start < 1 or start + length - 1 > N:
            return False
        for x in range(start, start + length):
            for y in range(start, start + length):
                if (x * y) % MOD == 1:
                    return False
        return True
    
    if not is_valid_interval(best_start, best_len):
        # Brute force fallback: find the longest valid interval
        best_start = 1
        best_len = 0
        for s in range(1, N + 1):
            if time.time() > deadline:
                break
            for l in range(1, N - s + 2):
                if time.time() > deadline:
                    break
                if is_valid_interval(s, l):
                    if l > best_len:
                        best_len = l
                        best_start = s
                else:
                    break
    
    return [best_start, best_len]
