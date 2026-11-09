import time

MAX_N = 10_000_000
TIME_LIMIT = 85.0

def construct():
    t_start = time.time()
    
    # Compute sigma(n) for n = 1..MAX_N
    sigma = [0] * (MAX_N + 1)
    for d in range(1, MAX_N + 1):
        for m in range(d, MAX_N + 1, d):
            sigma[m] += d
    
    # Search for best pair
    best_a = 1
    best_b = 1
    best_err = 0.5  # |1/1 - 3/2|
    
    for b in range(MAX_N, 0, -1):
        if time.time() > t_start + TIME_LIMIT - 3.0:
            break
        
        sb = sigma[b]
        
        # For this b, check k=1,2,... until k/(2b) >= best_err
        max_k = int(2 * b * best_err)
        if max_k < 1:
            max_k = 1
        # Cap max_k to avoid excessive work for large b
        if max_k > 100:
            max_k = 100
        
        found_for_b = False
        for k in range(1, max_k + 1):
            # Check a = (3b - k) / 2
            num = 3 * b - k
            if num > 0 and num % 2 == 0:
                a = num // 2
                if 1 <= a <= MAX_N and sigma[a] == sb:
                    err = k / (2 * b)
                    if err < best_err:
                        best_err = err
                        best_a = a
                        best_b = b
                    found_for_b = True
                    break
            
            # Check a = (3b + k) / 2
            num = 3 * b + k
            if num % 2 == 0:
                a = num // 2
                if 1 <= a <= MAX_N and sigma[a] == sb:
                    err = k / (2 * b)
                    if err < best_err:
                        best_err = err
                        best_a = a
                        best_b = b
                    found_for_b = True
                    break
        
        # If even k=1 doesn't beat best_err, and b is decreasing, 
        # for all smaller b, k=1 gives err = 1/(2b) which is larger
        if 1.0 / (2.0 * b) >= best_err:
            break
    
    return [int(best_a), int(best_b)]
