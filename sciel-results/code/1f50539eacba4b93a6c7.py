"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

MAX_N = 10_000_000
TIME_LIMIT = 80

def _compute_sigma(n_max):
    sigma = [0] * (n_max + 1)
    for d in range(1, n_max + 1):
        for m in range(d, n_max + 1, d):
            sigma[m] += d
    return sigma

def construct():
    t_start = time.time()
    sigma = _compute_sigma(MAX_N)
    
    best_error = float('inf')
    best_pair = None
    
    # Iterate over d = 2a - 3b, starting from small |d|
    # For each d, b must satisfy: 3b + d even, 1 <= (3b+d)/2 <= MAX_N
    # Error = |d| / (2b), so for fixed d, larger b gives smaller error
    
    # Let's try d in increasing order of |d|
    # For d = 0: a = 3b/2, requires b even. Error = 0.
    # Check if any b with sigma[3b/2] == sigma[b]
    
    for abs_d in range(0, 100000):
        if time.time() - t_start > TIME_LIMIT:
            break
        
        for d_sign in ([1, -1] if abs_d > 0 else [0]):
            d = abs_d * d_sign
            
            # b must have same parity as d
            # b_min: (3b + d) >= 2 => b >= (2 - d)/3
            # b_max: (3b + d) <= 2*MAX_N => b <= (2*MAX_N - d)/3
            # Also b >= 1, b <= MAX_N
            
            b_min = max(1, (2 - d + 2) // 3)  # ceil((2-d)/3)
            if (2 - d) % 3 != 0:
                b_min = max(1, (2 - d) // 3 + 1)
            else:
                b_min = max(1, (2 - d) // 3)
            
            b_max = min(MAX_N, (2 * MAX_N - d) // 3)
            if (2 * MAX_N - d) % 3 != 0:
                b_max = min(MAX_N, (2 * MAX_N - d) // 3)
            else:
                b_max = min(MAX_N, (2 * MAX_N - d) // 3)
            
            if b_min > b_max:
                continue
            
            # b must have same parity as d
            parity = d % 2
            if b_min % 2 != parity:
                b_min += 1
            
            # For fixed d, error = |d| / (2b), so we want largest b first
            # Iterate b from b_max down to b_min
            for b in range(b_max, b_min - 1, -2):
                if time.time() - t_start > TIME_LIMIT:
                    break
                a = (3 * b + d) // 2
                if a < 1 or a > MAX_N:
                    continue
                if sigma[a] == sigma[b]:
                    err = abs_d / (2 * b)
                    if err < best_error:
                        best_error = err
                        best_pair = (a, b)
                        if err == 0:
                            return [a, b]
            
            if time.time() - t_start > TIME_LIMIT:
                break
        
        if time.time() - t_start > TIME_LIMIT:
            break
    
    if best_pair is None:
        return [1, 1]
    
    return [best_pair[0], best_pair[1]]

# EVOLVE-BLOCK-END
