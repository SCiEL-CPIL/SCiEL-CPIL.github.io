import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort()
    
    # Filter divisors that are <= target
    divisors = [d for d in divisors if d <= target]
    num_divs = len(divisors)
    
    # DP arrays
    # min_count[s] = minimum number of divisors to sum to s
    # mask[s] = bitmask of divisors used
    INF = float('inf')
    min_count = [INF] * (target + 1)
    mask = [0] * (target + 1)
    
    min_count[0] = 0
    mask[0] = 0
    
    # Process each divisor
    for idx, d in enumerate(divisors):
        if time.time() > deadline:
            break
            
        # Update dp in reverse order to avoid using same divisor twice
        # We only need to update s where min_count[s-d] is not INF
        # Optimized loop:
        for s in range(target, d - 1, -1):
            prev_count = min_count[s - d]
            if prev_count != INF:
                new_count = prev_count + 1
                if new_count < min_count[s]:
                    min_count[s] = new_count
                    mask[s] = mask[s - d] | (1 << idx)
    
    # Extract the solution
    if min_count[target] == INF:
        # Fallback: Greedy (largest first)
        remainder = target
        chosen = []
        for d in reversed(divisors):
            if d <= remainder:
                chosen.append(d)
                remainder -= d
        return chosen
    
    # Reconstruct which divisors were used
    result_mask = mask[target]
    result = []
    for idx in range(num_divs):
        if result_mask & (1 << idx):
            result.append(divisors[idx])
            
    return result
