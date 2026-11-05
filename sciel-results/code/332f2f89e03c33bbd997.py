import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    n = 5040
    target = 4979
    
    divisors = [d for d in range(1, n+1) if n % d == 0]
    divisors.sort(reverse=True)
    
    # BFS to find minimum number of divisors
    # State: (current_sum, last_index) -> path?
    # To save memory, we just store sums and reconstruct later?
    # Or store paths.
    # Since depth is small (<=5), paths are small.
    
    # Level 0
    current_level = {0: []}
    
    max_depth = 10 # Safety
    
    for depth in range(1, max_depth + 1):
        next_level = {}
        for s, path in current_level.items():
            for i in range(len(divisors)):
                d = divisors[i]
                if s + d <= target:
                    new_s = s + d
                    new_path = path + [d]
                    if new_s == target:
                        return new_path
                    if new_s not in next_level:
                        next_level[new_s] = new_path
        if not next_level:
            break
        current_level = next_level
        if time.time() > deadline:
            break
            
    # Fallback to greedy if BFS fails or times out
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                return chosen
    return chosen
