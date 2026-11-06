"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def construct():
    n = 5040
    target = 4979
    
    # Generate divisors of 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Filter to only divisors <= target
    divisors = [d for d in divisors if d <= target]
    divisors.sort()
    num_divs = len(divisors)
    
    # Precompute bitmask for each divisor
    # We'll use a BFS approach with state (remaining_sum, used_mask)
    # But 60 divisors means 2^60 states which is too large.
    # However, we expect the solution to use few divisors (4-6 terms).
    # Let's use BFS with pruning.
    
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 90s limit
    
    # BFS: state = (remaining_sum, used_mask)
    # We want to minimize the number of divisors used.
    # BFS naturally finds the shortest path (fewest divisors) first.
    
    visited = set()
    queue = deque()
    
    # Initial state: remaining = target, no divisors used
    initial_state = (target, 0)
    queue.append(initial_state)
    visited.add(initial_state)
    
    # Store parent pointers for path reconstruction
    # parent[state] = (prev_state, divisor_index)
    parent = {initial_state: None}
    
    while queue:
        if time.time() > deadline:
            break
            
        remaining, mask = queue.popleft()
        
        if remaining == 0:
            # Reconstruct path
            path = []
            curr = (remaining, mask)
            while parent[curr] is not None:
                prev_state, idx = parent[curr]
                path.append(divisors[idx])
                curr = prev_state
            return path
            
        # Try adding each unused divisor
        for i in range(num_divs):
            if mask & (1 << i):
                continue  # Already used this divisor
            
            d = divisors[i]
            if d > remaining:
                continue  # Can't use if divisor exceeds remaining
            
            new_remaining = remaining - d
            new_mask = mask | (1 << i)
            new_state = (new_remaining, new_mask)
            
            if new_state not in visited:
                visited.add(new_state)
                parent[new_state] = ((remaining, mask), i)
                queue.append(new_state)
                
    # Fallback: try a greedy approach if BFS didn't find solution in time
    # Sort divisors in descending order and greedily pick
    remainder = target
    chosen = []
    for d in sorted(divisors, reverse=True):
        if d <= remainder:
            chosen.append(d)
            remainder -= d
        if remainder == 0:
            break
            
    if remainder == 0:
        return chosen
    else:
        # Last resort: return empty (should not happen for this instance)
        return []

# EVOLVE-BLOCK-END
