"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort(reverse=True)  # Sort descending for greedy-like search
    
    # Filter divisors <= target (can't use any divisor > target)
    divisors = [d for d in divisors if d <= target]
    num_divs = len(divisors)
    
    # Precompute suffix sums for efficient pruning
    suffix_sums = [0] * (num_divs + 1)
    for i in range(num_divs - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + divisors[i]
    
    # Try increasing counts from 1 upward
    for count in range(1, num_divs + 1):
        if time.time() > deadline:
            break
        
        found = None
        
        # Use recursive search with pruning for efficiency
        def search(idx, remaining, selected, depth):
            if time.time() > deadline:
                return None
            if remaining == 0:
                return selected[:]
            if depth == count:
                return None
            if idx >= num_divs:
                return None
            if remaining < 0:
                return None
            
            # Pruning: not enough divisors left
            if num_divs - idx < count - depth:
                return None
            
            # Pruning: maximum possible sum from remaining divisors is too small
            max_possible = suffix_sums[idx]
            if max_possible < remaining:
                return None
            
            # Try including divisors[idx]
            if divisors[idx] <= remaining:
                selected.append(divisors[idx])
                result = search(idx + 1, remaining - divisors[idx], selected, depth + 1)
                if result is not None:
                    return result
                selected.pop()
            
            # Try excluding divisors[idx]
            result = search(idx + 1, remaining, selected, depth + 1)
            if result is not None:
                return result
            
            return None
        
        found = search(0, target, [], 0)
        if found is not None:
            return found
    
    # Fallback: greedy solution (should always work)
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    assert remainder == 0, f"Greedy failed: remainder={remainder}"
    return chosen

# EVOLVE-BLOCK-END
