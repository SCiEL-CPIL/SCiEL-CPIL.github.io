import time
import random

def count_violations_fast(S):
    """
    Fast check: for each element a in S, check if a is the mean of any subset of >=2 other elements.
    Uses the fact that sum(T) = k*a for some k >= 2.
    """
    if len(S) < 3:
        return 0
        
    S_list = sorted(S)
    n = len(S_list)
    violations = 0
    
    for i, a in enumerate(S_list):
        others = [x for j, x in enumerate(S_list) if j != i]
        m = len(others)
        
        # We need to check if there's a subset of others with sum = k*a for some k in [2, m]
        # Use bitset DP but optimized: we only care about specific target sums
        # Max possible sum is sum(others)
        max_sum = sum(others)
        
        # For each k from 2 to m, target = k*a
        # We can do DP once and check all targets
        # dp[k] = bitset of achievable sums with exactly k elements
        
        # Optimization: if k*a > max_sum for all k >= 2, no violation
        # k*a > max_sum means k > max_sum/a
        max_k_to_check = min(m, max_sum // a) if a > 0 else 0
        if max_k_to_check < 2:
            continue
            
        # DP: dp[k] is integer bitmask
        # Start: dp[0] = 1 (sum 0)
        dp = [0] * (m + 1)
        dp[0] = 1
        
        for x in others:
            # Update in reverse to avoid using x multiple times
            for k in range(m - 1, -1, -1):
                if dp[k]:
                    dp[k + 1] |= (dp[k] << x)
        
        # Check for violations
        is_violation = False
        for k in range(2, max_k_to_check + 1):
            target = k * a
            if target > max_sum:
                break
            if (dp[k] >> target) & 1:
                is_violation = True
                break
                
        if is_violation:
            violations += 1
            
    return violations

def construct():
    deadline = time.time() + 55.0
    random.seed(42)
    
    # Phase 1: Greedy construction to get a large valid set
    # Start with a spread-out set to minimize collisions
    all_numbers = set(range(1, 513))
    
    # Use a structured initial set: pick numbers with good spacing
    # Try powers of 2-like structure or random with rejection
    current_set = set()
    
    # Greedy: add numbers that don't create violations
    candidates = list(range(1, 513))
    random.shuffle(candidates)
    
    # First pass: try to build a large set
    for c in candidates:
        test_set = current_set | {c}
        if count_violations_fast(test_set) == 0:
            current_set.add(c)
        if len(current_set) >= 48:
            break
    
    best_set = set(current_set)
    best_size = len(best_set)
    
    # Phase 2: Local search to improve
    # Try swaps and additions to maximize size while maintaining validity
    
    T = 50.0
    cooling = 0.9995
    
    while time.time() < deadline:
        if T < 0.1:
            T = 50.0  # Reheat
        
        # Generate neighbor
        move_type = random.random()
        new_set = set(current_set)
        
        if move_type < 0.6:  # Swap
            if not current_set:
                continue
            x = random.choice(list(current_set))
            available = list(all_numbers - current_set)
            if not available:
                continue
            y = random.choice(available)
            new_set.remove(x)
            new_set.add(y)
            
        elif move_type < 0.8:  # Add
            available = list(all_numbers - current_set)
            if not available:
                continue
            y = random.choice(available)
            new_set.add(y)
            
        else:  # Remove
            if not current_set:
                continue
            x = random.choice(list(current_set))
            new_set.remove(x)
        
        new_violations = count_violations_fast(new_set)
        new_size = len(new_set)
        
        # Cost: prioritize no violations, then maximize size
        old_violations = count_violations_fast(current_set)
        old_size = len(current_set)
        
        # Accept if: fewer violations, or same violations and larger size, or probabilistic
        delta_viol = new_violations - old_violations
        delta_size = new_size - old_size
        
        if delta_viol < 0 or (delta_viol == 0 and delta_size >= 0) or \
           (random.random() < pow(2.718281828, -(delta_viol * 100 - delta_size) / T)):
            current_set = new_set
            
            if new_violations == 0 and new_size > best_size:
                best_size = new_size
                best_set = set(new_set)
                
        T *= cooling
        
    # Return best valid set, or current if no valid found
    if count_violations_fast(best_set) == 0:
        return list(best_set)
    elif count_violations_fast(current_set) == 0:
        return list(current_set)
    else:
        # Return largest valid subset
        S_list = sorted(current_set)
        valid_set = set()
        for x in S_list:
            test = valid_set | {x}
            if count_violations_fast(test) == 0:
                valid_set.add(x)
        return list(valid_set) if valid_set else [1, 2, 3]
