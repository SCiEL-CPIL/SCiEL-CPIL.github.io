import time
import random
import sys

def _is_valid_fast(set_vals):
    """
    Check if no element in set_vals is the mean of any subset of >=2 other elements.
    Uses bitset DP for subset sums.
    """
    m = len(set_vals)
    if m < 3:
        return True
    
    for a in set_vals:
        others = [v for v in set_vals if v != a]
        n_others = len(others)
        
        max_target_sum = n_others * a
        if max_target_sum < 50000:
            mask = (1 << (max_target_sum + 1)) - 1
        else:
            mask = None
            
        dp = [0] * (n_others + 1)
        dp[0] = 1
        
        for v in others:
            for k in range(n_others, 0, -1):
                dp[k] |= (dp[k-1] << v)
                if mask:
                    dp[k] &= mask
        
        for k in range(2, n_others + 1):
            target = k * a
            if target <= max_target_sum:
                if (dp[k] >> target) & 1:
                    return False
            
    return True

def _cantor_base3_numbers(limit):
    """
    Generate numbers <= limit with only 0/1 digits in base 3.
    These are 3-AP-free, so no element is the mean of exactly 2 others.
    """
    results = []
    # Generate all numbers with base-3 digits 0 or 1 up to limit
    # Maximum number of digits: 3^5 = 243, 3^6 = 729 > 512
    # So we need up to 6 digits (positions 0-5)
    for mask in range(1, 1 << 6):  # 2^6 = 64 possibilities, minus 0
        num = 0
        for i in range(6):
            if mask & (1 << i):
                num += (3 ** i)
        if num <= limit and num > 0:
            results.append(num)
    return sorted(results)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 512
    MAX_K = 48
    
    # Step 1: Start with Cantor base-3 0/1 digit numbers (3-AP-free)
    # This gives us a large set where no element is the mean of exactly 2 others
    cantor_set = _cantor_base3_numbers(N)
    
    # cantor_set has at most 2^6 - 1 = 63 numbers, but filtered to <= 512
    # Let's see how many we get: 3^0=1, 3^1=3, 3^2=9, 3^3=27, 3^4=81, 3^5=243
    # All sums of distinct powers of 3 up to 3^5: max is 1+3+9+27+81+243 = 364 <= 512
    # So we get all 2^6 - 1 = 63 numbers? No, we need to check each sum <= 512
    # Actually all 63 non-empty subsets sum to at most 364, so all 63 are valid
    
    # But we only need at most 48. Start with all cantor numbers, then try to add more.
    # First, verify the cantor set is valid (it should be for 2-element means,
    # but we need to check for larger subsets too)
    
    # Actually, let's just start with a subset and grow greedily
    # Use the cantor numbers as a priority list, then try to add others
    
    # Build a candidate list: cantor numbers first, then other numbers
    cantor_set_sorted = sorted(cantor_set)
    other_numbers = [x for x in range(1, N+1) if x not in cantor_set]
    random.shuffle(other_numbers)
    
    # Try multiple strategies with different orderings
    best_set = []
    best_size = 0
    
    strategies = []
    
    # Strategy 1: Cantor numbers first, then others
    strategies.append(cantor_set_sorted + other_numbers)
    
    # Strategy 2: Random shuffle of all
    all_nums = list(range(1, N+1))
    random.shuffle(all_nums)
    strategies.append(all_nums)
    
    # Strategy 3: Descending order
    strategies.append(list(range(N, 0, -1)))
    
    # Strategy 4: Ascending order  
    strategies.append(list(range(1, N+1)))
    
    # Strategy 5: Cantor numbers in random order, then others
    cantor_shuffled = cantor_set_sorted[:]
    random.shuffle(cantor_shuffled)
    strategies.append(cantor_shuffled + other_numbers)
    
    random.shuffle(strategies)
    
    for strategy_idx, ordering in enumerate(strategies):
        if time.time() > deadline - 5:
            break
            
        selected = []
        
        # Greedy: add elements in order if valid
        for x in ordering:
            if time.time() > deadline - 5:
                break
            if len(selected) >= MAX_K:
                break
                
            if _is_valid_fast(selected + [x]):
                selected.append(x)
                
        current_size = len(selected)
        
        if current_size > best_size:
            best_size = current_size
            best_set = selected[:]
            
            if best_size == MAX_K:
                return best_set
                
        # Local search: try to improve by swapping
        if current_size < MAX_K and time.time() < deadline - 10:
            improved = True
            swap_attempts = 0
            while improved and swap_attempts < 200 and time.time() < deadline - 10:
                improved = False
                swap_attempts += 1
                
                # Try to add a random element
                remaining = [x for x in range(1, N+1) if x not in selected]
                if remaining:
                    for _ in range(10):
                        x = random.choice(remaining)
                        if _is_valid_fast(selected + [x]):
                            selected.append(x)
                            improved = True
                            if len(selected) >= MAX_K:
                                break
                        remaining.remove(x) if x in remaining else None
                        if len(selected) >= MAX_K:
                            break
                    
                if len(selected) >= MAX_K:
                    break
                
                # Try a swap
                if not improved and selected and remaining:
                    u_idx = random.randint(0, len(selected)-1)
                    u = selected[u_idx]
                    v = random.choice(remaining)
                    new_selected = selected[:]
                    new_selected[u_idx] = v
                    if _is_valid_fast(new_selected):
                        selected = new_selected
                        improved = True
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
            if best_size == MAX_K:
                return best_set
    
    return best_set[:MAX_K]

if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}")
    print(res[:10])
