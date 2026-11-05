import random

def construct():
    max_val = 512
    target_size = 48
    max_sum = 48 * 512  # Max possible sum for 48 elements
    
    # dp[k] is a bitset (integer) where bit s is 1 if sum s is achievable with k elements
    # We only care about k up to 48
    
    def is_safe(dp, current_size, candidate):
        # Check if candidate is the mean of any subset of size k >= 2
        # Mean = sum / k => sum = k * candidate
        for k in range(2, current_size + 1):
            target_sum = k * candidate
            if target_sum > max_sum:
                break
            # Check if bit target_sum is set in dp[k]
            if (dp[k] >> target_sum) & 1:
                return False
        return True

    def add_element(dp, x):
        # Update dp to include x
        # Iterate backwards to avoid using x multiple times in the same step
        # We need to know the current max size to limit iteration
        # But dp is a list, so we can just iterate down from len(dp)-1 to 0
        # However, we should only update up to the current max size + 1
        # For simplicity and speed, we iterate down from the highest possible index
        # Since we add at most 48 elements, we can just iterate down from 48 to 0
        for k in range(48, -1, -1):
            # If dp[k] is 0, shifting it does nothing useful, but it's cheap
            # We only update if dp[k] is non-zero
            if dp[k]:
                dp[k+1] |= (dp[k] << x)

    def greedy_greedy(order):
        dp = [0] * 50  # Index 0 to 49
        dp[0] = 1
        selected = []
        current_size = 0
        
        for x in order:
            if is_safe(dp, current_size, x):
                selected.append(x)
                add_element(dp, x)
                current_size += 1
                if current_size >= target_size:
                    break
        return selected

    # Strategy 1: Ascending
    best_set = greedy_greedy(range(1, max_val + 1))
    
    # Strategy 2: Descending
    set_desc = greedy_greedy(range(max_val, 0, -1))
    if len(set_desc) > len(best_set):
        best_set = set_desc

    # Strategy 3: Randomized (a few iterations)
    # Use a fixed seed for reproducibility in evaluation, or random if allowed.
    # The task says "Each evaluation must finish within 60 s".
    # Randomized runs are fast.
    for _ in range(10):
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        set_rand = greedy_greedy(candidates)
        if len(set_rand) > len(best_set):
            best_set = set_rand
            if len(best_set) >= target_size:
                break

    return best_set
