import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    def is_valid(nums):
        n = len(nums)
        if n == 0:
            return True
        
        dp = [0] * (n + 1)
        dp[0] = 1
        
        for i, x in enumerate(nums):
            for k in range(i + 1, 0, -1):
                dp[k] |= (dp[k - 1] << x)
        
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if dp[i] & dp[j]:
                    return False
        return True

    best_set = []
    current_set = set()
    
    # Initialize
    # Try to find a decent starting point
    for _ in range(100):
        x = random.randint(1, 256)
        if is_valid([x]):
            current_set.add(x)
            break
            
    stagnation_count = 0
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try to add
        candidates = [x for x in range(1, 257) if x not in current_set]
        if candidates:
            random.shuffle(candidates)
            for x in candidates:
                new_set = current_set | {x}
                if is_valid(list(new_set)):
                    current_set = new_set
                    improved = True
                    if len(current_set) > len(best_set):
                        best_set = list(current_set)
                    break
        
        if not improved:
            # 2. Try to swap
            swapped = False
            if current_set:
                to_remove = random.choice(list(current_set))
                temp_set = current_set - {to_remove}
                candidates = [x for x in range(1, 257) if x not in temp_set]
                random.shuffle(candidates)
                for x in candidates:
                    new_set = temp_set | {x}
                    if is_valid(list(new_set)):
                        current_set = new_set
                        swapped = True
                        if len(current_set) > len(best_set):
                            best_set = list(current_set)
                        break
            
            if not swapped:
                stagnation_count += 1
                # 3. Remove or Restart
                if stagnation_count > 50:
                    # Restart from a subset of best_set
                    if best_set:
                        # Keep a random subset of best_set
                        subset_size = max(1, len(best_set) // 2)
                        subset = random.sample(best_set, subset_size)
                        current_set = set(subset)
                    else:
                        current_set = {random.randint(1, 256)}
                    stagnation_count = 0
                elif current_set:
                    to_remove = random.choice(list(current_set))
                    current_set.remove(to_remove)
                else:
                    current_set = {random.randint(1, 256)}
            else:
                stagnation_count = 0
        else:
            stagnation_count = 0

    return best_set
