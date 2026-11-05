import time
import random
import math

def construct():
    deadline = time.time() + 55.0
    max_val = 512
    candidates = list(range(1, max_val + 1))
    
    def is_valid(S):
        if len(S) < 2:
            return True
        
        S_list = list(S)
        n = len(S_list)
        
        for i in range(n):
            x = S_list[i]
            # Create list of others
            # Optimization: We can iterate through S_list and skip x
            # But creating a list is O(N).
            others = []
            for j in range(n):
                if j != i:
                    others.append(S_list[j])
            
            if len(others) < 2:
                continue
                
            max_sum = sum(others)
            # Mask to limit bitset size
            mask = (1 << (max_sum + 1)) - 1
            
            # dp[k] is bitset of sums achievable with k elements
            dp = [0] * (len(others) + 1)
            dp[0] = 1
            current_max_k = 0
            
            for y in others:
                # Update dp from high k to low k
                # We only need to go up to current_max_k + 1
                for k in range(current_max_k + 1, 0, -1):
                    dp[k] |= (dp[k-1] << y) & mask
                current_max_k += 1
            
            # Check if any k >= 2 has sum k*x
            for k in range(2, len(others) + 1):
                target_sum = k * x
                if target_sum > max_sum:
                    break
                if (dp[k] >> target_sum) & 1:
                    return False
                    
        return True

    # 1. Initial Greedy Construction
    # Sort by value? Or random?
    # Let's try sorted ascending first, then descending, then random restarts for init
    best_set = []
    
    # Try a few initializations
    init_strategies = [
        lambda: list(range(1, max_val + 1)),
        lambda: list(range(max_val, 0, -1)),
        lambda: random.sample(candidates, max_val)
    ]
    
    for _ in range(3):
        random.shuffle(candidates)
        current_set = []
        for c in candidates:
            if is_valid(current_set + [c]):
                current_set.append(c)
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            
    best_size = len(best_set)
    
    # 2. Simulated Annealing
    current_state = set(best_set)
    T = 1.0
    cooling_rate = 0.9995
    min_T = 0.001
    last_improvement = time.time()
    no_improve_limit = 5.0 # Restart if no improvement for 5s
    
    while time.time() < deadline:
        # Pick a random element to flip
        idx = random.randint(1, max_val)
        
        if idx in current_state:
            new_state = current_state - {idx}
        else:
            new_state = current_state | {idx}
        
        # Check validity
        # Optimization: If size increases, we MUST check.
        # If size decreases, we might skip check if we are sure? No, removal can't create new violations,
        # but it can leave the set valid. So removal is always valid if the previous set was valid?
        # Wait. If S is valid, and we remove x, is S\{x} valid?
        # Yes. Removing elements cannot create a new subset that forms a mean.
        # So if we are removing an element, the new state is ALWAYS valid.
        # This is a huge optimization!
        
        if idx in current_state:
            # Removal: Always valid
            new_state = current_state - {idx}
            is_valid_state = True
        else:
            # Addition: Must check
            new_state = current_state | {idx}
            is_valid_state = is_valid(list(new_state))
            
        if is_valid_state:
            new_size = len(new_state)
            old_size = len(current_state)
            delta = new_size - old_size
            
            if delta > 0:
                current_state = new_state
                if new_size > best_size:
                    best_set = list(new_state)
                    best_size = new_size
                    last_improvement = time.time()
            elif delta == 0:
                # Swap
                # Accept with some probability to explore
                if random.random() < 0.5:
                    current_state = new_state
            else: # delta < 0 (Removal)
                # Accept with probability exp(delta / T)
                # delta is -1
                prob = math.exp(-1.0 / T)
                if random.random() < prob:
                    current_state = new_state
        else:
            # Invalid state, reject
            pass
        
        T *= cooling_rate
        
        # Restart condition
        if time.time() - last_improvement > no_improve_limit:
            # Reset to best_set and reheat
            current_state = set(best_set)
            T = 1.0
            last_improvement = time.time()

    return best_set
