import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    N = 32
    RANGE = 512
    MAX_D = 255  # max d such that 2*d < 512
    
    def score_bitset(mask):
        """Calculate the number of distinct d > 0 with a 3-AP in the set represented by mask."""
        count = 0
        # Iterate over all possible differences
        # d must be such that a, a+d, a+2d are in 1..512
        # max a = 512 - 2d
        # min a = 1
        # So 1 <= 512 - 2d => 2d <= 511 => d <= 255
        for d in range(1, MAX_D + 1):
            # Check if there exists a such that a, a+d, a+2d are in mask
            # This is equivalent to (mask & (mask >> d) & (mask >> 2*d)) != 0
            if (mask & (mask >> d) & (mask >> 2*d)) != 0:
                count += 1
        return count

    def random_init():
        """Generate a random set of N distinct integers in 1..RANGE."""
        s = random.sample(range(1, RANGE + 1), N)
        mask = 0
        for x in s:
            mask |= (1 << x)
        return mask, s

    def simulated_annealing(init_mask, init_set, time_limit):
        """Run Simulated Annealing starting from init_mask."""
        mask = init_mask
        s_set = set(init_set)
        current_score = score_bitset(mask)
        best_mask = mask
        best_score = current_score
        best_set = list(init_set)
        
        # SA Parameters
        T = 10.0
        cooling = 0.999
        min_T = 0.1
        
        # Pre-compute list of all values for random selection
        all_vals = list(range(1, RANGE + 1))
        
        while time.time() < time_limit and T > min_T:
            # Choose move type
            # 70% 1-swap, 30% 2-swap
            if random.random() < 0.7:
                # 1-Swap
                if not s_set:
                    break
                x = random.choice(list(s_set))
                # Pick y not in s_set
                # To be efficient, we can pick a random y and check, or maintain a list of available
                # Since N=32 and RANGE=512, picking random y and checking is fast
                while True:
                    y = random.choice(all_vals)
                    if y not in s_set:
                        break
                
                new_mask = mask ^ (1 << x) ^ (1 << y)
                new_score = score_bitset(new_mask)
                delta = new_score - current_score
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    mask = new_mask
                    s_set.remove(x)
                    s_set.add(y)
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = mask
                        best_set = list(s_set)
            else:
                # 2-Swap
                if len(s_set) < 2:
                    break
                x1, x2 = random.sample(list(s_set), 2)
                while True:
                    y1 = random.choice(all_vals)
                    if y1 not in s_set and y1 != x2:
                        break
                while True:
                    y2 = random.choice(all_vals)
                    if y2 not in s_set and y2 != x1 and y2 != y1:
                        break
                
                new_mask = mask ^ (1 << x1) ^ (1 << x2) ^ (1 << y1) ^ (1 << y2)
                new_score = score_bitset(new_mask)
                delta = new_score - current_score
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    mask = new_mask
                    s_set.remove(x1)
                    s_set.remove(x2)
                    s_set.add(y1)
                    s_set.add(y2)
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = mask
                        best_set = list(s_set)
            
            T *= cooling
        
        return best_mask, best_score, best_set

    best_overall_score = 0
    best_overall_set = []
    
    # Run multiple restarts
    num_restarts = 10
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        init_mask, init_set = random_init()
        
        # Give more time to earlier restarts? Or equal time?
        # Let's use equal time, but check deadline
        t_limit = min(time.time() + time_per_restart, deadline - 5)
        
        res_mask, res_score, res_set = simulated_annealing(init_mask, init_set, t_limit)
        
        if res_score > best_overall_score:
            best_overall_score = res_score
            best_overall_set = res_set
            
        # If we reach a very high score, we might stop early
        # Max possible is 255. If we get > 200, it's likely very good.
        if best_overall_score > 200:
            # We can continue to try to improve, but we are likely in a good basin
            # Let's just break if we have very little time left or score is perfect
            if best_overall_score >= 220:
                break

    # Fallback
    if not best_overall_set:
        best_overall_set = list(range(1, N + 1))
        
    # Final validation
    if len(set(best_overall_set)) != N or not all(1 <= x <= RANGE for x in best_overall_set):
        best_overall_set = list(range(1, N + 1))
        
    return sorted(best_overall_set)
