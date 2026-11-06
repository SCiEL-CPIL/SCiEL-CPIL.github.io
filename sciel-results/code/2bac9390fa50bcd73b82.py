import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # 5s margin
    
    # Try to find valid set of size 48, then 47, etc.
    for K in range(48, 0, -1):
        # Allocate time for this K
        # If K is large, give it more time?
        # Let's just try to find one quickly.
        # Initial random set
        S = set(random.sample(range(1, 513), K))
        
        # Calculate initial cost (number of violations)
        # We need a cost function that counts violations, not just boolean.
        # Cost = sum over a in S of (1 if a is mean of any subset else 0)
        # Actually, we just need to minimize cost to 0.
        
        def get_cost(S):
            cost = 0
            S_list = list(S)
            n = len(S_list)
            for i, a in enumerate(S_list):
                others = [x for x in S_list if x != a]
                m = len(others)
                bits = [0] * (m + 1)
                bits[0] = 1
                for t in others:
                    for k in range(m, 0, -1):
                        bits[k] |= (bits[k-1] << t)
                
                for k in range(2, m + 1):
                    target_sum = a * k
                    if (bits[k] >> target_sum) & 1:
                        cost += 1
                        break # Count violation per a
            return cost
        
        current_S = S
        current_cost = get_cost(current_S)
        
        if current_cost == 0:
            return list(current_S)
            
        T = 1.0
        cooling = 0.99
        step = 0
        max_steps = 10000 # Adjust based on speed
        
        while time.time() < deadline - 1.0 and step < max_steps:
            # Pick random move: swap one element
            x = random.choice(list(current_S))
            # Pick y not in S
            # Efficiently pick y from [1, 512] \ S
            # Since S is small, we can just sample and retry
            y = random.randint(1, 512)
            if y in current_S:
                continue
                
            new_S = set(current_S)
            new_S.remove(x)
            new_S.add(y)
            
            new_cost = get_cost(new_S)
            
            delta = new_cost - current_cost
            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                current_S = new_S
                current_cost = new_cost
                if current_cost == 0:
                    return list(current_S)
            
            T *= cooling
            step += 1
            
        # If we didn't find a valid set for this K, try smaller K
        # But we must ensure we return *something* valid if we run out of time
        # If we exit the loop for K=48 without success, we go to K=47.
        
    # Fallback: return a single element (always valid)
    return [1]
