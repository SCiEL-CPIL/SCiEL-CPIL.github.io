import random
import math

def construct():
    max_val = 256
    max_size = 60
    
    def compute_dp(S):
        dp = [0] * (max_size + 1)
        dp[0] = 1
        for x in S:
            # Update in reverse to avoid using x multiple times
            # But we are building from scratch for a specific set S
            # So we need to iterate k downwards?
            # No, if we iterate x one by one, we update dp based on previous state.
            # To do it in place, we must iterate k from max_size down to 0?
            # dp_new[k] = dp_old[k] | (dp_old[k-1] << x)
            # If we do it in place:
            # for k in range(max_size, -1, -1):
            #    dp[k] |= (dp[k-1] << x) if k>0 else 0
            # This works because dp[k-1] hasn't been updated for this x yet (since we go down).
            for k in range(max_size, -1, -1):
                if k > 0:
                    dp[k] |= (dp[k-1] << x)
                # else k=0, dp[0] stays 1
        return dp

    def count_violations(dp):
        violations = 0
        for i in range(max_size + 1):
            if dp[i] == 0:
                continue
            for j in range(i + 1, max_size + 1):
                if dp[j] == 0:
                    continue
                if (dp[i] & dp[j]) != 0:
                    violations += 1
        return violations

    def get_energy(S, dp):
        viol = count_violations(dp)
        size = len(S)
        # We want to maximize size, minimize violations.
        # Energy = Violations * LargeWeight - Size
        # If Violations > 0, it's bad.
        # Let's say Energy = viol * 1000 - size
        return viol * 1000 - size

    # Initial set using greedy
    def greedy_init():
        S = []
        dp = [0] * (max_size + 1)
        dp[0] = 1
        for x in range(1, max_val + 1):
            if len(S) >= max_size:
                break
            # Check if adding x is valid
            temp_dp = [0] * (max_size + 1)
            for k in range(max_size + 1):
                temp_dp[k] = dp[k]
                if k > 0:
                    temp_dp[k] |= (dp[k-1] << x)
            
            valid = True
            for i in range(max_size + 1):
                if temp_dp[i] == 0: continue
                for j in range(i + 1, max_size + 1):
                    if temp_dp[j] == 0: continue
                    if (temp_dp[i] & temp_dp[j]) != 0:
                        valid = False
                        break
                if not valid: break
            
            if valid:
                dp = temp_dp
                S.append(x)
        return S, dp

    S, dp = greedy_init()
    best_S = S[:]
    best_dp = dp[:]
    best_energy = get_energy(S, dp)
    
    current_S = S[:]
    current_dp = dp[:]
    current_energy = best_energy
    
    T = 10.0
    cooling = 0.99
    min_T = 0.01
    
    import time
    start_time = time.time()
    timeout = 55.0 # Leave buffer
    
    while T > min_T:
        if time.time() - start_time > timeout:
            break
            
        # Generate neighbor
        action = random.random()
        new_S = current_S[:]
        
        if action < 0.3:
            # Add
            candidates = [x for x in range(1, max_val + 1) if x not in new_S]
            if candidates:
                x = random.choice(candidates)
                new_S.append(x)
        elif action < 0.6:
            # Remove
            if new_S:
                x = random.choice(new_S)
                new_S.remove(x)
        else:
            # Swap
            if new_S:
                x = random.choice(new_S)
                candidates = [y for y in range(1, max_val + 1) if y not in new_S]
                if candidates:
                    y = random.choice(candidates)
                    new_S.remove(x)
                    new_S.append(y)
        
        # Evaluate neighbor
        new_dp = compute_dp(new_S)
        new_energy = get_energy(new_S, new_dp)
        
        delta = new_energy - current_energy
        
        if delta < 0:
            # Better
            current_S = new_S
            current_dp = new_dp
            current_energy = new_energy
            
            # Check if valid and better than best
            viol = count_violations(new_dp)
            if viol == 0 and len(new_S) > len(best_S):
                best_S = new_S[:]
                best_dp = new_dp[:]
                best_energy = new_energy
        else:
            # Worse
            # Accept with probability exp(-delta / T)
            if random.random() < math.exp(-delta / T):
                current_S = new_S
                current_dp = new_dp
                current_energy = new_energy
                
                # Even if worse, it might be valid?
                # If it's valid but smaller, we don't update best (we want max size).
                # But if it's valid and same size, it's a plateau.
                # We only care about valid sets for the "best" tracking.
                viol = count_violations(new_dp)
                if viol == 0 and len(new_S) > len(best_S):
                    best_S = new_S[:]
                    best_dp = new_dp[:]
                    best_energy = new_energy
        
        T *= cooling

    # Final check: return the best valid set found
    # best_S should be valid by construction of the update logic
    return best_S
