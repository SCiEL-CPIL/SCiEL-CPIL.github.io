"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin from 90s limit
    
    def update_dp(dp, x):
        """Update DP table by adding element x. dp[k] is bitmask of sums of size k."""
        n = len(dp)
        new_dp = [0] * (n + 1)
        new_dp[0] = 1
        for k in range(1, n + 1):
            val = dp[k] if k < n else 0
            if k > 0:
                val |= (dp[k-1] << x)
            new_dp[k] = val
        return new_dp

    def get_violations_and_size(selected, dp_all, dp_without):
        """
        Calculate the number of violations and the size of the set.
        Violation: a in S is mean of subset of S\{a} size >= 2.
        """
        n = len(selected)
        if n < 3:
            return 0, n
        
        violations = 0
        for i, a in enumerate(selected):
            others = selected[:i] + selected[i+1:]
            m = len(others)
            if m < 2:
                continue
            
            # dp_without[a] corresponds to S \ {a}
            dp_others = dp_without[a]
            
            for k in range(2, m + 1):
                target = k * a
                if target > 512 * k:
                    continue
                if (dp_others[k] >> target) & 1:
                    violations += 1
                    break # One violation per element is enough for energy calc? 
                          # Actually, we want to minimize total violations. 
                          # But for SA, knowing "is it valid" is often enough.
                          # Let's count total violations for better gradient.
        
        # Recalculate total violations properly if needed, 
        # but for SA step, we just need Delta E.
        return violations, n

    def greedy_incremental(order, max_size=48):
        selected = []
        n = 0
        dp_all = [1]
        dp_without = {}
        
        for x in order:
            if time.time() - start_time > deadline:
                break
            if n >= max_size:
                break
            
            valid = True
            # Check 1: x mean of S
            for k in range(2, n + 1):
                target = k * x
                if target > 512 * k:
                    continue
                if (dp_all[k] >> target) & 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check 2: a in S becomes invalid
            for a in selected:
                for k in range(2, n + 2):
                    target_sum = k * a - x
                    if target_sum < 0:
                        continue
                    if target_sum > 512 * (k - 1):
                        continue
                    if k - 1 < len(dp_without[a]):
                        if (dp_without[a][k-1] >> target_sum) & 1:
                            valid = False
                            break
                if not valid:
                    break
            
            if not valid:
                continue
            
            selected.append(x)
            n += 1
            old_dp_all = dp_all
            dp_all = update_dp(dp_all, x)
            
            for a in selected[:-1]:
                dp_without[a] = update_dp(dp_without[a], x)
            dp_without[x] = old_dp_all
        
        return selected, dp_all, dp_without

    def validate_set(s):
        if len(s) < 3:
            return True
        dp_all = [1]
        dp_without = {}
        for x in s:
            old_dp_all = dp_all
            dp_all = update_dp(dp_all, x)
            for a in s[:-1]:
                if a in dp_without:
                    dp_without[a] = update_dp(dp_without[a], x)
            dp_without[x] = old_dp_all
            
        for i, a in enumerate(s):
            others = s[:i] + s[i+1:]
            m = len(others)
            if m < 2:
                continue
            dp_others = dp_without[a]
            for k in range(2, m + 1):
                target = k * a
                if target > 512 * k:
                    continue
                if (dp_others[k] >> target) & 1:
                    return False
        return True

    # 1. Greedy Initialization
    best_set = []
    best_dp_all = [1]
    best_dp_without = {}
    best_len = 0
    
    # Try a few random shuffles for greedy
    for seed in range(20):
        if time.time() - start_time > deadline:
            break
        random.seed(seed)
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        
        res, dp_a, dp_w = greedy_incremental(candidates)
        if len(res) > best_len:
            best_set = res
            best_dp_all = dp_a
            best_dp_without = dp_w
            best_len = len(res)
            
        if best_len >= 48:
            break

    if not best_set:
        return []

    # 2. Simulated Annealing
    # State: S (list), dp_all, dp_without
    # We will work with a copy
    S = best_set[:]
    dp_all = best_dp_all[:]
    dp_without = {k: v[:] for k, v in best_dp_without.items()}
    
    current_energy = 0 # 0 violations, so energy is -size? 
    # Let's define Energy = Violations * 100 - Size
    # We want to minimize Energy.
    
    def calc_energy(S, dp_all, dp_without):
        n = len(S)
        if n < 3:
            return -n
        violations = 0
        for i, a in enumerate(S):
            dp_others = dp_without[a]
            m = n - 1
            for k in range(2, m + 1):
                target = k * a
                if target > 512 * k:
                    continue
                if (dp_others[k] >> target) & 1:
                    violations += 1
                    break
        return violations * 100 - n

    current_energy = calc_energy(S, dp_all, dp_without)
    
    T = 100.0
    cooling = 0.995
    min_T = 0.1
    max_steps = 50000
    
    step = 0
    while step < max_steps:
        if time.time() - start_time > deadline:
            break
        
        # Choose move type
        move_type = random.random()
        
        if move_type < 0.4:
            # Add
            candidates = [x for x in range(1, 513) if x not in S]
            if not candidates: continue
            x = random.choice(candidates)
            
            # Calculate Delta Energy for adding x
            # 1. Check if x is mean of S
            delta_viol = 0
            for k in range(2, len(S) + 1):
                target = k * x
                if target > 512 * k: continue
                if (dp_all[k] >> target) & 1:
                    delta_viol += 1
                    break
            
            # 2. Check if any a in S becomes invalid
            for a in S:
                # Check if a is mean of subset of S U {x} \ {a}
                # Subsets containing x: T U {x}, T subset S \ {a}
                # k a = sum(T) + x => sum(T) = k a - x
                for k in range(2, len(S) + 2):
                    target_sum = k * a - x
                    if target_sum < 0: continue
                    if target_sum > 512 * (k - 1): continue
                    if k - 1 < len(dp_without[a]):
                        if (dp_without[a][k-1] >> target_sum) & 1:
                            delta_viol += 1
                            break
                            
            delta_E = delta_viol * 100 - 1 # Size increases by 1
            
            if delta_E < 0 or (T > 0 and random.random() < math.exp(-delta_E / T)):
                # Accept
                S.append(x)
                old_dp_all = dp_all
                dp_all = update_dp(dp_all, x)
                for a in S[:-1]:
                    dp_without[a] = update_dp(dp_without[a], x)
                dp_without[x] = old_dp_all
                
                # Update Energy
                current_energy += delta_E
                if current_energy < 0 and delta_viol == 0:
                    # Improved
                    pass

        elif move_type < 0.7:
            # Remove
            if not S: continue
            x = random.choice(S)
            idx = S.index(x)
            
            # Calculate Delta Energy for removing x
            # Violations removed: 
            # 1. x was mean of S \ {x}
            # 2. a in S \ {x} was mean of subset involving x
            
            delta_viol = 0
            # x was invalid?
            # Check if x is mean of S \ {x}
            dp_others_x = dp_without[x]
            m = len(S) - 1
            x_was_invalid = False
            for k in range(2, m + 1):
                target = k * x
                if target > 512 * k: continue
                if (dp_others_x[k] >> target) & 1:
                    x_was_invalid = True
                    break
            
            # a in S \ {x} was invalid due to x?
            # If a is mean of T U {x}, then T subset S \ {a, x}
            # k a = sum(T) + x
            # We need to know if removing x fixes any a.
            # This is complex to calculate Delta exactly without re-evaluating.
            # Approximation: Just re-calculate energy? 
            # Or, assume removal reduces violations.
            
            # For simplicity and speed, let's just re-calculate energy after removal 
            # if we accept, or estimate.
            # Estimation: 
            # If x was invalid, we remove 1 violation.
            # If any a was invalid ONLY because of x, we remove that violation.
            
            # Let's just try the removal and calculate new energy if accepted?
            # No, SA needs Delta E for acceptance.
            
            # Let's approximate Delta E = -1 (size decreases) + (violations_fixed * 100)
            # violations_fixed is hard to calc exactly.
            
            # Alternative: Use a "Swap" move instead of pure Remove/Add to maintain size.
            # But we want to maximize size.
            
            # Let's stick to Add and Swap. Remove is risky for size.
            # Actually, if we have many violations, we need to remove.
            
            # Let's implement a simple "Repair" step instead of SA Remove.
            # If violations > 0, find a random violating element and remove it.
            
            if current_energy > 0:
                # Find a violating element
                # Re-scan to find one
                violator = None
                for a in S:
                    dp_others = dp_without[a]
                    m = len(S) - 1
                    for k in range(2, m + 1):
                        target = k * a
                        if target > 512 * k: continue
                        if (dp_others[k] >> target) & 1:
                            violator = a
                            break
                    if violator: break
                
                if violator:
                    # Remove violator
                    idx = S.index(violator)
                    S.pop(idx)
                    # Rebuild DP? 
                    # Rebuilding is O(N^2). 
                    # Given N is small, rebuilding is acceptable for a repair step.
                    dp_all = [1]
                    dp_without = {}
                    for y in S:
                        old_dp_all = dp_all
                        dp_all = update_dp(dp_all, y)
                        for b in S[:-1]:
                            if b in dp_without:
                                dp_without[b] = update_dp(dp_without[b], y)
                        dp_without[y] = old_dp_all
                    
                    current_energy = calc_energy(S, dp_all, dp_without)

        else:
            # Swap
            if not S: continue
            x = random.choice(S)
            candidates = [y for y in range(1, 513) if y not in S]
            if not candidates: continue
            y = random.choice(candidates)
            
            # Remove x, Add y
            # This is effectively Add y to S\{x}
            
            # Calculate Delta E
            # 1. Check y in S\{x}
            # 2. Check a in S\{x} in S\{x} U {y}
            
            # This is complex. 
            # Let's just do: Remove x (rebuild), Add y (incremental).
            # If time permits.
            
            # Given complexity, let's focus on Add and Repair.
            pass
        
        T *= cooling
        step += 1

    # Final Validation
    if not validate_set(S):
        # Repair
        valid_set = []
        for x in S:
            test = valid_set + [x]
            if validate_set(test):
                valid_set.append(x)
        S = valid_set

    return S

# EVOLVE-BLOCK-END
