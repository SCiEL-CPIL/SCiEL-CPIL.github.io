import random

def construct():
    def find_conflict(S):
        n = len(S)
        if n < 3:
            return None
        for i, a in enumerate(S):
            others = [y for j, y in enumerate(S) if j != i]
            m = len(others)
            if m < 2:
                continue
            
            dp = [0] * (m + 1)
            dp[0] = 1
            for y in others:
                for k in range(m, 0, -1):
                    dp[k] |= (dp[k-1] << y)
            
            max_sum = sum(others)
            for k in range(2, m + 1):
                target = k * a
                if target > max_sum:
                    break
                if (dp[k] >> target) & 1:
                    return i
        return None

    def is_valid(S):
        return find_conflict(S) is None

    max_size = 48
    n_max = 512
    best_set = []
    
    # Try to find a valid set of size 48
    # Use random restarts with repair
    
    for target_size in range(max_size, 35, -1):
        success = False
        # Increase attempts for larger sizes?
        attempts = 1000 if target_size > 40 else 500
        
        for _ in range(attempts):
            S = random.sample(range(1, n_max + 1), target_size)
            
            # Repair phase
            for step in range(2000):
                idx = find_conflict(S)
                if idx is None:
                    # Valid
                    # Try to grow if possible
                    current_size = len(S)
                    if current_size < max_size:
                        # Try to add elements
                        for _ in range(100):
                            if len(S) >= max_size:
                                break
                            candidates = [x for x in range(1, n_max + 1) if x not in S]
                            if not candidates:
                                break
                            x = random.choice(candidates)
                            new_S = S + [x]
                            if is_valid(new_S):
                                S = new_S
                            else:
                                # Maybe try swapping?
                                pass
                    
                    if len(S) > len(best_set):
                        best_set = S[:]
                        if len(best_set) == max_size:
                            return best_set
                    success = True
                    break
                
                # Repair
                if random.random() < 0.5:
                    victim_idx = idx
                else:
                    victim_idx = random.randint(0, len(S) - 1)
                
                old_val = S[victim_idx]
                new_val = random.randint(1, n_max)
                while new_val in S:
                    new_val = random.randint(1, n_max)
                S[victim_idx] = new_val
        
        if success and len(best_set) >= target_size:
            # If we found a valid set of this size, we might try to extend it in the next iteration?
            # But we are iterating down. If we found 47, we stop trying 46?
            # No, we want max. If we found 47, we should try to see if we can get 48 from it?
            # The loop above tries to grow.
            # If we are at target_size=47 and found a valid 47, and it didn't grow to 48,
            # we should move to target_size=46?
            # Actually, if we found a valid set of size K, we can just return it if we can't do better.
            # But we should exhaust time for 48 first.
            pass

    return best_set
