import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    max_n = 512
    max_size = 48

    # Precompute: for each pair (a, b), check if they form a 3-term AP with some c in [1, max_n]
    # i.e., a + b = 2*c, so c = (a+b)/2 must be integer and in range
    # This gives us the "conflict" structure: if a, b are in S, then c cannot be in S
    # (because c is the mean of {a, b})
    
    # Actually, the condition is: no element y in S is the mean of any subset T of S\{y} with |T|>=2
    # So for |T|=2: y = (a+b)/2 means a+b=2y, so a,b,y form a 3-term AP with y as the middle
    # For |T|=3: y = (a+b+c)/3 means a+b+c=3y
    # etc.
    
    # The 2-element subset case (3-term AP) is the most restrictive and common
    # Let's build a conflict graph where edge (a,b) exists if there's a c in [1,max_n] 
    # such that a+b=2c (i.e., c is the midpoint)
    
    # For the greedy, we want to avoid creating situations where some element becomes a mean
    # Key insight: if we ensure no 3-term AP exists in our set, we handle the |T|=2 case
    # But we also need to handle larger subsets
    
    # Strategy: Use a greedy that prioritizes numbers that don't create many 3-term AP conflicts
    # Then verify with full DP at the end
    
    def count_3ap_conflicts(x, current_set):
        """Count how many 3-term APs would be created if we add x to current_set"""
        count = 0
        S = set(current_set)
        for y in S:
            # Check if x, y, and some z in S form a 3-term AP
            # x is midpoint: y + z = 2x => z = 2x - y
            z = 2 * x - y
            if z in S and z != y:
                count += 1
            # y is midpoint: x + z = 2y => z = 2y - x
            z = 2 * y - x
            if z in S and z != x:
                count += 1
            # z is midpoint: x + y = 2z => z = (x+y)/2
            if (x + y) % 2 == 0:
                z = (x + y) // 2
                if z in S and z != x and z != y:
                    count += 1
        return count

    def is_valid_full(S):
        """Full validity check using subset-sum DP"""
        S_list = list(S)
        n = len(S_list)
        if n < 3:
            return True

        for i, y in enumerate(S_list):
            elements = [x for j, x in enumerate(S_list) if j != i]
            m = len(elements)
            
            dp = [0] * (m + 1)
            dp[0] = 1
            current_max_k = 0

            for x in elements:
                for k in range(current_max_k, -1, -1):
                    if dp[k]:
                        dp[k + 1] |= (dp[k] << x)
                current_max_k += 1

            for k in range(2, m + 1):
                target_sum = k * y
                if (dp[k] >> target_sum) & 1:
                    return False

        return True

    def greedy_conflict_aware(available, max_steps=48):
        """Greedy that picks the element with fewest 3-AP conflicts at each step"""
        S = []
        S_set = set()
        avail = list(available)
        
        for step in range(max_steps):
            if time.time() > deadline - 15.0:
                break
            
            best_x = None
            best_conflicts = float('inf')
            
            # Sample a subset of candidates to speed up
            if len(avail) > 50:
                sampled = random.sample(avail, 50)
            else:
                sampled = avail
            
            for x in sampled:
                conflicts = count_3ap_conflicts(x, S)
                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_x = x
            
            if best_x is not None:
                S.append(best_x)
                S_set.add(best_x)
                avail.remove(best_x)
        
        return S

    def local_search_improve(S, time_limit=20.0):
        """Try to improve S by removing one element and adding another"""
        S = list(S)
        S_set = set(S)
        end_time = time.time() + time_limit
        available = list(set(range(1, max_n + 1)) - S_set)
        
        improved = True
        while improved and time.time() < end_time:
            improved = False
            random.shuffle(available)
            
            for x in available:
                if time.time() > end_time:
                    break
                
                # Try removing each element y and adding x
                for i, y in enumerate(S):
                    if time.time() > end_time:
                        break
                    
                    new_S = S[:i] + S[i+1:] + [x]
                    if len(new_S) != len(S):
                        continue
                    
                    if is_valid_full(new_S):
                        S = new_S
                        S_set = set(S)
                        available = list(set(range(1, max_n + 1)) - S_set)
                        improved = True
                        break
                
                if improved:
                    break
        
        return S

    # Generate multiple diverse starting sets
    random.seed(42)
    candidates = []
    
    # Conflict-aware greedy with different random seeds
    for trial in range(20):
        if time.time() > deadline - 30.0:
            break
        
        # Different ordering strategies
        if trial == 0:
            order = list(range(max_n, 0, -1))
        elif trial == 1:
            order = list(range(1, max_n + 1))
        elif trial == 2:
            # Primes first
            def is_prime(n):
                if n < 2:
                    return False
                for i in range(2, int(n**0.5) + 1):
                    if n % i == 0:
                        return False
                return True
            primes = [x for x in range(1, max_n + 1) if is_prime(x)]
            others = [x for x in range(1, max_n + 1) if not is_prime(x)]
            order = primes + others
        else:
            order = list(range(1, max_n + 1))
            random.shuffle(order)
        
        S = greedy_conflict_aware(order)
        candidates.append(S)
    
    # Find the best candidate
    best_set = max(candidates, key=len)
    best_size = len(best_set)
    
    # If we haven't reached max_size, try to improve
    if best_size < max_size and time.time() < deadline - 10.0:
        improved = local_search_improve(best_set, time_limit=min(30.0, deadline - 10.0 - time.time()))
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_set)
    
    # Final verification
    if not is_valid_full(best_set):
        # Fallback: return a smaller valid set
        # Try removing elements until valid
        S = list(best_set)
        while S and not is_valid_full(S):
            # Remove the element that causes the most conflicts
            worst_idx = 0
            worst_conflicts = -1
            for i, y in enumerate(S):
                others = S[:i] + S[i+1:]
                # Count how many elements y is the mean of
                conflicts = 0
                for j, a in enumerate(others):
                    for k in range(j+1, len(others)):
                        b = others[k]
                        if (a + b) == 2 * y:
                            conflicts += 1
                if conflicts > worst_conflicts:
                    worst_conflicts = conflicts
                    worst_idx = i
            S.pop(worst_idx)
        best_set = S
    
    return best_set
