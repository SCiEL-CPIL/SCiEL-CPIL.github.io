import time
import random
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 85.0

    def check_x_not_mean_of_S(S, x):
        """Check if x is the mean of any subset of S with size >= 2."""
        n = len(S)
        if n < 2:
            return True
        max_sum = sum(S)
        # DP: dp[k] is a bitset where bit s is set if there's a subset of size k with sum s
        dp = [0] * (n + 1)
        dp[0] = 1
        for e in S:
            for k in range(n, 0, -1):
                dp[k] |= (dp[k-1] << e)
        for k in range(2, n + 1):
            target = k * x
            if target <= max_sum and (dp[k] >> target) & 1:
                return False
        return True

    def check_S_not_mean_with_x(S, x):
        """For each a in S, check a is not mean of any subset of (S\\{a} U {x}) with size >= 2 that includes x."""
        n = len(S)
        if n == 0:
            return True
        for a in S:
            others = [v for v in S if v != a]
            m = len(others)
            if m < 1:
                continue
            # Need: sum(U) = (j+1)*a - x for some U subset of others, j = |U| >= 1
            max_sum = sum(others)
            dp = [0] * (m + 1)
            dp[0] = 1
            for e in others:
                for k in range(m, 0, -1):
                    dp[k] |= (dp[k-1] << e)
            for j in range(1, m + 1):
                target = (j + 1) * a - x
                if 0 <= target <= max_sum and (dp[j] >> target) & 1:
                    return False
        return True

    def is_valid_add(S, x):
        """Check if adding x to S keeps the set mean-free."""
        if x in S:
            return False
        if not check_x_not_mean_of_S(S, x):
            return False
        if not check_S_not_mean_with_x(S, x):
            return False
        return True

    def quick_precheck(S, x):
        """Fast necessary condition: for pairs in S, 2x != a+b for distinct a,b in S.
        Also for triples: 3x != a+b+c, etc. But just check pairs first as fast filter."""
        n = len(S)
        if n < 2:
            return True
        # Check if 2x can be written as sum of 2 distinct elements from S
        # This is a necessary condition for x to be mean of a pair
        # Use set for O(n) check
        S_set = set(S)
        for a in S:
            b = 2 * x - a
            if b != a and b in S_set:
                return False
        return True

    # Strategy: Build set using structured ordering + greedy fill
    # Phase 1: Start with well-separated numbers (powers of 2 scaled)
    # Phase 2: Greedily add remaining candidates
    
    best_set = []
    best_len = 0

    # Multiple strategies
    strategies = []
    
    # Strategy A: Descending greedy
    strategies.append(('desc', list(range(512, 0, -1))))
    
    # Strategy B: Ascending greedy  
    strategies.append(('asc', list(range(1, 513))))
    
    # Strategy C: Powers of 2 first, then fill
    p2 = [2**i for i in range(1, 10) if 2**i <= 512]  # 2,4,8,...,256
    rest = [x for x in range(1, 513) if x not in p2]
    strategies.append(('p2', p2 + rest))
    
    # Strategy D: Randomized with different seeds
    random.seed(123)
    for seed in range(10):
        cands = list(range(1, 513))
        random.seed(seed * 1000 + 42)
        random.shuffle(cands)
        strategies.append((f'rand{seed}', cands))

    # Strategy E: Modular - pick numbers with specific residue classes
    # Numbers congruent to 1 mod 3: harder to form means
    mod_cands = [x for x in range(1, 513) if x % 3 == 1]
    mod_cands2 = [x for x in range(1, 513) if x % 3 == 2]
    strategies.append(('mod1', mod_cands + mod_cands2))

    for name, candidates in strategies:
        if time.time() > deadline:
            break
        
        current = []
        used = set()
        
        for x in candidates:
            if time.time() > deadline:
                break
            if x in used:
                continue
            if len(current) >= 48:
                break
            
            # Quick precheck first
            if not quick_precheck(current, x):
                continue
            
            # Full check
            if is_valid_add(current, x):
                current.append(x)
                used.add(x)
        
        if len(current) > best_len:
            best_len = len(current)
            best_set = current[:]
        
        if best_len >= 48:
            break
    
    # If we haven't reached 48, try local search to improve
    if best_len < 48 and time.time() < deadline - 5.0:
        # Try to add more elements by removing one and adding two
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            current = best_set[:]
            current_set = set(current)
            
            # Try removing one element and adding two new ones
            for i in range(len(current)):
                if time.time() > deadline - 5.0:
                    break
                removed = current[i]
                temp = [x for j, x in enumerate(current) if j != i]
                temp_set = set(temp)
                
                # Find candidates that could be added
                added = []
                for x in range(1, 513):
                    if x in temp_set or x == removed:
                        continue
                    if len(added) >= 2:
                        break
                    if not quick_precheck(temp + added, x):
                        continue
                    if is_valid_add(temp + added, x):
                        added.append(x)
                
                if len(added) >= 2:
                    new_set = temp + added
                    if len(new_set) > len(best_set):
                        best_set = new_set
                        best_len = len(new_set)
                        improved = True
                        break
    
    return best_set

if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")
    print(result[:10], "...", result[-10:])
