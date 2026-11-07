import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55 seconds safety margin
    
    best_set = [31, 32, 34]  # initial valid set
    best_size = 3
    
    def is_valid(S):
        """Check if set S satisfies the conditions."""
        if len(set(S)) != len(S):
            return False
        if any(x < 1 or x > 256 for x in S):
            return False
        
        # Generate all nonempty subset sums
        sums = set()
        n = len(S)
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += S[i]
            sums.add(s)
        
        # Check all sums are distinct (already ensured by set, but we need to check no two subsets give same sum)
        # Actually, the set removes duplicates, but we need to ensure no two different subsets have the same sum.
        # So we need to count: if any sum appears more than once, invalid.
        sum_count = {}
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += S[i]
            sum_count[s] = sum_count.get(s, 0) + 1
        if any(c > 1 for c in sum_count.values()):
            return False
        
        # Check no two distinct sums divide each other
        sum_list = sorted(sum_count.keys())
        for i in range(len(sum_list)):
            for j in range(i + 1, len(sum_list)):
                a, b = sum_list[i], sum_list[j]
                if b % a == 0:
                    return False
        return True
    
    def try_greedy():
        """Build a set greedily by trying to add largest possible elements."""
        S = []
        used = set()
        # Try adding elements from 256 down to 1
        for x in range(256, 0, -1):
            if x in used:
                continue
            new_S = S + [x]
            if is_valid(new_S):
                S = new_S
                used.add(x)
            if time.time() > deadline:
                break
        return S
    
    def local_search(S, max_iters=1000):
        """Local search: try swaps, adds, removes."""
        S = list(S)
        best = list(S)
        best_size = len(S)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            
            # Try adding a new element
            for x in range(256, 0, -1):
                if x not in S:
                    new_S = S + [x]
                    if is_valid(new_S):
                        S = new_S
                        best = list(S)
                        best_size = len(S)
                        improved = True
                        break
            
            # Try swapping
            if not improved:
                for i in range(len(S)):
                    for x in range(256, 0, -1):
                        if x not in S:
                            new_S = S[:i] + [x] + S[i+1:]
                            if is_valid(new_S):
                                S = new_S
                                best = list(S)
                                best_size = len(S)
                                improved = True
                                break
                    if improved:
                        break
            
            # Try removing
            if not improved and len(S) > 1:
                for i in range(len(S)):
                    new_S = S[:i] + S[i+1:]
                    if is_valid(new_S):
                        # Only remove if it allows a bigger addition later? For now, skip removals
                        pass
        
        return best
    
    # Try greedy first
    greedy_set = try_greedy()
    if len(greedy_set) > best_size:
        best_set = greedy_set
        best_size = len(best_set)
    
    # Try local search from greedy and from random starts
    for _ in range(10):
        if time.time() > deadline:
            break
        if best_size >= 15:  # if we already have a large set, stop
            break
        # Random start
        random_set = random.sample(range(1, 257), min(best_size, 10))
        # Try to build up from random set
        S = []
        for x in random_set:
            if is_valid(S + [x]):
                S.append(x)
        # Local search
        improved_set = local_search(S, max_iters=500)
        if len(improved_set) > best_size:
            best_set = improved_set
            best_size = len(best_set)
    
    # Final: try to add more to best_set
    for x in range(256, 0, -1):
        if x not in best_set:
            if is_valid(best_set + [x]):
                best_set.append(x)
    
    return best_set
