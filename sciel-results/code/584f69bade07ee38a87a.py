import random
import time
from fractions import Fraction
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    # Precompute reciprocals
    recip = {i: Fraction(1, i) for i in range(1, 25)}
    
    def is_valid(subset):
        """Check if no element's reciprocal equals sum of reciprocals of a subset of others."""
        s = list(subset)
        n = len(s)
        if n == 0:
            return True
        
        for i in range(n):
            target = recip[s[i]]
            others = [s[j] for j in range(n) if j != i]
            # Check if target can be formed by a subset of others
            if _can_make(target, others):
                return False
        return True
    
    def _can_make(target, others):
        """Check if target can be formed by a subset of others' reciprocals."""
        if target == 0:
            return True
        if not others:
            return False
        
        # Sort others by value descending for better pruning
        others_sorted = sorted(others, key=lambda x: -recip[x])
        # Actually sort by reciprocal value descending (larger reciprocals first)
        others_sorted = sorted(others, key=lambda x: -recip[x])
        
        # Compute suffix sums for pruning
        n = len(others_sorted)
        suffix_sum = [Fraction(0)] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_sum[i] = suffix_sum[i + 1] + recip[others_sorted[i]]
        
        def dfs(idx, remaining):
            if remaining == 0:
                return True
            if idx == n:
                return False
            # Prune: if remaining > suffix_sum[idx], impossible
            if remaining > suffix_sum[idx]:
                return False
            # Prune: if remaining < 0, impossible
            if remaining < 0:
                return False
            
            # Option 1: include others_sorted[idx]
            r = recip[others_sorted[idx]]
            if r <= remaining:
                if dfs(idx + 1, remaining - r):
                    return True
            
            # Option 2: exclude
            if dfs(idx + 1, remaining):
                return True
            
            return False
        
        return dfs(0, target)
    
    # Start with the seed solution
    best_set = list(range(13, 25))
    best_score = len(best_set)
    
    # Try to improve via local search
    # Strategy: try adding elements from [1..12] one by one
    # Then try swaps
    
    def try_grow(current_set):
        """Try to add elements to current_set while maintaining validity."""
        s = list(current_set)
        for candidate in range(1, 25):
            if candidate in s:
                continue
            new_set = s + [candidate]
            if is_valid(new_set):
                s = new_set
        return s
    
    # Greedy grow from seed
    grown = try_grow(best_set)
    if len(grown) > best_score:
        best_set = grown
        best_score = len(best_set)
    
    # Try multiple random restarts
    random.seed(42)
    for _ in range(200):
        if time.time() > deadline:
            break
        
        # Random initial set: pick random subset of [1..24]
        size = random.randint(10, 18)
        init_set = random.sample(range(1, 25), size)
        
        # Check validity, if valid, try to grow
        if is_valid(init_set):
            grown = try_grow(init_set)
            if len(grown) > best_score:
                best_set = grown
                best_score = len(best_set)
        
        # Also try: start from best_set, remove one random element, then regrow
        if len(best_set) > 1:
            modified = list(best_set)
            remove_idx = random.randint(0, len(modified) - 1)
            modified.pop(remove_idx)
            # Also try swapping: remove one, add another
            if random.random() < 0.5 and len(modified) >= 1:
                # Swap
                removed = modified[random.randint(0, len(modified) - 1)]
                modified.remove(removed)
                candidates = [x for x in range(1, 25) if x not in modified]
                if candidates:
                    modified.append(random.choice(candidates))
            
            if is_valid(modified):
                grown = try_grow(modified)
                if len(grown) > best_score:
                    best_set = grown
                    best_score = len(best_set)
    
    # Final check
    if not is_valid(best_set):
        # Fallback to seed
        best_set = list(range(13, 25))
    
    return list(best_set)
