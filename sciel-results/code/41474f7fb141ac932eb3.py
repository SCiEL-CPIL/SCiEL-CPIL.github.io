import time
from math import factorial

TARGET = factorial(12)

def get_divisors(n):
    divisors = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
        i += 1
    return divisors

ALL_DIVISORS = get_divisors(TARGET)

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # Filter divisors > 12
    # We will filter by M inside the search function
    
    # Binary search on M
    low = 13
    high = TARGET
    best_factors = None
    
    # We can tighten low.
    # The product of the smallest possible factors > 12.
    # 13 * 14 * ... 
    # But we don't know the count.
    
    while low <= high and time.time() < deadline:
        mid = (low + high) // 2
        sol = find_solution(mid)
        if sol:
            best_factors = sol
            high = mid - 1
        else:
            low = mid + 1
            
    if best_factors is None:
        # Fallback: [TARGET] is always a valid solution (1 factor, >12)
        return [TARGET]
        
    return best_factors

def find_solution(M):
    # Candidates: divisors of TARGET such that 12 < d <= M
    candidates = [d for d in ALL_DIVISORS if 12 < d <= M]
    
    if not candidates:
        return None
        
    # Sort descending to pick large factors first
    candidates.sort(reverse=True)
    
    # Precompute suffix products for pruning
    # suffix_prod[i] = product of candidates[i:]
    # Note: Product can get very large, but Python handles big ints.
    # However, we only care if it's >= remaining.
    # If suffix_prod[i] < remaining, we can prune.
    
    n = len(candidates)
    suffix_prod = [1] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_prod[i] = suffix_prod[i + 1] * candidates[i]
        
    # DFS
    # state: idx, remaining
    # remaining is the target product we still need to form
    
    # Optimization: 
    # If remaining == 1, success.
    # If idx == n, fail.
    # If suffix_prod[idx] < remaining, fail.
    
    # We also need to ensure distinct factors. 
    # Since we iterate idx forward, we never reuse a candidate.
    
    # One more pruning: 
    # If candidates[idx] > remaining, we can't pick it. 
    # But since candidates are sorted descending, if candidates[idx] > remaining, 
    # then all candidates[j] for j < idx are also > remaining? 
    # No, we are at idx. candidates[idx] is the largest remaining candidate.
    # If candidates[idx] > remaining, we MUST skip it.
    # Actually, if candidates[idx] > remaining, we cannot pick it. 
    # We just skip to idx+1.
    
    # What if remaining is not divisible by any candidate from idx onwards?
    # Hard to check.
    
    def dfs(idx, remaining):
        if remaining == 1:
            return []
        
        if idx == n:
            return None
            
        # Pruning: Max possible product of remaining candidates
        if suffix_prod[idx] < remaining:
            return None
            
        # Pruning: If current candidate is too large to divide remaining?
        # If candidates[idx] > remaining, we can't use it.
        # Since sorted desc, if candidates[idx] > remaining, 
        # we can skip all candidates that are > remaining?
        # No, we just skip this one.
        
        # Try picking candidates[idx]
        c = candidates[idx]
        if c <= remaining and remaining % c == 0:
            sub_sol = dfs(idx + 1, remaining // c)
            if sub_sol is not None:
                return [c] + sub_sol
                
        # Try skipping candidates[idx]
        sub_sol = dfs(idx + 1, remaining)
        if sub_sol is not None:
            return sub_sol
            
        return None
        
    sol = dfs(0, TARGET)
    return sol
