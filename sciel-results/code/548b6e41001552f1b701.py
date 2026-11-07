import time
import math

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Precompute: for each residue r in [0, 27720), and for each L from 1 to 144,
    # check if interval (m, m+L] contains distinct b_1..b_12 with i|b_i.
    # But we only need to find the max f(12, m) over m in [0, 1e6].
    # Since pattern repeats every 27720, we can scan one period and extrapolate.
    
    # Actually, let's just scan all m in [0, 1000000] but with a very fast f(12,m)
    # using precomputed divisibility masks.
    
    n = 12
    max_L = 144
    
    # Precompute for each i in 1..12, the residues mod i
    # For a given m and L, b_i must be in (m, m+L] and divisible by i.
    # So for each i, the candidates are: ceil((m+1)/i)*i, then +i, etc., up to m+L.
    
    # Fast approach: for each L from 1 to max_L, check if matching exists.
    # But we want the MINIMUM L such that matching exists. So for each m, 
    # we binary search or linear scan L from 1 upward.
    
    # Even faster: precompute for each m mod 27720 the value f(12, m).
    # Then for any m, f(12, m) = f(12, m % 27720).
    
    period = 27720  # lcm(1,2,...,12)
    
    # Precompute f for one period
    f_values = [0] * period
    
    # For each residue r in [0, period), compute f(12, r)
    # We'll use a helper that checks if a given interval (m, m+L] has a valid matching.
    
    def has_matching(m, L):
        """Check if (m, m+L] contains distinct b_1..b_12 with i|b_i."""
        # For each i, find all multiples of i in (m, m+L]
        # Then check if there's a perfect matching.
        
        # Build adjacency: for each i, list of candidate b's (as indices in a global list)
        # But b's must be distinct. Let's collect all candidate b's first.
        
        candidates = {}  # i -> list of b values
        all_bs = set()
        
        for i in range(1, n + 1):
            # First multiple of i > m
            start = (m // i + 1) * i
            if start <= m:
                start += i
            candidates[i] = []
            b = start
            while b <= m + L:
                if b not in all_bs:
                    candidates[i].append(b)
                else:
                    candidates[i].append(b)  # still add, but track duplicates
                all_bs.add(b)
                b += i
        
        # Now we need to check if there's a system of distinct representatives.
        # Use DFS with memoization or Hall's theorem.
        # For n=12, DFS is feasible if we order by smallest candidate set first.
        
        # Order i's by length of candidates[i]
        items = sorted(range(1, n + 1), key=lambda i: len(candidates[i]))
        
        # If any i has no candidates, fail
        for i in items:
            if not candidates[i]:
                return False
        
        # DFS with backtracking
        used = set()
        
        def dfs(idx):
            if idx == n:
                return True
            i = items[idx]
            for b in candidates[i]:
                if b not in used:
                    used.add(b)
                    if dfs(idx + 1):
                        return True
                    used.remove(b)
            return False
        
        return dfs(0)
    
    # Precompute f for one period
    for r in range(period):
        # f(12, r) is the least L >= 1 such that has_matching(r, L) is True
        found = False
        for L in range(1, max_L + 1):
            if time.time() > deadline:
                # Return best found so far; fill rest with 1
                f_values[r] = 1
                continue
            if has_matching(r, L):
                f_values[r] = L
                found = True
                break
        if not found:
            f_values[r] = max_L  # should not happen per problem statement
    
    # Now find the m in [0, 1000000] that maximizes f(12, m)
    # Since f(12, m) = f_values[m % period], just find max in f_values
    max_f = max(f_values)
    
    # Find an m in [0, 1000000] with f(12, m) == max_f
    # Pick the smallest such m
    for m in range(1000001):
        if f_values[m % period] == max_f:
            return int(m)
    
    # Fallback (should not reach here)
    return int(0)
