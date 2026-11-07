import math
from functools import reduce

def construct():
    n = 12
    # Calculate LCM of 1..12
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    
    period = reduce(lcm, range(1, n + 1))
    # period is 27720

    def can_match(m, L):
        # Build adjacency list for left nodes 1..n
        # Right nodes are offsets 1..L representing numbers m+1 ... m+L
        # adj[i] is list of offsets j (0-indexed) such that i divides (m+j+1)
        adj = [[] for _ in range(n)]
        for i in range(1, n + 1):
            # We need m+j+1 to be divisible by i
            # m+j+1 ≡ 0 (mod i) => j+1 ≡ -m (mod i) => j ≡ -m-1 (mod i)
            rem = (-m - 1) % i
            # j is in 0..L-1 corresponding to offset 1..L
            # j = rem, rem+i, ...
            # Note: if rem == 0, j=0 is a candidate (offset 1)
            # Wait, j is 0-indexed offset.
            # Number is m + (j+1).
            # Condition: i | (m + j + 1).
            # j+1 ≡ -m (mod i)
            # j ≡ -m - 1 (mod i)
            
            j = rem
            while j < L:
                adj[i-1].append(j)
                j += i
        
        # Bipartite matching
        # match_r[j] = i that is matched to right node j (offset index j)
        match_r = [-1] * L
        
        def dfs(i, seen):
            for j in adj[i]:
                if seen[j]:
                    continue
                seen[j] = True
                if match_r[j] == -1 or dfs(match_r[j], seen):
                    match_r[j] = i
                    return True
            return False
        
        result = 0
        for i in range(n):
            seen = [False] * L
            if dfs(i, seen):
                result += 1
        return result == n

    best_m = 0
    best_L = 0
    
    # We only need to check m in [0, period-1]
    # For each m, find minimal L such that can_match(m, L) is True.
    # L is at most 144 (empirically and theoretically bounded for n=12)
    # We can binary search L in [1, 144]
    
    for m in range(period):
        # Binary search for minimal L
        lo, hi = 1, 144
        found_L = -1
        
        # Optimization: If L=144 doesn't work, something is wrong, but it should.
        # Actually, for any m, there is always a solution.
        
        while lo <= hi:
            mid = (lo + hi) // 2
            if can_match(m, mid):
                found_L = mid
                hi = mid - 1
            else:
                lo = mid + 1
        
        if found_L != -1:
            if found_L > best_L:
                best_L = found_L
                best_m = m
        else:
            # Fallback: linear search if binary search fails (should not happen)
            # Or just assume 144 if it fails? No, let's just skip or handle.
            # Given the problem statement, L <= 144 is guaranteed for valid inputs.
            pass

    # The problem asks for m in 0..1000000.
    # best_m is in 0..27719, which is within range.
    return int(best_m)

if __name__ == "__main__":
    construct()
