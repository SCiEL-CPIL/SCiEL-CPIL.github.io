import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _compute_lcm_range(n):
    result = 1
    for i in range(1, n + 1):
        result = _lcm(result, i)
    return result

def _has_perfect_matching(m, L, n=12):
    """
    Check if there's a perfect matching in interval (m, m+L] for indices 1..n.
    Left side: indices 1..n
    Right side: positions 1..L (representing numbers m+1 to m+L)
    Edge (i, j) exists if (m + j) % i == 0
    """
    # adj[i] is a bitmask of valid positions j (1..L) for index i
    # We use 0-indexed bits for positions, so position j corresponds to bit (j-1)
    adj = [0] * (n + 1)
    
    # Precompute masks for each i
    for i in range(1, n + 1):
        mask = 0
        # Find first multiple of i in (m, m+L]
        # smallest multiple > m is m + (i - m % i) % i
        # but simpler: iterate j or calculate
        # j such that m+j is divisible by i
        # (m+j) % i == 0 => j % i == (-m) % i
        rem = (-m) % i
        # j can be rem, rem+i, rem+2i, ...
        # j must be in 1..L
        # if rem == 0, j can be i, 2i, ... (since j>=1)
        # if rem != 0, j can be rem, rem+i, ...
        
        # Actually, just iterating j is fine for small L
        for j in range(1, L + 1):
            if (m + j) % i == 0:
                mask |= (1 << (j - 1))
        adj[i] = mask
        if mask == 0:
            return False

    # DFS-based bipartite matching (Kuhn's algorithm)
    # match_pos[k] = which index i is matched to position k (0-indexed pos)
    # We want to match all indices 1..n to distinct positions
    # Let's match indices to positions.
    # match[i] = position j (0-indexed) that index i is matched to
    match = [-1] * (n + 1)
    
    def dfs(i, seen):
        for j in range(L):
            if (adj[i] & (1 << j)) and not seen[j]:
                seen[j] = True
                if match[j] == -1 or dfs(match[j], seen):
                    match[j] = i
                    return True
        return False

    # We need to match all n indices.
    # Standard Kuhn's: iterate over left nodes, try to find augmenting path.
    # But here we have fixed left size n and right size L.
    # We want to see if we can match all n left nodes.
    
    # Initialize match for positions
    # match_pos[j] = i
    # Let's stick to matching left to right.
    # We'll use a standard approach:
    # for each left node u, try to find a matching.
    
    # Actually, the above `dfs` is slightly modified for our needs.
    # Let's use a cleaner implementation.
    
    # match_right[j] = i
    match_right = [-1] * L
    
    def try_match(i, visited):
        for j in range(L):
            if (adj[i] & (1 << j)) and not visited[j]:
                visited[j] = True
                if match_right[j] == -1 or try_match(match_right[j], visited):
                    match_right[j] = i
                    return True
        return False
    
    matched_count = 0
    for i in range(1, n + 1):
        visited = [False] * L
        if try_match(i, visited):
            matched_count += 1
        else:
            return False
            
    return matched_count == n

def construct():
    n = 12
    max_start = 1000000
    period = _compute_lcm_range(n)
    
    best_L = 0
    best_m_residues = []
    
    # Precompute f(12, m) for m in [0, period)
    for m in range(period):
        # f(12, m) is the least L such that matching exists
        # L is at most period? Actually much smaller.
        # We can bound L. The maximum gap for 12 distinct multiples is small.
        # Let's search L from 1 upwards.
        # Upper bound for L: In worst case, we might need to go further.
        # But since we just need the max, we can stop if we find a very large one?
        # No, we need exact f(12, m) to compare.
        # However, we know f(12, m) <= 144 from the prompt.
        # So we can just loop L from 1 to 150.
        
        found = False
        for L in range(1, 150):
            if _has_perfect_matching(m, L, n):
                if L > best_L:
                    best_L = L
                    best_m_residues = [m]
                elif L == best_L:
                    best_m_residues.append(m)
                found = True
                break
        
        # If not found in 150, it's unlikely given the prompt, but let's be safe
        if not found:
            # This shouldn't happen for n=12, but if it does, L is larger
            # We can extend search, but 150 is safe based on "at most 144"
            pass

    # Find the largest m <= max_start such that m % period is in best_m_residues
    best_m = -1
    for res in best_m_residues:
        # m = k * period + res
        # We want max m <= max_start
        # k = (max_start - res) // period
        if max_start >= res:
            k = (max_start - res) // period
            m_candidate = k * period + res
            if m_candidate > best_m:
                best_m = m_candidate
    return best_m
