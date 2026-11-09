import numpy as np

def construct():
    n = 512
    max_size = 48
    
    # Conflict graph approach: pre-compute mean relationships and use greedy
    # independent set selection based on conflict degrees.
    
    # Pre-compute: for each a, how many pairs {x,y} have a as midpoint
    midpoint_degree = {}
    for a in range(1, n + 1):
        deg = 0
        for x in range(1, n + 1):
            if x == a:
                continue
            y = 2 * a - x
            if y != a and y != x and 1 <= y <= n:
                deg += 1
        midpoint_degree[a] = deg
    
    def is_valid_addition(selected, candidate):
        """Check if adding candidate to selected keeps the set valid."""
        new_set = set(selected) | {candidate}
        new_list = list(new_set)
        
        for a in new_list:
            others = [x for x in new_list if x != a]
            if len(others) < 2:
                continue
            
            others_set = set(others)
            
            # k=2: check if a is midpoint of any pair in others
            for x in others:
                y = 2 * a - x
                if y != x and y in others_set:
                    return False
            
            # k=3: check if a is mean of any triple in others
            for x in others:
                rem = 3 * a - x
                for y in others:
                    if y == x:
                        continue
                    z = rem - y
                    if z != x and z != y and z in others_set:
                        return False
            
            # k=4: use pair-sums
            if len(others) >= 4:
                pair_sums = {}
                for i, x in enumerate(others):
                    for j in range(i + 1, len(others)):
                        y = others[j]
                        s = x + y
                        if s not in pair_sums:
                            pair_sums[s] = []
                        pair_sums[s].append((i, j))
                
                for s, pairs in pair_sums.items():
                    complement = 4 * a - s
                    if complement in pair_sums:
                        for (i1, j1) in pairs:
                            for (i2, j2) in pair_sums[complement]:
                                if len({i1, j1, i2, j2}) == 4:
                                    return False
            
            # For k >= 5, use DP
            m = len(others)
            if m >= 5:
                max_k = min(m, 15)
                dp = [set() for _ in range(max_k + 1)]
                dp[0].add(0)
                for x in others:
                    for k in range(max_k - 1, -1, -1):
                        if dp[k]:
                            for s in list(dp[k]):
                                dp[k + 1].add(s + x)
                for k in range(2, max_k + 1):
                    if k * a in dp[k]:
                        return False
                
                # Check remaining k values if m > max_k
                if m > max_k:
                    dp_full = [set() for _ in range(m + 1)]
                    dp_full[0].add(0)
                    for x in others:
                        for k in range(m - 1, -1, -1):
                            if dp_full[k]:
                                for s in list(dp_full[k]):
                                    dp_full[k + 1].add(s + x)
                    for k in range(max_k + 1, m + 1):
                        if k * a in dp_full[k]:
                            return False
        
        return True
    
    # Greedy independent set: select in order of increasing conflict degree
    candidates_sorted = sorted(range(1, n + 1), key=lambda x: midpoint_degree[x])
    
    selected = []
    for x in candidates_sorted:
        if len(selected) >= max_size:
            break
        if is_valid_addition(selected, x):
            selected.append(x)
    
    if len(selected) >= max_size:
        return selected[:max_size]
    
    # Try descending order as fallback
    selected = []
    for x in range(n, 0, -1):
        if len(selected) >= max_size:
            break
        if is_valid_addition(selected, x):
            selected.append(x)
    
    if len(selected) >= max_size:
        return selected[:max_size]
    
    # Try ascending order
    selected = []
    for x in range(1, n + 1):
        if len(selected) >= max_size:
            break
        if is_valid_addition(selected, x):
            selected.append(x)
    
    return selected[:max_size]
