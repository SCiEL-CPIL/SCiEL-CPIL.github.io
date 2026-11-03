import time
import itertools
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "1231213"
    if n == 4:
        return "12341231421341231423124312"
    if n == 5:
        # Standard construction is optimal for 5
        return _standard_superperm(5)
    if n == 6:
        # Standard construction is optimal for 6
        return _standard_superperm(6)
    
    if n == 7:
        # Try to find a shorter path using TSP heuristic on permutations of 1..6
        # If time runs out, fallback to standard
        try:
            result = _tsp_superperm(7, deadline)
            if result is not None:
                return result
        except Exception:
            pass
        return _standard_superperm(7)
    
    # Fallback for other n (not expected in this task)
    return _standard_superperm(n)

def _standard_superperm(n):
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Calculate overlap
            overlap = 0
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))

def _tsp_superperm(n, deadline):
    # This function attempts to find a shorter superpermutation for n=7
    # by optimizing the order of permutations of 1..6
    
    inner_n = n - 1
    symbols = list(range(1, inner_n + 1))
    perms = list(itertools.permutations(symbols))
    N = len(perms)
    
    # Precompute blocks: P + [n] + P
    blocks = []
    for p in perms:
        block = list(p) + [n] + list(p)
        blocks.append(block)
        
    # Calculate overlap cost matrix
    # cost[i][j] = length of block[j] - max_overlap(block[i], block[j])
    # We want to minimize total length, which is sum of (len(block[j]) - overlap)
    # Actually, total length = len(block[0]) + sum_{k=1}^{N-1} (len(block[k]) - overlap(block[k-1], block[k]))
    
    # To speed up overlap calculation, we can use string matching or simple loops
    # Since N=720, N^2 = 518400 pairs. Each overlap check is O(n).
    # 518400 * 14 ~ 7 million ops. This is fast.
    
    cost = [[0]*N for _ in range(N)]
    for i in range(N):
        b_i = blocks[i]
        s_i = "".join(map(str, b_i))
        for j in range(N):
            b_j = blocks[j]
            s_j = "".join(map(str, b_j))
            max_ov = 0
            # Max possible overlap is len(b_j) - 1 (can't be full match as they are distinct blocks mostly)
            # Actually, if b_i == b_j, overlap is len(b_j). But i!=j in TSP.
            limit = min(len(s_i), len(s_j))
            # Check suffix of s_i against prefix of s_j
            # We can use KMP or just simple slicing for small strings
            for ov in range(limit, -1, -1):
                if ov == 0:
                    max_ov = 0
                    break
                if s_i[len(s_i)-ov:] == s_j[:ov]:
                    max_ov = ov
                    break
            cost[i][j] = len(b_j) - max_ov
            
    # Initialize tour using Nearest Neighbor
    tour = [0]
    unvisited = set(range(1, N))
    current = 0
    while unvisited:
        next_node = min(unvisited, key=lambda x: cost[current][x])
        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node
        
    # 2-Opt Improvement
    def total_cost(tour):
        total = len(blocks[tour[0]])
        for i in range(len(tour) - 1):
            total += cost[tour[i]][tour[i+1]]
        return total
        
    best_tour = tour
    best_cost = total_cost(tour)
    
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        for i in range(N - 1):
            if time.time() > deadline - 10.0:
                break
            for j in range(i + 1, N):
                # Reverse segment tour[i+1..j]
                # Check if it reduces cost
                # Cost before: cost[tour[i]][tour[i+1]] + cost[tour[j]][tour[j+1]] (if j+1 exists)
                # Note: The path is not a cycle in the construction, it's a path.
                # But we can treat it as a path.
                
                # Indices in tour
                u = tour[i]
                v = tour[i+1]
                w = tour[j]
                x = tour[j+1] if j + 1 < N else None
                
                c1 = cost[u][v]
                c2 = cost[w][x] if x is not None else 0
                
                # New edges: u -> w and v -> x (reversed segment)
                # Wait, 2-opt on a path:
                # Original: ... u, v, ..., w, x ...
                # New: ... u, w, ..., v, x ...
                # Cost change:
                # Remove: cost[u][v] + cost[w][x] (if x exists)
                # Add: cost[u][w] + cost[v][x] (if x exists)
                
                new_c1 = cost[u][w]
                new_c2 = cost[v][x] if x is not None else 0
                
                if new_c1 + new_c2 < c1 + c2:
                    tour[i+1:j+1] = reversed(tour[i+1:j+1])
                    improved = True
                    best_cost = total_cost(tour)
                    
    # Construct the string
    final_word = list(blocks[best_tour[0]])
    for i in range(1, N):
        b_prev = blocks[best_tour[i-1]]
        b_curr = blocks[best_tour[i]]
        s_prev = "".join(map(str, b_prev))
        s_curr = "".join(map(str, b_curr))
        
        max_ov = 0
        limit = min(len(s_prev), len(s_curr))
        for ov in range(limit, -1, -1):
            if ov == 0:
                max_ov = 0
                break
            if s_prev[len(s_prev)-ov:] == s_curr[:ov]:
                max_ov = ov
                break
        final_word += b_curr[max_ov:]
        
    result_str = "".join(map(str, final_word))
    
    # Verify
    if not _verify_superperm(result_str, n):
        return None
        
    return result_str

def _verify_superperm(word, n):
    if len(word) < n:
        return False
    seen = set()
    for i in range(len(word) - n + 1):
        window = tuple(int(c) for c in word[i:i+n])
        if len(set(window)) == n:
            seen.add(window)
    # Check if all permutations are present
    # For n=7, there are 5040 perms.
    # We can just check the count if we assume no duplicates in valid windows?
    # No, better to check if len(seen) == n!
    import math
    return len(seen) == math.factorial(n)
