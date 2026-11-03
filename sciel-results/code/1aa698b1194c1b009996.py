import itertools
import time
import sys

def construct(n):
    if n <= 4:
        # Classic construction is optimal for small n
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
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # Convert to strings for faster comparison/slicing
    perm_strs = ["".join(map(str, p)) for p in perms]
    
    # Build overlap matrix for O(1) lookup
    # ov_matrix[i][j] = largest k (0..n-1) such that perm_strs[i][n-k:] == perm_strs[j][:k]
    ov_matrix = [[0] * N for _ in range(N)]
    
    # Precompute suffixes and prefixes for faster comparison
    # Actually, for n=7, we can just compare substrings.
    # To speed up, we can group by suffixes/prefixes.
    
    # Let's just do the naive precomputation but optimized.
    # For each i, for each j, check overlaps from n-1 down to 1.
    
    # Optimization: Group permutations by their suffix of length k and prefix of length k.
    # suffix_map[k][s] = list of indices i such that perm_strs[i][n-k:] == s
    # prefix_map[k][p] = list of indices j such that perm_strs[j][:k] == p
    
    # Then for each i, for k from n-1 down to 1:
    #   s = perm_strs[i][n-k:]
    #   if s in prefix_map[k]:
    #       for j in prefix_map[k][s]:
    #           ov_matrix[i][j] = k
    #           break # Only need the largest k, but we are iterating k downwards so first hit is max?
    #           Wait, we want the max k for each pair (i,j).
    #           If we iterate k from n-1 down to 1, the first time we find a match for (i,j) is the max k.
    #           But we need to fill the matrix.
    #           Actually, we can just fill it.
    
    suffix_maps = [dict() for _ in range(n)]
    prefix_maps = [dict() for _ in range(n)]
    
    for k in range(1, n):
        for idx, s in enumerate(perm_strs):
            suf = s[n-k:]
            pre = s[:k]
            if suf not in suffix_maps[k]:
                suffix_maps[k][suf] = []
            suffix_maps[k][suf].append(idx)
            
            if pre not in prefix_maps[k]:
                prefix_maps[k][pre] = []
            prefix_maps[k][pre].append(idx)
    
    # Now fill ov_matrix
    # For each i, for each k from n-1 down to 1:
    #   s = perm_strs[i][n-k:]
    #   if s in prefix_maps[k]:
    #       for j in prefix_maps[k][s]:
    #           if ov_matrix[i][j] == 0: # Not set yet, so this is the max k
    #               ov_matrix[i][j] = k
    
    for i in range(N):
        s_i = perm_strs[i]
        for k in range(n-1, 0, -1):
            suf = s_i[n-k:]
            if suf in prefix_maps[k]:
                for j in prefix_maps[k][suf]:
                    if ov_matrix[i][j] == 0:
                        ov_matrix[i][j] = k
                    else:
                        # Already found a larger k? No, we are going from large to small.
                        # So if it's not 0, it was set by a larger k.
                        pass

    # Nearest Neighbor
    unvisited = set(range(N))
    unvisited.remove(0)
    path = [0]
    current = 0
    
    for _ in range(N - 1):
        best_next = -1
        best_overlap = -1
        for j in unvisited:
            ov = ov_matrix[current][j]
            if ov > best_overlap:
                best_overlap = ov
                best_next = j
        path.append(best_next)
        unvisited.remove(best_next)
        current = best_next
        
    # 2-Opt Improvement
    # Precompute edge costs? No, we have ov_matrix.
    # Cost of edge (u, v) is n - ov_matrix[u][v].
    
    # 2-Opt: Reverse segment [i, j]
    # Old edges: (path[i-1], path[i]) and (path[j], path[j+1])
    # New edges: (path[i-1], path[j]) and (path[i], path[j+1])
    # Internal edges are reversed.
    # Cost change = (cost(i-1, i) + cost(j, j+1) - cost(i-1, j) - cost(i, j+1))
    #               + sum_{k=i}^{j-1} (cost(path[k], path[k+1]) - cost(path[k+1], path[k]))
    
    # Let's do a few passes of 2-Opt.
    # To speed up, we can precompute the "asymmetry" of edges: asym[u][v] = cost(u,v) - cost(v,u) = ov[v][u] - ov[u][v]
    # Then the internal cost change is sum of asym[path[k+1]][path[k]] for k=i to j-1.
    # Wait, cost(u,v) - cost(v,u) = (n - ov[u][v]) - (n - ov[v][u]) = ov[v][u] - ov[u][v].
    # So we want to minimize total cost.
    # Delta = (cost(i-1, i) + cost(j, j+1) - cost(i-1, j) - cost(i, j+1))
    #        + sum_{k=i}^{j-1} (cost(path[k], path[k+1]) - cost(path[k+1], path[k]))
    
    # Let's implement this.
    
    def edge_cost(u, v):
        return n - ov_matrix[u][v]
        
    # Precompute asymmetry matrix?
    # asym[u][v] = ov_matrix[v][u] - ov_matrix[u][v]
    # This is O(N^2) memory and time. 25M ints. 200MB? Maybe too much.
    # But we can compute it on the fly?
    # The sum part is the bottleneck.
    # We can use prefix sums of asymmetry along the path.
    
    # Let's just do a simple 1-Opt (swap adjacent) first, which is O(N) per pass.
    # Swap i, i+1:
    # Old: (i-1, i), (i, i+1), (i+1, i+2)
    # New: (i-1, i+1), (i+1, i), (i, i+2)
    # Delta = cost(i-1, i+1) + cost(i+1, i) + cost(i, i+2)
    #        - (cost(i-1, i) + cost(i, i+1) + cost(i+1, i+2))
    
    improved = True
    max_iterations = 10
    iterations = 0
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        for i in range(1, N - 1):
            u = path[i-1]
            v = path[i]
            w = path[i+1]
            x = path[i+2] if i+2 < N else None
            
            old_cost = edge_cost(u, v) + edge_cost(v, w)
            if x is not None:
                old_cost += edge_cost(w, x)
                
            new_cost = edge_cost(u, w) + edge_cost(w, v)
            if x is not None:
                new_cost += edge_cost(v, x)
                
            if new_cost < old_cost:
                path[i], path[i+1] = path[i+1], path[i]
                improved = True
                
    # Build string from path
    res = perm_strs[path[0]]
    for i in range(1, N):
        ov = ov_matrix[path[i-1]][path[i]]
        res += perm_strs[path[i]][ov:]
    return res
