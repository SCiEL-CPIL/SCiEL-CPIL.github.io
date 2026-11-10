"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Deadline guard: 90s limit, use 80s to be safe
    deadline = time.time() + 80
    
    # Precompute modular inverses for 1..10 mod 101
    # inv[x] = x^{-1} mod 101
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, 101)
        
    # Precompute all non-empty subsets of {1, ..., 10}
    # There are 2^10 - 1 = 1023 subsets.
    subsets = []
    for mask in range(1, 1 << 10):
        terms = []
        s = 0
        for x in range(1, 11):
            if mask & (1 << (x - 1)):
                terms.append(x)
                s = (s + inv[x]) % 101
        subsets.append((s, len(terms), terms))
    
    # Sort subsets by length (ascending), then by sum for determinism
    subsets.sort(key=lambda x: (x[1], x[0]))
    
    # Hopcroft-Karp algorithm for maximum bipartite matching
    # Left side: residues 0..100
    # Right side: subsets (indices 0..1022)
    # Edge exists if subset sum == residue
    
    # Build adjacency list: for each residue, which subsets can represent it
    # For efficiency, group subsets by their sum
    from collections import defaultdict
    sum_to_subsets = defaultdict(list)
    for idx, (s, length, terms) in enumerate(subsets):
        sum_to_subsets[s].append(idx)
    
    # For each residue, the available subsets are those with sum == residue
    # We want to assign distinct subsets to residues to minimize the maximum length
    
    # Binary search on k: is it possible to assign subsets of length <= k to all 101 residues?
    # Actually, since we want to minimize the maximum length, we can try k = 1, 2, ..., 10
    # and find the smallest k such that a perfect matching exists using only subsets of length <= k
    
    # Hopcroft-Karp implementation
    # U = residues (0..100), V = subset indices (0..1022)
    # matchU[u] = matched v or -1
    # matchV[v] = matched u or -1
    
    def hopcroft_karp(max_len):
        # Filter subsets to only those with length <= max_len
        # Build adjacency: for each residue u, list of subset indices v where sum==u and len<=max_len
        adj = [[] for _ in range(101)]
        for r in range(101):
            for idx in sum_to_subsets.get(r, []):
                if subsets[idx][1] <= max_len:
                    adj[r].append(idx)
        
        n_u = 101
        n_v = len(subsets)
        
        matchU = [-1] * n_u
        matchV = [-1] * n_v
        dist = [0] * n_u
        
        from collections import deque
        
        def bfs():
            queue = deque()
            for u in range(n_u):
                if matchU[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = float('inf')
            
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    mu = matchV[v]
                    if mu == -1:
                        found = True
                    elif dist[mu] == float('inf'):
                        dist[mu] = dist[u] + 1
                        queue.append(mu)
            return found
        
        def dfs(u):
            for v in adj[u]:
                mu = matchV[v]
                if mu == -1 or (dist[mu] == dist[u] + 1 and dfs(mu)):
                    matchU[u] = v
                    matchV[v] = u
                    return True
            dist[u] = float('inf')
            return False
        
        matching = 0
        while bfs():
            for u in range(n_u):
                if matchU[u] == -1:
                    if dfs(u):
                        matching += 1
        
        return matching == n_u, matchU
    
    # Try increasing max_len from 1 to 10
    best_matchU = None
    for k in range(1, 11):
        if time.time() > deadline - 5:
            break
        success, matchU = hopcroft_karp(k)
        if success:
            best_matchU = matchU
            break
    
    # If we found a perfect matching, use it
    if best_matchU is not None:
        result = []
        for r in range(101):
            v = best_matchU[r]
            result.append(subsets[v][2])
        return result
    
    # Fallback: greedy assignment (original approach)
    best_rep = [None] * 101
    for s, length, terms in subsets:
        if best_rep[s] is None or length < len(best_rep[s]):
            best_rep[s] = terms
    
    result = []
    for r in range(101):
        if best_rep[r] is None:
            best_rep[r] = [1]
        result.append(best_rep[r])
    
    return result
# EVOLVE-BLOCK-END
