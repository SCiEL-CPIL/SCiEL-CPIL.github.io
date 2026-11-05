"""Combined program: Exact MWC via B&B with Coloring Bounds."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    N = 512
    
    # 1. Precomputation
    # Sieve for primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return is_prime
    
    is_prime = sieve(511)
    
    # Prime factors
    prime_factors = [set() for _ in range(N)]
    for p in range(2, N):
        if is_prime[p]:
            for multiple in range(p, N, p):
                prime_factors[multiple].add(p)
                
    # Weights
    weights = [0.0] * N
    for a in range(1, N):
        weights[a] = 1.0 / (512 - a)
        
    # Adjacency Bitmasks (Coprime Graph)
    # adj[i] has bit j set if gcd(i, j) == 1
    adj = [0] * N
    for i in range(1, N):
        for j in range(i+1, N):
            if gcd(i, j) == 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Sort nodes by weight descending
    nodes = list(range(1, N))
    nodes.sort(key=lambda x: weights[x], reverse=True)
    
    best_score = 0.0
    best_set = []
    
    # Greedy Initial Solution
    def greedy_initial():
        current_clique = 0
        current_score = 0.0
        selected = []
        for v in nodes:
            if (adj[v] & current_clique) == current_clique:
                current_clique |= (1 << v)
                current_score += weights[v]
                selected.append(v)
        return selected, current_score
        
    best_set, best_score = greedy_initial()
    
    # Coloring Bound
    def color_bound(cands):
        if not cands:
            return 0.0
        if len(cands) < 10:
            return sum(weights[v] for v in cands)
            
        # Greedy coloring
        # cands is sorted by weight desc
        colors = {}
        max_color = 0
        # Optimization: Only check against previously colored nodes?
        # No, coloring requires checking all neighbors in the subgraph.
        # But we can iterate through cands in order.
        
        for v in cands:
            used_colors = set()
            for u in cands:
                if u == v:
                    continue
                if (adj[v] >> u) & 1:
                    if u in colors:
                        used_colors.add(colors[u])
            c = 1
            while c in used_colors:
                c += 1
            colors[v] = c
            if c > max_color:
                max_color = c
                
        max_w_per_color = [0.0] * (max_color + 1)
        for v in cands:
            c = colors[v]
            if weights[v] > max_w_per_color[c]:
                max_w_per_color[c] = weights[v]
                
        return sum(max_w_per_color)

    # Branch and Bound
    def dfs(cands, current_clique, current_score, depth):
        nonlocal best_score, best_set
        
        if time.time() > deadline:
            return
            
        if not cands:
            if current_score > best_score:
                best_score = current_score
                # Reconstruct set from mask
                s = []
                mask = current_clique
                # We can just keep the list, but mask is passed.
                # Actually, passing the list is safer for reconstruction if we don't want to iterate 512 bits.
                # But 512 bits is fast.
                for i in range(1, N):
                    if (current_clique >> i) & 1:
                        s.append(i)
                best_set = s
            return
            
        # Upper Bound
        # If small, use sum. Else, use coloring.
        if len(cands) < 30:
            ub = sum(weights[v] for v in cands)
        else:
            ub = color_bound(cands)
            
        if current_score + ub <= best_score + 1e-15:
            return
            
        # Iterate candidates
        # cands is sorted by weight desc
        for idx in range(len(cands)):
            if time.time() > deadline:
                return
                
            v = cands[idx]
            
            # Check if v is coprime to current clique
            if (adj[v] & current_clique) != current_clique:
                continue
                
            # New candidates: 
            # Must be coprime to v, and from cands[idx+1:]
            # (We only consider nodes after v in the sorted list to avoid duplicates)
            
            new_cands = []
            # Optimization: Filter cands[idx+1:] by adjacency to v
            # Since cands is sorted, we just iterate.
            for u in cands[idx+1:]:
                if (adj[v] >> u) & 1:
                    new_cands.append(u)
                    
            dfs(new_cands, current_clique | (1 << v), current_score + weights[v], depth + 1)
            
            # The loop continues to explore branches where v is NOT included
            # (implicitly, by considering the next candidate as the "first" included node in that branch)
            
    # Start B&B
    # We start with the full list of nodes
    dfs(nodes, 0, 0.0, 0)
    
    return best_set
# EVOLVE-BLOCK-END
