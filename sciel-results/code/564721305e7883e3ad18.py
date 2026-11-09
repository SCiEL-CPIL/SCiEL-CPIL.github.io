import time
import math

def construct():
    start_time = time.time()
    deadline = 60.0  # Safety margin
    
    # Target: Maximize size of S subset [1, 255] containing 255, gcd(a,b)>1 for all a!=b in S.
    # 255 = 3 * 5 * 17.
    # All elements in S must share a factor with 255.
    
    n = 255
    target = n
    
    # Candidates: x in [1, n-1] such that gcd(x, n) > 1
    candidates = []
    for x in range(1, n):
        if math.gcd(x, n) > 1:
            candidates.append(x)
    
    # Map candidates to indices for bitset operations or adjacency list
    # We want to find max clique in this induced subgraph.
    # Size of candidates is ~126.
    
    # Build adjacency
    # adj[i] is a set of indices j such that gcd(candidates[i], candidates[j]) > 1
    m = len(candidates)
    adj = [set() for _ in range(m)]
    
    for i in range(m):
        ci = candidates[i]
        for j in range(i + 1, m):
            cj = candidates[j]
            if math.gcd(ci, cj) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Simple Branch and Bound for Max Clique
    # We use a greedy coloring heuristic for upper bound (Tomita's algorithm)
    # But for 126 nodes, even a simpler bound might work if we order well.
    # Let's try a standard recursive search with a simple bound: current_size + len(remaining)
    
    best_clique = []
    
    # Heuristic: Start with the known good solution (multiples of 3)
    # This helps with pruning if we use a lower bound.
    init_clique = [i for i, x in enumerate(candidates) if x % 3 == 0]
    # Verify init_clique is a clique
    is_clique = True
    for i in range(len(init_clique)):
        for j in range(i+1, len(init_clique)):
            if init_clique[j] not in adj[init_clique[i]]:
                is_clique = False
                break
        if not is_clique:
            break
    
    if is_clique:
        best_clique = init_clique.copy()
    
    # Order candidates by degree (descending) to help pruning
    degrees = [len(adj[i]) for i in range(m)]
    order = sorted(range(m), key=lambda i: -degrees[i])
    
    # Remap indices? Or just use the order in recursion.
    # To keep it simple and fast, let's just run a search.
    # Given the small size, we can afford a bit of computation.
    
    # Recursive function
    # P: list of candidate indices available
    # current: list of indices in current clique
    def expand(P, current):
        nonlocal best_clique
        if time.time() > deadline:
            return
        
        # Pruning: if len(current) + len(P) <= len(best_clique), return
        if len(current) + len(P) <= len(best_clique):
            return
        
        if not P:
            if len(current) > len(best_clique):
                best_clique = current.copy()
            return
        
        # Pick a vertex from P
        # Heuristic: pick vertex with highest degree in P
        # v_idx is index in 'order' list? No, let's pick from P directly.
        # To speed up, we can pre-sort P by degree in P.
        
        # Simple pick: first element
        v = P[0]
        P_rest = P[1:]
        
        # New P: neighbors of v in P_rest
        P_new = [u for u in P_rest if u in adj[v]]
        
        # Recurse
        expand(P_new, current + [v])
        
        # Remove v from P and recurse (exclude v)
        expand(P_rest, current)
        
    # Initialize P with all candidates
    # We should order P initially by degree
    P_init = order[:]
    
    # To make the search faster, we can use a better coloring bound.
    # But let's try this simple one first. If it's too slow, we fall back to heuristic.
    
    # Actually, 2^126 is too big for simple B&B without strong pruning.
    # The bound `len(current) + len(P)` is weak.
    # We need a coloring bound.
    
    # Let's implement a simple greedy coloring for the subgraph induced by P.
    def greedy_coloring(vertices, adj_sets):
        # vertices: list of node indices
        # Returns max color used (which is an upper bound on clique size in this subgraph)
        if not vertices:
            return 0
        
        colors = {}
        max_color = 0
        # Sort vertices by degree in the subgraph (descending)
        # For simplicity, just iterate
        for v in vertices:
            # Find smallest available color
            used_colors = set()
            for u in adj_sets[v]:
                if u in colors:
                    used_colors.add(colors[u])
            
            c = 1
            while c in used_colors:
                c += 1
            colors[v] = c
            if c > max_color:
                max_color = c
        
        return max_color

    def expand_color(P, current):
        nonlocal best_clique
        if time.time() > deadline:
            return
        
        # Coloring bound
        color_bound = greedy_coloring(P, adj)
        if len(current) + color_bound <= len(best_clique):
            return
        
        if not P:
            if len(current) > len(best_clique):
                best_clique = current.copy()
            return
        
        # Pick vertex v from P
        # Heuristic: pick vertex with highest degree in P
        # Calculate degrees in P
        degs = {}
        for v in P:
            degs[v] = len([u for u in P if u != v and u in adj[v]])
        
        v = max(P, key=lambda x: degs[x])
        P_rest = [u for u in P if u != v]
        
        # Neighbors of v in P_rest
        P_new = [u for u in P_rest if u in adj[v]]
        
        # Recurse with v included
        expand_color(P_new, current + [v])
        
        # Recurse with v excluded
        expand_color(P_rest, current)

    # Start search
    # We already have a good lower bound (84 nodes in subgraph, total 85)
    # Let's run the search.
    # To avoid infinite loops or slow paths, we rely on the deadline.
    
    # Note: The simple coloring might be slow if P is large.
    # But P shrinks quickly.
    
    # Let's run it.
    expand_color(P_init, [])
    
    # Construct result
    # best_clique contains indices into 'candidates'
    result = [candidates[i] for i in best_clique]
    result.append(target)
    
    # Sort for consistency
    result.sort()
    
    # Verify validity (optional, but good for safety)
    # if len(result) > 1:
    #    for i in range(len(result)):
    #        for j in range(i+1, len(result)):
    #            if math.gcd(result[i], result[j]) == 1:
    #                # Fallback to multiples of 3
    #                return list(range(3, 256, 3))
                
    return result

# Fallback if search fails or times out badly (shouldn't happen with deadline)
# The function above handles the deadline by returning current best.
