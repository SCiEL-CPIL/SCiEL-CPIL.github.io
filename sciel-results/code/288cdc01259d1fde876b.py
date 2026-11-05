import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Helper to convert bitmask adj to edge list
    def bitmask_to_edges(n, adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    # Check if graph has any K4
    def has_k4(n, adj):
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        c ^= lsb
                        rest = c
                        if (adj[w] & rest):
                            return True
        return False

    # Check if graph has independent set of size 6
    def has_ind6(n, adj):
        # Find independent set of size 6 via backtracking
        def find_is(candidates, size_needed):
            if size_needed == 0:
                return True
            if len(candidates) < size_needed:
                return False
            if not candidates:
                return False
            
            # Heuristic: pick vertex with highest degree in candidate subgraph
            # to prune more
            v = candidates[0]
            # New candidates must be non-neighbors of v in G
            new_candidates = [x for x in candidates[1:] if not ((adj[v] >> x) & 1)]
            if find_is(new_candidates, size_needed - 1):
                return True
            
            # Try excluding v
            return find_is(candidates[1:], size_needed)
        
        return find_is(list(range(n)), 6)

    # Penalty: number of K4s + number of independent 6-sets (approximate)
    def penalty(n, adj):
        k4_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        c ^= lsb
                        rest = c
                        k4_count += (adj[w] & rest).bit_count()
        k4_count //= 6
        
        # Count independent 6-sets (K6 in complement)
        ind6_count = 0
        mask = (1 << n) - 1
        # Build complement adjacency
        comp_adj = [0] * n
        for i in range(n):
            comp_adj[i] = mask & ~(adj[i] | (1 << i))
        
        # Count K6 in complement using similar logic
        for u in range(n):
            for v in range(u + 1, n):
                if (comp_adj[u] >> v) & 1:
                    common = comp_adj[u] & comp_adj[v]
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        c ^= lsb
                        rest = c
                        # Count K4 in complement within rest that include w
                        for w2 in range(w + 1, n):
                            if (rest >> w2) & 1 and (comp_adj[w] >> w2) & 1:
                                common2 = comp_adj[w] & comp_adj[w2] & rest
                                c2 = common2
                                while c2:
                                    lsb2 = c2 & -c2
                                    w3 = lsb2.bit_length() - 1
                                    c2 ^= lsb2
                                    rest2 = c2
                                    if (comp_adj[w3] & rest2):
                                        ind6_count += 1
        ind6_count //= 120  # Each K6 counted 6! / 2 = 360 times? Let me recalculate.
        # Actually this counting is getting complex. Let's just use boolean checks for validity
        # and use a simpler penalty.
        
        # Simpler: just check validity
        if has_k4(n, adj):
            return 1000
        if has_ind6(n, adj):
            return 100
        return 0

    # Local search: flip edges to minimize violations
    def local_search(n, adj, max_steps=5000):
        current_adj = adj[:]
        current_penalty = penalty(n, current_adj)
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            if current_penalty == 0:
                return current_adj, 0
            
            # Try flipping a random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            # Flip edge (u,v)
            if (current_adj[u] >> v) & 1:
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
            else:
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
            
            new_penalty = penalty(n, current_adj)
            
            if new_penalty <= current_penalty:
                current_penalty = new_penalty
            else:
                # Revert
                if (current_adj[u] >> v) & 1:
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
                else:
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
        
        return current_adj, current_penalty

    # Try to find valid graphs for increasing n
    best_n = 17
    best_adj = None
    
    # Start with seed graph for n=17
    squares = {x * x % 17 for x in range(1, 17)}
    adj17 = [0] * 17
    for i in range(17):
        for j in range(i + 1, 17):
            if (j - i) % 17 in squares:
                adj17[i] |= (1 << j)
                adj17[j] |= (1 << i)
    best_adj = adj17
    
    # Try to extend to larger n
    for target_n in [18, 19, 20, 22, 25, 28, 30, 32, 35]:
        if time.time() > deadline - 15:
            break
        
        # Initialize: start from best known and add vertices
        if best_adj is not None and best_n < target_n:
            # Extend graph
            new_adj = [0] * target_n
            for i in range(best_n):
                new_adj[i] = best_adj[i]
            # Connect new vertices randomly
            for i in range(best_n, target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        new_adj[i] |= (1 << j)
                        new_adj[j] |= (1 << i)
                # Connect to old vertices
                for j in range(best_n):
                    if random.random() < 0.5:
                        new_adj[i] |= (1 << j)
                        new_adj[j] |= (1 << i)
        else:
            # Random initialization
            new_adj = [0] * target_n
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        new_adj[i] |= (1 << j)
                        new_adj[j] |= (1 << i)
        
        # Local search
        result_adj, result_penalty = local_search(target_n, new_adj, max_steps=3000)
        
        if result_penalty == 0:
            best_n = target_n
            best_adj = result_adj
            print(f"Found valid graph with n={target_n}")
    
    # Return best found
    if best_adj is not None:
        edges = bitmask_to_edges(best_n, best_adj)
        return {'n': best_n, 'edges': edges}
    
    # Fallback
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
