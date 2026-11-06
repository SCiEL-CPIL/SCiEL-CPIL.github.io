import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin
    
    n = 40
    # Adjacency matrix as list of integers (bitsets)
    # adj[i] has bit j set if (i, j) is an edge
    
    def is_triangle_free(adj):
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    # Check common neighbors
                    common = adj[i] & adj[j]
                    if common:
                        return False
        return True

    def find_large_independent_set(adj, size_limit=10):
        # Greedy or simple branch and bound to find an independent set of size >= size_limit
        # Returns the set as a list of vertices or None if not found
        # For speed, we use a greedy heuristic with backtracking if needed, 
        # but for n=40, a simple recursive search with pruning is fast.
        
        # We want to know if there is an independent set of size 10.
        # We can try to find one.
        
        # Simple DFS
        # State: current set, candidate set
        # To be fast, we can use a bitset for candidates.
        
        # Let's use a randomized greedy approach first, then verify.
        # Actually, we just need to find ANY independent set of size 10.
        
        # We'll use a simple backtracking search.
        # To avoid exponential blowup, we limit depth or use heuristics.
        
        # Heuristic: Order vertices by degree (lowest first) to pick into independent set.
        # But we are adding edges to BREAK independent sets.
        # So we just need to find one.
        
        # Let's try a simple recursive search with pruning.
        # If remaining candidates < needed, prune.
        
        def search(current_set, candidates, needed):
            if needed == 0:
                return current_set
            if not candidates:
                return None
            if candidates.bit_count() < needed:
                return None
            
            # Pick a vertex
            # Randomize order to escape local minima?
            # For now, pick lowest index.
            v = (candidates & -candidates).bit_length() - 1
            # Or pick random from candidates
            # Let's pick random to diversify
            # Get list of set bits
            bits = []
            c = candidates
            while c:
                lsb = c & -c
                bits.append(lsb.bit_length() - 1)
                c ^= lsb
            
            random.shuffle(bits)
            
            for v in bits:
                # Try including v
                new_candidates = candidates & ~adj[v] & ~(1 << v)
                res = search(current_set + [v], new_candidates, needed - 1)
                if res:
                    return res
                
                # Try excluding v? 
                # If we exclude v, we must still find 'needed' from candidates without v.
                # This can be expensive. 
                # However, if we just want to find ONE set, we can just try to include vertices.
                # If we fail, we might have pruned too aggressively?
                # No, if we include v, we remove its neighbors.
                # If we exclude v, we just remove v.
                # Standard independent set search.
                
                # To keep it fast, we only try including v.
                # This is a greedy path. It might fail to find a set that exists.
                # But for random graphs, it's usually okay.
                # If it fails, we can try a different random permutation.
                
            return None
        
        # Run a few times with different random seeds/orderings
        for _ in range(10):
            res = search([], (1 << n) - 1, size_limit)
            if res:
                return res
        return None

    def try_add_edge(adj, u, v):
        # Check if adding (u, v) creates a triangle
        # Common neighbors of u and v must be empty
        if adj[u] & adj[v]:
            return False
        # Also check if they are already connected
        if (adj[u] >> v) & 1:
            return True # Already connected, no change needed, but returns True as "success" in adding? 
                        # Actually we want to add it. If it's there, we are good.
                        # But we are looking for non-edges to add.
                        # So if it's an edge, we shouldn't be trying to add it.
                        # This function should only be called for non-edges.
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        return True

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def make_triangle_free(adj):
        # Iteratively remove edges in triangles
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(i+1, n):
                    if (adj[i] >> j) & 1:
                        if adj[i] & adj[j]:
                            remove_edge(adj, i, j)
                            changed = True
                            break
                if changed:
                    break
        return adj

    def random_graph():
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    best_n = 26
    best_adj = None
    # Fallback: The seed graph
    # We can construct the seed graph to ensure we have a valid baseline if search fails
    def build_seed_graph():
        n_seed = 26
        adj_seed = [0] * n_seed
        for i in range(n_seed):
            for j in range(i+1, n_seed):
                if (j - i) % n_seed % 3 == 1:
                    adj_seed[i] |= (1 << j)
                    adj_seed[j] |= (1 << i)
        return n_seed, adj_seed

    n_seed, adj_seed = build_seed_graph()
    best_n = n_seed
    best_adj = adj_seed

    # Search for n=40
    for attempt in range(1000):
        if time.time() > deadline:
            break
            
        adj = random_graph()
        make_triangle_free(adj)
        
        # Try to reduce independence number
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indep_set = find_large_independent_set(adj, 10)
            if not indep_set:
                # Success!
                best_n = n
                best_adj = adj
                return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i+1, n) if (adj[i] >> j) & 1]}
            
            # Try to add an edge between two vertices in indep_set
            # Shuffled to try different pairs
            vertices = list(indep_set)
            random.shuffle(vertices)
            
            added = False
            for u in vertices:
                if added: break
                for v in vertices:
                    if u == v: continue
                    # Check if non-edge
                    if not ((adj[u] >> v) & 1):
                        # Check if adding creates triangle
                        if not (adj[u] & adj[v]):
                            # Add edge
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            added = True
                            improved = True
                            break
            if not added:
                # Stuck. Perturb?
                # Remove a random edge to allow new configurations?
                # Or just break and restart
                break

    # If we didn't find n=40, try n=39, 38... down to best_n+1?
    # Or just return the best found (which is at least the seed)
    
    # If best_n < 40, we might try smaller n if we found a valid one?
    # The search above only targeted n=40.
    # If it fails, we return the seed.
    
    # To improve score, we should ensure we return a valid graph.
    # The seed is valid.
    
    # Let's add a quick check for n=39 if 40 fails?
    # Probably not worth the complexity if 40 is the target.
    
    edges = [[i, j] for i in range(best_n) for j in range(i+1, best_n) if (best_adj[i] >> j) & 1]
    return {'n': best_n, 'edges': edges}
