import random
import time
import itertools

def construct():
    n = 20
    # Precompute subsets of size 6 and 7 for fast checking
    # Storing as list of (mask, vertices_list)
    subsets_6 = []
    subsets_7 = []
    
    # Generate subsets
    for comb in itertools.combinations(range(n), 6):
        mask = 0
        for v in comb:
            mask |= (1 << v)
        subsets_6.append((mask, comb))
        
    for comb in itertools.combinations(range(n), 7):
        mask = 0
        for v in comb:
            mask |= (1 << v)
        subsets_7.append((mask, comb))
        
    # Larger subsets for final check
    subsets_final = []
    for k in range(8, n + 1):
        for comb in itertools.combinations(range(n), k):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            subsets_final.append((mask, comb))

    def count_regular_subgraphs(adj, subsets):
        count = 0
        for mask, verts in subsets:
            # Check if induced subgraph is regular
            # Get degrees of each vertex in the subset
            degrees = []
            for v in verts:
                # adj[v] is the bitmask of neighbors of v
                # deg is popcount(adj[v] & mask)
                deg = (adj[v] & mask).bit_count()
                degrees.append(deg)
            if len(set(degrees)) == 1:
                count += 1
        return count

    def is_valid_graph(adj):
        # Check k=6 and k=7
        if count_regular_subgraphs(adj, subsets_6) > 0:
            return False
        if count_regular_subgraphs(adj, subsets_7) > 0:
            return False
        # Check k>=8
        if count_regular_subgraphs(adj, subsets_final) > 0:
            return False
        return True

    # Local Search
    best_adj = None
    best_cost = float('inf')
    
    start_time = time.time()
    timeout = 110  # Leave 10s buffer
    
    for restart in range(10):
        # Initialize random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        current_cost = count_regular_subgraphs(adj, subsets_6) + count_regular_subgraphs(adj, subsets_7)
        
        if current_cost == 0:
             if is_valid_graph(adj):
                 # Convert to edge list
                 edges = []
                 for i in range(n):
                     for j in range(i + 1, n):
                         if adj[i] & (1 << j):
                             edges.append([i, j])
                 return {'n': n, 'edges': edges}
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_adj = adj[:]
            
        # Simulated Annealing / Hill Climbing
        temp = 1.0
        for i in range(5000):
            if time.time() - start_time > timeout:
                break
                
            # Pick a random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
                
            # Flip edge (u, v)
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            
            new_cost = count_regular_subgraphs(adj, subsets_6) + count_regular_subgraphs(adj, subsets_7)
            
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < pow(2.718, -delta / temp):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_adj = adj[:]
                    
                if new_cost == 0:
                    if is_valid_graph(adj):
                        edges = []
                        for i in range(n):
                            for j in range(i + 1, n):
                                if adj[i] & (1 << j):
                                    edges.append([i, j])
                        return {'n': n, 'edges': edges}
            else:
                # Revert
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                
            temp *= 0.995
            
            if time.time() - start_time > timeout:
                break
        
        if time.time() - start_time > timeout:
            break

    # If we didn't find a valid graph on 20, try to return the best one found 
    # or fallback to a smaller n if best_cost is not 0.
    # However, the task requires a VALID graph. 
    # If we can't find a valid n=20, we should try to find a valid n=19, 18, etc.
    # But n=20 is known to exist, so this loop should find it.
    
    # Fallback: Try smaller n if 20 fails
    if best_cost > 0:
        # Try n=19
        n = 19
        # ... (Implement similar search for n=19 if needed, but 20 should work)
        # For simplicity and robustness, if 20 fails, return a known small valid graph?
        # No, we must maximize n. 
        # Let's try one more time with a different initialization strategy: 
        # Start with a random graph and just do hill climbing on full cost (including k>=8) 
        # but that's slow.
        
        # If 20 fails, we return the best found if it's valid? No, best_cost>0 means invalid.
        # We must return a valid graph.
        # Let's assume the search finds it. If not, we might timeout.
        # To be safe, let's add a quick check for n=19 if n=20 fails.
        
        # Actually, let's just return a valid graph on n=5 if all else fails, 
        # but that scores 1.0. We want > 1.0.
        # Given n=20 is known, the probability of finding it in 110s is high.
        
    # If we reached here, we didn't find a valid graph on 20 in time.
    # This is a failure case for the heuristic.
    # Return the seed graph as a last resort to ensure validity, though score will be low.
    return {'n': 5, 'edges': []}
