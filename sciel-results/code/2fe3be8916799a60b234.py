import random
import itertools
import time

def construct():
    max_n = 22
    start_time = time.time()
    time_limit = 115.0
    
    def is_valid_graph(adj, n):
        """
        Check if the graph defined by adj (list of int bitmasks) on n vertices
        has no regular induced subgraph on k >= 6 vertices.
        """
        # Iterate k from 6 to n
        for k in range(6, n + 1):
            for combo in itertools.combinations(range(n), k):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                
                # Check if G[mask] is regular
                # Get first vertex
                lsb = mask & -mask
                v0 = lsb.bit_length() - 1
                d0 = (adj[v0] & mask).bit_count()
                
                # Check all other vertices
                m = mask
                is_regular = True
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    if v != v0:
                        dv = (adj[v] & mask).bit_count()
                        if dv != d0:
                            is_regular = False
                            break
                    m ^= lsb
                
                if is_regular:
                    return False
        return True

    # Try to find a valid graph for n = 22, 21, 20
    for target_n in [22, 21, 20]:
        if time.time() - start_time > time_limit:
            break
            
        # Try different probabilities
        for p in [0.3, 0.4, 0.5, 0.6, 0.7]:
            if time.time() - start_time > time_limit:
                break
                
            # Try multiple random graphs
            for attempt in range(50):
                if time.time() - start_time > time_limit:
                    break
                
                adj = [0] * target_n
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if random.random() < p:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                
                if is_valid_graph(adj, target_n):
                    # Found a valid graph
                    edges = [[i, j] for i in range(target_n) for j in range(i + 1, target_n) if (adj[i] >> j) & 1]
                    return {'n': target_n, 'edges': edges}
                    
    # Fallback: return a small valid graph (e.g., n=5 or n=6 if we can find one quickly, or just n=1)
    # But we should try to return the best we found.
    # If we didn't find one for 20, we might have found one for 21 or 22 in previous loops? No, we break on first success.
    # If we failed all, we should return a trivial valid graph to avoid 0 score?
    # A graph with n=5 is always valid (no subset of size >= 6).
    # Let's return n=5.
    return {'n': 5, 'edges': []}
