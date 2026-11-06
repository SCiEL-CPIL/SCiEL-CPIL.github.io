import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_n = 26
    best_edges = [[i, j] for i in range(best_n) for j in range(i + 1, best_n) if (j - i) % best_n % 3 == 1]
    
    target_n = 39
    
    def is_triangle_free(adj, n):
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    for w in adj[v]:
                        if w > v and u in adj[w]:
                            return False
        return True

    def has_independent_set_10(adj, n):
        # Greedy lower bound
        avail = set(range(n))
        count = 0
        while avail:
            # Pick vertex with min degree in avail
            v = min(avail, key=lambda x: len([y for y in adj[x] if y in avail]))
            count += 1
            if count >= 10:
                return True
            avail = avail - {v} - adj[v]
        return False

    for restart in range(100):
        if time.time() > deadline:
            break
            
        adj = [set() for _ in range(target_n)]
        
        # Randomly order vertices and edges
        vertices = list(range(target_n))
        random.shuffle(vertices)
        
        edges = []
        for i in range(target_n):
            for j in range(i+1, target_n):
                edges.append((i, j))
        random.shuffle(edges)
        
        for u, v in edges:
            # Check if adding (u, v) creates a triangle
            # Common neighbors of u and v
            common = adj[u] & adj[v]
            if not common:
                adj[u].add(v)
                adj[v].add(u)
                
        # Check if triangle-free (should be by construction)
        # Check alpha
        if not has_independent_set_10(adj, target_n):
            # Greedy says alpha < 10. 
            # We should verify with a more exact method if possible, 
            # but greedy is a good filter.
            # To be safe, let's do a quick exact check for this candidate.
            # If exact check passes, return.
            
            # Exact check for alpha >= 10
            # Use the recursive search from before but with a tight timeout
            def exact_check(adj, n, k, timeout):
                import sys
                sys.setrecursionlimit(10000)
                start_t = time.time()
                
                def search(avail, count):
                    if count == k:
                        return True
                    if not avail:
                        return False
                    if time.time() > start_t + timeout:
                        return None # Timeout
                    v = min(avail, key=lambda x: len([y for y in adj[x] if y in avail]))
                    new_avail = avail - {v} - adj[v]
                    res = search(new_avail, count + 1)
                    if res is True:
                        return True
                    if res is None:
                        return None
                    res = search(avail - {v}, count)
                    if res is True:
                        return True
                    if res is None:
                        return None
                    return False
                
                return search(set(range(n)), 0)

            res = exact_check(adj, target_n, 10, 5.0)
            if res is False: # No independent set of size 10
                edges_list = [[i, j] for i in range(target_n) for j in adj[i] if i < j]
                return {'n': target_n, 'edges': edges_list}
            elif res is None:
                # Timeout, assume it might be valid? Or invalid?
                # If we assume valid and it's invalid, score 0.
                # If we assume invalid, we might miss a valid one.
                # Given the rarity of valid graphs, missing one is better than submitting invalid.
                pass

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
