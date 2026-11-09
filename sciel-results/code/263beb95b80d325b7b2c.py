import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    n = 20
    best_edges = 0
    best_adj = None
    
    def can_add(u, v, adj):
        """
        Returns True if adding edge (u, v) keeps the graph K3,3-free.
        Assumes u and v are not currently connected.
        """
        # 1. Check triples (u, x, y) and (v, x, y) for x < y, x,y != u,v
        for x in range(n):
            if x == u or x == v:
                continue
            for y in range(x + 1, n):
                if y == u or y == v:
                    continue
                
                # Check if v is a common neighbor of x and y
                v_in_xy = (adj[x] >> v) & 1 and (adj[y] >> v) & 1
                # Check if u is a common neighbor of x and y
                u_in_xy = (adj[x] >> u) & 1 and (adj[y] >> u) & 1
                
                if v_in_xy:
                    # Adding (u,v) adds v to N(u).
                    # For triple (u, x, y), v is a new common neighbor if v in N(x) and N(y).
                    # Count of common neighbors for (u,x,y) becomes:
                    # old_count(u,x,y) + 1 (for v)
                    # old_count(u,x,y) = |N(u) & N(x) & N(y)|
                    count_uxy = (adj[u] & adj[x] & adj[y]).bit_count()
                    if count_uxy + 1 >= 3:
                        return False
                    
                    # For triple (v, x, y), u is a new common neighbor if u in N(x) and N(y).
                    # But we are in the case v_in_xy. Does u_in_xy matter for (v,x,y)?
                    # Adding (u,v) adds u to N(v).
                    # If u is in N(x) and N(y), then u becomes a common neighbor of (v,x,y).
                    # Count of common neighbors for (v,x,y) becomes:
                    # old_count(v,x,y) + (1 if u_in_xy else 0)
                    count_vxy = (adj[v] & adj[x] & adj[y]).bit_count()
                    if u_in_xy:
                        if count_vxy + 1 >= 3:
                            return False
                    else:
                        if count_vxy >= 3:
                            return False

                elif u_in_xy:
                    # Adding (u,v) adds v to N(u).
                    # For triple (u, x, y), v is a new common neighbor if v in N(x) and N(y).
                    # Here v_in_xy is False, so v is NOT a new common neighbor for (u,x,y).
                    # Count remains old_count(u,x,y).
                    count_uxy = (adj[u] & adj[x] & adj[y]).bit_count()
                    if count_uxy >= 3:
                        return False
                        
                    # For triple (v, x, y), u is a new common neighbor if u in N(x) and N(y).
                    # Here u_in_xy is True, so u IS a new common neighbor for (v,x,y).
                    count_vxy = (adj[v] & adj[x] & adj[y]).bit_count()
                    if count_vxy + 1 >= 3:
                        return False

                else:
                    # Neither u nor v is in N(x) & N(y).
                    # Adding (u,v) does not change common neighbors for (u,x,y) or (v,x,y) 
                    # via the new edge endpoint being in N(x)&N(y).
                    # However, we must ensure the graph was valid before. 
                    # Since we maintain validity, old counts are < 3.
                    # So these are safe.
                    pass

        # 2. Check triples (u, v, x) for x != u, v
        for x in range(n):
            if x == u or x == v:
                continue
            # Common neighbors of u, v, x in new graph:
            # N(u)_new = N(u) U {v}. v not in N(v). So v not in N(u)_new & N(v)_new.
            # N(v)_new = N(v) U {u}. u not in N(u). So u not in N(u)_new & N(v)_new.
            # So intersection is N(u) & N(v) & N(x).
            count_uxy = (adj[u] & adj[v] & adj[x]).bit_count()
            if count_uxy >= 3:
                return False

        return True

    def is_k33_free(adj):
        # Full check for validation
        for i in range(n):
            for j in range(i + 1, n):
                common_ij = adj[i] & adj[j]
                if common_ij.bit_count() < 3:
                    continue
                for k in range(j + 1, n):
                    common = common_ij & adj[k]
                    if common.bit_count() >= 3:
                        return False
        return True

    def edge_count(adj):
        return sum(x.bit_count() for x in adj) // 2

    # Greedy construction with multiple restarts
    for restart in range(200):
        if time.time() > deadline - 10.0:
            break
        
        adj = [0] * n
        # Random order of edges
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            if can_add(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Local Search: Try to add more edges
        improved = True
        while improved:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((adj[u] >> v) & 1):
                        if can_add(u, v, adj):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            improved = True
        
        # Local Search: Try swaps (remove one, add one)
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            
            existing = [(u, v) for u in range(n) for v in range(u + 1, n) if (adj[u] >> v) & 1]
            missing = [(u, v) for u in range(n) for v in range(u + 1, n) if not (adj[u] >> v) & 1]
            
            if not existing or not missing:
                break
                
            random.shuffle(existing)
            random.shuffle(missing)
            
            for u1, v1 in existing[:10]:
                for u2, v2 in missing[:10]:
                    # Remove u1-v1
                    adj[u1] &= ~(1 << v1)
                    adj[v1] &= ~(1 << u1)
                    
                    if can_add(u2, v2, adj):
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                        # Accept
                        break
                    else:
                        # Revert
                        adj[u1] |= (1 << v1)
                        adj[v1] |= (1 << u1)
                else:
                    continue
                break
        
        ec = edge_count(adj)
        if ec > best_edges:
            best_edges = ec
            best_adj = adj[:]
            
        # Optional: Verify validity (should be true by construction)
        # if not is_k33_free(adj):
        #     print("Invalid graph generated!")

    if best_adj:
        return [[u, v] for u in range(n) for v in range(u + 1, n) if (best_adj[u] >> v) & 1]
    else:
        # Fallback: Empty graph or known small valid graph
        # A bipartite graph K_{10,10} is K_{3,3}-free? No, K_{3,3} is a subgraph of K_{10,10}.
        # We need a graph with no K_{3,3}.
        # A random sparse graph is likely valid.
        # Let's return a set of edges from a known valid structure or just empty.
        # Empty graph is valid.
        return []
