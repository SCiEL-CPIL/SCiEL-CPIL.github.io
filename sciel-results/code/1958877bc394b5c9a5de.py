import random
import time
import math
from itertools import combinations

def construct():
    n = 20
    # adj[u] is a bitmask of neighbors of u
    adj = [0] * n
    
    def popcount(x):
        return bin(x).count('1')
    
    def is_valid_addition(u, v):
        # Check if adding edge (u, v) violates the constraint
        # Constraint: For all triples {a,b,c}, |N(a) & N(b) & N(c)| <= 2
        # Adding (u,v) updates N(u) and N(v).
        # Only triples containing u or v are affected.
        
        # Triples {u, w, z} where w, z != u, v
        # New common neighbors for {u, w, z} = (N(u) | (1<<v)) & N(w) & N(z)
        # Old common neighbors = N(u) & N(w) & N(z)
        # The new common neighbors are Old UNION ( {v} & N(w) & N(z) )
        # So the count increases by 1 if v is in N(w) and N(z).
        # We need to ensure that if v is in N(w) & N(z), then the old count was < 2.
        # i.e., if v in N(w) & N(z), then popcount(N(u) & N(w) & N(z)) must be <= 1.
        
        # Similarly for triples {v, w, z}
        # If u in N(w) & N(z), then popcount(N(v) & N(w) & N(z)) must be <= 1.
        
        # Iterate over all pairs w, z
        for w in range(n):
            if w == u or w == v:
                continue
            for z in range(w + 1, n):
                if z == u or z == v:
                    continue
                
                # Check triple {u, w, z}
                if (adj[w] >> v) & 1 and (adj[z] >> v) & 1:
                    # v is a neighbor of both w and z
                    # Check if u already has 2 common neighbors with w and z
                    common = adj[u] & adj[w] & adj[z]
                    if popcount(common) >= 2:
                        return False
                
                # Check triple {v, w, z}
                if (adj[w] >> u) & 1 and (adj[z] >> u) & 1:
                    # u is a neighbor of both w and z
                    # Check if v already has 2 common neighbors with w and z
                    common = adj[v] & adj[w] & adj[z]
                    if popcount(common) >= 2:
                        return False
        
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        return 1

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        return 1

    def edge_count():
        count = 0
        for u in range(n):
            count += popcount(adj[u])
        return count // 2

    def local_search(adj_local):
        """Try to improve by swapping edges"""
        improved = True
        max_swaps = 100
        while improved and max_swaps > 0:
            improved = False
            max_swaps -= 1
            # Try removing a random edge and adding a different one
            current_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj_local[u] >> v) & 1:
                        current_edges.append((u, v))
            
            if not current_edges:
                break
                
            random.shuffle(current_edges)
            
            for u, v in current_edges:
                # Try removing (u, v)
                adj_local[u] &= ~(1 << v)
                adj_local[v] &= ~(1 << u)
                
                # Try adding a random non-edge
                all_non_edges = [(a, b) for a in range(n) for b in range(a + 1, n) 
                                if not ((adj_local[a] >> b) & 1) and (a, b) != (u, v)]
                if not all_non_edges:
                    # Restore
                    adj_local[u] |= (1 << v)
                    adj_local[v] |= (1 << u)
                    break
                    
                random.shuffle(all_non_edges)
                
                for a, b in all_non_edges:
                    # Check if we can add (a, b)
                    can_add = True
                    for w in range(n):
                        if w == a or w == b:
                            continue
                        for z in range(w + 1, n):
                            if z == a or z == b:
                                continue
                            if (adj_local[w] >> b) & 1 and (adj_local[z] >> b) & 1:
                                common = adj_local[a] & adj_local[w] & adj_local[z]
                                if popcount(common) >= 2:
                                    can_add = False
                                    break
                            if (adj_local[w] >> a) & 1 and (adj_local[z] >> a) & 1:
                                common = adj_local[b] & adj_local[w] & adj_local[z]
                                if popcount(common) >= 2:
                                    can_add = False
                                    break
                        if not can_add:
                            break
                    if can_add:
                        # Add edge (a, b)
                        adj_local[a] |= (1 << b)
                        adj_local[b] |= (1 << a)
                        improved = True
                        break
                    # Restore original state for next attempt
                if not improved:
                    # Restore removed edge
                    adj_local[u] |= (1 << v)
                    adj_local[v] |= (1 << u)
                else:
                    break
        return adj_local

    best_edges = []
    best_count = 0
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Generate all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Multiple random restarts
    restarts = 0
    while time.time() < deadline:
        random.shuffle(all_edges)
        adj_local = [0] * n
        for u, v in all_edges:
            # Use local adj for checking
            local_adj = adj_local
            can_add = True
            for w in range(n):
                if w == u or w == v:
                    continue
                for z in range(w + 1, n):
                    if z == u or z == v:
                        continue
                    if (local_adj[w] >> v) & 1 and (local_adj[z] >> v) & 1:
                        common = local_adj[u] & local_adj[w] & local_adj[z]
                        if popcount(common) >= 2:
                            can_add = False
                            break
                    if (local_adj[w] >> u) & 1 and (local_adj[z] >> u) & 1:
                        common = local_adj[v] & local_adj[w] & local_adj[z]
                        if popcount(common) >= 2:
                            can_add = False
                            break
                if not can_add:
                    break
            if can_add:
                adj_local[u] |= (1 << v)
                adj_local[v] |= (1 << u)
        
        current_count = edge_count()
        
        # Try local search to improve
        adj_improved = local_search(adj_local[:])
        improved_count = edge_count()
        # Recount for improved
        improved_count = 0
        for u in range(n):
            improved_count += popcount(adj_improved[u])
        improved_count //= 2
        
        if improved_count > current_count:
            adj_local = adj_improved
            current_count = improved_count
        
        if current_count > best_count:
            best_count = current_count
            # Save current adjacency
            best_adj = adj_local[:]
        
        restarts += 1
        # If we have enough time, do more restarts
        
    # Reconstruct edge list from best_adj
    if best_count > 0:
        for u in range(n):
            for v in range(u + 1, n):
                if (best_adj[u] >> v) & 1:
                    best_edges.append([u, v])
    
    # Fallback to seed if something went wrong or no edges found (unlikely)
    if not best_edges:
        return [[a, b] for a in range(2) for b in range(2, 20)]
        
    return best_edges
