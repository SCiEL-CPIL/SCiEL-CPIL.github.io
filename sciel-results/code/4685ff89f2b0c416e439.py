import numpy as np
import time
import random

def construct():
    n = 20
    deadline = time.time() + 130  # Safety margin (150s limit)
    
    # Precompute edge indices: 190 possible edges
    edge_list = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_edges = len(edge_list)  # 190
    edge_idx = {}
    for i, (u, v) in enumerate(edge_list):
        edge_idx[(u, v)] = i
        edge_idx[(v, u)] = i
    
    # Precompute: for each edge e=(u,v), which other edges share a vertex with u or v
    # An edge f=(a,b) conflicts with e=(u,v) if adding both creates a K2,3
    # Two edges e1=(u1,v1) and e2=(u2,v2) conflict if they share exactly one endpoint
    # and the other two endpoints have >= 2 common neighbors in the current graph.
    # Actually, the constraint is: no two vertices have 3+ common neighbors.
    # So for any pair of vertices (u,v), the number of w with (u,w) and (v,w) both present <= 2.
    
    # For bitset approach: represent graph as bitset of 190 edges.
    # To check if adding edge e is valid, we need to verify that for all pairs (u,v) 
    # where e contributes to common neighbors, the count stays <= 2.
    
    # Better approach: maintain adjacency bitsets per vertex (20 bits each, fits in int)
    # adj[v] = bitmask of neighbors of v
    
    def is_valid_adj(adj):
        """Check no K2,3: for every pair (u,v), common neighbors <= 2"""
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if bin(common).count('1') > 2:
                    return False
        return True
    
    def count_edges_from_adj(adj):
        return sum(bin(adj[u]).count('1') for u in range(n)) // 2
    
    def try_add_edge(adj, u, v):
        """Try to add edge (u,v). Returns True if added."""
        if adj[u] & (1 << v):
            return False
        
        # Check all pairs that could be affected
        # Adding (u,v) affects common-neighbor counts for:
        # - pair (u, w) for each w in adj[v] (w is now common neighbor of u and v)
        # - pair (v, w) for each w in adj[u]
        # - pair (u, v) itself
        
        # For pair (u, v): common neighbors = adj[u] & adj[v] (before adding)
        # After adding, adj[u] gains v, adj[v] gains u
        # New common = (adj[u] | (1<<v)) & (adj[v] | (1<<u))
        #            = (adj[u] & adj[v]) | (adj[u] & (1<<u)) | ((1<<v) & adj[v]) | ((1<<v) & (1<<u))
        # Since u != v: = (adj[u] & adj[v]) | (adj[u] & (1<<u)) | ((1<<v) & adj[v])
        # adj[u] & (1<<u) = 0 (no self-loops), (1<<v) & adj[v] = 0
        # So new common for (u,v) = adj[u] & adj[v] (unchanged!)
        
        # For pair (u, w) where w in adj[v]:
        # Before: common(u,w) = adj[u] & adj[w]
        # After: adj[u] gains v, so common(u,w) gains v if v in adj[w]
        # v in adj[w] iff w in adj[v] (symmetric), which is true by construction
        # So new common(u,w) = old common(u,w) + 1
        # Need: old common(u,w) + 1 <= 2, i.e., old common(u,w) <= 1
        
        # For pair (v, w) where w in adj[u]:
        # Similarly, new common(v,w) = old common(v,w) + 1
        # Need: old common(v,w) <= 1
        
        # Check pairs (u, w) for w in adj[v]
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v] & (1 << w):  # w is neighbor of v
                common_uw = adj[u] & adj[w]
                if bin(common_uw).count('1') >= 2:
                    return False
        
        # Check pairs (v, w) for w in adj[u]
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[u] & (1 << w):  # w is neighbor of u
                common_vw = adj[v] & adj[w]
                if bin(common_vw).count('1') >= 2:
                    return False
        
        # Add the edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        return True
    
    def remove_edge(adj, u, v):
        """Remove edge (u,v)."""
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def simulated_annealing(adj, budget):
        """Simulated annealing with remove-and-refill moves."""
        best_adj = adj[:]
        best_edges = count_edges_from_adj(adj)
        current_edges = best_edges
        
        T = 5.0  # Initial temperature
        cooling = 0.999
        
        while time.time() < budget:
            T *= cooling
            if T < 0.01:
                T = 0.01
            
            move_type = random.random()
            
            if move_type < 0.3:
                # Try to add a random non-edge
                u, v = random.sample(range(n), 2)
                if not (adj[u] & (1 << v)):
                    # Backup
                    adj_backup = adj[:]
                    if try_add_edge(adj, u, v):
                        new_edges = count_edges_from_adj(adj)
                        if new_edges > best_edges:
                            best_edges = new_edges
                            best_adj = adj[:]
                        current_edges = new_edges
                    else:
                        adj[:] = adj_backup
            
            elif move_type < 0.6:
                # Remove a random edge, then try to add 1-2 edges (refill)
                edges = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u] & (1 << v)]
                if not edges:
                    continue
                ru, rv = random.choice(edges)
                adj_backup = adj[:]
                remove_edge(adj, ru, rv)
                
                # Try to add some edges
                added = 0
                for _ in range(3):
                    u, v = random.sample(range(n), 2)
                    if try_add_edge(adj, u, v):
                        added += 1
                
                new_edges = count_edges_from_adj(adj)
                delta = new_edges - current_edges
                
                if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                    current_edges = new_edges
                    if new_edges > best_edges:
                        best_edges = new_edges
                        best_adj = adj[:]
                else:
                    adj[:] = adj_backup
            
            else:
                # Swap: remove one edge, add another
                edges = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u] & (1 << v)]
                non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if not (adj[u] & (1 << v))]
                
                if not edges or not non_edges:
                    continue
                
                ru, rv = random.choice(edges)
                au, av = random.choice(non_edges)
                
                adj_backup = adj[:]
                remove_edge(adj, ru, rv)
                
                if try_add_edge(adj, au, av):
                    new_edges = count_edges_from_adj(adj)
                    delta = new_edges - current_edges
                    if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                        current_edges = new_edges
                        if new_edges > best_edges:
                            best_edges = new_edges
                            best_adj = adj[:]
                    else:
                        adj[:] = adj_backup
                else:
                    adj[:] = adj_backup
        
        return best_adj, best_edges
    
    # Initial solution: 5 disjoint K4s
    adj = [0] * n
    for k in range(5):
        for a in range(4):
            for b in range(a + 1, 4):
                u = 4 * k + a
                v = 4 * k + b
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    
    # Greedy fill
    changed = True
    while changed:
        changed = False
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    if try_add_edge(adj, u, v):
                        changed = True
    
    best_adj = adj[:]
    best_edges = count_edges_from_adj(adj)
    
    # Multiple SA restarts
    restarts = 0
    while time.time() < deadline - 10:
        # Perturb the best solution
        perturbed = best_adj[:]
        for _ in range(random.randint(3, 8)):
            edges = [(u, v) for u in range(n) for v in range(u + 1, n) if perturbed[u] & (1 << v)]
            if not edges:
                break
            ru, rv = random.choice(edges)
            remove_edge(perturbed, ru, rv)
        
        # Run SA for a time slice
        slice_end = min(time.time() + 5, deadline - 10)
        sa_adj, sa_edges = simulated_annealing(perturbed, slice_end)
        
        if sa_edges > best_edges:
            best_edges = sa_edges
            best_adj = sa_adj[:]
        
        restarts += 1
    
    # Final polish: greedy fill on best solution
    final_adj = best_adj[:]
    changed = True
    while changed:
        changed = False
        for u in range(n):
            for v in range(u + 1, n):
                if not (final_adj[u] & (1 << v)):
                    if try_add_edge(final_adj, u, v):
                        changed = True
    
    if count_edges_from_adj(final_adj) > best_edges:
        best_adj = final_adj
        best_edges = count_edges_from_adj(final_adj)
    
    # Convert to edge list
    edges_list = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                edges_list.append([u, v])
    
    return edges_list
