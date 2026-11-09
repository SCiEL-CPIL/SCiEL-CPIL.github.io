import random
import time
from itertools import combinations

def construct():
    n = 20
    deadline = time.time() + 110
    
    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Bitmasks for neighbors
    # adj[i] is a bitmask of neighbors of i
    
    def count_common(u, v, adj):
        # Returns count of common neighbors of u and v
        return bin(adj[u] & adj[v]).count('1')
        
    def count_triple_common(a, b, c, adj):
        # Returns count of common neighbors of a, b, c
        return bin(adj[a] & adj[b] & adj[c]).count('1')

    def check_validity(adj):
        """
        Returns the number of violations (triples with >= 3 common neighbors).
        """
        violations = 0
        for a, b, c in combinations(range(n), 3):
            if count_triple_common(a, b, c, adj) >= 3:
                violations += 1
        return violations

    def can_add_fast(u, v, adj):
        """
        Check if adding edge (u, v) keeps the graph K3,3-free.
        Uses the logic:
        Adding (u, v) makes v a common neighbor of {u, x, y} for x, y in N(v).
        And u a common neighbor of {v, x, y} for x, y in N(u).
        """
        # Check 1: v becomes common neighbor of {u, x, y}
        # x, y must be neighbors of v.
        # We need to ensure that for no pair x, y in N(v), 
        # the triple {u, x, y} already has 2 common neighbors (excluding v, since v was not in N(u)).
        # Actually, if {u, x, y} has k common neighbors in the current graph,
        # and v is connected to x and y, then after adding (u, v), v becomes a common neighbor.
        # So the new count is k + 1.
        # We require k + 1 <= 2 => k <= 1.
        
        # So for all x, y in N(v):
        # count_common_neighbors(u, x, y) <= 1
        
        # Note: count_common_neighbors(u, x, y) is |N(u) & N(x) & N(y)|.
        
        # We iterate over pairs of neighbors of v.
        nv = [i for i in range(n) if adj[v] & (1 << i)]
        for i in range(len(nv)):
            x = nv[i]
            for j in range(i+1, len(nv)):
                y = nv[j]
                # Count common neighbors of u, x, y
                # If this count is >= 2, then adding v makes it >= 3.
                if count_triple_common(u, x, y, adj) >= 2:
                    return False
        
        # Check 2: u becomes common neighbor of {v, x, y}
        # Similarly for x, y in N(u)
        nu = [i for i in range(n) if adj[u] & (1 << i)]
        for i in range(len(nu)):
            x = nu[i]
            for j in range(i+1, len(nu)):
                y = nu[j]
                if count_triple_common(v, x, y, adj) >= 2:
                    return False
        
        return True

    def add_edge(u, v, adj):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(u, v, adj):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def edge_count(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    def greedy_random():
        adj = [0] * n
        edges = all_pairs[:]
        random.shuffle(edges)
        for u, v in edges:
            if can_add_fast(u, v, adj):
                add_edge(u, v, adj)
        return adj

    def local_search(adj, time_limit):
        start_time = time.time()
        current_edges = edge_count(adj)
        best_edges = current_edges
        best_adj = adj[:]
        
        # Simple hill climbing: try to add edges, then try swaps
        while time.time() - start_time < time_limit:
            improved = False
            
            # Try adding any edge
            for u, v in all_pairs:
                if not (adj[u] & (1 << v)):
                    if can_add_fast(u, v, adj):
                        add_edge(u, v, adj)
                        current_edges += 1
                        improved = True
            
            if not improved:
                # Try 1-for-1 swaps: remove (u, v), add (x, y)
                # Or 1-for-2: remove (u, v), add (x, y), (x, z)
                # Let's try random removals and re-additions
                edges_present = [(i, j) for i, j in all_pairs if adj[i] & (1 << j)]
                if not edges_present:
                    break
                
                # Try removing a random edge and seeing if we can add more
                for _ in range(100): # Limit attempts per iteration
                    if time.time() - start_time > time_limit:
                        break
                    
                    u, v = random.choice(edges_present)
                    remove_edge(u, v, adj)
                    current_edges -= 1
                    
                    # Try to add edges
                    added = 0
                    # Sort candidates by potential gain? Just random for now
                    candidates = [e for e in all_pairs if not (adj[e[0]] & (1 << e[1]))]
                    random.shuffle(candidates)
                    
                    for x, y in candidates:
                        if can_add_fast(x, y, adj):
                            add_edge(x, y, adj)
                            current_edges += 1
                            added += 1
                            if added > 1: # Net gain
                                break
                            
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                    elif current_edges < best_edges:
                        # Undo if it got worse? Or keep if it's a plateau?
                        # Let's revert if we didn't improve
                        # This is a greedy local search, so we revert if not improved
                        # But we already added some edges. We need to revert all changes.
                        # This is complex. Let's just keep the best and restart local search from best?
                        # Or just revert to best_adj if current is worse.
                        adj = best_adj[:]
                        current_edges = best_edges
                    else:
                        # Plateau, keep going?
                        pass
            
            if not improved and current_edges == best_edges:
                # Stuck, try a random perturbation?
                # For now, break and restart global search
                break
                
        return best_adj, best_edges

    best_edges = 0
    best_adj = [0] * n
    
    # Run multiple restarts
    while time.time() < deadline:
        adj = greedy_random()
        edges = edge_count(adj)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
            
        # Local search
        remaining = deadline - time.time()
        if remaining > 5:
            adj, edges = local_search(adj, min(2.0, remaining * 0.2))
            if edges > best_edges:
                best_edges = edges
                best_adj = adj[:]
                
        if best_edges > 90:
            break

    # Final polish
    if time.time() < deadline - 2:
        for u, v in all_pairs:
            if not (best_adj[u] & (1 << v)):
                if can_add_fast(u, v, best_adj):
                    add_edge(u, v, best_adj)
                    
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result.append([i, j])
    return result
