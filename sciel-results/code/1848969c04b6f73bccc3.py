"""K3,3-free graph construction on 20 vertices."""
import random
import time
import itertools

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 100  # safety margin before 150s kill

    n = 20
    # Precompute all triples of vertices for K3,3 check
    # A K3,3 exists iff some 3 vertices have >=3 common neighbors
    # (excluding themselves, but common neighbors won't include the triple members in a simple graph)
    
    def is_k33_free(adj):
        """Check if graph with adjacency bitmasks is K3,3-free."""
        # For every triple (i,j,k), count common neighbors
        for i in range(n):
            for j in range(i+1, n):
                common_ij = adj[i] & adj[j]
                # Remove i and j from consideration (they aren't their own neighbors)
                # Actually in a simple graph, adj[i] doesn't include i, so common_ij doesn't include i or j
                # But we need common neighbors of i, j, k excluding i, j, k
                for k in range(j+1, n):
                    common_ijk = common_ij & adj[k]
                    # Remove k from common_ijk (k is not its own neighbor, so this is automatic)
                    # Count bits
                    if common_ijk.bit_count() >= 3:
                        return False
        return True

    def count_edges(adj):
        """Count edges in adjacency bitmasks."""
        total = 0
        for i in range(n):
            total += adj[i].bit_count()
        return total // 2

    def adj_to_edges(adj):
        """Convert adjacency bitmasks to edge list."""
        edges = []
        for u in range(n):
            v = adj[u]
            while v:
                low = v & (-v)
                vv = low.bit_length() - 1
                if vv > u:  # only add once
                    edges.append([u, vv])
                v ^= low
        return edges

    def try_add_edge(adj, u, v):
        """Try to add edge (u,v), return True if graph remains K3,3-free."""
        old_u = adj[u]
        old_v = adj[v]
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        if not is_k33_free(adj):
            adj[u] = old_u
            adj[v] = old_v
            return False
        return True

    def remove_edge(adj, u, v):
        """Remove edge (u,v)."""
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def try_swap(adj, remove_pair, add_pair):
        """Try to remove edge remove_pair and add add_pair."""
        u1, v1 = remove_pair
        u2, v2 = add_pair
        old_u1 = adj[u1]
        old_v1 = adj[v1]
        old_u2 = adj[u2]
        old_v2 = adj[v2]
        
        remove_edge(adj, u1, v1)
        adj[u2] |= (1 << v2)
        adj[v2] |= (1 << u2)
        
        if not is_k33_free(adj):
            adj[u1] = old_u1
            adj[v1] = old_v1
            adj[u2] = old_u2
            adj[v2] = old_v2
            return False
        return True

    best_adj = [0] * n
    best_count = 0
    best_edges = []

    # Strategy: Randomized greedy with local search
    # Multiple restarts
    
    random.seed(42)
    
    all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    num_restarts = 0
    for restart in range(500):  # max restarts, but time-limited
        if time.time() > deadline - 5:
            break
        num_restarts += 1
        
        # Start from empty or random sparse graph
        adj = [0] * n
        
        # Phase 1: Greedy addition
        random.shuffle(all_possible_edges)
        edge_count = 0
        for u, v in all_possible_edges:
            if time.time() > deadline - 10:
                break
            if try_add_edge(adj, u, v):
                edge_count += 1
        
        # Phase 2: Local search - try swaps to add more edges
        improved = True
        swap_rounds = 0
        while improved and time.time() < deadline - 5:
            improved = False
            swap_rounds += 1
            
            # Build list of non-edges and edges for swap attempts
            non_edges = []
            edges_list = []
            for u in range(n):
                for v in range(u+1, n):
                    if adj[u] & (1 << v):
                        edges_list.append((u, v))
                    else:
                        non_edges.append((u, v))
            
            # Try random swaps: remove an edge, add a non-edge
            random.shuffle(non_edges)
            random.shuffle(edges_list)
            
            for ne in non_edges:
                if time.time() > deadline - 5:
                    break
                for e in edges_list:
                    if time.time() > deadline - 5:
                        break
                    if ne[0] == e[0] and ne[1] == e[1]:
                        continue
                    if try_swap(adj, e, ne):
                        improved = True
                        break
                if improved:
                    break
            
            if not improved and swap_rounds > 3:
                break
        
        # Update best
        current_count = count_edges(adj)
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
            best_edges = adj_to_edges(adj)
            print(f"Restart {restart}: {best_count} edges", flush=True)
    
    # Fallback: if no edges found, return K2,18
    if not best_edges:
        return [[a, b] for a in range(2) for b in range(2, 20)]
    
    return best_edges
# EVOLVE-BLOCK-END
