"""K3,3-free graph on 20 vertices: maximize edges via local search with bitmask acceleration."""
import time
import random
from itertools import combinations

N = 20
# Total possible edges
ALL_EDGES = [(i, j) for i in range(N) for j in range(i + 1, N)]
EDGE_INDEX = {(i, j): k for k, (i, j) in enumerate(ALL_EDGES)}
EDGE_COUNT = len(ALL_EDGES)  # 190

# Precompute for each pair of vertices, the set of common neighbors
# common_neighbors[u][v] = bitmask of vertices that are neighbors of both u and v
# We'll maintain this incrementally.

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    random.seed(42)
    
    # Bitmask representation: adj[v] is a bitmask of neighbors of v
    # For 20 vertices, we can use Python integers as bitmasks
    
    def is_k33_free(adj):
        """Check if graph with adjacency bitmasks adj is K3,3-free.
        A K3,3 exists iff there exist three vertices a,b,c such that
        |N(a) ∩ N(b) ∩ N(c)| >= 3.
        """
        for a in range(N):
            for b in range(a + 1, N):
                ab = adj[a] & adj[b]
                if ab.bit_count() < 2:
                    continue
                # Need to check if any c has at least 3 common neighbors with a and b
                # Actually we need: for some triple (a,b,c), |N(a)&N(b)&N(c)| >= 3
                # Let's iterate over c
                for c in range(b + 1, N):
                    abc = ab & adj[c]
                    if abc.bit_count() >= 3:
                        return False
        return True
    
    def count_common3(adj, a, b, c):
        """Count common neighbors of three vertices."""
        return (adj[a] & adj[b] & adj[c]).bit_count()
    
    def try_add_edge(adj, u, v):
        """Try to add edge (u,v). Returns True if valid."""
        if adj[u] >> v & 1:
            return True  # already present
        # Adding edge (u,v) means u becomes neighbor of v and vice versa
        # We need to check that no triple containing u and v (or where u,v are in the neighborhood)
        # creates a K3,3.
        # A K3,3 with the new edge: the new edge connects u and v.
        # In K3,3, we need 3 vertices on each side. The new edge is in the graph.
        # If we add edge (u,v), it could create a K3,3 where {u,v,...} is one side.
        # More precisely: adding edge (u,v) can create a K3,3 iff there exist vertices
        # x,y such that x and y are common neighbors of u and v (in the new graph),
        # and there's a third vertex w that is also a common neighbor, making 3 common neighbors.
        # Wait, let me think more carefully.
        # 
        # K3,3 has bipartition {a,b,c} and {d,e,f} with all 9 edges present.
        # After adding edge (u,v), a K3,3 is created if:
        # Case 1: u and v are on the same side of the K3,3.
        #   Then we need a third vertex w on that side, and 3 vertices on the other side
        #   that are all neighbors of u, v, and w.
        # Case 2: u and v are on opposite sides.
        #   Then u is on one side, v on the other. We need 2 more vertices on u's side
        #   and 2 more on v's side, such that all cross-edges exist.
        #
        # This is complex. Let's just do the full check by temporarily adding the edge.
        
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        # Check all triples
        for a in range(N):
            for b in range(a + 1, N):
                ab = adj[a] & adj[b]
                if ab.bit_count() < 2:
                    continue
                for c in range(b + 1, N):
                    if (ab & adj[c]).bit_count() >= 3:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        return False
        return True
    
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def edge_count(adj):
        total = 0
        for i in range(N):
            total += adj[i].bit_count()
        return total // 2
    
    # Greedy construction
    adj = [0] * N
    # Start with a random ordering of edges
    edges_order = ALL_EDGES[:]
    random.shuffle(edges_order)
    
    for u, v in edges_order:
        if try_add_edge(adj, u, v):
            pass  # edge added
        # else: edge not added, try next
    
    current_edges = edge_count(adj)
    best_adj = adj[:]
    best_edges = current_edges
    
    # Local search
    def local_search(adj, max_iter=5000):
        improved = True
        iters = 0
        while improved and iters < max_iter:
            improved = False
            iters += 1
            if time.time() > deadline - 10:
                break
            # Try to add edges
            for u, v in ALL_EDGES:
                if time.time() > deadline - 10:
                    break
                if adj[u] >> v & 1:
                    continue
                if try_add_edge(adj, u, v):
                    improved = True
                    # Continue trying to add more edges
            # Try to remove and re-add (edge swap)
            if time.time() > deadline - 10:
                break
            for u1, v1 in ALL_EDGES:
                if time.time() > deadline - 10:
                    break
                if not (adj[u1] >> v1 & 1):
                    continue
                for u2, v2 in ALL_EDGES:
                    if time.time() > deadline - 10:
                        break
                    if adj[u2] >> v2 & 1:
                        continue
                    if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
                        continue
                    # Try swap: remove (u1,v1), add (u2,v2)
                    remove_edge(adj, u1, v1)
                    if try_add_edge(adj, u2, v2):
                        # Now try to add more edges
                        more_improved = True
                        while more_improved:
                            more_improved = False
                            for u3, v3 in ALL_EDGES:
                                if time.time() > deadline - 10:
                                    break
                                if adj[u3] >> v3 & 1:
                                    continue
                                if try_add_edge(adj, u3, v3):
                                    more_improved = True
                        improved = True
                    else:
                        # Revert
                        adj[u2] &= ~(1 << v2)
                        adj[v2] &= ~(1 << u2)
                        remove_edge(adj, u1, v1)  # already removed, need to re-add
                        adj[u1] |= (1 << v1)
                        adj[v1] |= (1 << u1)
        return adj
    
    # Run local search
    adj = local_search(adj)
    current_edges = edge_count(adj)
    if current_edges > best_edges:
        best_adj = adj[:]
        best_edges = current_edges
    
    # Kick-based restarts
    restart_count = 0
    while time.time() < deadline - 15:
        restart_count += 1
        # Create a perturbation: remove some random edges and re-add
        adj_perturbed = best_adj[:]
        # Remove k random edges
        k = random.randint(3, 8)
        present_edges = [(u, v) for u, v in ALL_EDGES if adj_perturbed[u] >> v & 1]
        random.shuffle(present_edges)
        for i in range(min(k, len(present_edges))):
            u, v = present_edges[i]
            remove_edge(adj_perturbed, u, v)
        
        # Run local search on perturbed graph
        adj_perturbed = local_search(adj_perturbed, max_iter=3000)
        pert_edges = edge_count(adj_perturbed)
        
        if pert_edges > best_edges:
            best_adj = adj_perturbed[:]
            best_edges = pert_edges
            # Reset restart count on improvement
            restart_count = 0
    
    # Final verification
    assert is_k33_free(best_adj), "Graph is not K3,3-free!"
    
    # Convert to edge list
    result = []
    for i in range(N):
        for j in range(i + 1, N):
            if best_adj[i] >> j & 1:
                result.append([i, j])
    
    return result


if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
