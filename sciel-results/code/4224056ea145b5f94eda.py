"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

def _is_k23_free(adj):
    """Check if graph with adjacency matrix adj has no K2,3."""
    n = adj.shape[0]
    # For each pair of vertices (u, v), count common neighbors
    # If any pair has >= 3 common neighbors, K2,3 exists
    # We can compute common neighbors via (adj @ adj) but need to be careful
    # Actually, number of common neighbors of u and v = sum over w of adj[u,w]*adj[v,w]
    # This is (adj @ adj.T)[u,v] - but adj is symmetric so adj @ adj
    # However, we need to exclude u and v themselves from being counted as common neighbors
    # adj[u,u] = 0 for simple graph, so adj[u,w] for w=u is 0, similarly for v
    # So common neighbors count = (adj @ adj)[u, v] for u != v
    
    common = adj @ adj
    for u in range(n):
        for v in range(u+1, n):
            if common[u, v] >= 3:
                return False
    return True

def _count_edges(adj):
    return int(adj.sum() // 2)

def _paley_graph(n):
    """
    Construct a Paley-type graph on n vertices.
    For prime power q, vertices are 0..q-1, edge between u,v if (v-u) is a quadratic residue mod q.
    We'll use n=20, but 20 is not a prime power. Instead, we'll try to construct on a suitable field size.
    
    Actually, let's use a different approach: use a finite field of size q >= 20, embed 20 vertices,
    and define edges based on quadratic residues.
    
    Alternatively, use a known construction: for n=20, we can try the complement of a known K2,3-free graph,
    or use a random heuristic with spectral guidance.
    
    Let's try: use GF(19) (prime), embed 19 vertices, then add a 20th vertex carefully.
    But 19 is odd prime, Paley graph on 19 vertices has 19*18/4 = 81 edges (since (q-1)/2 = 9 neighbors per vertex).
    Check if Paley(19) is K2,3-free: in Paley graphs, the number of common neighbors of two vertices depends on whether
    they are adjacent. For Paley graph on q vertices (q ≡ 1 mod 4), two adjacent vertices have (q-5)/4 common neighbors,
    and two non-adjacent vertices have (q-1)/4 common neighbors.
    For q=19: adjacent: (19-5)/4 = 3.5... wait, that's not integer. Let me recalculate.
    
    Actually for Paley graph on q vertices where q ≡ 1 mod 4:
    - If u and v are adjacent: common neighbors = (q-5)/4
    - If u and v are non-adjacent: common neighbors = (q-1)/4
    
    For q=17: (17-5)/4=3, (17-1)/4=4. So non-adjacent pairs have 4 common neighbors -> K2,3 exists.
    For q=13: (13-5)/4=2, (13-1)/4=3. Non-adjacent pairs have 3 common neighbors -> K2,3 exists (since we need >=3).
    
    Hmm, so Paley graphs may not be K2,3-free. Let me reconsider.
    
    Actually, the condition is "no two vertices have three common neighbors", meaning common neighbors < 3, i.e., <= 2.
    
    For Paley(13): non-adjacent have 3 common neighbors -> violates.
    For Paley(17): non-adjacent have 4 -> violates.
    
    So Paley graphs don't work directly. Let's try a different spectral approach.
    
    Alternative: use a bipartite-like construction or a random graph with edge removal to satisfy K2,3-free.
    """
    pass

def construct():
    n = 20
    # Start with a random dense graph and iteratively remove edges to satisfy K2,3-free
    # Then try to add back edges where possible
    
    np.random.seed(42)
    
    # Start with a fairly dense random graph
    adj = np.zeros((n, n), dtype=int)
    for u in range(n):
        for v in range(u+1, n):
            if np.random.random() < 0.7:
                adj[u, v] = adj[v, u] = 1
    
    # Function to check and fix K2,3 violations
    def fix_k23(adj):
        changed = True
        while changed:
            changed = False
            common = adj @ adj
            for u in range(n):
                for v in range(u+1, n):
                    if common[u, v] >= 3:
                        # Find common neighbors
                        common_neighbors = [w for w in range(n) if adj[u, w] and adj[v, w]]
                        if len(common_neighbors) >= 3:
                            # Remove an edge from one of the common neighbors to break the K2,3
                            # Remove edge (u, common_neighbors[0])
                            adj[u, common_neighbors[0]] = 0
                            adj[common_neighbors[0], u] = 0
                            changed = True
        return adj
    
    adj = fix_k23(adj)
    
    # Now try to add edges back
    def try_add_edges(adj):
        improved = True
        while improved:
            improved = False
            for u in range(n):
                for v in range(u+1, n):
                    if adj[u, v] == 0:
                        # Check if adding (u,v) creates a K2,3
                        # Adding (u,v) means u and v become adjacent.
                        # We need to check all pairs (a,b) where a,b are common neighbors of some pair
                        # Actually, adding edge (u,v) could create K2,3 in several ways:
                        # 1. u and v themselves: they now share common neighbors. If they already have >=2 common neighbors, adding the edge doesn't change common neighbors count, but wait...
                        #    Actually, common neighbors of u and v don't change when we add edge (u,v).
                        #    The issue is: for any pair (x,y), adding (u,v) might increase their common neighbors if u or v is a common neighbor of x and y.
                        #    
                        # Let me think more carefully. When we add edge (u,v):
                        # - For the pair (u,v): their common neighbors are unchanged.
                        # - For any other pair (x,y) where x != u,v and y != u,v: common neighbors unchanged.
                        # - For pairs involving u: (u, w) for w != v. The common neighbors of u and w are unchanged.
                        # - Wait, no. Adding edge (u,v) doesn't change the neighbor sets of any vertex except it adds v to N(u) and u to N(v).
                        #   So common neighbors of (u, w) for w != v: a vertex z is a common neighbor of u and w if z in N(u) and z in N(w). 
                        #   N(u) gains v, so if v in N(w), then v becomes a new common neighbor of u and w.
                        #   Similarly, N(v) gains u, so if u in N(w), then u becomes a new common neighbor of v and w.
                        #   Also, for pair (v, w) with w != u: N(v) gains u, so if u in N(w), u becomes common neighbor of v and w.
                        #   For pair (u, v): no change in common neighbors.
                        #   For pair (w1, w2) where neither is u or v: no change.
                        #
                        # So we need to check:
                        # - For each w in N(v) (neighbors of v, excluding u): check if u and w now have >=3 common neighbors
                        # - For each w in N(u) (neighbors of u, excluding v): check if v and w now have >=3 common neighbors
                        # - Also check if u and v themselves have >=3 common neighbors (they did before, still do)
                        
                        # Check u and v common neighbors
                        common_uv = [z for z in range(n) if adj[u, z] and adj[v, z]]
                        if len(common_uv) >= 3:
                            continue
                        
                        # Check pairs (u, w) for w in N(v), w != u
                        ok = True
                        for w in range(n):
                            if w == u or w == v:
                                continue
                            if adj[v, w]:  # w is neighbor of v
                                # After adding (u,v), u and w share common neighbors
                                # Current common neighbors of u and w:
                                common_uw = [z for z in range(n) if adj[u, z] and adj[w, z]]
                                # Plus v if v is neighbor of u (which it will be) and v is neighbor of w (yes)
                                # So new common neighbors = common_uw + [v]
                                if len(common_uw) + 1 >= 3:
                                    ok = False
                                    break
                            if adj[u, w]:  # w is neighbor of u
                                # After adding (u,v), v and w share common neighbors
                                common_vw = [z for z in range(n) if adj[v, z] and adj[w, z]]
                                # Plus u if u is neighbor of v (yes) and u is neighbor of w (yes)
                                if len(common_vw) + 1 >= 3:
                                    ok = False
                                    break
                        if ok:
                            adj[u, v] = 1
                            adj[v, u] = 1
                            improved = True
        return adj
    
    adj = try_add_edges(adj)
    
    # Convert to edge list
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if adj[u, v]:
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
