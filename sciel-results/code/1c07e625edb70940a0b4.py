"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    n = 31
    target_edges = 91

    def count_c4(adj_sets):
        """Count 4-cycles: for each non-adjacent pair {u,v}, add C(|N(u)&N(v)|, 2).
        Each C4 is counted exactly once (via its two pairs of opposite vertices,
        but we only count non-adjacent pairs, so each C4 contributes exactly once
        per pair of opposite non-adjacent vertices... actually each C4 has exactly
        2 pairs of opposite vertices, both non-adjacent, so it's counted twice).
        """
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                if v not in adj_sets[u]:
                    common = len(adj_sets[u] & adj_sets[v])
                    if common >= 2:
                        total += common * (common - 1) // 2
        return total // 2  # Each C4 counted twice (once per opposite pair)

    def build_polarity_graph():
        """Construct a polarity graph on GF(5) giving a C4-free graph with 90 edges.
        Polarity graph of the unitary polarity on the projective plane of order 5.
        Vertices: 0..29 for points (i,j) with i in GF(5), j in GF(5), plus vertex 30 for the line at infinity.
        Actually for q=5, the projective plane has q^2+q+1 = 31 points.
        A polarity graph connects two points if they are conjugate under the polarity.
        For the standard polarity on PG(2, q), the graph has q^2+q+1 vertices and
        each vertex has degree q+1 or q (for self-conjugate points).
        
        For q=5: 31 vertices, each has degree 5 or 6.
        Total edges = (31*5 + number_of_self_conjugate) / 2... 
        
        Actually, let me use a different well-known construction.
        The Erdos-Renyi construction: vertices are pairs (a,b) in GF(q) x GF(q) plus one extra.
        For q=5: 26 + 1 = 27... not 31.
        
        Let me use the polarity graph directly.
        PG(2,5) has 31 points. The standard polarity x0*x2 + x1^2 = 0 (or similar).
        Two points are adjacent if each lies on the polar line of the other.
        Self-conjugate points (isotropic) are also adjacent to themselves? No, we don't have self-loops.
        
        For the standard polarity on PG(2,q) with q odd:
        - There are q+1 self-conjugate (isotropic) points
        - The graph is C4-free
        - Number of edges: (q^2+q+1)(q+1)/2 - (q+1)/2 ... 
        
        Let me just enumerate it properly.
        """
        q = 5
        # Represent points in PG(2,q) as triples (x0, x1, x2) not all zero, modulo scaling.
        # Canonical form: make first non-zero coordinate = 1.
        points = []
        seen = set()
        for x0 in range(q):
            for x1 in range(q):
                for x2 in range(q):
                    if x0 == 0 and x1 == 0 and x2 == 0:
                        continue
                    # Normalize: find first non-zero
                    if x0 != 0:
                        canon = (1, x1 * pow(x0, q-2, q) % q, x2 * pow(x0, q-2, q) % q)
                    elif x1 != 0:
                        canon = (0, 1, x2 * pow(x1, q-2, q) % q)
                    else:
                        canon = (0, 0, 1)
                    if canon not in seen:
                        seen.add(canon)
                        points.append(canon)
        
        assert len(points) == 31, f"Expected 31 points, got {len(points)}"
        
        # Define a polarity: for point (x0,x1,x2), polar line is:
        # x0*X2 + x1*X1 + x2*X0 = 0  (symmetric bilinear form)
        # Two points p=(a0,a1,a2) and r=(b0,b1,b2) are conjugate if:
        # a0*b2 + a1*b1 + a2*b0 = 0
        
        def are_conjugate(p, r):
            return (p[0]*r[2] + p[1]*r[1] + p[2]*r[0]) % q == 0
        
        def is_self_conjugate(p):
            return (p[0]*p[2] + p[1]*p[1] + p[2]*p[0]) % q == 0
        
        # Build adjacency
        adj = [set() for _ in range(31)]
        edges = []
        for i in range(31):
            for j in range(i+1, 31):
                if are_conjugate(points[i], points[j]):
                    adj[i].add(j)
                    adj[j].add(i)
                    edges.append([i, j])
        
        c4_count = count_c4(adj)
        return edges, adj, c4_count

    def local_search_swap(adj, edges, current_c4, max_time):
        """Try to reduce C4 count by swapping edges."""
        edge_set = set()
        for e in edges:
            edge_set.add(tuple(sorted(e)))
        
        best_c4 = current_c4
        best_adj = [s.copy() for s in adj]
        best_edges = list(edges)
        
        # Precompute all possible edges
        all_possible = []
        for i in range(n):
            for j in range(i+1, n):
                all_possible.append((i, j))
        
        # For each edge, try removing it and adding a non-edge
        # that reduces C4 count
        random.shuffle(all_possible)
        
        while time.time() < max_time:
            improved = False
            for i in range(n):
                for j in range(i+1, n):
                    if (i, j) not in edge_set:
                        # Try adding (i,j) and removing some other edge
                        # First, find which edges to remove
                        # Simplest: just try adding this edge (if we have < target)
                        pass
            
            # Better approach: try swapping - remove one edge, add another
            for e_idx in range(len(best_edges)):
                if time.time() > max_time:
                    break
                u, v = best_edges[e_idx]
                # Remove edge (u,v)
                best_adj[u].discard(v)
                best_adj[v].discard(u)
                edge_set.discard((u, v))
                
                c4_after_remove = count_c4(best_adj)
                
                # Try adding various non-edges
                for i in range(n):
                    for j in range(i+1, n):
                        if (i, j) not in edge_set and (i, j) != (u, v):
                            best_adj[i].add(j)
                            best_adj[j].add(i)
                            edge_set.add((i, j))
                            c4_after_add = count_c4(best_adj)
                            
                            if c4_after_add < best_c4:
                                best_c4 = c4_after_add
                                best_edges = best_edges.copy()
                                best_edges[e_idx] = [i, j]
                                improved = True
                                break
                            else:
                                best_adj[i].discard(j)
                                best_adj[j].discard(i)
                                edge_set.discard((i, j))
                
                if improved:
                    break
                else:
                    # Restore edge
                    best_adj[u].add(v)
                    best_adj[v].add(u)
                    edge_set.add((u, v))
            
            if not improved:
                break
        
        return best_edges, best_adj, best_c4

    # Step 1: Build the polarity graph (C4-free, 90 edges)
    core_edges, core_adj, core_c4 = build_polarity_graph()
    
    if len(core_edges) != 90:
        # Fallback: use a different construction
        # Try random C4-free construction
        core_edges = []
        core_adj = [set() for _ in range(n)]
        all_pairs = list(combinations(range(n), 2))
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if len(core_edges) >= 90:
                break
            # Check if adding (u,v) creates a C4
            common = core_adj[u] & core_adj[v]
            if len(common) < 2:
                core_edges.append([u, v])
                core_adj[u].add(v)
                core_adj[v].add(u)
        core_c4 = count_c4(core_adj)
    
    # Step 2: Add the 91st edge to minimize total C4
    # Try each possible non-edge and pick the one that adds the fewest C4s
    edge_set = set()
    for e in core_edges:
        edge_set.add(tuple(sorted(e)))
    
    best_add_c4 = float('inf')
    best_add_edge = None
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) not in edge_set:
                # Count C4s created by adding (i,j)
                # A C4 is created for each common neighbor pair
                common = core_adj[i] & core_adj[j]
                new_c4s = len(common) * (len(common) - 1) // 2
                if new_c4s < best_add_c4:
                    best_add_c4 = new_c4s
                    best_add_edge = (i, j)
    
    if best_add_edge is not None:
        core_edges.append(list(best_add_edge))
        core_adj[best_add_edge[0]].add(best_add_edge[1])
        core_adj[best_add_edge[1]].add(best_add_edge[0])
    
    current_c4 = count_c4(core_adj)
    
    # Step 3: Local search to reduce C4 count further
    if time.time() < deadline - 5:
        max_search_time = min(deadline - 5 - time.time(), 60)
        if max_search_time > 2:
            core_edges, core_adj, current_c4 = local_search_swap(
                core_adj, core_edges, current_c4, time.time() + max_search_time
            )
    
    # Ensure exactly 91 distinct edges
    edge_set = set()
    unique_edges = []
    for e in core_edges:
        key = tuple(sorted(e))
        if key not in edge_set:
            edge_set.add(key)
            unique_edges.append(list(e))
    
    # If we have more than 91, trim
    if len(unique_edges) > target_edges:
        unique_edges = unique_edges[:target_edges]
    
    # If we have fewer than 91, add random edges
    if len(unique_edges) < target_edges:
        adj_temp = [set() for _ in range(n)]
        for e in unique_edges:
            adj_temp[e[0]].add(e[1])
            adj_temp[e[1]].add(e[0])
        for i in range(n):
            for j in range(i+1, n):
                if len(unique_edges) >= target_edges:
                    break
                if (i, j) not in edge_set:
                    unique_edges.append([i, j])
                    edge_set.add((i, j))
            if len(unique_edges) >= target_edges:
                break
    
    return unique_edges[:target_edges]
# EVOLVE-BLOCK-END
