"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill

    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_one(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set(edges)
        
        # Build adjacency list for K (0 to k-1) and I (k to n-1)
        # K is the clique part, I is the independent set part
        adj_K = [set() for _ in range(k)]
        adj_I = [set() for _ in range(n - k)] # indexed 0 to n-k-1, corresponding to vertex k+i
        
        # Separate edges
        edges_K = []
        edges_I = []
        for u, v in edges:
            if u < k and v < k:
                edges_K.append((u, v))
                adj_K[u].add(v)
                adj_K[v].add(u)
            elif u >= k and v >= k:
                # Should not happen in split graph definition (I is independent)
                # But just in case
                pass
            else:
                # Cross edge
                # Ensure u is in K, v in I for consistent indexing
                if u < k:
                    u_k = u
                    v_i = v - k
                else:
                    u_k = v
                    v_i = u - k
                edges_I.append((u_k, v_i))
                adj_K[u_k].add(v_i + k) # Store original vertex id in adj_K for convenience? No, let's keep separate.
                # Let's store neighbors in I for each K vertex
                # And neighbors in K for each I vertex
                pass
        
        # Re-do adjacency properly
        neighbors_K_in_K = [set() for _ in range(k)]
        neighbors_K_in_I = [set() for _ in range(k)] # Set of indices in I (0 to n-k-1)
        neighbors_I_in_K = [set() for _ in range(n - k)] # Set of indices in K (0 to k-1)
        
        for u, v in edges:
            if u < k and v < k:
                neighbors_K_in_K[u].add(v)
                neighbors_K_in_K[v].add(u)
            else:
                if u < k:
                    u_k = u
                    v_i = v - k
                else:
                    u_k = v
                    v_i = u - k
                neighbors_K_in_I[u_k].add(v_i)
                neighbors_I_in_K[v_i].add(u_k)

        # Identify active I vertices (those with degree > 0)
        active_I = [i for i in range(n - k) if len(neighbors_I_in_K[i]) > 0]
        
        # List of Type-B cliques
        type_b_cliques = []
        # Set of edges in K covered by Type-B cliques
        covered_K_edges = set()
        
        # Greedy assignment of K-edges to Type-B cliques
        # Sort active_I by size of N_K(v) descending to prioritize large cliques?
        # Or just process in order.
        # Let's sort by size of N_K(v) descending to try to cover more early?
        # Actually, to maximize disjointness, maybe smaller first?
        # Let's try sorting by size of N_K(v) descending.
        active_I_sorted = sorted(active_I, key=lambda i: len(neighbors_I_in_K[i]), reverse=True)
        
        for i in active_I_sorted:
            N_K_i = neighbors_I_in_K[i]
            # We want to select a subset S of N_K_i such that no edge within S is in covered_K_edges
            # S must be a clique in K (which it is, since N_K_i subset of K)
            # We want to maximize the number of new edges covered.
            # Greedy: Add vertices one by one.
            S = []
            # Sort candidates in N_K_i
            # To be deterministic and fast, just iterate
            candidates = list(N_K_i)
            # Heuristic: sort candidates by degree in K descending?
            candidates.sort(key=lambda u: len(neighbors_K_in_K[u]), reverse=True)
            
            for u in candidates:
                # Check if u can be added to S
                # u can be added if (u, w) is NOT in covered_K_edges for all w in S
                can_add = True
                for w in S:
                    e = (min(u, w), max(u, w))
                    if e in covered_K_edges:
                        can_add = False
                        break
                if can_add:
                    S.append(u)
                    # Mark edges between u and existing S as covered
                    for w in S[:-1]: # S[:-1] is S before adding u
                        e = (min(u, w), max(u, w))
                        covered_K_edges.add(e)
            
            if len(S) >= 1:
                # The clique is {i+k} U S
                # Note: If S is empty, the clique is just {i+k}, which has size 1.
                # Task requires cliques of size >= 2.
                # If S is empty, we cannot form a valid clique for this i?
                # But i has neighbors, so S should not be empty unless all neighbors' edges are covered?
                # If S is empty, it means all edges from i to N_K(i) are... wait.
                # S is subset of N_K(i).
                # If S is empty, we don't add a clique.
                # But the cross edges (u, i+k) for u in N_K(i) need to be covered!
                # My strategy covers K-edges with Type-B. Cross edges are ALWAYS covered by Type-B cliques.
                # So if I don't form a clique, I haven't covered the cross edges.
                # So I MUST form a clique for every active i.
                # If S is empty, the clique is {i+k}. Size 1. INVALID.
                # This implies my greedy logic might leave S empty if all potential edges are covered?
                # No, S is built from N_K(i).
                # If S is empty, it means we didn't add any vertex.
                # Why? Because for every u in N_K(i), (u, w) was covered for some w in S?
                # Initially S is empty. So for the first u, S is empty, so loop `for w in S` doesn't run. can_add is True.
                # So S will always have at least one vertex if N_K(i) is not empty.
                # So S is never empty for active i.
                
                clique_vertices = [i + k] + S
                type_b_cliques.append(clique_vertices)
        
        # Now, cover remaining K-edges with Type-A cliques
        remaining_K_edges = []
        for u in range(k):
            for v in neighbors_K_in_K[u]:
                if u < v:
                    if (u, v) not in covered_K_edges:
                        remaining_K_edges.append((u, v))
        
        type_a_cliques = []
        if remaining_K_edges:
            # Build graph of remaining K-edges
            # We need to cover these edges with cliques.
            # This is minimum clique cover of the graph G_rem = (K, remaining_K_edges).
            # Equivalent to minimum coloring of complement graph \bar{G}_rem.
            
            # Build complement adjacency
            # Vertices 0..k-1
            # Edge (u,v) in \bar{G}_rem if (u,v) NOT in remaining_K_edges and u != v
            # Note: If (u,v) was in covered_K_edges, it is NOT in remaining_K_edges, so it IS in \bar{G}_rem.
            # If (u,v) was never an edge in K, it is NOT in remaining_K_edges, so it IS in \bar{G}_rem.
            
            # We only care about vertices that are incident to at least one remaining edge?
            # No, isolated vertices in G_rem are cliques of size 1 in G_rem?
            # No, clique cover of G_rem.
            # If a vertex has no remaining edges, it doesn't need to be in any clique?
            # "Every edge must lie in exactly one listed clique."
            # Vertices without edges don't need to be in cliques?
            # "Every listed set must be a clique... every edge must lie in exactly one..."
            # It doesn't say every vertex must be in a clique.
            # So we only need to cover the edges.
            
            # Extract vertices involved in remaining_K_edges
            involved_vertices = set()
            for u, v in remaining_K_edges:
                involved_vertices.add(u)
                involved_vertices.add(v)
            
            if not involved_vertices:
                pass
            else:
                # Map involved vertices to 0..m-1
                v_list = sorted(involved_vertices)
                v_map = {v: i for i, v in enumerate(v_list)}
                m = len(v_list)
                
                # Build complement graph adjacency for coloring
                # comp_adj[i] is set of j such that (v_list[i], v_list[j]) is NOT an edge in G_rem
                # i.e., (v_list[i], v_list[j]) IS in covered_K_edges OR was never an edge in K.
                
                # Actually, for coloring the complement, we want to find independent sets in \bar{G}_rem.
                # Independent set in \bar{G}_rem is a set of vertices with NO edges in \bar{G}_rem.
                # No edges in \bar{G}_rem means ALL pairs are edges in G_rem.
                # So Independent set in \bar{G}_rem = Clique in G_rem.
                # So we are finding a partition of V into cliques of G_rem.
                
                # Let's use B&B to find minimum coloring of \bar{G}_rem.
                # \bar{G}_rem edge (u,v) exists if (u,v) is NOT in remaining_K_edges.
                
                comp_adj = [set() for _ in range(m)]
                for i in range(m):
                    for j in range(i + 1, m):
                        u = v_list[i]
                        v = v_list[j]
                        e = (min(u, v), max(u, v))
                        if e not in set(remaining_K_edges):
                            # Not in remaining edges, so in complement
                            comp_adj[i].add(j)
                            comp_adj[j].add(i)
                
                # B&B Coloring
                # State: color of each vertex, or -1
                # We want to minimize number of colors.
                
                best_colors = m # Upper bound
                best_solution = None
                
                # Precompute neighbors for speed
                # Use bitmasks for colors? m <= 30.
                
                def color_bnb(vertex, colors_used, color_assign):
                    nonlocal best_colors, best_solution
                    
                    if vertex == m:
                        if colors_used < best_colors:
                            best_colors = colors_used
                            best_solution = list(color_assign)
                        return
                    
                    # Try colors 0 to colors_used - 1
                    # And color colors_used (new color)
                    
                    # Optimization: If vertex has no neighbors in comp_adj, it can take any color.
                    # If it has neighbors, it can take a color not used by neighbors.
                    
                    # Try existing colors
                    for c in range(colors_used):
                        # Check if color c is available
                        available = True
                        for neighbor in comp_adj[vertex]:
                            if color_assign[neighbor] == c:
                                available = False
                                break
                        if available:
                            color_assign[vertex] = c
                            color_bnb(vertex + 1, colors_used, color_assign)
                            color_assign[vertex] = -1
                            if best_colors <= colors_used:
                                return # Prune if we already found a solution with <= colors
                    
                    # Try new color
                    color_assign[vertex] = colors_used
                    color_bnb(vertex + 1, colors_used + 1, color_assign)
                    color_assign[vertex] = -1
                
                # Initialize
                color_assign = [-1] * m
                # Sort vertices by degree in comp_adj descending for better pruning?
                # Standard B&B often benefits from ordering.
                # For now, natural order.
                
                color_bnb(0, 0, color_assign)
                
                # Reconstruct cliques from best_solution
                # best_solution[v] is the color (clique index)
                # Group vertices by color
                cliques_by_color = {}
                for v_idx in range(m):
                    c = best_solution[v_idx]
                    if c not in cliques_by_color:
                        cliques_by_color[c] = []
                    cliques_by_color[c].append(v_list[v_idx])
                
                for c, verts in cliques_by_color.items():
                    if len(verts) >= 2:
                        type_a_cliques.append(verts)
                    else:
                        # If clique size 1, it covers no edges.
                        # But we only included vertices involved in edges.
                        # So every vertex in v_list has at least one remaining edge.
                        # If a clique has size 1, it means that vertex was isolated in G_rem?
                        # No, v_list contains vertices with remaining edges.
                        # If a vertex u has a remaining edge (u,v), and u is in a clique of size 1,
                        # then v must be in a different clique?
                        # But (u,v) is an edge in G_rem. It must be covered.
                        # If u is in clique C1={u} and v in C2={v}, the edge (u,v) is not covered.
                        # So this case should not happen if the coloring is valid for \bar{G}_rem.
                        # In \bar{G}_rem, u and v are NOT connected (since (u,v) in G_rem).
                        # So they can have the same color.
                        # So u and v should be in the same clique.
                        # So cliques of size 1 should not exist for vertices with edges.
                        pass
        
        # Combine results
        all_cliques = type_b_cliques + type_a_cliques
        
        # Final validation and cleanup
        valid_cliques = []
        for c in all_cliques:
            if len(c) >= 2:
                # Ensure all are ints
                clean_c = [int(x) for x in c]
                # Verify it's a clique
                is_clique = True
                for i in range(len(clean_c)):
                    for j in range(i + 1, len(clean_c)):
                        e = (min(clean_c[i], clean_c[j]), max(clean_c[i], clean_c[j]))
                        if e not in edge_set:
                            is_clique = False
                            break
                    if not is_clique:
                        break
                if is_clique:
                    valid_cliques.append(clean_c)
        
        # Fallback if something went wrong
        if not valid_cliques:
             return [[u, v] for u, v in edges]
             
        return valid_cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > DEADLINE:
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
            
        cliques = solve_one(*spec)
        # Ensure types are correct
        clean_cliques = []
        for c in cliques:
            if len(c) >= 2:
                clean_cliques.append([int(x) for x in c])
        results.append(clean_cliques)
        
    return results
# EVOLVE-BLOCK-END
