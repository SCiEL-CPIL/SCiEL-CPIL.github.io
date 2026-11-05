import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin for 120s limit

    def random_graph(n, p, seed):
        from random import Random
        rng = Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(edges_set, n):
        if not edges_set:
            return []
        
        # 1. Build adjacency structure for fast clique expansion
        # adj[i] is a set of neighbors j > i? No, let's use a list of sets for all neighbors.
        adj = [set() for _ in range(n)]
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
            
        # 2. Enumerate all maximal cliques using Bron-Kerbosch with Pivoting
        # In dense graphs (p=0.55, n=64), the number of maximal cliques is manageable.
        # For sparse graphs, this could explode, but the problem guarantees dense graphs.
        
        all_maximal_cliques = []
        
        def bron_kerbosch(R, P, X):
            if time.time() > DEADLINE:
                return
            
            if not P and not X:
                # R is a maximal clique
                if len(R) >= 2:
                    all_maximal_cliques.append(list(R))
                return
            
            # Choose pivot u from P union X to minimize branching
            # We want u that maximizes |P ∩ N(u)|
            U = P.union(X)
            if not U:
                return
            
            # Simple heuristic for pivot: pick node in U with max degree in P
            # Since U is a set, iterating is O(|U|). |U| <= n.
            best_u = None
            max_inter = -1
            for u in U:
                # Count how many nodes in P are neighbors of u
                # P is a set, adj[u] is a set
                inter = len(P.intersection(adj[u]))
                if inter > max_inter:
                    max_inter = inter
                    best_u = u
            
            # Branch on all v in P \ N(best_u)
            # If best_u is in P, P \ N(best_u) excludes best_u's neighbors in P.
            # If best_u is not in P, we branch on all P.
            if best_u in P:
                to_branch = P.difference(adj[best_u])
            else:
                to_branch = P
            
            for v in list(to_branch):
                bron_kerbosch(R.union([v]), P.intersection(adj[v]), X.intersection(adj[v]))
                P.remove(v)
                X.add(v)
        
        # Initialize: R = empty, P = all vertices, X = empty
        all_vertices = set(range(n))
        bron_kerbosch(set(), all_vertices, set())
        
        # 3. Greedy Set Cover
        # We have a universe of edges: edges_set
        # We have a collection of sets: all_maximal_cliques (each clique covers the edges within it)
        # We want to select the minimum number of cliques such that every edge is covered by exactly one selected clique.
        # Note: The problem requires "every edge must lie in exactly one listed clique".
        # Standard Set Cover allows an element to be covered by multiple sets. 
        # However, if we select a clique C, it covers all edges in C. 
        # If two selected cliques share an edge, that edge is in two cliques, which is INVALID.
        # Therefore, we must select a set of pairwise edge-disjoint cliques.
        
        # This is the "Exact Cover" or "Packing" variant.
        # Greedy approach:
        # 1. Calculate for each maximal clique the set of edges it covers.
        # 2. Filter out cliques that are subsets of other cliques? 
        #    Actually, if C1 is a subset of C2, C2 covers more edges. 
        #    But we need disjointness. 
        #    If we pick C2, we cannot pick any clique that shares an edge with C2.
        
        # Let's represent each maximal clique by its edge set.
        clique_edges = []
        for clique in all_maximal_cliques:
            c_edges = set()
            for i in range(len(clique)):
                for j in range(i+1, len(clique)):
                    u, v = clique[i], clique[j]
                    if u > v: u, v = v, u
                    c_edges.add((u, v))
            # Only keep if it's a valid clique in the original graph (it is, by construction)
            # And non-empty
            if c_edges:
                clique_edges.append(c_edges)
        
        # Sort cliques by size descending? Or by some heuristic?
        # Larger cliques cover more edges, so they are "better" if they don't conflict.
        # Let's sort by size descending to greedily pick big ones first.
        clique_edges.sort(key=lambda x: len(x), reverse=True)
        
        uncovered = set(edges_set)
        selected_cliques = []
        
        # We need to be careful: if we pick a clique, we remove its edges.
        # Any other clique that has ANY edge in common with the picked clique is now "conflicting" 
        # if we were to pick it later? No, if we pick C1, we remove C1's edges. 
        # If C2 shares an edge e with C1, e is now removed from uncovered.
        # C2 might still have other edges in uncovered. 
        # BUT, the condition is "every edge must lie in EXACTLY one listed clique".
        # If we select C1 and C2, and they share edge e, then e is in C1 and e is in C2. Violation.
        # So we cannot select two cliques that share an edge.
        
        # This makes it a "Maximum Set Packing" problem on the edge sets, with the goal of covering all edges.
        # Since we must cover ALL edges, it's a "Set Partition" problem where the sets must be disjoint.
        
        # Greedy Set Partition:
        # 1. Start with all edges uncovered.
        # 2. Pick a clique C that covers the most uncovered edges.
        # 3. Remove C's edges from uncovered.
        # 4. Mark C as "used". Any other clique that shares an edge with C is now "invalid" 
        #    because it would cause a conflict if selected later? 
        #    Wait. If C1 and C2 share an edge, we can't pick both.
        #    If we pick C1, we can still pick C3 if C3 is disjoint from C1.
        #    What about C2? C2 shares an edge with C1. So C2 is forbidden.
        #    What if C2 shares an edge with C3? C3 is disjoint from C1, so C3 is allowed.
        #    But C2 is forbidden because of C1.
        
        # So, we maintain a list of "available" cliques.
        # When we pick a clique C, we remove all edges in C from 'uncovered'.
        # We also remove from 'available' any clique that intersects with C (on edges).
        
        # Optimization: 
        # Instead of checking intersection with all remaining cliques (O(K^2 * E)), 
        # we can just check if a clique has ANY edge in 'uncovered' that is NOT in C? 
        # No, that's not right. 
        # If C1 and C2 share edge e. We pick C1. e is removed from uncovered.
        # C2 still has other edges in uncovered. 
        # If we pick C2 later, e is not in uncovered, so C2 would only cover its other edges.
        # BUT, the output list would contain C1 and C2. 
        # The verifier checks: "every edge must lie in exactly one listed clique".
        # Edge e is in C1 (listed) and in C2 (listed). Violation.
        # So yes, we must forbid any clique that shares ANY edge with a previously selected clique.
        
        # Algorithm:
        available_cliques = [set(c) for c in clique_edges] # List of sets of edges
        # To speed up intersection checks, we might just iterate. K is small (~hundreds).
        
        while uncovered:
            if time.time() > DEADLINE:
                # Fallback: cover remaining edges as singletons
                for e in uncovered:
                    selected_cliques.append([e[0], e[1]])
                break
            
            # Find the available clique that covers the most uncovered edges
            best_clique_idx = -1
            best_count = 0
            
            for i, c_edges in enumerate(available_cliques):
                if not c_edges:
                    continue
                # Count how many edges in c_edges are in uncovered
                # intersection = len(c_edges & uncovered)
                # If this is 0, skip
                # We also need to ensure this clique doesn't conflict with already selected ones?
                # We already removed conflicting cliques from available_cliques?
                # No, we didn't implement that removal yet. 
                # Let's do it: 
                # When we select a clique, we remove from available_cliques any clique that intersects with it.
                
                inter_count = len(c_edges & uncovered)
                if inter_count > best_count:
                    best_count = inter_count
                    best_clique_idx = i
            
            if best_clique_idx == -1 or best_count == 0:
                # No clique covers any uncovered edge? 
                # This implies the uncovered edges don't form any clique found in our maximal list?
                # That shouldn't happen if we found ALL maximal cliques. 
                # Every edge is in at least one maximal clique.
                # If best_count is 0, it means the maximal cliques containing these edges 
                # were already "blocked" by previous selections?
                # If so, we have a conflict. 
                # Fallback: cover remaining edges individually.
                for e in uncovered:
                    selected_cliques.append([e[0], e[1]])
                break
            
            selected_c_edges = available_cliques[best_clique_idx]
            
            # Add this clique to result
            # We need the vertex list, not just edge set. 
            # We stored edge sets in available_cliques. 
            # We need to map back to vertices or store vertices alongside.
            # Let's store (vertex_list, edge_set) in available_cliques.
            
            # Re-implementing available_cliques structure to hold vertices
            # This requires refactoring the loop above.
            
            # Let's restart the inner loop logic with better data structure.
            break # Break out of while to refactor
            
        # --- Refactored Logic ---
        
        # Re-initialize
        uncovered = set(edges_set)
        selected_cliques = []
        
        # Structure: list of (vertex_list, edge_set)
        candidates = []
        for clique in all_maximal_cliques:
            c_edges = set()
            for i in range(len(clique)):
                for j in range(i+1, len(clique)):
                    u, v = clique[i], clique[j]
                    if u > v: u, v = v, u
                    c_edges.add((u, v))
            if c_edges:
                candidates.append((list(clique), c_edges))
        
        # Sort by size descending initially
        candidates.sort(key=lambda x: len(x[1]), reverse=True)
        
        while uncovered:
            if time.time() > DEADLINE:
                for e in uncovered:
                    selected_cliques.append([e[0], e[1]])
                break
            
            best_idx = -1
            best_cover = 0
            
            for i, (v_list, c_edges) in enumerate(candidates):
                if not c_edges:
                    continue
                # Check if this clique conflicts with already selected ones?
                # We remove conflicting candidates from the list, so we don't need to check.
                
                cover_count = len(c_edges & uncovered)
                if cover_count > best_cover:
                    best_cover = cover_count
                    best_idx = i
            
            if best_idx == -1 or best_cover == 0:
                # Fallback
                for e in uncovered:
                    selected_cliques.append([e[0], e[1]])
                break
            
            selected_v, selected_e = candidates[best_idx]
            selected_cliques.append(selected_v)
            
            # Remove covered edges from uncovered
            uncovered.difference_update(selected_e)
            
            # Remove conflicting candidates
            # A candidate conflicts if it shares ANY edge with selected_e
            # Note: It doesn't matter if the shared edge is in uncovered or not.
            # If it shares an edge with a selected clique, it's invalid.
            
            new_candidates = []
            for i, (v_list, c_edges) in enumerate(candidates):
                if i == best_idx:
                    continue
                if c_edges & selected_e:
                    # Conflict
                    continue
                new_candidates.append((v_list, c_edges))
            
            candidates = new_candidates
            
            # Optimization: If candidates is empty, we can't cover remaining edges with maximal cliques
            # without conflict. This implies the remaining edges must be covered by smaller cliques
            # that were not maximal? No, any clique is contained in a maximal clique.
            # If all maximal cliques covering an edge conflict with selected ones, 
            # it means the edge is "trapped". 
            # In that case, we must cover it with a singleton or small clique.
            if not candidates:
                # Cover remaining edges greedily as singletons or small groups?
                # Singletons are safe.
                for e in uncovered:
                    selected_cliques.append([e[0], e[1]])
                break

        return selected_cliques

    # Instance 1
    n1, p1, s1 = 32, 0.6, 601
    e1 = random_graph(n1, p1, s1)
    c1 = solve_instance(e1, n1)
    
    # Instance 2
    n2, p2, s2 = 64, 0.55, 602
    e2 = random_graph(n2, p2, s2)
    c2 = solve_instance(e2, n2)
    
    return [c1, c2]

if __name__ == "__main__":
    res = construct()
    print(len(res[0]), len(res[1]))
