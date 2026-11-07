"""Clique partition of split graphs: Greedy packing with local improvement."""
from random import Random
import time
import itertools

def construct():
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _solve_split(n, k, p, seed, deadline):
        """
        Split graph: C = [0, k) is a clique, I = [k, n) is independent.
        """
        # Generate exact adjacency
        rng = Random(seed)
        adj_I = [set() for _ in range(n)]
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    adj_I[v].add(u)
        
        # adj_C is a complete graph, so we don't need to store it explicitly, 
        # but we need to track which edges are covered.
        
        cliques = []
        
        # Track which edges in C are already covered
        covered_C_edges = set()
        
        # Helper to check if a set S forms a valid clique in the remaining available graph
        def is_valid_subset(S):
            S_list = list(S)
            for i in range(len(S_list)):
                for j in range(i + 1, len(S_list)):
                    u, w = S_list[i], S_list[j]
                    if u > w:
                        u, w = w, u
                    if (u, w) in covered_C_edges:
                        return False
            return True

        def add_covered_edges(S):
            S_list = list(S)
            for i in range(len(S_list)):
                for j in range(i + 1, len(S_list)):
                    u, w = S_list[i], S_list[j]
                    if u > w:
                        u, w = w, u
                    covered_C_edges.add((u, w))

        # Process each vertex in I
        # Order I vertices by degree (descending) to potentially pack larger cliques first?
        # Or ascending? Let's try descending degree for I vertices to handle "dense" ones first.
        i_vertices = list(range(k, n))
        i_vertices.sort(key=lambda v: len(adj_I[v]), reverse=True)
        
        for v in i_vertices:
            remaining_neighbors = set(adj_I[v])
            
            while remaining_neighbors:
                if time.time() > deadline:
                    # Fallback: cover rest with 2-cliques
                    for u in remaining_neighbors:
                        cliques.append([v, u])
                    break
                
                # Find a maximal subset S of remaining_neighbors such that
                # all edges within S are not covered.
                S = set()
                # Sort neighbors to be deterministic
                for u in sorted(remaining_neighbors):
                    new_S = S | {u}
                    if is_valid_subset(new_S):
                        S = new_S
                
                if not S:
                    S = {next(iter(remaining_neighbors))}
                
                # Form clique {v} U S
                clique = [v] + sorted(S)
                cliques.append(clique)
                
                # Update covered edges
                add_covered_edges(S)
                
                # Remove S from remaining neighbors
                remaining_neighbors -= S
            
            # If broke out of while due to timeout, cover remaining N_i with 2-cliques
            if remaining_neighbors:
                for u in remaining_neighbors:
                    cliques.append([v, u])

        # Cover remaining edges in C
        all_C_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                all_C_edges.add((u, w))
        
        rem_edges = all_C_edges - covered_C_edges
        
        # Partition rem_edges into cliques
        rem_edges_list = list(rem_edges)
        
        # Simple greedy for remaining edges
        while rem_edges_list:
            if time.time() > deadline:
                for e in rem_edges_list:
                    cliques.append(sorted([e[0], e[1]]))
                break
            
            e = rem_edges_list.pop(0)
            u, w = e
            S = {u, w}
            
            for x in range(k):
                if x in S:
                    continue
                can_add = True
                for y in S:
                    a, b = x, y
                    if a > b:
                        a, b = b, a
                    if (a, b) not in rem_edges:
                        can_add = False
                        break
                if can_add:
                    S.add(x)
            
            clique = sorted(S)
            cliques.append(clique)
            
            S_list = list(S)
            edges_to_remove = set()
            for i in range(len(S_list)):
                for j in range(i + 1, len(S_list)):
                    a, b = S_list[i], S_list[j]
                    if a > b:
                        a, b = b, a
                    edges_to_remove.add((a, b))
            
            rem_edges_list = [e for e in rem_edges_list if e not in edges_to_remove]

        # Local Improvement Phase
        # 1. Try to merge cliques
        # A merge is possible if:
        # - Both are pure C-cliques and their union is a clique (always true if edges are available? No, they must be disjoint in edges, but vertices can overlap? No, partition of edges means vertices can be in multiple cliques, but edges are unique. Wait, the problem says "partition of edges". Vertices can be in multiple cliques.
        # - One is {v} U S and the other is {v'} U S'. Cannot merge if v != v' because v, v' not connected.
        # - One is {v} U S and the other is S' (pure C). Can merge if v is connected to all u in S'. The new clique is {v} U S U S'.
        #   This is valid if S U S' is a clique in C (always true) and v is connected to all in S'.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(cliques)):
                for j in range(i + 1, len(cliques)):
                    c1 = set(cliques[i])
                    c2 = set(cliques[j])
                    
                    # Check if union is a clique
                    # Condition:
                    # 1. No two vertices in c1 U c2 are non-adjacent.
                    # 2. Since it's a split graph, if c1 and c2 both contain I-vertices, they must be the same I-vertex (else not adjacent).
                    
                    v1 = [x for x in c1 if x >= k]
                    v2 = [x for x in c2 if x >= k]
                    
                    if len(v1) > 1 or len(v2) > 1:
                        continue # Should not happen
                    
                    if v1 and v2:
                        if v1[0] != v2[0]:
                            continue # Not adjacent
                        # Same I-vertex, so union is {v} U (S1 U S2).
                        # Always a clique if S1 U S2 is a clique in C.
                        # S1, S2 are subsets of C, so S1 U S2 is a clique in C.
                        # So we can always merge two cliques containing the same I-vertex.
                        new_clique = sorted(c1 | c2)
                        # Remove c1, c2, add new_clique
                        cliques.pop(j)
                        cliques.pop(i)
                        cliques.append(new_clique)
                        improved = True
                        break
                    elif v1:
                        # c1 has I-vertex v, c2 is pure C
                        v = v1[0]
                        # Check if v is connected to all in c2
                        if all(u in adj_I[v] for u in c2):
                            new_clique = sorted(c1 | c2)
                            cliques.pop(j)
                            cliques.pop(i)
                            cliques.append(new_clique)
                            improved = True
                            break
                    elif v2:
                        # c2 has I-vertex v, c1 is pure C
                        v = v2[0]
                        if all(u in adj_I[v] for u in c1):
                            new_clique = sorted(c1 | c2)
                            cliques.pop(j)
                            cliques.pop(i)
                            cliques.append(new_clique)
                            improved = True
                            break
                    else:
                        # Both pure C. Union is always a clique in C.
                        # But we must ensure we don't double cover edges?
                        # The problem is edge partition.
                        # If we merge two pure C cliques, we are just combining the sets of vertices.
                        # The edges covered by the new clique are all pairs in c1 U c2.
                        # The edges covered by c1 were pairs in c1.
                        # The edges covered by c2 were pairs in c2.
                        # The new clique covers pairs in c1, pairs in c2, AND pairs between c1 and c2.
                        # The pairs between c1 and c2 were NOT covered by c1 or c2 (since they were separate cliques in the partition).
                        # So merging them would "double cover" the edges between c1 and c2 if they were already covered?
                        # No, in a partition, each edge is in exactly one clique.
                        # If c1 and c2 are disjoint in edges, then no edge between c1 and c2 is covered by c1 or c2.
                        # But wait, if an edge (u, v) with u in c1, v in c2 exists in the graph, it MUST be covered by some clique.
                        # If it was covered by c1 or c2, then u and v would have to be in the same clique, which they are not.
                        # So the edge (u, v) must be covered by another clique, or it's not in the graph (impossible, C is a clique).
                        # So if we merge c1 and c2, we are covering the edges between them.
                        # But those edges were already covered by some other clique in the partition!
                        # So we CANNOT simply merge two pure C cliques unless the edges between them are NOT covered by any other clique.
                        # This implies we need to track edge coverage carefully.
                        # My current `cliques` list is a partition, so each edge is in exactly one clique.
                        # If I merge c1 and c2, the new clique covers edges within c1, within c2, and between c1 and c2.
                        # The edges between c1 and c2 were previously covered by some other clique(s).
                        # So I would be double-covering them.
                        # Thus, I can only merge if the edges between c1 and c2 are NOT covered by any other clique.
                        # This is hard to check without the full edge coverage map.
                        # Let's skip merging pure C cliques for now to be safe.
                        pass
            
            if improved:
                continue
        
        # 2. Try to move vertices?
        # This is complex. Let's stick to merging for now.
        
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    out = []
    start_time = time.time()
    deadline = start_time + 140.0
    
    for spec in specs:
        out.append(_solve_split(*spec, deadline))
    
    # Ensure types are correct
    final_results = []
    for cliques in out:
        clean_cliques = []
        for clique in cliques:
            clean_cliques.append([int(v) for v in clique])
        final_results.append(clean_cliques)
    
    return final_results
