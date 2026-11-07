"""Clique partition of split graphs: Greedy packing with conflict avoidance."""
from random import Random
import time

def construct():
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _solve_split(n, k, p, seed):
        """
        Split graph: C = [0, k) is a clique, I = [k, n) is independent.
        Strategy:
        1. For each v in I, greedily form cliques {v} U S where S is a subset of N(v).
           S is chosen to be a maximal clique in the graph of available edges in C.
           This avoids double-covering edges in C.
        2. Cover any remaining edges in C with additional cliques.
        """
        # Generate exact adjacency
        rng = Random(seed)
        adj_I = [set() for _ in range(n)]
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    adj_I[v].add(u)
        
        cliques = []
        
        # Track which edges in C are already covered
        # Edge (u, w) with u, w in [0, k)
        covered_C_edges = set()
        
        # Helper to check if a set S forms a valid clique in the remaining available graph
        # i.e., all pairs in S are NOT in covered_C_edges
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
        for v in range(k, n):
            remaining_neighbors = set(adj_I[v])
            
            while remaining_neighbors:
                # Find a maximal subset S of remaining_neighbors such that
                # all edges within S are not covered.
                # Greedy: Start with empty, add vertices one by one if valid.
                # To maximize size, we can try to add vertices in order of high degree in remaining graph?
                # Simple greedy: Just pick any, try to expand.
                
                # Let's try to find a maximal clique in the graph G_rem induced by remaining_neighbors
                # where edge (u,w) exists if (u,w) not in covered_C_edges.
                
                # Since k is small (<=30), we can do a simple greedy expansion.
                S = set()
                # Sort neighbors to be deterministic
                for u in sorted(remaining_neighbors):
                    # Try adding u to S
                    new_S = S | {u}
                    if is_valid_subset(new_S):
                        S = new_S
                
                if not S:
                    # Should not happen, single vertex is always valid
                    S = {next(iter(remaining_neighbors))}
                
                # Form clique {v} U S
                clique = [v] + sorted(S)
                cliques.append(clique)
                
                # Update covered edges
                add_covered_edges(S)
                
                # Remove S from remaining neighbors
                remaining_neighbors -= S
        
        # Cover remaining edges in C
        # These are edges (u, w) in C not in covered_C_edges
        all_C_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                all_C_edges.add((u, w))
        
        rem_edges = all_C_edges - covered_C_edges
        
        # Partition rem_edges into cliques
        # Since rem_edges is a subgraph of a clique, we can just greedily pick maximal cliques
        # Or simply add each edge as a size-2 clique if rem_edges is small.
        # Given the greedy packing above, rem_edges should be very small or empty.
        
        # To be safe and efficient, if rem_edges is small, use size-2 cliques.
        # If large, use greedy maximal clique.
        
        rem_edges_list = list(rem_edges)
        
        # Simple greedy for remaining edges
        while rem_edges_list:
            # Pick first edge
            e = rem_edges_list.pop(0)
            u, w = e
            
            # Try to expand to a clique containing u, w using remaining edges
            S = {u, w}
            
            # Find candidates: vertices x such that (x, u) and (x, w) are in rem_edges
            # Note: rem_edges is a set of tuples.
            # We need to check if (x, y) is in rem_edges for all y in S.
            
            # Since rem_edges is likely small, we can just check all vertices in C
            for x in range(k):
                if x in S:
                    continue
                # Check if x can be added to S
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
            
            # Add clique S
            clique = sorted(S)
            cliques.append(clique)
            
            # Remove all edges within S from rem_edges_list
            S_list = list(S)
            edges_to_remove = set()
            for i in range(len(S_list)):
                for j in range(i + 1, len(S_list)):
                    a, b = S_list[i], S_list[j]
                    if a > b:
                        a, b = b, a
                    edges_to_remove.add((a, b))
            
            rem_edges_list = [e for e in rem_edges_list if e not in edges_to_remove]
        
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    out = []
    for spec in specs:
        out.append(_solve_split(*spec))
    return out
