"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    DEADLINE = 145  # Safety margin from 150s limit
    
    def instance(n, p, seed):
        rng = Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

    def bron_kerbosch_maximal_cliques(adj, n):
        """
        Enumerates all maximal cliques in a graph using Bron-Kerbosch with pivot.
        adj: list of sets of neighbors for each vertex.
        n: number of vertices.
        Returns a list of sets, each set is a maximal clique.
        """
        cliques = []
        
        def expand(R, P, X):
            if not P and not X:
                if len(R) >= 2: # Only care about cliques of size >= 2
                    cliques.append(frozenset(R))
                return
            
            # Choose pivot u from P union X to minimize branching
            P_union_X = P.union(X)
            if P_union_X:
                # Pick vertex with max degree in P_union_X
                u = max(P_union_X, key=lambda v: len(adj[v].intersection(P_union_X)))
                candidates = P.difference(adj[u])
            else:
                candidates = P
            
            for v in list(candidates):
                # R union {v}
                new_R = R.union([v])
                # P intersection adj[v]
                new_P = P.intersection(adj[v])
                # X intersection adj[v]
                new_X = X.intersection(adj[v])
                
                expand(new_R, new_P, new_X)
                
                # Move v from P to X
                P.remove(v)
                X.add(v)
        
        # Initial call: R=empty, P=all vertices, X=empty
        P_initial = set(range(n))
        expand(set(), P_initial, set())
        return cliques

    def solve_min_transversal(n, cliques):
        """
        Finds a small set of vertices that intersects every clique in `cliques`.
        Uses Greedy Set Cover followed by Local Search.
        """
        if not cliques:
            return []
        
        # Map clique to index for fast lookup
        clique_indices = {c: i for i, c in enumerate(cliques)}
        
        # Precompute for each vertex, which cliques it covers
        vertex_cliques = [set() for _ in range(n)]
        for c in cliques:
            idx = clique_indices[c]
            for v in c:
                vertex_cliques[v].add(idx)
        
        # Greedy Set Cover
        selected = set()
        covered = set()
        remaining_cliques = set(range(len(cliques)))
        
        # To speed up, we can sort vertices by initial coverage, but dynamic selection is better.
        # We'll just pick the best vertex each time.
        # Optimization: Instead of iterating all n vertices each step, we can maintain a count.
        # Given small n (<=40), O(n * |cliques|) per step is fine.
        
        while remaining_cliques:
            best_v = -1
            best_count = -1
            for v in range(n):
                if v in selected:
                    continue
                # Count how many remaining cliques v covers
                count = len(vertex_cliques[v].intersection(remaining_cliques))
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v == -1:
                break # Should not happen if graph is connected enough, but safe
            selected.add(best_v)
            # Update remaining cliques
            remaining_cliques.difference_update(vertex_cliques[best_v])
        
        # Local Search: Remove redundant vertices
        changed = True
        while changed:
            changed = False
            for v in list(selected):
                # Check if v is redundant
                # v is redundant if all cliques it covers are covered by other selected vertices
                is_redundant = True
                for c_idx in vertex_cliques[v]:
                    # Check if c_idx is covered by some other u in selected - {v}
                    covered_by_other = False
                    for u in selected:
                        if u == v:
                            continue
                        if c_idx in vertex_cliques[u]:
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        is_redundant = False
                        break
                if is_redundant:
                    selected.remove(v)
                    changed = True
            
            # Check time
            if time.time() > DEADLINE:
                break
        
        # Local Search: Swap to potentially enable further removals
        # This is more complex. A simple swap: replace v with u if valid.
        # If we swap, size is same. But it might make another vertex redundant.
        # Given time constraints, let's try a few iterations of swap+remove.
        
        for _ in range(10): # Limit iterations
            if time.time() > DEADLINE:
                break
            improved = False
            for v in list(selected):
                # Try replacing v with u
                for u in range(n):
                    if u in selected:
                        continue
                    # Tentative swap
                    new_selected = set(selected)
                    new_selected.remove(v)
                    new_selected.add(u)
                    
                    # Check validity
                    valid = True
                    for c in cliques:
                        # Does new_selected intersect c?
                        if not new_selected.intersection(c):
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # If valid, perform swap and try to reduce
                    selected = new_selected
                    improved = True
                    # Try to remove redundant vertices again
                    sub_changed = True
                    while sub_changed:
                        sub_changed = False
                        for w in list(selected):
                            is_redundant = True
                            for c_idx in vertex_cliques[w]:
                                covered_by_other = False
                                for z in selected:
                                    if z == w:
                                        continue
                                    if c_idx in vertex_cliques[z]:
                                        covered_by_other = True
                                        break
                                if not covered_by_other:
                                    is_redundant = False
                                    break
                            if is_redundant:
                                selected.remove(w)
                                sub_changed = True
                    break # Restart outer loop if we made a change
            if not improved:
                break

        return sorted(selected)

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > DEADLINE:
            # Fallback to trivial solution if time runs out
            edges = instance(n, p, seed)
            touched = {x for e in edges for x in e}
            out.append(sorted(touched))
            continue
            
        edges = instance(n, p, seed)
        
        # Build adjacency list
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Enumerate maximal cliques
        cliques = bron_kerbosch_maximal_cliques(adj, n)
        
        # Solve transversal
        transversal = solve_min_transversal(n, cliques)
        
        # Validation: Ensure every maximal clique of size >= 2 is hit
        # This is a safety check. If it fails, fall back to all vertices.
        valid = True
        for c in cliques:
            if not set(transversal).intersection(c):
                valid = False
                break
        if not valid:
            # Fallback: use all vertices involved in any edge
            touched = {x for e in edges for x in e}
            transversal = sorted(touched)
            
        out.append(transversal)
        
    return out
# EVOLVE-BLOCK-END
