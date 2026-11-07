"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        # Clique on first k
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        # Cross edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return sorted(edges)

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback to seed-like result if timeout
            # But we should try to finish current
            pass
            
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        edge_set = set(edges)
        
        # Separate internal and cross edges
        internal_edges = set()
        cross_edges = {} # v -> set of u
        
        for u, v in edges:
            if u < k and v < k:
                internal_edges.add((u, v))
            else:
                # One is in C, one in I
                if u < k:
                    cross_edges.setdefault(v, set()).add(u)
                else:
                    cross_edges.setdefault(u, set()).add(v)
                    
        uncovered_internal = set(internal_edges)
        cliques = []
        
        # Process each vertex in I
        for v in range(k, n):
            if v not in cross_edges:
                continue
            
            neighbors = cross_edges[v]
            # We need to cover all neighbors with cliques {v} U S
            # Greedy: pick largest subset S of remaining neighbors
            # such that all internal edges within S are in uncovered_internal
            
            remaining = set(neighbors)
            
            while remaining:
                # Try to find a large clique in the graph of 'remaining'
                # where edges exist in uncovered_internal
                # Actually, any subset is a clique in C.
                # We just need the edges to be uncovered.
                
                # Heuristic: Pick one node, then try to add others if edges are uncovered
                # Or just pick all if no conflicts?
                # Conflict: if (u1, u2) is in uncovered_internal, we can include both.
                # If (u1, u2) is NOT in uncovered_internal (i.e., already covered or doesn't exist),
                # we cannot include both in the SAME clique if we want to claim that edge?
                # No, if (u1, u2) is already covered, we can't include it again.
                # But if we include u1 and u2 in the clique, the edge (u1, u2) is "in" the clique.
                # So we must ensure (u1, u2) is in uncovered_internal.
                
                # So we need to find a subset S of remaining such that
                # for all u, w in S, (u, w) in uncovered_internal.
                
                # Since p is high, most edges are uncovered initially.
                # Let's try to take all remaining if possible.
                valid_subset = set(remaining)
                
                # Check if all pairs in valid_subset have edges in uncovered_internal
                # If not, we need to remove some.
                # For simplicity and speed, let's just take one vertex at a time if we can't be sure,
                # OR try to take all, and if conflict, shrink.
                
                # Efficient check:
                # Iterate through pairs.
                # If any pair is not in uncovered_internal, we have a conflict.
                
                # Let's try a simple greedy:
                # Start with empty S.
                # For each u in remaining:
                #   Check if (u, w) is in uncovered_internal for all w in current S.
                #   If yes, add u to S.
                
                S = set()
                for u in remaining:
                    ok = True
                    for w in S:
                        e = (min(u, w), max(u, w))
                        if e not in uncovered_internal:
                            ok = False
                            break
                    if ok:
                        S.add(u)
                    
                if not S:
                    # Fallback: take one
                    u = next(iter(remaining))
                    S = {u}
                
                # Form clique
                clique = [v] + list(S)
                cliques.append(clique)
                
                # Mark internal edges in S as covered
                for i in range(len(S)):
                    for j in range(i+1, len(S)):
                        u1 = list(S)[i]
                        u2 = list(S)[j]
                        e = (min(u1, u2), max(u1, u2))
                        if e in uncovered_internal:
                            uncovered_internal.remove(e)
                            
                remaining -= S
                
        # Cover remaining internal edges
        # These are edges in C not covered by I-cliques
        # We can group them into cliques.
        # Since C is a clique, we can just take all remaining vertices?
        # No, we only need to cover the edges.
        # If we take a set of vertices V_rem, the clique covers all edges within V_rem.
        # We want to cover exactly the edges in uncovered_internal.
        # If we take V_rem = all vertices incident to uncovered_internal,
        # we might cover extra edges that are already covered?
        # No, if an edge is already covered, it's not in uncovered_internal.
        # But if we form a clique on V_rem, it includes ALL edges between vertices in V_rem.
        # If some edge (u,w) in V_rem is NOT in uncovered_internal (i.e. already covered),
        # then adding it to the new clique would double-cover it.
        
        # So we must be careful.
        # If uncovered_internal is small, we can just add edges as size-2 cliques.
        # This is safe.
        for e in uncovered_internal:
            cliques.append(list(e))
            
        results.append(cliques)
        
    return results

# EVOLVE-BLOCK-END
