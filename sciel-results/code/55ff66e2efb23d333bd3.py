"""Clique partitions of three fixed split (chordal) graphs."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random

    t0 = time.time()
    DEADLINE = t0 + 110.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def partition_cliques(n, k, p, seed):
        edges = instance(n, k, p, seed)
        cliques = []
        
        # Compute N(v) for each v in I
        N = {}
        for v in range(k, n):
            nbrs = set()
            for u in range(k):
                if (u, v) in edges:
                    nbrs.add(u)
            N[v] = nbrs
        
        # Track which C-C edges are covered
        covered_cc = set()
        
        # Helper to get all C-C edges within a set of vertices
        def get_cc_edges(vertices):
            verts = sorted(vertices)
            res = set()
            for i in range(len(verts)):
                for j in range(i+1, len(verts)):
                    res.add((verts[i], verts[j]))
            return res

        # Process each v in I
        for v in range(k, n):
            nbrs = N[v]
            if not nbrs:
                continue
            
            # We need to cover all cross edges (u, v) for u in nbrs.
            # This requires u to be in a clique with v.
            # We partition nbrs into subsets S such that the C-C edges within S
            # are not already covered.
            
            remaining_nbrs = set(nbrs)
            
            while remaining_nbrs:
                # Find a maximal subset S of remaining_nbrs such that
                # all C-C edges within S are not in covered_cc.
                # This is equivalent to finding a maximal clique in the graph
                # induced by remaining_nbrs where edges exist if NOT in covered_cc.
                
                # Simple greedy for maximal clique in the "available" graph
                # Graph G_avail on remaining_nbrs: edge (u,w) exists if (u,w) not in covered_cc.
                # We want a large clique.
                
                # Start with an arbitrary vertex
                start = next(iter(remaining_nbrs))
                S = {start}
                
                # Try to add vertices to S one by one
                # To make it efficient, we can just pick vertices that are connected
                # to all current members of S in G_avail.
                
                changed = True
                while changed:
                    changed = False
                    candidates = list(remaining_nbrs - S)
                    for u in candidates:
                        # Check if u is connected to all in S in G_avail
                        is_connected = True
                        for w in S:
                            e = (min(u, w), max(u, w))
                            if e in covered_cc:
                                is_connected = False
                                break
                        if is_connected:
                            S.add(u)
                            changed = True
                            # We could break or continue to find a larger one.
                            # Let's continue to find a maximal one.
                
                # Form clique S U {v}
                clique = sorted(S) + [v]
                cliques.append(clique)
                
                # Mark C-C edges within S as covered
                new_edges = get_cc_edges(S)
                covered_cc.update(new_edges)
                
                # Remove S from remaining_nbrs
                remaining_nbrs -= S
        
        # Now cover remaining C-C edges
        remaining_edges = []
        for u in range(k):
            for u2 in range(u+1, k):
                if (u, u2) in edges and (u, u2) not in covered_cc:
                    remaining_edges.append((u, u2))
        
        # Cover remaining edges with 2-cliques for simplicity and correctness
        for e in remaining_edges:
            cliques.append(list(e))
        
        return cliques

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > DEADLINE:
            break
        cliques = partition_cliques(*spec)
        cleaned = []
        for c in cliques:
            cl = [int(x) for x in c]
            if len(cl) >= 2 and len(set(cl)) == len(cl):
                cleaned.append(cl)
        result.append(cleaned)

    return result
# EVOLVE-BLOCK-END
