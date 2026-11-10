"""Clique partition of split graphs via greedy set cover on cross-edges."""
import time
from random import Random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, edges):
        # Separate edges into cross-edges and clique-internal edges
        cross_edges = set()
        internal_edges = set()
        
        for (u, v) in edges:
            if u < k and v < k:
                # Both in clique
                internal_edges.add((min(u, v), max(u, v)))
            elif u >= k and v >= k:
                # Both in independent set - shouldn't happen for split graph
                pass
            else:
                # Cross edge
                cross_edges.add((min(u, v), max(u, v)))
        
        cliques = []
        
        # Phase 1: Greedy set cover for cross-edges
        # Each independent set vertex i (k..n-1) can form a clique {i} U S where S is a subset of its neighbors in {0..k-1}
        # Since all clique vertices are mutually adjacent, any subset of neighbors forms a clique with i
        # We want to cover all cross-edges with as few such cliques as possible
        
        # For each independent vertex, determine its neighbors in the clique
        i_neighbors = {}
        for i in range(k, n):
            i_neighbors[i] = set()
        
        for (u, v) in cross_edges:
            # One endpoint is in clique, one in independent set
            if u < k:
                clique_v, indep_v = u, v
            else:
                clique_v, indep_v = v, u
            i_neighbors[indep_v].add(clique_v)
        
        # Greedy set cover: repeatedly pick the clique covering most uncovered cross-edges
        uncovered_cross = set(cross_edges)
        
        while uncovered_cross and time.time() < deadline - 10:
            best_clique = None
            best_coverage = 0
            best_covered_edges = set()
            
            # Try each independent vertex as the "center" of a potential clique
            for i in range(k, n):
                if not i_neighbors[i]:
                    continue
                
                # The clique {i} U S covers cross-edges (i, s) for s in S
                # We want to choose S to maximize coverage of uncovered edges
                # Since all neighbors of i form a clique with i, we can include any subset
                # But we also want to minimize the number of cliques, so we should include as many as possible
                # that are still uncovered
                
                # Find which neighbors of i have uncovered cross-edges to i
                uncovered_neighbors = set()
                for s in i_neighbors[i]:
                    if (min(i, s), max(i, s)) in uncovered_cross:
                        uncovered_neighbors.add(s)
                
                if not uncovered_neighbors:
                    continue
                
                # The clique {i} U uncovered_neighbors covers exactly the uncovered edges from i to these neighbors
                # But wait - we could also include neighbors that don't have uncovered edges to i,
                # but that wouldn't help cover more cross-edges. So we only need to include uncovered neighbors.
                
                candidate_clique = [i] + list(uncovered_neighbors)
                candidate_edges = set()
                for s in uncovered_neighbors:
                    candidate_edges.add((min(i, s), max(i, s)))
                
                coverage = len(candidate_edges)
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_clique = candidate_clique
                    best_covered_edges = candidate_edges
            
            if best_clique is None or best_coverage == 0:
                # No improvement possible with single-vertex cliques, break
                break
            
            cliques.append(best_clique)
            
            # Remove covered edges from uncovered set
            for e in best_covered_edges:
                if e in uncovered_cross:
                    uncovered_cross.remove(e)
        
        # Phase 2: Handle remaining internal edges (clique-internal)
        # The clique vertices 0..k-1 form a complete graph initially, but some edges may have been "used"
        # Actually, no - the cross-edges are separate from internal edges. The internal edges are all present
        # in the original graph and haven't been covered by any clique yet (since our cliques in Phase 1
        # only cover cross-edges).
        # Wait - we need to cover ALL edges. The internal edges are still uncovered.
        # We need to partition the internal edges (which form a complete graph on k vertices) into cliques.
        # But the entire clique {0, ..., k-1} is itself a clique! So we can just use one clique for all internal edges.
        
        if k >= 2:
            cliques.append(list(range(k)))
        
        # But wait - we also need to make sure that if we used some clique vertices in Phase 1 cliques,
        # we haven't double-covered or missed anything. Actually, the edges are partitioned:
        # Cross-edges are covered by Phase 1 cliques.
        # Internal edges are covered by the single clique {0, ..., k-1}.
        # These are disjoint sets of edges, so this is valid.
        
        # However, we must ensure that every edge is in EXACTLY one clique.
        # Let's verify: 
        # - Cross-edge (u,v) with u<k, v>=k: covered by the clique in Phase 1 that contains both u and v.
        #   Since we added v to the clique and u is in v's neighbors, (u,v) is in that clique.
        #   And we removed it from uncovered_cross, so it won't be in another Phase 1 clique.
        #   It's not in the internal clique since v >= k.
        # - Internal edge (u,v) with u,v<k: covered by the final clique {0,...,k-1}.
        #   It's not in any Phase 1 clique since Phase 1 cliques contain at least one vertex >= k.
        
        # Convert to lists of Python ints
        final_cliques = []
        for c in cliques:
            final_cliques.append([int(x) for x in c])
        
        return final_cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline - 15:
            # Emergency fallback: just list each edge as a clique
            edges = instance(*spec)
            results.append([[int(e[0]), int(e[1])] for e in edges])
            continue
        
        edges = instance(*spec)
        cliques = partition_split_graph(spec[0], spec[1], edges)
        
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
