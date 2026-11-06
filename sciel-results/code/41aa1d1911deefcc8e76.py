"""
Clique Partition via Weighted Greedy Set Cover and Local Refinement.
Strategy:
1. Pre-compute all valid cliques of size 3, 4, and 5.
   (Size 2 is the base case for any remaining edge).
2. Formulate as a Set Cover problem:
   - Universe U: All edges in the graph.
   - Sets S: Each clique C covers the set of edges induced by C.
   - Cost: Each clique has cost 1.
3. Use Weighted Greedy Set Cover:
   - Repeatedly pick the clique that covers the most *uncovered* edges per unit cost (which is 1).
   - This is equivalent to maximizing newly covered edges.
4. Local Refinement:
   - After the greedy phase, try to replace k cliques in the cover with 1 larger clique 
     if the union of their vertices forms a valid clique and covers exactly the union of their edges.
   - This handles cases where greedy made suboptimal local choices.
"""
import time
import random
from itertools import combinations

_START = time.time()
DEADLINE = 145.0

def _remaining():
    return DEADLINE - time.time()

def _generate_graph(n, p, seed):
    """Generate the same graph as the verifier."""
    rng = random.Random(seed)
    adj = [[False] * n for _ in range(n)]
    edge_list = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u][v] = adj[v][u] = True
                edge_list.append((u, v))
    return adj, edge_list

def _is_clique(vertices, adj):
    """Check if a list of vertices forms a clique."""
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            if not adj[vertices[i]][vertices[j]]:
                return False
    return True

def _find_all_cliques(adj, n, max_size=5):
    """
    Find all cliques of size 3 up to max_size.
    We don't need size 2 explicitly in the 'candidate sets' for the greedy 
    because any edge not covered by a larger clique can just be its own set.
    However, including size 2 allows the greedy to potentially merge edges 
    into triangles if beneficial, but usually size >= 3 is better for count minimization.
    Actually, for Set Cover, we want to minimize the number of sets.
    A triangle (3 vertices, 3 edges) is better than 3 edges (3 sets).
    A K4 (4 vertices, 6 edges) is better than 4 triangles (4 sets) or 6 edges.
    
    We will generate candidates of size 3, 4, and 5.
    """
    candidates = []
    vertices = list(range(n))
    
    # Size 3
    for tri in combinations(vertices, 3):
        if _is_clique(tri, adj):
            candidates.append(frozenset(tri))
            
    # Size 4
    for quad in combinations(vertices, 4):
        if _is_clique(quad, adj):
            candidates.append(frozenset(quad))
            
    # Size 5
    if max_size >= 5:
        for pent in combinations(vertices, 5):
            if _is_clique(pent, adj):
                candidates.append(frozenset(pent))
                
    return candidates

def _clique_edges(clique, adj):
    """Return the set of edges covered by a clique."""
    edges = set()
    verts = list(clique)
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            edges.add((min(verts[i], verts[j]), max(verts[i], verts[j])))
    return edges

def _solve_set_cover(adj, edge_list, n):
    """
    Greedy Set Cover to partition edges into cliques.
    """
    if not edge_list:
        return []
        
    # Map edge to index for fast lookup? No, use sets of tuples.
    all_edges = set(edge_list)
    uncovered_edges = set(all_edges)
    
    # Generate candidate cliques (size 3, 4, 5)
    # Note: Generating all combinations for n=64 size 5 is C(64,5) ~ 7.6M. 
    # Checking clique validity is O(k^2). This might be too slow.
    # Optimization: Only generate candidates from high-degree regions or use a faster method.
    # For n=32, C(32,5) ~ 201k. This is fine.
    # For n=64, C(64,5) ~ 7.6M. This is likely too slow for Python in 150s if done naively.
    
    # Alternative: Generate candidates on the fly or restrict to "dense" subgraphs?
    # Or, just use size 3 and 4 for n=64 to save time, and rely on local search to find size 5?
    # Let's check timing. 7.6M * 10 ops ~ 76M ops. In Python, ~10-50s. Risky.
    
    # Strategy:
    # 1. If n <= 40, generate size 3, 4, 5.
    # 2. If n > 40, generate size 3, 4. And maybe size 5 only if time permits or via local search.
    
    max_size = 5 if n <= 40 else 4
    
    candidates = _find_all_cliques(adj, n, max_size=max_size)
    
    # Pre-compute edges for each candidate
    cand_edges = {}
    for c in candidates:
        cand_edges[c] = _clique_edges(c, adj)
        
    chosen_cliques = []
    
    # Greedy Phase
    # To speed up, we can sort candidates by size descending initially? 
    # No, greedy picks based on *uncovered* edges.
    
    # For efficiency, we can't re-scan all candidates every time if the list is huge.
    # But for n=32, list is small. For n=64, list of size 3 and 4 is C(64,3)+C(64,4) ~ 41k + 635k ~ 676k.
    # Scanning 676k candidates * (number of steps) might be slow.
    # Number of steps is roughly E/3 ~ 1000. 676k * 1000 = 676M ops. Too slow.
    
    # Optimization: 
    # Instead of scanning all candidates every step, we can maintain a priority queue?
    # But the "weight" (uncovered edges) changes as we pick cliques.
    
    # Alternative Approach for n=64:
    # 1. Greedy extraction of maximal cliques (like the original code) but smarter.
    # 2. Then Local Search.
    
    # Let's stick to the Set Cover idea but optimize the candidate generation and selection.
    # Actually, for n=64, p=0.55, the graph is dense. Max cliques are likely small (size 5-7).
    # The number of triangles is high.
    
    # Revised Plan for n=64:
    # 1. Find all triangles.
    # 2. Greedy: Pick triangle with most uncovered edges? 
    #    Wait, a triangle covers 3 edges. A K4 covers 6.
    #    If we only have triangles, we cover 3 edges per clique.
    #    If we find a K4, we cover 6 edges per clique.
    
    # Let's try a hybrid:
    # 1. Find all maximal cliques of size >= 3.
    # 2. Use a greedy heuristic to select them.
    
    # To make it fast:
    # We will iterate. In each iteration, we look for the best clique to add.
    # To avoid O(N^k) per step, we can pre-compute all cliques of size 3 and 4.
    # For n=64, 676k candidates is manageable if we don't rescan all of them every time.
    # But we DO need to rescan to find the one with max uncovered edges.
    
    # Trick: 
    # If we pick a clique, it removes its edges from 'uncovered'.
    # We can update the "score" of candidates that intersect with the picked clique.
    # This is complex.
    
    # Simpler Trick:
    # For n=64, just use the original greedy extraction (find max clique in remaining graph) 
    # but improve the "max clique" finder to be more global or use the set cover logic 
    # only on the *remaining* edges after a few greedy steps?
    
    # Actually, the prompt asks for a DIFFERENT strategy. 
    # Set Cover is different. Let's make it work within time limits.
    
    # If n > 40, we will restrict candidates to size 3 and 4.
    # And we will limit the number of greedy steps or use a faster selection.
    
    # Let's try: 
    # 1. Pre-compute all cliques of size 3 and 4.
    # 2. Greedy: 
    #    - Maintain a set of uncovered edges.
    #    - In each step, find the candidate clique that covers the most uncovered edges.
    #    - To do this fast, we can just scan all candidates. 
    #    - For n=64, ~600k candidates. 
    #    - If we need ~200 cliques (E/3), 200 * 600k = 120M ops. 
    #    - In Python, 120M simple ops might take 10-20s. This is feasible.
    
    # Let's implement this carefully.
    
    # Pre-compute candidate edges
    # candidates is a list of frozensets
    # cand_edges is a dict frozenset -> set of edges
    
    # To speed up "uncovered edges count", we can just compute the intersection size.
    # len(cand_edges[c] & uncovered_edges)
    
    while uncovered_edges:
        if _remaining() < 10:
            # Fallback: cover remaining edges as individual edges
            for e in uncovered_edges:
                chosen_cliques.append([e[0], e[1]])
            break
            
        best_clique = None
        best_count = 0
        
        # Scan all candidates
        # Optimization: If we have many candidates, this is the bottleneck.
        # We can filter candidates that have NO uncovered edges? 
        # A candidate has uncovered edges if any of its edges are in uncovered_edges.
        
        for c in candidates:
            # Quick check: does this clique have any uncovered edges?
            # We can check if the intersection is non-empty.
            # To be faster, we can pre-compute the edges as a list and check?
            # set intersection is C-optimized in Python.
            
            # If the clique is already fully covered, skip.
            # We can't easily check "fully covered" without intersection.
            
            inter = cand_edges[c] & uncovered_edges
            count = len(inter)
            
            if count > best_count:
                best_count = count
                best_clique = c
                
        if best_clique is None or best_count == 0:
            # No candidate covers any uncovered edge? 
            # This implies uncovered edges are not part of any triangle/quad.
            # They must be edges that don't form triangles/quads with other uncovered edges?
            # Or they are isolated edges.
            # Just take them as edges.
            for e in uncovered_edges:
                chosen_cliques.append([e[0], e[1]])
            break
            
        # Select best_clique
        chosen_cliques.append(list(best_clique))
        
        # Remove its edges from uncovered
        uncovered_edges -= cand_edges[best_clique]
        
    return chosen_cliques

def _local_refinement(cliques, adj, edge_list):
    """
    Try to merge cliques.
    1. Check if union of any two cliques is a clique.
    2. Check if union of any three cliques is a clique (and covers exactly their union of edges).
    """
    # Convert to sets for easier manipulation
    current_cliques = [set(c) for c in cliques]
    
    # Remove duplicates (shouldn't happen, but just in case)
    # And empty sets
    
    changed = True
    while changed:
        if _remaining() < 5:
            break
        changed = False
        
        # Try pairwise merges
        n_cliques = len(current_cliques)
        for i in range(n_cliques):
            if _remaining() < 5:
                break
            if not current_cliques[i]:
                continue
            for j in range(i + 1, n_cliques):
                if _remaining() < 5:
                    break
                if not current_cliques[j]:
                    continue
                
                union_verts = current_cliques[i] | current_cliques[j]
                if _is_clique(list(union_verts), adj):
                    # Check if union of edges is exactly the union of edges of i and j
                    # Since they are disjoint in terms of edge partition (by definition of partition),
                    # and the union of vertices is a clique, the edges covered by the union
                    # are exactly the edges within union_verts.
                    # We must ensure that NO OTHER clique in the partition covers edges 
                    # inside union_verts that are NOT in clique i or j.
                    # Wait, the partition is a partition of edges. 
                    # If we merge i and j, we are combining their edge sets.
                    # The new clique covers all edges in union_verts.
                    # Are there edges in union_verts that are covered by OTHER cliques?
                    # If so, merging is invalid because it would double cover or cover edges 
                    # that belong to other cliques.
                    
                    # Let's check this condition.
                    union_edges = _clique_edges(union_verts, adj)
                    edges_i = _clique_edges(current_cliques[i], adj)
                    edges_j = _clique_edges(current_cliques[j], adj)
                    
                    # The edges covered by the merged clique must be exactly edges_i U edges_j
                    if union_edges == (edges_i | edges_j):
                        # Valid merge
                        current_cliques[i] = union_verts
                        current_cliques[j] = set()
                        changed = True
                        break
            if changed:
                break
        
        # If no pairwise merge, try triple merges? 
        # This is O(N^3) which might be slow if N is large.
        # But N (number of cliques) is much smaller than n (vertices).
        # For n=64, E ~ 1000. If avg clique size is 3 edges, N ~ 333.
        # 333^3 is 37M. Too slow for Python loops.
        # So we stick to pairwise merges and maybe some random triple attempts.
        
        # Random Triple Merge Attempt
        if not changed and n_cliques > 3:
            if _remaining() > 10:
                # Pick 3 random distinct non-empty cliques
                non_empty = [k for k in range(n_cliques) if current_cliques[k]]
                if len(non_empty) >= 3:
                    i, j, k = random.sample(non_empty, 3)
                    union_verts = current_cliques[i] | current_cliques[j] | current_cliques[k]
                    if _is_clique(list(union_verts), adj):
                        union_edges = _clique_edges(union_verts, adj)
                        edges_ijk = _clique_edges(current_cliques[i], adj) | _clique_edges(current_cliques[j], adj) | _clique_edges(current_cliques[k], adj)
                        if union_edges == edges_ijk:
                            current_cliques[i] = union_verts
                            current_cliques[j] = set()
                            current_cliques[k] = set()
                            changed = True

    # Filter out empty sets
    final_cliques = [sorted(list(c)) for c in current_cliques if c]
    return final_cliques

def _verify_partition(cliques, edge_list, adj):
    """Verify every edge is in exactly one clique and each clique is valid."""
    n_edges = len(edge_list)
    edge_cover = [0] * n_edges
    edge_idx = {(min(u,v), max(u,v)): i for i, (u,v) in enumerate(edge_list)}
    
    for clique in cliques:
        # Check clique validity
        if not _is_clique(clique, adj):
            return False
        # Mark edges
        for a in range(len(clique)):
            for b in range(a + 1, len(clique)):
                e = (min(clique[a], clique[b]), max(clique[a], clique[b]))
                if e in edge_idx:
                    edge_cover[edge_idx[e]] += 1
                else:
                    # Edge in clique not in graph? Impossible if clique is from graph.
                    return False
    
    for c in edge_cover:
        if c != 1:
            return False
    return True

def construct():
    global _START
    _START = time.time()
    
    results = []
    
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if _remaining() < 20:
            # Fallback: just output each edge as its own clique
            adj, edge_list = _generate_graph(n, p, seed)
            results.append([[u, v] for u, v in edge_list])
            continue
        
        adj, edge_list = _generate_graph(n, p, seed)
        
        # 1. Set Cover Greedy
        cliques = _solve_set_cover(adj, edge_list, n)
        
        # 2. Local Refinement
        cliques = _local_refinement(cliques, adj, edge_list)
        
        # 3. Verify
        if not _verify_partition(cliques, edge_list, adj):
            # Fallback to each edge its own clique
            cliques = [[u, v] for u, v in edge_list]
            
        results.append(cliques)
        
    return results

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Instance 1: {len(res[0])} cliques")
    print(f"Instance 2: {len(res[1])} cliques")
