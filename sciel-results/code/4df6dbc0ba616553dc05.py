"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_START = time.time()
_LIMIT = 110.0  # safety margin under 150s wall-clock

def _deadline_ok():
    return (time.time() - _START) < _LIMIT

def _random_edges(n, p, seed):
    from random import Random
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _solve_instance(n, p, seed):
    edges = _random_edges(n, p, seed)
    if not edges:
        return []
    
    # Build adjacency matrix for fast lookup
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    
    # The set of all edges in the graph
    all_edges = set()
    for u, v in edges:
        if u < v:
            all_edges.add((u, v))
        else:
            all_edges.add((v, u))
    
    # Initial partition: every edge is its own clique (size 2)
    # This is always a valid clique partition
    partition = [list(e) for e in all_edges]
    
    # To speed up merging, we can group potential candidates.
    # A merge of clique A and clique B is valid if A ∪ B is a clique.
    # This implies:
    # 1. A is a clique (true by construction if we maintain invariant)
    # 2. B is a clique
    # 3. Every vertex in A is connected to every vertex in B.
    
    # We will use a heuristic local search.
    # We maintain a list of cliques.
    # We repeatedly try to find two cliques that can be merged.
    # To avoid O(K^2) checks where K is number of cliques (which starts at E ~ n^2/2),
    # we can prioritize cliques with higher "potential" or just sample.
    # However, for n=64, E is ~1100. K starts at 1100.
    # O(K^2) is ~1.2M pairs. Checking each pair involves checking edges between them.
    # If cliques are small, this is fast. As they grow, K decreases.
    
    # Let's try a smarter approach:
    # Instead of checking all pairs, we can try to grow a clique by adding vertices from other cliques?
    # No, the partition constraint is strict: every edge must be in EXACTLY one clique.
    # So we can't just "add" vertices. We must "merge" entire cliques.
    
    # Strategy:
    # 1. Start with all edges as separate cliques.
    # 2. While time permits:
    #    a. Pick a random pair of cliques (C1, C2).
    #    b. Check if C1 ∪ C2 is a clique.
    #    c. If yes, merge them into one clique.
    #    d. If no, discard.
    #    
    # Problem: Random pairs are rarely mergeable initially because most edges don't form triangles.
    # We need a better heuristic to find mergeable pairs.
    
    # Alternative Strategy: "Greedy Merge"
    # Sort cliques by size (descending).
    # For each clique C, try to merge it with another clique D if C ∪ D is a clique.
    # This is still O(K^2) per pass.
    
    # Let's use a "Candidate Generation" approach.
    # Two cliques C1={u,v} and C2={x,y} can be merged if u-x, u-y, v-x, v-y are all edges.
    # This means the 4 vertices form a K4 (complete graph on 4 vertices).
    # So, we are looking for K4s in the graph to merge two edge-cliques.
    # More generally, if we have a clique of size k and a clique of size m,
    # they can merge if the induced subgraph on their union is complete.
    
    # Since the graph is dense (p=0.55, 0.6), there are many cliques.
    # Let's try a local search that starts with a "Greedy Peeling" result (like the parent)
    # and then tries to "split and re-merge" or just "merge" if possible?
    # Actually, the parent's peeling is already quite good.
    # The inspiration 2 (score 3.73) used a merge heuristic and got better than parent (3.52).
    # Wait, the parent in the prompt is 3.52. Inspiration 2 is 3.73.
    # The prompt says "Current Program (score 3.5198)".
    # So the current program is the peeling one.
    # Inspiration 2 (merge heuristic) scored 3.7323.
    # This suggests merging from a trivial start is BETTER than peeling?
    # Or maybe the implementation of inspiration 2 was better tuned.
    
    # Let's implement a robust "Iterative Merging" local search.
    # 1. Start with all edges as cliques.
    # 2. Use a priority queue or just random sampling with bias towards larger cliques?
    #    Actually, merging two small cliques is easier to verify.
    # 3. To find mergeable pairs efficiently:
    #    For a given clique C, any other clique D that can merge with it must consist of vertices
    #    that are all connected to all vertices in C.
    #    Let N(C) be the set of vertices connected to ALL vertices in C.
    #    Then D must be a subset of N(C) and D itself must be a clique (which it is, by invariant).
    #    So, for each clique C, we can identify all other cliques D that are subsets of N(C).
    #    If we find one, we merge.
    
    # This might be complex to maintain dynamically.
    
    # Let's try a simpler heuristic:
    # 1. Start with all edges as cliques.
    # 2. Repeatedly pick a random clique C.
    # 3. Identify all vertices in N(C) (neighbors of C in the complement of C).
    # 4. Check if any other clique D is entirely contained in N(C).
    #    We can maintain a mapping from vertex to list of cliques containing it.
    #    For a clique C, find the intersection of the cliques containing each vertex in C? No.
    #    We need cliques D such that D ⊆ N(C).
    #    This means for every v in D, v is in N(C).
    #    
    #    Let's invert: For each vertex v, maintain the set of cliques containing v.
    #    For a candidate clique C, the set of valid merge partners D must be such that
    #    for all v in D, v is connected to all u in C.
    #    
    #    This is getting complicated to implement efficiently in Python within time limits.
    
    # Let's go back to the "Peeling" approach but improve the selection heuristic.
    # The current peeling picks the edge with max common neighbors.
    # Maybe we should pick the clique that maximizes the number of edges covered?
    # Or use a "Weighted" peeling where we prioritize edges that are "harder" to cover?
    
    # Actually, let's look at the scores again.
    # Parent (Peeling): 3.52
    # Insp 2 (Merge from trivial): 3.73
    # Insp 1 (Step 1: Peeling prone to local optima...): 3.84
    #
    # It seems Insp 1 (which likely improved peeling or used a different greedy) got the best score.
    # The prompt says Insp 1 uses "materiall" (typo for materially?) and defines same functions.
    #
    # Let's try a **Simulated Annealing** or **Random Restart** on the Peeling approach.
    # Or better: **Greedy with Lookahead**.
    
    # New Strategy: **Maximal Clique Enumeration + Set Cover**
    # 1. Enumerate all maximal cliques in the graph.
    # 2. Use a greedy set cover algorithm to select the minimum number of maximal cliques
    #    that cover all edges.
    #    Note: The cliques in the partition don't have to be maximal in the graph,
    #    but they must be disjoint in terms of edges.
    #    Wait, "every edge must lie in exactly one listed clique".
    #    This is an edge partition.
    #    If we pick a clique C, we remove all edges in C.
    #    The next clique must be formed from the remaining edges.
    #    So the cliques are not necessarily maximal in the original graph, but they are maximal
    #    in the residual graph at the time of selection.
    #
    # The "Set Cover" framing is slightly different: we want to cover E with disjoint cliques.
    # This is exactly what peeling does.
    #
    # Improvement: Instead of picking ONE best edge, pick a set of edges that form a clique
    # and have high "density" in the residual graph.
    
    # Let's implement a **Multi-Start Greedy Peeling** with different random tie-breakers
    # or different initial edge orderings, and keep the best result.
    
    def _greedy_peeling(edges_list, adj, n, time_limit_frac=1.0):
        """
        Performs greedy peeling.
        Returns the partition (list of cliques).
        """
        remaining_edges = set()
        for u, v in edges_list:
            if u < v:
                remaining_edges.add((u, v))
            else:
                remaining_edges.add((v, u))
        
        if not remaining_edges:
            return []
            
        partition = []
        start_time = time.time()
        limit = time_limit_frac * _LIMIT * 0.9 # Use 90% of limit for safety
        
        while remaining_edges:
            if (time.time() - start_time) > limit:
                # Fallback: add remaining as 2-cliques
                for e in remaining_edges:
                    partition.append(list(e))
                break
                
            # Heuristic: Find edge (u,v) that maximizes the size of the clique it can form.
            # We approximate this by counting common neighbors in remaining_edges.
            best_edge = None
            best_score = -1
            best_clique = None
            
            # To speed up, we can sample a subset of edges if there are many
            # But for n=64, E~1100, iterating all is fine (1100 * 64 ~ 70k ops per step)
            # Number of steps is at most E, so worst case 1100 * 70k = 77M ops.
            # In Python, this might be too slow (seconds to minutes).
            # We need to optimize the inner loop.
            
            # Optimization: Use bitsets for adjacency?
            # Python integers can act as bitsets.
            # adj_bits[u] is an integer where bit v is set if (u,v) is in remaining_edges.
            
            # Let's rebuild the bitsets for remaining edges.
            # This is O(E) per step.
            # Total O(E^2). For E=1100, 1.2M ops. This is fast.
            
            # Let's use bitsets for the "common neighbors" calculation.
            # common_neighbors(u, v) = adj_bits[u] & adj_bits[v]
            # popcount of this gives the number of common neighbors.
            
            # We'll maintain adj_bits dynamically.
            # Initially, build from edges_list.
            
            adj_bits = [0] * n
            for u, v in edges_list:
                if u < v:
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                else:
                    adj_bits[v] |= (1 << u)
                    adj_bits[u] |= (1 << v)
            
            # We need to update adj_bits when we remove edges.
            # Removing a clique C means for each u in C, we clear bits for all v in C.
            
            while remaining_edges:
                if (time.time() - start_time) > limit:
                    for e in remaining_edges:
                        partition.append(list(e))
                    break
                    
                # Find best edge
                best_edge = None
                best_common_count = -1
                best_u = -1
                best_v = -1
                
                # Iterate over remaining edges
                # To make this faster, we can iterate over a list of remaining edges
                # and update it.
                
                # Convert set to list for iteration? No, set iteration is fine.
                # But we need to be careful with removal.
                
                # Let's pick a random edge first to see if we can improve,
                # or just the first one?
                # The heuristic "max common neighbors" is good.
                
                # Optimization: Instead of checking ALL remaining edges,
                # check a sample? No, for small n, check all.
                
                # We can speed up common neighbor count using bit operations.
                # But we need to update adj_bits as we go?
                # No, adj_bits reflects the ORIGINAL graph?
                # No, adj_bits must reflect the REMAINING graph.
                # So we must update adj_bits when we remove a clique.
                
                # Let's restructure:
                # 1. Maintain adj_bits for remaining graph.
                # 2. Maintain remaining_edges set.
                # 3. Find best edge (u,v) by maximizing popcount(adj_bits[u] & adj_bits[v]).
                #    Note: adj_bits[u] & adj_bits[v] includes u and v? No, no self loops.
                #    It includes all w such that (u,w) and (v,w) are in remaining graph.
                # 4. Expand to maximal clique in remaining graph.
                # 5. Remove edges of clique from remaining_edges and adj_bits.
                
                # To find best edge efficiently:
                # We can precompute for each edge its "potential" (common neighbors).
                # But this changes as edges are removed.
                
                # Let's just iterate.
                # For n=64, E~1100.
                # Iterating 1100 edges, doing a bit AND and popcount is very fast.
                
                for u, v in remaining_edges:
                    # Ensure u < v for consistency in bit logic?
                    # adj_bits is symmetric.
                    common = adj_bits[u] & adj_bits[v]
                    count = common.bit_count()
                    if count > best_common_count:
                        best_common_count = count
                        best_edge = (u, v)
                        best_u = u
                        best_v = v
                        
                if best_edge is None:
                    break
                    
                u, v = best_edge
                
                # Expand to maximal clique
                clique = [u, v]
                # Candidates are common neighbors
                candidates_mask = adj_bits[u] & adj_bits[v]
                
                # We want to greedily add vertices to the clique.
                # A vertex w can be added if it is connected to all current clique members.
                # Since we start with u, v, and candidates are connected to both,
                # we just need to check connectivity among candidates and to new additions.
                
                # Let's maintain a mask of valid candidates.
                valid_mask = candidates_mask
                
                while True:
                    if not valid_mask:
                        break
                    
                    # Pick a candidate w from valid_mask
                    # Heuristic: pick w with highest degree within valid_mask
                    # deg(w) = popcount(valid_mask & adj_bits[w])
                    
                    best_w = -1
                    best_deg = -1
                    
                    # Iterate through bits in valid_mask
                    temp_mask = valid_mask
                    while temp_mask:
                        # Get lowest set bit
                        lsb = temp_mask & -temp_mask
                        w = lsb.bit_length() - 1
                        deg = (valid_mask & adj_bits[w]).bit_count()
                        if deg > best_deg:
                            best_deg = deg
                            best_w = w
                        temp_mask ^= lsb
                    
                    if best_w == -1:
                        break
                        
                    clique.append(best_w)
                    valid_mask ^= (1 << best_w)
                    
                    # Update valid_mask: remove any vertex not connected to best_w
                    valid_mask &= adj_bits[best_w]
                    
                # Add clique to partition
                partition.append(clique)
                
                # Remove edges from remaining_edges and adj_bits
                for i in range(len(clique)):
                    ui = clique[i]
                    for j in range(i+1, len(clique)):
                        vj = clique[j]
                        if ui < vj:
                            rem_edge = (ui, vj)
                        else:
                            rem_edge = (vj, ui)
                        if rem_edge in remaining_edges:
                            remaining_edges.remove(rem_edge)
                        # Update adj_bits
                        adj_bits[ui] &= ~(1 << vj)
                        adj_bits[vj] &= ~(1 << ui)
                        
        return partition

    # Multi-start: Run greedy peeling a few times with slight variations?
    # The greedy is deterministic given the tie-breaking.
    # We can randomize the order of edges in the initial set to break ties?
    # Or just run it once. It's usually good.
    
    # Let's try one run.
    partition = _greedy_peeling(edges, adj, n)
    
    # Optional: Local Search on the result?
    # Try to merge any two cliques in the partition if their union is a clique.
    # This can only improve the score (reduce count).
    
    # Check for merges
    changed = True
    while changed and _deadline_ok():
        changed = False
        # Convert partition to list of sets for easier union check
        cliques = [set(c) for c in partition]
        
        # We want to find i != j such that cliques[i] | cliques[j] is a clique.
        # This is O(K^2 * S^2) where K is num cliques, S is avg size.
        # K is small (maybe 50-100). S is small.
        # So this is fast.
        
        merged = False
        new_partition = []
        used = [False] * len(cliques)
        
        for i in range(len(cliques)):
            if used[i]:
                continue
            for j in range(i+1, len(cliques)):
                if used[j]:
                    continue
                union = cliques[i] | cliques[j]
                # Check if union is a clique
                is_clique = True
                for u in union:
                    for v in union:
                        if u >= v:
                            continue
                        if not adj[u][v]:
                            is_clique = False
                            break
                    if not is_clique:
                        break
                if is_clique:
                    new_partition.append(list(union))
                    used[i] = True
                    used[j] = True
                    changed = True
                    merged = True
                    break
            if merged:
                break
            
        if changed:
            # Rebuild partition
            for i in range(len(cliques)):
                if not used[i]:
                    new_partition.append(list(cliques[i]))
            partition = new_partition
        else:
            # No more simple merges
            break
            
    return partition

def construct():
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        results.append(_solve_instance(n, p, seed))
    return results
# EVOLVE-BLOCK-END
