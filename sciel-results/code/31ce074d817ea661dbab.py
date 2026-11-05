import time
import sys
import random
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
        
        # 1. Compute a vertex ordering.
        # Strategy: Descending degree in the graph.
        # We compute degrees based on the original graph edges.
        degrees = [0] * n
        for u, v in edges_set:
            degrees[u] += 1
            degrees[v] += 1
        
        # Sort vertices by degree descending. 
        # Ties broken by vertex index for determinism.
        order = sorted(range(n), key=lambda x: (-degrees[x], x))
        
        # Map vertex to its position in the order
        pos = [0] * n
        for i, v in enumerate(order):
            pos[v] = i
            
        # 2. Iterate through vertices in this order.
        # For each vertex v, we look at the subgraph induced by v and all vertices 
        # that come AFTER v in the order.
        # We want to find a large clique in this subgraph that includes v.
        # Any edge (v, w) where w comes after v in the order MUST be covered by a clique 
        # that includes v (or a clique that includes w and was already processed? 
        # No, if we process v first, we can cover edges (v, w). 
        # If we process w first, we would have covered (v, w) in w's clique.
        # So, we must decide: does the edge (v, w) belong to the clique of the EARLIER vertex or the LATER one?
        
        # Standard Clique Cover via Ordering:
        # We assign each edge (u, v) to the clique of the vertex that appears FIRST in the order.
        # So for vertex v_i, we only care about edges (v_i, v_j) where j > i.
        # We want to form a clique C_i containing v_i and some subset of {v_j | j > i}.
        # All edges within C_i that are of the form (v_i, v_j) or (v_a, v_b) where a,b > i 
        # are NOT necessarily covered by C_i if we only assign edges to the FIRST vertex.
        
        # Correction: The problem requires EVERY edge to be in EXACTLY one clique.
        # If we use the "assign edge to first vertex" rule:
        # Edge (u, v) with pos[u] < pos[v] is assigned to the "clique of u".
        # But a "clique" is a set of vertices. If u is in clique C_u, and v is in C_u, 
        # then edge (u, v) is covered by C_u.
        # What if v is NOT in C_u? Then edge (u, v) is NOT covered by C_u.
        # It must be covered by some other clique. But by our rule, it's assigned to u's turn.
        # So, if we decide to form C_u, we MUST include v in C_u if we want to cover (u, v) with C_u.
        # If we don't include v in C_u, then (u, v) remains uncovered?
        # No, the rule "assign to first vertex" implies that the clique associated with the first vertex 
        # MUST cover all edges from that vertex to later vertices? 
        # That is impossible if the later vertices don't form a clique with each other.
        
        # Therefore, the "Ordering" approach needs a different framing:
        # We are partitioning the EDGES.
        # Let's stick to the Greedy Set Cover but make it MUCH faster and smarter.
        
        # Revised Strategy: 
        # 1. Use a fast Maximal Clique Finder (Bron-Kerbosch with pivot) to find ALL maximal cliques?
        #    For n=64, p=0.55, the number of maximal cliques can be large, but manageable?
        #    Actually, in dense graphs, there are fewer maximal cliques.
        #    However, finding ALL maximal cliques might be expensive.
        
        # 2. Better: **Greedy with Local Improvement**.
        #    Start with a greedy partition.
        #    Then, try to merge cliques or move edges to reduce count.
        #    Merging two cliques C1, C2 is only possible if C1 U C2 is a clique.
        #    This is rare.
        
        # 3. **Different Framing: Vertex Cover Dual?**
        #    No.
        
        # Let's try **Iterative Greedy with Better Heuristic**:
        # The current program's weakness is likely the O(E * n^2) search for the best clique.
        # We can precompute the "potential" of each edge.
        
        # Let's try a **Randomized Restart** approach with a very fast heuristic.
        # Heuristic: 
        # 1. Pick a random edge (u, v).
        # 2. Expand to a maximal clique using a greedy order (e.g., by degree).
        # 3. Cover those edges.
        # 4. Repeat.
        # This is O(K * (n+E)) per run. Very fast.
        # We can run this multiple times with different seeds and keep the best.
        
        def greedy_partition(edges_set, n, rand_seed):
            rng = random.Random(rand_seed)
            uncovered = set(edges_set)
            cliques = []
            
            # Precompute adjacency list for original graph for fast neighbor checks
            # But we need "uncovered" neighbors.
            # We'll maintain a set of uncovered edges.
            
            while uncovered:
                if time.time() > DEADLINE:
                    # Fallback
                    for u, v in uncovered:
                        cliques.append([u, v])
                    break
                
                # Pick a random edge from uncovered to start a clique
                # To be deterministic and reproducible for the evaluator, 
                # we should use a fixed seed or deterministic pick.
                # Let's use a deterministic pick: the edge with the smallest (u, v) in the set?
                # Or the edge with the highest "clique potential"?
                
                # Let's pick the first edge in a sorted list for determinism, 
                # but random restarts need randomness. 
                # Since the evaluator runs construct() once, we can use a fixed seed for the inner randomness.
                
                # Let's pick an edge. To avoid bias, we can pick the edge 
                # whose endpoints have the highest degree in the UNCOVERED graph.
                
                # Compute degrees in uncovered graph
                unc_deg = [0] * n
                for u, v in uncovered:
                    unc_deg[u] += 1
                    unc_deg[v] += 1
                
                # Find edge with max (unc_deg[u] + unc_deg[v])
                best_edge = None
                best_val = -1
                # Sample a few edges to save time? Or scan all?
                # Scanning all is O(E). E is ~2000. Fast.
                for u, v in uncovered:
                    val = unc_deg[u] + unc_deg[v]
                    if val > best_val:
                        best_val = val
                        best_edge = (u, v)
                
                if best_edge is None:
                    break
                    
                u, v = best_edge
                
                # Expand (u, v) to a maximal clique in the uncovered graph.
                # Candidates: w such that (u,w) in uncovered and (v,w) in uncovered.
                # We want to add as many w as possible while keeping it a clique.
                
                # Get candidates
                cand_set = set()
                for w in range(n):
                    if w == u or w == v: continue
                    a, b = (u, w) if u < w else (w, u)
                    c, d = (v, w) if v < w else (w, v)
                    if (a, b) in uncovered and (c, d) in uncovered:
                        cand_set.add(w)
                
                # Greedily add vertices to the clique.
                # Order candidates by their degree in the candidate subgraph?
                # Or just by global degree?
                # Let's sort by unc_deg descending.
                sorted_cands = sorted(cand_set, key=lambda x: unc_deg[x], reverse=True)
                
                current_clique = {u, v}
                
                for w in sorted_cands:
                    # Check if w is connected to all in current_clique
                    is_connected = True
                    for x in current_clique:
                        if x == w: continue
                        a, b = (w, x) if w < x else (x, w)
                        if (a, b) not in uncovered:
                            is_connected = False
                            break
                    if is_connected:
                        current_clique.add(w)
                
                # Remove all edges in current_clique from uncovered
                clique_list = list(current_clique)
                for i in range(len(clique_list)):
                    for j in range(i + 1, len(clique_list)):
                        a, b = clique_list[i], clique_list[j]
                        if a > b: a, b = b, a
                        if (a, b) in uncovered:
                            uncovered.remove((a, b))
                
                cliques.append(clique_list)
                
            return cliques

        # Run greedy partition with a few different strategies/seeds and pick the best
        # Since we need determinism, we'll try a few fixed variations.
        
        best_cliques = None
        best_count = float('inf')
        
        # Variation 1: Standard greedy (highest degree sum edge)
        c1 = greedy_partition(edges_set, n, 1)
        if len(c1) < best_count:
            best_count = len(c1)
            best_cliques = c1
            
        # Variation 2: Randomized greedy (pick random edge, but deterministic seed)
        # We can modify the edge selection to be random with a fixed seed.
        def greedy_partition_random(edges_set, n, rand_seed, edge_pick_strategy='random'):
            rng = random.Random(rand_seed)
            uncovered = set(edges_set)
            cliques = []
            
            while uncovered:
                if time.time() > DEADLINE:
                    for u, v in uncovered:
                        cliques.append([u, v])
                    break
                
                unc_deg = [0] * n
                for u, v in uncovered:
                    unc_deg[u] += 1
                    unc_deg[v] += 1
                
                if edge_pick_strategy == 'random':
                    # Pick a random edge
                    edge_list = list(uncovered)
                    u, v = edge_list[rng.randint(0, len(edge_list) - 1)]
                else:
                    # Pick max degree sum
                    best_edge = None
                    best_val = -1
                    for u, v in uncovered:
                        val = unc_deg[u] + unc_deg[v]
                        if val > best_val:
                            best_val = val
                            best_edge = (u, v)
                    u, v = best_edge
                
                cand_set = set()
                for w in range(n):
                    if w == u or w == v: continue
                    a, b = (u, w) if u < w else (w, u)
                    c, d = (v, w) if v < w else (w, v)
                    if (a, b) in uncovered and (c, d) in uncovered:
                        cand_set.add(w)
                
                # Sort candidates by degree in candidate subgraph (approx by unc_deg)
                sorted_cands = sorted(cand_set, key=lambda x: unc_deg[x], reverse=True)
                
                current_clique = {u, v}
                for w in sorted_cands:
                    is_connected = True
                    for x in current_clique:
                        if x == w: continue
                        a, b = (w, x) if w < x else (x, w)
                        if (a, b) not in uncovered:
                            is_connected = False
                            break
                    if is_connected:
                        current_clique.add(w)
                
                clique_list = list(current_clique)
                for i in range(len(clique_list)):
                    for j in range(i + 1, len(clique_list)):
                        a, b = clique_list[i], clique_list[j]
                        if a > b: a, b = b, a
                        if (a, b) in uncovered:
                            uncovered.remove((a, b))
                
                cliques.append(clique_list)
                
            return cliques

        # Try a few random seeds
        for seed in [42, 123, 456]:
            c_rand = greedy_partition_random(edges_set, n, seed, 'random')
            if len(c_rand) < best_count:
                best_count = len(c_rand)
                best_cliques = c_rand
        
        # Also try a "Lowest Degree" strategy? Sometimes picking isolated edges first helps?
        # Probably not for dense graphs.
        
        return best_cliques

    # Instance 1
    n1, p1, s1 = 32, 0.6, 601
    e1 = random_graph(n1, p1, s1)
    c1 = solve_instance(e1, n1)
    
    # Instance 2
    n2, p2, s2 = 64, 0.55, 602
    e2 = random_graph(n2, p2, s2)
    c2 = solve_instance(e2, n2)
    
    return [c1, c2]
