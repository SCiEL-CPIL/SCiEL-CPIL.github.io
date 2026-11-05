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
        
        # Use a set of frozensets for cliques? No, lists are required for output.
        # Internally, we track uncovered edges.
        uncovered = set(edges_set)
        cliques = []
        
        # Precompute global adjacency for fast "is edge present in original graph" check
        # But we need "is edge present in UNCOVERED graph".
        # Since n is small (32, 64), we can maintain a dynamic adjacency structure or just check the set.
        # Checking (u,v) in uncovered is O(1).
        
        # Optimization: Precompute all pairs of vertices? No, just iterate.
        
        while uncovered:
            if time.time() > DEADLINE:
                # Fallback: add all remaining as edges
                for u, v in uncovered:
                    cliques.append([u, v])
                break
            
            best_clique = None
            best_score = -1
            best_pair = None
            
            # Iterate over all uncovered edges to find the "best" one to extend.
            # "Best" is defined by the size of the clique we can form extending it.
            # To avoid O(E^2) or worse, we limit the search or use a heuristic.
            # However, n is small. E is at most n(n-1)/2.
            # For n=64, E ~ 2000.
            # If we iterate all edges and for each find the max clique extension, it might be slow.
            
            # Heuristic: Sample a subset of edges or just iterate all but limit the extension search.
            # Let's iterate all uncovered edges.
            
            # To speed up "find max clique extension for edge (u,v)":
            # 1. Get candidates C = {w | (u,w) in uncovered and (v,w) in uncovered}
            # 2. Find a large clique in the subgraph induced by C.
            
            # We can optimize by not checking every single edge if we find a "good" one early?
            # No, we want the global maximum to minimize count.
            # But checking max clique is NP-hard. We use a greedy approximation for the extension.
            
            # Let's try to be smarter:
            # Instead of iterating all edges, maybe iterate all pairs of vertices?
            # If (u,v) is not in uncovered, it can't be the core of a new clique covering uncovered edges
            # unless the clique is just {u,v} and we cover nothing? No, the clique must cover at least the edge (u,v).
            # So (u,v) MUST be in uncovered.
            
            # Let's iterate through uncovered edges.
            # To make it faster, we can break early if we find a clique of size K?
            # Max possible size is n.
            
            # Let's collect candidate pairs.
            # In dense graphs, most edges are covered quickly.
            
            # Optimization: 
            # If we find a clique of size 5, it covers 10 edges.
            # A clique of size 6 covers 15 edges.
            
            # Let's iterate all uncovered edges.
            for u, v in list(uncovered):
                # Find candidates
                # w must be connected to u and v in the uncovered graph.
                candidates = []
                
                # We need to check neighbors of u in uncovered graph.
                # Since we don't have a dynamic adj list, we scan?
                # Scanning all w for every u,v is O(n) * O(E) = O(nE).
                # n=64, E=2000. 64*2000 = 128,000 operations per edge? No.
                # For each edge (u,v), we check all w (n).
                # Total work: sum_{(u,v) in U} n = |U| * n.
                # |U| starts at ~2000. 2000 * 64 = 128,000 checks.
                # Inside the check, we do set lookups.
                # Then we do a greedy clique search on candidates.
                # Candidates size is at most n-2.
                # Greedy clique search is O(n^2) or similar.
                # Total: |U| * (n + n^2) ~ 2000 * 4000 = 8,000,000 operations per iteration of the main loop?
                # The main loop runs at most |E| / 2 times? No, fewer.
                # If we cover 10 edges per clique, we run 200 times.
                # 200 * 8,000,000 = 1.6 billion operations. This might be too slow in Python (120s limit).
                
                # We need a faster heuristic.
                # 1. Don't check ALL edges. Sample? Or pick the edge with highest "potential"?
                # 2. Faster candidate generation?
                
                # Alternative Strategy:
                # Pick a random edge? No, deterministic is better for reproducibility.
                # Pick the edge (u,v) that minimizes something?
                
                # Let's optimize the candidate generation.
                # Precompute neighbors in the UNCOVERED graph?
                # Updating adj list on removal is expensive.
                
                # Let's try a different heuristic:
                # Instead of iterating all edges, iterate all vertices u.
                # For each u, look at its uncovered neighbors N_u.
                # For each pair (v, w) in N_u, check if (v,w) is uncovered.
                # This finds triangles containing u.
                # This is still expensive.
                
                # Let's stick to iterating edges but optimize the inner loop.
                # We can limit the number of edges we consider per "batch" or just accept the cost if n is small enough.
                # n=32 is fast. n=64 is the bottleneck.
                
                # Let's try to limit the search:
                # If we find a clique of size >= 4, it's quite good.
                # We can stop searching for better cliques for this specific "pass" if we find a big one?
                # No, we need the GLOBAL best to minimize count.
                
                # Let's try a "Greedy Set Cover" approach on a subset of candidates.
                # 1. Generate a list of "candidate cliques".
                #    - For each uncovered edge (u,v), compute a greedy maximal clique extension.
                #    - Keep the top K (e.g., 50) largest candidate cliques.
                # 2. Pick the largest one.
                # 3. Remove its edges.
                
                # This reduces the inner loop cost if we don't check all edges?
                # We still have to check all edges to generate candidates.
                
                # Can we skip edges?
                # If we pick an edge (u,v) that is "isolated" (no common neighbors), it's size 2.
                # We can quickly check degree of u and v in uncovered graph?
                
                # Let's implement a faster version:
                # 1. Build a dynamic adjacency list for the uncovered graph.
                #    Update it when we remove edges.
                #    This makes "get neighbors of u" O(degree) instead of O(n).
                
                # Let's restart solve_instance with this optimization.
                pass
            
            # --- RE-IMPLEMENTATION WITH DYNAMIC ADJACENCY ---
            # The previous block was a dry run. Let's write the actual code below.
            # I will replace the body of solve_instance.
            
            # Since I can't easily "replace" in this thought process, I will structure the final code 
            # to use the dynamic adjacency list from the start.
            
            # For the sake of this specific iteration block, I will just use the list(uncovered) approach 
            # but add a time check and a "top K" selection to prune the search space if it's too slow.
            # Actually, for n=64, O(n^3) is 262,144. O(E*n^2) is 2000*4000 = 8M.
            # 8M simple ops in Python is ~0.05s.
            # If we do this 200 times, that's 10s. This is feasible!
            
            # So the O(E * n^2) approach is likely fine.
            # Let's proceed with iterating all uncovered edges.
            
            # Optimization: Use a local variable for uncovered set for speed.
            unc = uncovered
            
            for u, v in list(unc):
                # Find candidates: w such that (u,w) in unc and (v,w) in unc
                # We can intersect neighbors.
                # But we don't have dynamic neighbors.
                # We can just loop w from 0 to n-1.
                
                candidates = []
                for w in range(n):
                    if w == u or w == v:
                        continue
                    # Check (u,w)
                    a, b = (u, w) if u < w else (w, u)
                    if (a, b) not in unc:
                        continue
                    # Check (v,w)
                    a, b = (v, w) if v < w else (w, v)
                    if (a, b) not in unc:
                        continue
                    candidates.append(w)
                
                if not candidates:
                    # Clique is just [u, v]
                    # We can't do better than size 2 for this edge right now.
                    # But maybe another edge allows a bigger clique.
                    # We record this as a candidate clique of size 2.
                    # But we don't need to store all size 2 cliques, just note that max_size might be 2.
                    # If best_score is already > 2, we skip.
                    if best_score < 2:
                        best_score = 2
                        best_clique = [u, v]
                    continue
                
                # We have candidates. Find a large clique in the induced subgraph of candidates.
                # Greedy: Sort by degree in the induced subgraph?
                # Or just greedy addition.
                
                # To speed up, let's limit the size of the clique we try to build?
                # No, we want the largest.
                
                # Greedy Clique Construction:
                # 1. Start with empty set S.
                # 2. Sort candidates by "potential" (e.g., degree in candidates).
                # 3. Add w to S if w is connected to all in S.
                
                # Calculate degrees in candidate subgraph
                # This is O(|C|^2).
                cand_set = set(candidates)
                deg = {w: 0 for w in candidates}
                for w1 in candidates:
                    for w2 in candidates:
                        if w1 == w2:
                            continue
                        a, b = (w1, w2) if w1 < w2 else (w2, w1)
                        if (a, b) in unc:
                            deg[w1] += 1
                
                # Sort by degree descending
                sorted_cands = sorted(candidates, key=lambda x: deg[x], reverse=True)
                
                current_s = set()
                for w in sorted_cands:
                    # Check if w is connected to all in current_s
                    ok = True
                    for x in current_s:
                        a, b = (w, x) if w < x else (x, w)
                        if (a, b) not in unc:
                            ok = False
                            break
                    if ok:
                        current_s.add(w)
                
                clique = [u, v] + list(current_s)
                size = len(clique)
                
                if size > best_score:
                    best_score = size
                    best_clique = clique
                    
                    # Early termination if we found a "very large" clique?
                    # Max size is n.
                    # If size == n, we can't do better.
                    if size == n:
                        break # Break inner loop? No, we are in the edge loop.
                        # We can break the edge loop if we found a clique of size n?
                        # Yes, because no clique can be larger.
            
            # After iterating all edges, pick the best_clique found.
            if best_clique is None:
                # Should not happen if uncovered is not empty
                u, v = next(iter(uncovered))
                best_clique = [u, v]
            
            # Remove edges in best_clique
            m = len(best_clique)
            for i in range(m):
                for j in range(i + 1, m):
                    a, b = best_clique[i], best_clique[j]
                    if a > b:
                        a, b = b, a
                    if (a, b) in uncovered:
                        uncovered.remove((a, b))
            
            cliques.append(best_clique)
            
        return cliques

    # Instance 1
    n1, p1, s1 = 32, 0.6, 601
    e1 = random_graph(n1, p1, s1)
    c1 = solve_instance(e1, n1)
    
    # Instance 2
    n2, p2, s2 = 64, 0.55, 602
    e2 = random_graph(n2, p2, s2)
    c2 = solve_instance(e2, n2)
    
    return [c1, c2]
