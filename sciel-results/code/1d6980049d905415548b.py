# EVOLVE-BLOCK-START
import time
import random
import sys
from itertools import combinations

def construct():
    deadline = time.time() + 145  # 5s safety margin before 150s kill
    
    def build_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj
    
    def bron_kerbosch_maximal_cliques(adj):
        n = len(adj)
        maximal_cliques = []
        
        def expand(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(frozenset(R))
                return
            
            # Pivot selection: choose vertex with max |P ∩ N(v)|
            pivot = max(P | X, key=lambda v: len(P & adj[v])) if (P | X) else None
            candidates = P - adj[pivot] if pivot is not None else P
            
            for v in list(candidates):
                expand(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        expand(frozenset(), set(range(n)), set())
        return maximal_cliques
    
    def solve_instance(cliques, n, time_limit):
        if not cliques:
            return set()
        
        num_cliques = len(cliques)
        
        # Map each clique to its index
        clique_list = list(cliques)
        
        # For each vertex, which cliques does it cover?
        vertex_cliques = {v: set() for v in range(n)}
        for i, clique in enumerate(clique_list):
            for v in clique:
                vertex_cliques[v].add(i)
        
        # Precompute: for each clique, how many vertices cover it?
        # This helps in weighting "hard" cliques
        clique_cover_count = [0] * num_cliques
        for v in range(n):
            for i in vertex_cliques[v]:
                clique_cover_count[i] += 1
        
        # Initial Weights: Inverse of cover count. 
        # Cliques covered by few vertices are "harder" to satisfy and get higher weight.
        # This acts as a Lagrangian-like penalty for leaving hard cliques uncovered.
        clique_weights = [1.0 / max(1, c) for c in clique_cover_count]
        
        # Greedy Selection with Weighted Marginal Utility
        selected = set()
        covered = set()
        
        # To speed up, we can just iterate. n is small (<=40).
        # Number of cliques is usually small in random graphs with these parameters.
        
        while len(covered) < num_cliques:
            if time.time() > time_limit:
                break
                
            best_v = None
            best_score = -1
            
            for v in range(n):
                if v in selected:
                    continue
                
                # Calculate marginal weighted utility
                score = 0.0
                for i in vertex_cliques[v]:
                    if i not in covered:
                        score += clique_weights[i]
                
                if score > best_score:
                    best_score = score
                    best_v = v
            
            if best_v is None or best_score <= 0:
                break
                
            selected.add(best_v)
            for i in vertex_cliques[best_v]:
                covered.add(i)
            
            # Optional: Update weights? 
            # A simple dynamic update: reduce weight of covered cliques to 0 (already done by 'covered' check).
            # We could also increase weights of uncovered cliques that are now "more exposed" 
            # but standard greedy with static inverse-degree weights is a strong baseline.
            
        # Local Search Polish
        # We try to remove vertices if possible, then swap.
        # To make this fast, we maintain a count of how many times each clique is covered.
        
        # Build coverage counts for the current selection
        cov_count = [0] * num_cliques
        for v in selected:
            for i in vertex_cliques[v]:
                cov_count[i] += 1
        
        # Check if all are covered (should be, but just in case)
        for i in range(num_cliques):
            if cov_count[i] == 0:
                # Fallback: add a vertex from this clique
                for v in clique_list[i]:
                    if v not in selected:
                        selected.add(v)
                        for j in vertex_cliques[v]:
                            cov_count[j] += 1
                        break
        
        # Local Search Loop
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1. Try removing a vertex
            for v in list(selected):
                # Check if v is the only one covering any clique
                is_critical = False
                for i in vertex_cliques[v]:
                    if cov_count[i] == 1:
                        is_critical = True
                        break
                
                if not is_critical:
                    # Safe to remove
                    selected.remove(v)
                    for i in vertex_cliques[v]:
                        cov_count[i] -= 1
                    improved = True
                    break # Restart scan after change
            
            if improved:
                continue
                
            # 2. Try swapping: remove v1, add v2
            # This is O(N^2) per iteration. With N<=40, N^2=1600. 
            # If we do this 1000 times, it's 1.6M ops, which is fast.
            found_swap = False
            for v1 in list(selected):
                # Temporarily remove v1
                for i in vertex_cliques[v1]:
                    cov_count[i] -= 1
                
                # Check if coverage is still valid? No, we are looking for a swap.
                # We need to find v2 that covers the "lost" cliques and maybe more, 
                # such that we don't increase size (size stays same in swap) 
                # but wait, the goal is to MINIMIZE size.
                # A swap (remove 1, add 1) doesn't change size.
                # A "swap" that reduces size is effectively "remove 1, add 0" (covered above) 
                # or "remove 2, add 1".
                
                # Let's stick to "remove 1" first. 
                # If "remove 1" fails, try "remove 1, add 1" to potentially enable future removals?
                # Or just try "remove 2, add 1"?
                
                # Let's try a specific improvement: 
                # Find v2 not in selected such that adding v2 allows removing v1 AND v3?
                # That's complex.
                
                # Simpler: Try to replace v1 with v2 if v2 covers all cliques that v1 uniquely covers, 
                # and v2 covers some other cliques that are currently covered by only 1 other vertex (making them critical)?
                # No, that's complicated.
                
                # Let's just try "remove 1, add 1" if it allows a subsequent "remove 1".
                # Actually, the most effective local search for Set Cover is often:
                # 1. Remove redundant vertices.
                # 2. Try to replace a pair of vertices with a single vertex.
                
                # Let's implement "Replace 2 with 1".
                # For every pair (v1, v2) in selected:
                #   Let C_unique = {c | cov_count[c] == 1 and c in vertex_cliques[v1] or vertex_cliques[v2]}
                #   Actually, if we remove v1 and v2, we lose coverage for cliques where cov_count was 1.
                #   Let Lost = {c | c in (VC[v1] U VC[v2]) and cov_count[c] == 1}
                #   If Lost is empty, we can remove both. (Handled by repeated single removals).
                #   If Lost is not empty, we need a vertex v3 NOT in selected that covers ALL of Lost.
                #   If such v3 exists, we can remove v1, v2 and add v3. Size decreases by 1.
                
                # This is O(N^3) roughly. N=40, 64000 ops. Very fast.
                
                # Restore v1 coverage
                for i in vertex_cliques[v1]:
                    cov_count[i] += 1
                
                # Now try "Replace 2 with 1"
                # We need to iterate pairs. This is expensive inside the loop if done naively.
                # Let's just do the single removal pass again, and if it fails, try the 2-for-1.
                
                # To keep it simple and fast, let's just do multiple passes of single removal.
                # The 2-for-1 is a significant improvement.
                
                # Let's restructure the local search to be cleaner.
                pass 
            
            # If we didn't improve in the single removal pass, try 2-for-1 swaps
            if not improved:
                # Try to find v1, v2 in selected and v3 not in selected
                # such that V3 covers all cliques that are uniquely covered by v1 or v2.
                # And adding V3 doesn't break anything else (it only adds coverage).
                
                # Optimization: Precompute "unique coverage" sets?
                # For each v in selected, Unique(v) = {c | c in VC[v] and cov_count[c] == 1}
                
                # We want to find v1, v2, v3 such that Unique(v1) U Unique(v2) is subset of VC[v3].
                # If found:
                #   Remove v1, v2 from selected.
                #   Add v3 to selected.
                #   Update cov_counts.
                
                # This is O(N^3) per pass.
                
                # Let's implement this.
                unique_covers = {}
                for v in selected:
                    uc = set()
                    for i in vertex_cliques[v]:
                        if cov_count[i] == 1:
                            uc.add(i)
                    unique_covers[v] = uc
                
                found_2_for_1 = False
                # Iterate pairs in selected
                sel_list = list(selected)
                for idx1 in range(len(sel_list)):
                    if found_2_for_1: break
                    v1 = sel_list[idx1]
                    uc1 = unique_covers[v1]
                    
                    for idx2 in range(idx1 + 1, len(sel_list)):
                        v2 = sel_list[idx2]
                        uc2 = unique_covers[v2]
                        
                        combined_unique = uc1 | uc2
                        
                        if not combined_unique:
                            # If no unique covers, we could have removed them individually.
                            # This case should have been handled by single removals.
                            continue
                        
                        # Find v3 not in selected that covers combined_unique
                        for v3 in range(n):
                            if v3 in selected:
                                continue
                            
                            # Check if v3 covers all in combined_unique
                            covers_all = True
                            for c in combined_unique:
                                if c not in vertex_cliques[v3]:
                                    covers_all = False
                                    break
                            
                            if covers_all:
                                # Perform swap
                                selected.remove(v1)
                                selected.remove(v2)
                                selected.add(v3)
                                
                                # Update counts
                                for i in vertex_cliques[v1]:
                                    cov_count[i] -= 1
                                for i in vertex_cliques[v2]:
                                    cov_count[i] -= 1
                                for i in vertex_cliques[v3]:
                                    cov_count[i] += 1
                                
                                found_2_for_1 = True
                                break
                        if found_2_for_1:
                            break
                    if found_2_for_1:
                        break
                
                if found_2_for_1:
                    improved = True
                    continue # Restart local search loop
        
        return selected

    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for n, p, seed in instances:
        if time.time() > deadline:
            # Emergency fallback: return all vertices
            results.append(list(range(n)))
            continue
        
        adj = build_graph(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(adj)
        
        # Filter to only cliques with >= 2 vertices
        cliques = [c for c in cliques if len(c) >= 2]
        
        if not cliques:
            results.append([])
            continue
        
        # Allocate time for this instance
        # We have 3 instances. Let's give each ~40s?
        # Or just use the remaining time?
        # The previous code used a fixed 10s limit per instance.
        # Let's use a dynamic limit: remaining time / remaining instances.
        remaining_instances = 3 - len(results)
        time_limit = min(40.0, (deadline - time.time()) / remaining_instances - 1)
        
        selected = solve_instance(cliques, n, time_limit)
        
        # Verify coverage
        covered_cliques = set()
        for clique in cliques:
            for v in selected:
                if v in clique:
                    covered_cliques.add(clique)
                    break
        
        if len(covered_cliques) != len(cliques):
            # Fallback: add vertices to cover remaining
            for clique in cliques:
                if clique not in covered_cliques:
                    for v in clique:
                        if v not in selected:
                            selected.add(v)
                            break
        
        results.append(sorted(selected))
    
    return results
# EVOLVE-BLOCK-END
