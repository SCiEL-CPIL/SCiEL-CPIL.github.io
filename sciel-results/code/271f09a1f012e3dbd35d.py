import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        # Core clique edges
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        # Cross edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, k, p, seed):
        rng = random.Random(seed)
        
        # Build adjacency for core and cross
        # core_adj[u] is set of neighbors in core
        core_adj = [set() for _ in range(k)]
        for u in range(k):
            for v in range(u + 1, k):
                core_adj[u].add(v)
                core_adj[v].add(u)
                
        # cross_adj[i] is set of neighbors in core for vertex i in I
        cross_adj = [set() for _ in range(n)]
        for i in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_adj[i].add(u)
                    core_adj[u].add(i) # Not strictly needed for core logic, but good for full graph view if needed
                    # Actually, we only care about core edges availability.
        
        # We need to track which core edges are still "available" to be covered by I-cliques.
        # A core edge (u,v) is available if it hasn't been used in any I-clique yet.
        # Initially all core edges are available.
        available_core_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                available_core_edges.add((u, v))
        
        cliques = []
        
        # Process I vertices
        # Sort by degree in core to handle high-degree vertices first? 
        # Or random order? 
        # Let's try sorting by degree descending to pack larger cliques early.
        i_vertices = list(range(k, n))
        i_vertices.sort(key=lambda x: len(cross_adj[x]), reverse=True)
        
        for i in i_vertices:
            if time.time() > deadline:
                break
            N_i = cross_adj[i]
            if not N_i:
                continue
                
            # Find largest clique S in N_i such that all edges in S are in available_core_edges
            # Greedy heuristic:
            # Candidates = N_i
            # S = []
            # While candidates not empty:
            #   Pick v in candidates with max degree in available_core_edges restricted to candidates
            #   If v has edges to all in S? No, S is being built.
            #   Actually, we want S to be a clique in the graph of available_core_edges.
            #   So we need (u,v) in available_core_edges for all u,v in S.
            
            candidates = set(N_i)
            S = []
            
            while candidates:
                # Pick vertex with max "potential" - i.e., connected to all current S via available edges
                # And ideally high degree in remaining candidates to allow more additions
                best_v = None
                best_score = -1
                
                for v in candidates:
                    # Check if v can connect to all in S
                    valid = True
                    for u in S:
                        edge = (min(u, v), max(u, v))
                        if edge not in available_core_edges:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Score: number of candidates w such that (v,w) is available
                    # This helps pick a vertex that allows adding more later
                    count = 0
                    for w in candidates:
                        if w == v: continue
                        edge = (min(v, w), max(v, w))
                        if edge in available_core_edges:
                            count += 1
                            
                    if count > best_score:
                        best_score = count
                        best_v = v
                        
                if best_v is None:
                    # No vertex can be added to S
                    break
                    
                S.append(best_v)
                candidates.remove(best_v)
                
                # Remove from candidates any vertex that cannot connect to best_v
                # Actually, we just check validity in the next iteration.
                # But to speed up, we can prune.
                # Let's just rely on the check.
                
            if S:
                clique = [i] + sorted(S)
                cliques.append(clique)
                
                # Remove edges in S from available_core_edges
                for idx_u in range(len(S)):
                    for idx_v in range(idx_u + 1, len(S)):
                        u = S[idx_u]
                        v = S[idx_v]
                        edge = (min(u, v), max(u, v))
                        available_core_edges.discard(edge)
            else:
                # If no clique found (should not happen if N_i not empty, as single vertex is a clique of size 1? 
                # But we need at least 2 vertices in final clique. 
                # If S is empty, we can't form a clique with i and S.
                # But wait, if N_i is not empty, we can always pick one vertex u. 
                # Then S=[u]. Edge (u,u) doesn't exist. 
                # So S=[u] is always valid.
                # Why would best_v be None? 
                # Ah, if S is empty, the condition "connected to all in S" is vacuously true.
                # So best_v should be found.
                pass

        # Cover remaining core edges
        if time.time() > deadline:
             # Fallback: cover remaining edges as 2-cliques
             for e in available_core_edges:
                 cliques.append(list(e))
             return cliques

        remaining_edges = set(available_core_edges)
        
        while remaining_edges:
            if time.time() > deadline:
                for e in remaining_edges:
                    cliques.append(list(e))
                break
                
            # Find max clique in remaining_edges
            # Build adj for remaining graph
            rem_adj = defaultdict(set)
            for u, v in remaining_edges:
                rem_adj[u].add(v)
                rem_adj[v].add(u)
                
            # Greedy max clique
            candidates = set(rem_adj.keys())
            if not candidates:
                break
                
            S = []
            while candidates:
                # Pick vertex with max degree in candidates
                best_v = None
                best_deg = -1
                for v in candidates:
                    deg = len(rem_adj[v] & candidates)
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                            
                if best_v is None:
                    break
                    
                S.append(best_v)
                candidates.remove(best_v)
                
                # Keep only neighbors of best_v
                candidates = candidates & rem_adj[best_v]
                
            if len(S) < 2:
                # If no clique of size >= 2 found, pick an edge
                u, v = next(iter(remaining_edges))
                S = [u, v]
            else:
                S = sorted(S)
                
            cliques.append(S)
            
            # Remove edges in S
            for idx_u in range(len(S)):
                for idx_v in range(idx_u + 1, len(S)):
                    u = S[idx_u]
                    v = S[idx_v]
                    edge = (min(u, v), max(u, v))
                    remaining_edges.discard(edge)
                    
        return cliques

    results = []
    specs = [(30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)]
    
    for spec in specs:
        if time.time() > deadline:
            # If we run out of time, we must return something valid.
            # But we should have finished previous ones.
            # Just break and handle remaining with trivial solution?
            # The contract says "Any broken rule makes the whole result invalid".
            # So we must ensure all 3 are solved.
            # If we are tight, we should skip complex search for remaining.
            pass
            
        cliques = solve_instance(*spec)
        
        # Validate and clean types
        clean_cliques = []
        for c in cliques:
            # Ensure list of ints
            clean_cliques.append([int(v) for v in c])
        results.append(clean_cliques)
        
    return results

