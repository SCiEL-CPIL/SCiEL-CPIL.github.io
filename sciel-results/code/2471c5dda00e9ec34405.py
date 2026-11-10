"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def greedy_partition(n, edges, seed_heur, strategy="max_deg"):
        """
        Greedy clique partition.
        strategy: 'max_deg' - pick vertex with max remaining degree
                  'random' - pick random vertex with edges
                  'min_fill' - pick vertex minimizing fill-in (approx)
        """
        neighbors = [set() for _ in range(n)]
        for u, v in edges:
            neighbors[u].add(v)
            neighbors[v].add(u)
        
        remaining_edges = set(edges)
        cliques = []
        rng_heur = Random(seed_heur)
        
        while remaining_edges:
            # Find candidate vertices that still have remaining edges
            active = set()
            for u, v in remaining_edges:
                active.add(u)
                active.add(v)
            
            if not active:
                break
            
            if strategy == "max_deg":
                # Pick vertex with max degree in remaining graph
                max_deg = -1
                candidates = []
                for u in active:
                    deg = 0
                    for v in neighbors[u]:
                        if (min(u,v), max(u,v)) in remaining_edges:
                            deg += 1
                    if deg > max_deg:
                        max_deg = deg
                        candidates = [u]
                    elif deg == max_deg and deg > 0:
                        candidates.append(u)
                v = rng_heur.choice(candidates) if candidates else rng_heur.choice(list(active))
                
            elif strategy == "random":
                v = rng_heur.choice(list(active))
                
            else:  # min_fill approximation
                # Pick vertex with fewest "conflicts" - approximate by min degree
                min_deg = float('inf')
                candidates = []
                for u in active:
                    deg = 0
                    for w in neighbors[u]:
                        if (min(u,w), max(u,w)) in remaining_edges:
                            deg += 1
                    if deg < min_deg:
                        min_deg = deg
                        candidates = [u]
                    elif deg == min_deg:
                        candidates.append(u)
                v = rng_heur.choice(candidates) if candidates else rng_heur.choice(list(active))
            
            # Build clique containing v
            C = [v]
            P = set(neighbors[v])
            # Only keep vertices that share an edge with v in remaining graph
            P = {u for u in P if (min(v,u), max(v,u)) in remaining_edges}
            
            while P:
                # Find u in P that maximizes common neighbors in P
                max_score = -1
                tie_candidates = []
                for u in P:
                    score = len(neighbors[u] & P)
                    if score > max_score:
                        max_score = score
                        tie_candidates = [u]
                    elif score == max_score:
                        tie_candidates.append(u)
                
                u = rng_heur.choice(tie_candidates)
                C.append(u)
                P = P & neighbors[u]
            
            # Remove edges of C from remaining
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u, v2 = C[i], C[j]
                    e = (u, v2) if u < v2 else (v2, u)
                    if e in remaining_edges:
                        remaining_edges.remove(e)
                    neighbors[u].discard(v2)
                    neighbors[v2].discard(u)
                    
            cliques.append(C)
            
        return cliques

    def local_search_improve(n, edges, cliques, seed_ls, max_iters=50):
        """
        Try to merge adjacent cliques or reassign vertices to reduce count.
        Simple local search: try removing one clique and redistributing its edges.
        """
        if len(cliques) <= 1:
            return cliques
        
        rng_ls = Random(seed_ls)
        best = cliques[:]
        best_count = len(cliques)
        
        # Build edge -> clique mapping
        def build_edge_map(clq_list):
            emap = {}
            for idx, clq in enumerate(clq_list):
                for i in range(len(clq)):
                    for j in range(i+1, len(clq)):
                        u, v = clq[i], clq[j]
                        e = (u, v) if u < v else (v, u)
                        emap[e] = idx
            return emap
        
        # Try to merge pairs of cliques if their union is a clique
        for _ in range(max_iters):
            merged = False
            for i in range(len(best)):
                for j in range(i+1, len(best)):
                    # Check if union of cliques i and j is a clique
                    union = set(best[i]) | set(best[j])
                    is_clique = True
                    for u in union:
                        for v in union:
                            if u >= v:
                                continue
                            if (u, v) not in edges:
                                is_clique = False
                                break
                        if not is_clique:
                            break
                    if is_clique:
                        new_cliques = best[:i] + best[i+1:j] + best[j+1:] + [list(union)]
                        if len(new_cliques) < best_count:
                            best = new_cliques
                            best_count = len(best)
                            merged = True
                            break
                if merged:
                    break
            if not merged:
                break
        
        return best

    def solve_instance(n, p, seed, time_limit):
        edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        best_cliques = None
        best_count = float('inf')
        
        strategies = ["max_deg", "random", "min_fill"]
        attempt = 0
        while time.time() < time_limit:
            strat = strategies[attempt % len(strategies)]
            heur_seed = seed + attempt * 7919
            cliques = greedy_partition(n, edges, heur_seed, strat)
            attempt += 1
            
            if len(cliques) < best_count:
                best_count = len(cliques)
                best_cliques = cliques[:]
                
            # Try local search improvement occasionally
            if attempt % 3 == 0 and best_cliques is not None:
                improved = local_search_improve(n, edges, best_cliques, seed + attempt * 31, max_iters=20)
                if len(improved) < best_count:
                    best_count = len(improved)
                    best_cliques = improved
        
        if best_cliques is None:
            best_cliques = [[u, v] for u, v in edges]
            
        return best_cliques

    # Allocate time: 40s for n=32, rest for n=64
    t1_limit = start_time + 40.0
    res1 = solve_instance(32, 0.6, 601, t1_limit)
    
    t2_limit = deadline
    res2 = solve_instance(64, 0.55, 602, t2_limit)
    
    return [res1, res2]
# EVOLVE-BLOCK-END
