"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def random_graph(n, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def _greedy_partition_fast(n, edges, seed_heur, mode="max_common"):
        """Fast greedy clique partition using bitmasks for n<=64."""
        neighbors = [0] * n  # bitmask of neighbors
        for u, v in edges:
            neighbors[u] |= (1 << v)
            neighbors[v] |= (1 << u)
        
        remaining_edges = set(edges)
        cliques = []
        rng_heur = Random(seed_heur)
        
        while remaining_edges:
            if mode == "max_common":
                best_edge = None
                best_score = -1
                edge_list = list(remaining_edges)
                rng_heur.shuffle(edge_list)
                
                for u, v in edge_list:
                    common = neighbors[u] & neighbors[v]
                    score = bin(common).count('1')
                    if score > best_score:
                        best_score = score
                        best_edge = (u, v)
            
            elif mode == "min_common":
                best_edge = None
                best_score = float('inf')
                edge_list = list(remaining_edges)
                rng_heur.shuffle(edge_list)
                
                for u, v in edge_list:
                    common = neighbors[u] & neighbors[v]
                    score = bin(common).count('1')
                    if score < best_score:
                        best_score = score
                        best_edge = (u, v)
            else:
                best_edge = None
                best_score = -1
                edge_list = list(remaining_edges)
                rng_heur.shuffle(edge_list)
                for u, v in edge_list:
                    common = neighbors[u] & neighbors[v]
                    score = bin(common).count('1')
                    if score > best_score:
                        best_score = score
                        best_edge = (u, v)

            if best_edge is None:
                break
                
            u, v = best_edge
            C = [u, v]
            P = neighbors[u] & neighbors[v]
            
            # Greedily expand the clique
            while P:
                # Find w in P with max degree in P
                max_deg = -1
                best_w = -1
                P_list = []
                temp = P
                while temp:
                    lsb = temp & (-temp)
                    w = lsb.bit_length() - 1
                    P_list.append(w)
                    temp ^= lsb
                
                for w in P_list:
                    deg = bin(neighbors[w] & P).count('1')
                    if deg > max_deg:
                        max_deg = deg
                        best_w = w
                
                if best_w < 0:
                    break
                    
                C.append(best_w)
                P = P & neighbors[best_w]
            
            # Remove edges of C from remaining graph
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    a, b = C[i], C[j]
                    e = (a, b) if a < b else (b, a)
                    if e in remaining_edges:
                        remaining_edges.remove(e)
                        neighbors[a] &= ~(1 << b)
                        neighbors[b] &= ~(1 << a)
            
            cliques.append(C)
            
        return cliques

    def solve_instance(n, p, seed, time_limit):
        edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        best_cliques = None
        best_count = float('inf')
        
        attempt = 0
        modes = ["max_common", "min_common"]
        
        while time.time() < time_limit - 5:  # Leave 5s for local improvement
            heur_seed = seed + attempt * 1000 + 42
            mode = modes[attempt % len(modes)]
            cliques = _greedy_partition_fast(n, edges, heur_seed, mode)
            
            if len(cliques) < best_count:
                best_count = len(cliques)
                best_cliques = cliques
            
            attempt += 1
                
        if best_cliques is None:
            # Fallback: each edge is a clique
            best_cliques = [[u, v] for u, v in edges]
        
        return best_cliques

    # Allocate time: 40s for n=32, rest for n=64
    t1_limit = start_time + 40.0
    res1 = solve_instance(32, 0.6, 601, t1_limit)
    
    t2_limit = deadline
    res2 = solve_instance(64, 0.55, 602, t2_limit)
    
    return [res1, res2]
# EVOLVE-BLOCK-END
