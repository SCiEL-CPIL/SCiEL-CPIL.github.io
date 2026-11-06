import random
import time

def construct():
    start_time = time.time()
    TIME_LIMIT = 115  # Safe margin

    def get_edges(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = get_edges(n, p, seed)
        if not edges:
            return []
        
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        cliques = []
        remaining_edges = set(edges)
        
        while remaining_edges:
            if time.time() - start_time > TIME_LIMIT:
                # Fallback: return remaining edges as individual cliques
                for e in remaining_edges:
                    cliques.append(list(e))
                return cliques

            best_edge = None
            best_score = -1
            best_cand = set()
            
            # Find the edge with the most common neighbors in the remaining graph
            for u, v in remaining_edges:
                common = adj[u] & adj[v]
                valid = set()
                for w in common:
                    e_uw = (min(u, w), max(u, w))
                    e_vw = (min(v, w), max(v, w))
                    if e_uw in remaining_edges and e_vw in remaining_edges:
                        valid.add(w)
                
                if len(valid) > best_score:
                    best_score = len(valid)
                    best_edge = (u, v)
                    best_cand = valid
            
            if best_edge is None:
                break
            
            u, v = best_edge
            clique = {u, v}
            
            # Calculate degrees in remaining graph for candidates to sort them
            deg_in_rem = {}
            for w in best_cand:
                cnt = 0
                for x in adj[w]:
                    if (min(w, x), max(w, x)) in remaining_edges:
                        cnt += 1
                deg_in_rem[w] = cnt
            
            # Sort candidates by degree descending to prefer well-connected nodes
            sorted_cands = sorted(best_cand, key=lambda x: deg_in_rem.get(x, 0), reverse=True)
            
            for w in sorted_cands:
                can_add = True
                for v_c in clique:
                    e = (min(w, v_c), max(w, v_c))
                    if e not in remaining_edges:
                        can_add = False
                        break
                if can_add:
                    clique.add(w)
            
            cliques.append(list(clique))
            
            # Remove edges in clique
            clique_list = list(clique)
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    e = (min(clique_list[i], clique_list[j]), max(clique_list[i], clique_list[j]))
                    remaining_edges.discard(e)
        
        return cliques

    res1 = solve_instance(32, 0.6, 601)
    res2 = solve_instance(64, 0.55, 602)
    
    return [res1, res2]
