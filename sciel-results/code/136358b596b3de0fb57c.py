import time
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 140

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].append(u)
        return neighbors

    def solve_instance(n, k, p, seed):
        neighbors = instance(n, k, p, seed)
        N_sets = [set(neighbors[v]) for v in range(k, n)]
        
        # All pairs in K
        pairs = []
        pair_idx = {}
        for u in range(k):
            for w in range(u + 1, k):
                pair_idx[(u, w)] = len(pairs)
                pairs.append((u, w))
        
        # For each I-vertex, which pairs is it adjacent to?
        I = n - k
        i_adj_pairs = [set() for _ in range(I)]
        for i in range(I):
            N_v = N_sets[i]
            for u in N_v:
                for w in N_v:
                    if u < w:
                        i_adj_pairs[i].add(pair_idx[(u, w)])
        
        # Max bipartite matching: pairs on left, I-vertices on right
        # Edge if I-vertex is adjacent to both vertices of the pair
        # Use Hopcroft-Karp or simple augmenting path
        
        match_pair = [-1] * len(pairs)  # pair -> i
        match_i = [-1] * I  # i -> pair
        
        def try_augment(p, visited):
            for i in i_adj_pairs_list[p]:
                if visited[i]:
                    continue
                visited[i] = True
                if match_i[i] == -1 or try_augment(match_i[i], visited):
                    match_pair[p] = i
                    match_i[i] = p
                    return True
            return False
        
        # Build adjacency list for pairs
        i_adj_pairs_list = [list(s) for s in i_adj_pairs]
        # Wait, that's I-vertex -> pairs. We need pair -> I-vertices
        pair_to_i = [set() for _ in range(len(pairs))]
        for i in range(I):
            for p in i_adj_pairs[i]:
                pair_to_i[p].add(i)
        pair_to_i_list = [list(s) for s in pair_to_i]
        
        def try_augment(p, visited):
            for i in pair_to_i_list[p]:
                if visited[i]:
                    continue
                visited[i] = True
                if match_i[i] == -1 or try_augment(match_i[i], visited):
                    match_pair[p] = i
                    match_i[i] = p
                    return True
            return False
        
        matching_size = 0
        for p in range(len(pairs)):
            visited = [False] * I
            if try_augment(p, visited):
                matching_size += 1
        
        # Build solution
        cliques = []
        
        # I-vertex cliques
        i_to_clique = []
        for i in range(I):
            v = k + i
            if match_i[i] != -1:
                p = match_i[i]
                u, w = pairs[p]
                i_to_clique.append([u, w, v])
            else:
                # Single K-vertex
                N_v = N_sets[i]
                if N_v:
                    u = min(N_v)
                    i_to_clique.append([u, v])
                else:
                    i_to_clique.append([0, v])
        
        cliques.extend(i_to_clique)
        
        # K-vertex cover for uncovered vertices
        covered = set()
        for i in range(I):
            if match_i[i] != -1:
                u, w = pairs[match_i[i]]
                covered.add(u)
                covered.add(w)
        
        rem = [u for u in range(k) if u not in covered]
        rem.sort()
        
        j = 0
        while j < len(rem):
            if j + 1 < len(rem):
                cliques.append([rem[j], rem[j + 1]])
                j += 2
            else:
                # Odd: merge last vertex into first I-clique
                cliques[0].append(rem[j])
                j += 1
        
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            n, k, p, seed = spec
            neighbors = instance(n, k, p, seed)
            cliques = []
            for v in range(k, n):
                for u in neighbors[v]:
                    cliques.append([u, v])
            for u in range(k):
                for w in range(u + 1, k):
                    cliques.append([u, w])
            results.append(cliques)
            continue
        cliques = solve_instance(*spec)
        results.append([[int(x) for x in c] for c in cliques])
    
    return results
