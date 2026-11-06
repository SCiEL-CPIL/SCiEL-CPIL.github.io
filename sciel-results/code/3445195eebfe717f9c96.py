import time
from random import Random

def _random_graph(n, p, seed):
    rng = Random(seed)
    adj = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    return adj

def _greedy_clique(adj, start, n):
    """
    Greedy maximal clique expansion starting from `start`.
    Uses the heuristic: pick vertex in candidates with max degree within candidates.
    """
    clique = [start]
    cand = adj[start]
    
    while cand:
        best = -1
        best_deg = -1
        temp = cand
        while temp:
            low = temp & (-temp)
            u = low.bit_length() - 1
            # Degree of u within cand
            deg = (adj[u] & cand).bit_count()
            if deg > best_deg:
                best_deg = deg
                best = u
            temp ^= low
        
        if best == -1:
            break
            
        clique.append(best)
        cand &= adj[best]
        
    return clique

def _solve_instance(adj, n, deadline):
    rem = [adj[i] for i in range(n)]
    remaining_edges = sum(rem[i].bit_count() for i in range(n)) // 2
    cliques = []
    
    if remaining_edges == 0:
        return cliques
    
    # Number of parallel greedy attempts per step
    # Higher is better quality, lower is faster
    # Let's start with 5
    NUM_STARTS = 5 
    
    while remaining_edges > 0:
        if time.time() > deadline:
            break
            
        # Find top K vertices by degree in remaining graph
        degs = [(rem[i].bit_count(), i) for i in range(n)]
        degs.sort(reverse=True)
        
        # If max degree is 0, break (should be handled by remaining_edges check)
        if degs[0][0] == 0:
            break
            
        # If max degree is 1, we can just take the edge
        if degs[0][0] == 1:
             v = degs[0][1]
             low = rem[v] & (-rem[v])
             u = low.bit_length() - 1
             cliques.append([v, u])
             rem[v] &= ~(1 << u)
             rem[u] &= ~(1 << v)
             remaining_edges -= 1
             continue
             
        # Try multiple starts
        best_clique = []
        best_size = 0
        
        # Pick top NUM_STARTS vertices with degree > 0
        starts = [v for d, v in degs[:NUM_STARTS] if d > 0]
        
        # Also maybe add some random vertices? 
        # For now, just top degree.
        
        for s in starts:
            c = _greedy_clique(rem, s, n)
            if len(c) > best_size:
                best_size = len(c)
                best_clique = c
        
        # Fallback if something went wrong
        if not best_clique or len(best_clique) < 2:
             v = degs[0][1]
             low = rem[v] & (-rem[v])
             u = low.bit_length() - 1
             best_clique = [v, u]
             best_size = 2

        # Remove edges of best_clique
        m = len(best_clique)
        cliques.append(best_clique)
        for i in range(m):
            for j in range(i + 1, m):
                a, b = best_clique[i], best_clique[j]
                rem[a] &= ~(1 << b)
                rem[b] &= ~(1 << a)
        remaining_edges -= m * (m - 1) // 2
        
    # Handle remaining edges
    if remaining_edges > 0:
        for i in range(n):
            if time.time() > deadline:
                break
            temp = rem[i]
            while temp:
                if time.time() > deadline:
                    break
                low = temp & (-temp)
                j = low.bit_length() - 1
                cliques.append([i, j])
                rem[i] &= ~(1 << j)
                rem[j] &= ~(1 << i)
                temp &= temp - 1
                rem[i] &= ~(1 << j)
                
    return cliques

def construct():
    deadline = time.time() + 140
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline:
            results.append([])
            continue
        adj = _random_graph(n, p, seed)
        cliques = _solve_instance(adj, n, deadline)
        results.append(cliques)
    return results
