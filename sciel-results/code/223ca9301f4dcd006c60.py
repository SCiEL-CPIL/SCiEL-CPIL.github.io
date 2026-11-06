import time
import random
from math import gcd, lcm

def _build_adjacency(n=512):
    """
    Builds an adjacency list for the graph where vertices are 1..n
    and an edge exists if lcm(u, v) <= n.
    """
    adj = [[] for _ in range(n + 1)]
    for u in range(1, n + 1):
        for v in range(u + 1, n + 1):
            if lcm(u, v) <= n:
                adj[u].append(v)
                adj[v].append(u)
    return adj

def _greedy_clique(adj, n, random_seed=None):
    """
    Greedy construction of a clique.
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    # Vertices 1..n
    remaining = list(range(1, n + 1))
    # Sort by degree descending initially to help greedy
    remaining.sort(key=lambda x: len(adj[x]), reverse=True)
    
    clique = []
    # We need to efficiently find candidates connected to all in clique
    # For small N, checking all remaining is fine.
    
    while remaining:
        # Find vertex in remaining with max connections to other remaining vertices
        # AND connected to all in clique (guaranteed if we filter remaining correctly)
        
        # Optimization: Filter remaining to only those connected to all in clique
        # This is done implicitly if we remove non-neighbors when adding to clique.
        
        # Pick best vertex from remaining
        # Heuristic: Max degree in remaining subgraph
        best_v = None
        max_deg = -1
        for v in remaining:
            deg = 0
            for u in remaining:
                if u == v: continue
                if u in adj[v]:
                    deg += 1
            if deg > max_deg:
                max_deg = deg
                best_v = v
        
        if best_v is None:
            break
            
        clique.append(best_v)
        
        # Remove best_v and all non-neighbors of best_v from remaining
        neighbors = set(adj[best_v])
        new_remaining = []
        for u in remaining:
            if u == best_v:
                continue
            if u in neighbors:
                new_remaining.append(u)
        remaining = new_remaining
        
    return clique

def _local_search(adj, n, clique, max_swaps=1000):
    """
    Local search to improve clique size.
    1. Try to add any vertex.
    2. Try to swap a vertex in clique with one outside to potentially enable future additions.
    """
    clique_set = set(clique)
    improved = True
    swaps_done = 0
    
    while improved and swaps_done < max_swaps:
        improved = False
        
        # 1. Try to add
        added = False
        for v in range(1, n + 1):
            if v in clique_set:
                continue
            # Check if v connects to all in clique
            if all(u in adj[v] for u in clique_set):
                clique_set.add(v)
                added = True
                improved = True
                break # Restart loop after addition
        
        if added:
            swaps_done = 0 # Reset swap counter on improvement
            continue
            
        # 2. Try to swap
        # Find a vertex u in clique and v outside such that (clique - {u} + {v}) is a clique
        # This keeps size same. We accept swaps randomly or based on some heuristic
        # to escape local optima.
        candidates_for_swap = []
        for u in list(clique_set):
            temp_clique = clique_set - {u}
            for v in range(1, n + 1):
                if v in temp_clique:
                    continue
                # Check if v connects to all in temp_clique
                if all(w in adj[v] for w in temp_clique):
                    candidates_for_swap.append((u, v))
        
        if candidates_for_swap:
            # Pick a random swap
            u, v = random.choice(candidates_for_swap)
            clique_set.remove(u)
            clique_set.add(v)
            improved = True
            swaps_done += 1
        else:
            # No swaps possible, stuck
            break
            
    return list(clique_set)

def construct():
    start_time = time.time()
    deadline = 50.0 # Leave margin for return
    
    n = 512
    adj = _build_adjacency(n)
    
    best_clique = []
    
    # Run multiple restarts
    while time.time() < deadline:
        # Randomize initial order or use random restart
        seed = int(time.time() * 1000) % (2**32)
        clique = _greedy_clique(adj, n, random_seed=seed)
        
        # Local search
        clique = _local_search(adj, n, clique, max_swaps=200)
        
        if len(clique) > len(best_clique):
            best_clique = clique
            
        # If we found a very large clique, we might stop early?
        # Max possible is 26. If we hit 26, stop.
        if len(best_clique) >= 26:
            break

    # Fallback: If we didn't find anything good, return divisors of 420 (size 24)
    if len(best_clique) < 24:
        # Generate divisors of 420
        divs = []
        for i in range(1, int(420**0.5) + 1):
            if 420 % i == 0:
                divs.append(i)
                if i != 420 // i:
                    divs.append(420 // i)
        best_clique = sorted(divs)

    return sorted(best_clique)
