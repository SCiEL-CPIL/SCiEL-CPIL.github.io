import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin
    
    # Candidates: numbers in [1,255] that share a factor with 255
    # Actually, we can include any number, but it must share a factor with 255 to be in a clique with 255
    # 255 = 3 * 5 * 17
    # So any number in the set must share a factor with 255, i.e., be divisible by 3, 5, or 17
    
    # Precompute gcd relationships
    # For efficiency, precompute prime factors or just use gcd
    
    # Greedy: start with 255, add vertices with max degree in current subgraph
    candidates = [i for i in range(1, 256) if math.gcd(i, 255) > 1]
    
    # Precompute adjacency
    n = len(candidates)
    # Map value to index
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    
    # Adjacency matrix (or list of sets)
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Greedy clique: start with 255
    idx_255 = val_to_idx[255]
    clique = {idx_255}
    
    # Available vertices
    available = set(range(n)) - clique
    
    # Greedy: repeatedly add the vertex with the most neighbors in the current clique
    while available:
        # Check deadline
        if time.time() > deadline:
            break
        
        best_vertex = None
        best_count = -1
        for v in available:
            count = len(adj[v] & clique)
            if count > best_count:
                best_count = count
                best_vertex = v
        
        if best_vertex is None or best_count < len(clique):
            break  # Can't add any more
        
        clique.add(best_vertex)
        available.remove(best_vertex)
    
    # Local search: try to improve
    # For each vertex not in clique, check if it can be added
    # For each pair (in_clique, not_in_clique), try swap
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding
        for v in list(available):
            if adj[v] >= clique:  # v is connected to all in clique
                clique.add(v)
                available.remove(v)
                improved = True
                break
        
        # Try swapping: remove u from clique, add v
        if not improved:
            for u in list(clique):
                if time.time() > deadline:
                    break
                for v in available:
                    # Check if v can replace u
                    # v must be connected to all in clique - {u}
                    if adj[v] >= (clique - {u}):
                        clique.remove(u)
                        clique.add(v)
                        available.remove(v)
                        available.add(u)
                        improved = True
                        break
                if improved:
                    break
    
    # Convert back to values
    result = [candidates[i] for i in clique]
    return result
