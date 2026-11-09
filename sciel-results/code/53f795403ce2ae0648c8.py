import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety
    
    # Start with Paley 17 as baseline
    best_n = 17
    best_edges = _paley_17()
    
    # Try to build larger graphs using simulated annealing with delta penalties
    for target_n in range(35, 17, -1):
        if time.time() > deadline - 15:
            break
        
        time_alloc = min(30.0, (deadline - time.time()) * 0.7)
        if time_alloc < 5.0:
            break
        
        success, edges = _sa_construct(target_n, time_alloc, deadline)
        if success:
            best_n = target_n
            best_edges = edges
            if target_n >= 35:
                break
    
    return {'n': best_n, 'edges': best_edges}

def _paley_17():
    squares = {x * x % 17 for x in range(1, 17)}
    return [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]

def _sa_construct(n, time_alloc, deadline):
    """
    Use simulated annealing with delta-based penalty updates.
    Penalty = (# K4s) * W_K4 + (# independent 6-sets) * W_IND6
    We track penalties incrementally for O(n^4) per flip instead of O(n^6).
    """
    W_K4 = 1000
    W_IND6 = 1000
    
    num_restarts = max(2, int(time_alloc / 10.0))
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Initialize with a random graph
        adj = np.zeros((n, n), dtype=np.uint8)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.4:
                    adj[i, j] = 1
                    adj[j, i] = 1
        
        # Compute initial penalties
        k4_count, ind6_count = _count_violations(adj, n)
        penalty = k4_count * W_K4 + ind6_count * W_IND6
        
        if penalty == 0:
            return True, _adj_to_edges(adj, n)
        
        # Simulated annealing
        T = 100.0
        T_min = 0.01
        cooling = 0.9995
        
        steps_per_check = 1000
        step_count = 0
        
        while time.time() < deadline - 3:
            # Pick a random edge to flip
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            # Compute delta penalty from flipping edge (i,j)
            delta_k4, delta_ind6 = _flip_delta(adj, n, i, j)
            delta_penalty = delta_k4 * W_K4 + delta_ind6 * W_IND6
            
            # Accept if improving or with probability exp(-delta/T)
            if delta_penalty <= 0 or random.random() < np.exp(-delta_penalty / T):
                # Accept the flip
                adj[i, j] = 1 - adj[i, j]
                adj[j, i] = adj[i, j]
                k4_count += delta_k4
                ind6_count += delta_ind6
                penalty = k4_count * W_K4 + ind6_count * W_IND6
                
                if penalty == 0:
                    return True, _adj_to_edges(adj, n)
            
            step_count += 1
            if step_count % steps_per_check == 0:
                T = max(T_min, T * cooling)
        
        # Final check
        if k4_count == 0 and ind6_count == 0:
            return True, _adj_to_edges(adj, n)
    
    return False, None

def _count_violations(adj, n):
    """Count K4s and independent 6-sets in the graph."""
    k4_count = 0
    ind6_count = 0
    
    # Count K4s: for each pair (i,j) that is an edge, count triangles in common neighbors
    # Actually, let's just do a direct count for accuracy
    # For n up to 35, we can be smarter
    
    # K4 count: for each 4-tuple, check if all 6 edges exist
    # This is O(n^4) which is fine for n=35
    for a in range(n):
        for b in range(a + 1, n):
            if not adj[a, b]:
                continue
            for c in range(b + 1, n):
                if not adj[a, c] or not adj[b, c]:
                    continue
                for d in range(c + 1, n):
                    if adj[a, d] and adj[b, d] and adj[c, d]:
                        k4_count += 1
    
    # Independent 6-set count: for each 6-tuple, check if no edges exist
    # O(n^6) is too slow for n=35. Use complement graph approach.
    # In complement, independent 6-set in G = K6 in complement.
    # For n=35, C(35,6) = 1,623,160 which is manageable.
    for a in range(n):
        for b in range(a + 1, n):
            if adj[a, b]:
                continue
            for c in range(b + 1, n):
                if adj[a, c] or adj[b, c]:
                    continue
                for d in range(c + 1, n):
                    if adj[a, d] or adj[b, d] or adj[c, d]:
                        continue
                    for e in range(d + 1, n):
                        if adj[a, e] or adj[b, e] or adj[c, e] or adj[d, e]:
                            continue
                        for f in range(e + 1, n):
                            if (adj[a, f] or adj[b, f] or adj[c, f] or 
                                adj[d, f] or adj[e, f]):
                                continue
                            ind6_count += 1
    
    return k4_count, ind6_count

def _flip_delta(adj, n, i, j):
    """
    Compute the change in (#K4) and (#ind6) when flipping edge (i,j).
    If adj[i,j] is currently 1, we're removing the edge.
    If adj[i,j] is currently 0, we're adding the edge.
    
    Adding edge (i,j):
    - New K4s: for each pair (k,l) where k,l > max(i,j) or k,l < min(i,j), 
      if k~l and k~i and k~j and l~i and l~j, then {i,j,k,l} is a new K4.
      Actually: {i,j,k,l} forms K4 iff all 6 edges present. Since we're adding (i,j),
      we need: (i,k), (j,k), (i,l), (j,l), (k,l) all present.
    - New independent 6-sets: none directly (adding an edge can only destroy independent sets)
      Wait, adding an edge can DESTROY independent sets that contained both i and j.
      So delta_ind6 is negative (or zero) when adding an edge.
      Specifically: independent 6-sets containing both i and j are destroyed.
      For each 4-tuple (a,b,c,d) not containing i or j, if {i,j,a,b,c,d} was independent
      (all pairs non-adjacent), it's now destroyed.
      So we need: a,b,c,d all non-adjacent to i, all non-adjacent to j, 
      and {a,b,c,d} independent among themselves.
    
    Removing edge (i,j):
    - Destroys K4s: for each pair (k,l), if {i,j,k,l} was a K4, it's destroyed.
      Need: (i,k), (j,k), (i,l), (j,l), (k,l) all present.
    - Creates independent 6-sets: for each 4-tuple (a,b,c,d) not containing i or j,
      if a,b,c,d all non-adjacent to i, all non-adjacent to j, and {a,b,c,d} independent,
      then {i,j,a,b,c,d} becomes a new independent 6-set.
    """
    adding = not adj[i, j]  # True if we're adding the edge
    
    delta_k4 = 0
    delta_ind6 = 0
    
    if adding:
        # Adding edge (i,j): creates K4s, destroys ind6-sets
        # K4s created: find pairs (k,l) such that all edges among {i,j,k,l} except (i,j) exist
        # i.e., (i,k),(j,k),(i,l),(j,l),(k,l) all exist
        for k in range(n):
            if k == i or k == j:
                continue
            if not adj[i, k] or not adj[j, k]:
                continue
            for l in range(k + 1, n):
                if l == i or l == j:
                    continue
                if not adj[i, l] or not adj[j, l]:
                    continue
                if adj[k, l]:
                    delta_k4 += 1
        
        # Ind6-sets destroyed: find 4-tuples (a,b,c,d) where all are non-adjacent to i and j
        # and {a,b,c,d} is independent
        non_adj_i = [v for v in range(n) if v != i and v != j and not adj[i, v]]
        non_adj_j = [v for v in range(n) if v != i and v != j and not adj[j, v]]
        # Intersection: vertices non-adjacent to both i and j
        candidates = [v for v in non_adj_i if v in set(non_adj_j)]
        
        # Count independent 4-sets in candidates
        cands = candidates
        for a in range(len(cands)):
            for b in range(a + 1, len(cands)):
                if adj[cands[a], cands[b]]:
                    continue
                for c in range(b + 1, len(cands)):
                    if adj[cands[a], cands[c]] or adj[cands[b], cands[c]]:
                        continue
                    for d in range(c + 1, len(cands)):
                        if (adj[cands[a], cands[d]] or adj[cands[b], cands[d]] or 
                            adj[cands[c], cands[d]]):
                            continue
                        delta_ind6 -= 1  # Destroying an independent 6-set
    else:
        # Removing edge (i,j): destroys K4s, creates ind6-sets
        # K4s destroyed: same logic as above
        for k in range(n):
            if k == i or k == j:
                continue
            if not adj[i, k] or not adj[j, k]:
                continue
            for l in range(k + 1, n):
                if l == i or l == j:
                    continue
                if not adj[i, l] or not adj[j, l]:
                    continue
                if adj[k, l]:
                    delta_k4 -= 1
        
        # Ind6-sets created: same candidate logic
        non_adj_i = [v for v in range(n) if v != i and v != j and not adj[i, v]]
        non_adj_j = [v for v in range(n) if v != i and v != j and not adj[j, v]]
        candidates = [v for v in non_adj_i if v in set(non_adj_j)]
        
        cands = candidates
        for a in range(len(cands)):
            for b in range(a + 1, len(cands)):
                if adj[cands[a], cands[b]]:
                    continue
                for c in range(b + 1, len(cands)):
                    if adj[cands[a], cands[c]] or adj[cands[b], cands[c]]:
                        continue
                    for d in range(c + 1, len(cands)):
                        if (adj[cands[a], cands[d]] or adj[cands[b], cands[d]] or 
                            adj[cands[c], cands[d]]):
                            continue
                        delta_ind6 += 1  # Creating an independent 6-set
    
    return delta_k4, delta_ind6

def _adj_to_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                edges.append([i, j])
    return edges
