import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    # Fallback: Paley 37
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    best_n = q
    best_adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                best_adj[i] |= (1 << j)
                best_adj[j] |= (1 << i)
    
    # Try to find larger graphs
    for target_n in [46, 45, 44, 43]:
        if time.time() > deadline - 30: # Leave 30s for cleanup
            break
            
        # Run local search
        result = _local_search(target_n, best_adj, deadline)
        if result is not None:
            n, adj = result
            if n > best_n:
                best_n = n
                best_adj = adj
                # If we found 46, we can stop? No, 46 is max possible? 
                # R(5,5) <= 46. So 46 is the max possible.
                if n == 46:
                    break

    # Convert best_adj to edges
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj[i] >> j) & 1:
                edges.append((i, j))
                
    return best_n, edges

def _local_search(target_n, init_adj, deadline):
    # Initialize graph of size target_n
    # If target_n > 37, extend init_adj (37 Paley) with random vertices
    n = target_n
    adj = [0] * n
    
    # Copy Paley part
    for i in range(37):
        adj[i] = init_adj[i]
        
    # Add random edges for new vertices and between new and old
    # Vertices 37 to n-1 are new
    for i in range(37, n):
        # Connect to existing 0..36
        for j in range(37):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        # Connect to other new vertices
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Local Search Loop
    while time.time() < deadline - 5:
        bad_set, is_clique = _find_bad_set(n, adj)
        if bad_set is None:
            return n, adj
            
        # Flip a random edge in bad_set
        # bad_set is a tuple of 5 vertices
        # Pick a random pair
        u, v = random.sample(bad_set, 2)
        
        # Flip edge (u, v)
        if (adj[u] >> v) & 1:
            # Edge exists, remove it
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            # Edge doesn't exist, add it
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    return None

def _find_bad_set(n, adj):
    # Find a K5 or Independent 5-set
    # Randomize vertex order to avoid bias
    vertices = list(range(n))
    random.shuffle(vertices)
    
    for v in vertices:
        # Check for K5 containing v
        # Neighbors of v
        neighbors = []
        for u in range(n):
            if u != v and ((adj[v] >> u) & 1):
                neighbors.append(u)
                
        # Check if neighbors contain a K4
        # Iterate 4-subsets of neighbors
        if len(neighbors) >= 4:
            for k4 in itertools.combinations(neighbors, 4):
                if _is_k4(k4, adj):
                    return (v,) + k4, True
                    
        # Check for Independent 5-set containing v
        # Non-neighbors of v
        non_neighbors = []
        for u in range(n):
            if u != v and not ((adj[v] >> u) & 1):
                non_neighbors.append(u)
                
        # Check if non_neighbors contain an Independent 4-set
        if len(non_neighbors) >= 4:
            for i4 in itertools.combinations(non_neighbors, 4):
                if _is_indep4(i4, adj):
                    return (v,) + i4, False
                    
    return None, False

def _is_k4(vertices, adj):
    a, b, c, d = vertices
    # Check all 6 edges
    if ((adj[a] >> b) & 1) and ((adj[a] >> c) & 1) and ((adj[a] >> d) & 1) and \
       ((adj[b] >> c) & 1) and ((adj[b] >> d) & 1) and \
       ((adj[c] >> d) & 1):
        return True
    return False

def _is_indep4(vertices, adj):
    a, b, c, d = vertices
    # Check all 6 non-edges
    if not ((adj[a] >> b) & 1) and not ((adj[a] >> c) & 1) and not ((adj[a] >> d) & 1) and \
       not ((adj[b] >> c) & 1) and not ((adj[b] >> d) & 1) and \
       not ((adj[c] >> d) & 1):
        return True
    return False
