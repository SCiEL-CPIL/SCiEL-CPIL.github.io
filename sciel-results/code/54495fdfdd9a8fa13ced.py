import time
import random
from itertools import combinations

def _paley_graph_bitmask(q):
    """Return list of bitmasks for Paley graph on q vertices."""
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _count_violations(n, adj):
    """
    Count the number of K5s and independent 5-sets.
    adj: list of ints, bit j set if (i,j) is an edge.
    Returns (k5_count, ind5_count)
    """
    k5_count = 0
    ind5_count = 0
    
    # Precompute non-adjacency masks
    # non_adj[i] has bit j set if (i,j) is NOT an edge (and i!=j)
    non_adj = [0] * n
    for i in range(n):
        # Mask for all vertices except i
        all_mask = (1 << n) - 1
        all_mask ^= (1 << i)
        non_adj[i] = all_mask ^ adj[i]
        
    # Check for K5
    # A K5 exists if there is a vertex v such that its neighborhood contains a K4.
    for v in range(n):
        nbrs_mask = adj[v]
        # Extract list of neighbors
        nbrs = []
        m = nbrs_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            nbrs.append(idx)
            m ^= lsb
            
        if len(nbrs) < 4:
            continue
            
        # Check for K4 in nbrs
        # Iterate over pairs in nbrs
        k = len(nbrs)
        for i in range(k):
            a = nbrs[i]
            for j in range(i + 1, k):
                b = nbrs[j]
                # Common neighbors of a and b within nbrs
                common = adj[a] & adj[b] & nbrs_mask
                # common must contain at least 2 vertices c, d such that c-d is an edge
                # Get list of common neighbors
                common_list = []
                cm = common
                while cm:
                    lsb = cm & -cm
                    idx = lsb.bit_length() - 1
                    common_list.append(idx)
                    cm ^= lsb
                    
                if len(common_list) < 2:
                    continue
                    
                # Check if any pair in common_list is an edge
                # If yes, a,b,c,d form a K4.
                # Since a,b are in nbrs of v, and a,b,c,d form K4, then v,a,b,c,d form K5.
                # We just need to know if ANY such K4 exists to count violations?
                # Actually, for local search, we want to minimize the count.
                # Counting all K5s is expensive.
                # Let's just return 1 if found, 0 if not? 
                # No, for gradient-like descent, a count is better.
                # But counting all K5s in a graph of size 40 is still heavy.
                # Let's approximate: if we find one K5 involving v, we add 1.
                # This is a heuristic score.
                
                # To speed up, we can just check if common_list has an edge.
                # If it does, we found a K5.
                # We can break early if we just want to know if valid.
                # But for optimization, we want the count.
                # Let's count the number of K4s in the neighborhood.
                for x in range(len(common_list)):
                    c = common_list[x]
                    for y in range(x + 1, len(common_list)):
                        d = common_list[y]
                        if (adj[c] >> d) & 1:
                            k5_count += 1
                            
    # Check for Independent 5
    # An independent 5 in G is a K5 in complement.
    # So check if non_adj[v] contains a K4 in the complement graph.
    # Wait, non_adj[v] are the vertices not connected to v.
    # We need an independent set of size 4 in the subgraph induced by non_adj[v].
    # This is equivalent to a K4 in the complement of the subgraph induced by non_adj[v].
    # Let comp_adj[i] = non_adj[i] restricted to non_adj[v]?
    # Actually, easier:
    # An independent set {v, a, b, c, d} means no edges between any of them.
    # So a, b, c, d must be in non_adj[v].
    # And a, b, c, d must be mutually non-adjacent in G.
    # So we look for a K4 in the graph H where edges are non-edges of G, restricted to non_adj[v].
    
    for v in range(n):
        non_nbrs_mask = non_adj[v]
        non_nbrs = []
        m = non_nbrs_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            non_nbrs.append(idx)
            m ^= lsb
            
        if len(non_nbrs) < 4:
            continue
            
        # Check for independent 4-set in non_nbrs
        # This is a K4 in the complement graph restricted to non_nbrs.
        k = len(non_nbrs)
        for i in range(k):
            a = non_nbrs[i]
            for j in range(i + 1, k):
                b = non_nbrs[j]
                # a and b must be non-adjacent in G
                if (adj[a] >> b) & 1:
                    continue
                
                # Common non-neighbors of a and b within non_nbrs
                # c must be non-adjacent to a and b
                # d must be non-adjacent to a, b, c
                # common_non = non_adj[a] & non_adj[b] & non_nbrs_mask
                common_non = non_adj[a] & non_adj[b] & non_nbrs_mask
                common_list = []
                cm = common_non
                while cm:
                    lsb = cm & -cm
                    idx = lsb.bit_length() - 1
                    common_list.append(idx)
                    cm ^= lsb
                    
                if len(common_list) < 2:
                    continue
                    
                for x in range(len(common_list)):
                    c = common_list[x]
                    for y in range(x + 1, len(common_list)):
                        d = common_list[y]
                        # c and d must be non-adjacent
                        if (adj[c] >> d) & 1:
                            continue
                        # a,b,c,d are mutually non-adjacent.
                        # So v,a,b,c,d is an independent set.
                        ind5_count += 1
                        
    return k5_count, ind5_count

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    best_n = 0
    best_adj = None
    
    # Try to find a valid graph for n from 42 down to 37
    # We prioritize higher n.
    for n in range(42, 36, -1):
        if time.time() > deadline - 10:
            break
            
        # If we already have a valid graph with n, we can stop if we are looking for max n
        # But we want the MAX n. So we try 42 first.
        # If we find a valid 42, we return it.
        # If not, we try 41, etc.
        
        found_valid = False
        
        # Number of restarts depends on time left
        # Let's allow a few seconds per n
        time_per_n = 60
        restarts = 0
        
        while time.time() < deadline - 5 and not found_valid:
            restarts += 1
            
            # Initialize
            if n == 37 and restarts == 1:
                adj = _paley_graph_bitmask(n)
            else:
                adj = [0] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.5:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
            
            # Local search
            # We will try to minimize violations
            k5, ind5 = _count_violations(n, adj)
            current_score = k5 + ind5
            
            if current_score == 0:
                found_valid = True
                best_n = n
                best_adj = adj
                break
            
            # Local search steps
            # Time budget per restart
            restart_deadline = time.time() + 5
            steps = 0
            while time.time() < restart_deadline and current_score > 0:
                steps += 1
                
                # Pick a random edge to flip
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                
                # Flip edge (i, j)
                if (adj[i] >> j) & 1:
                    # Remove edge
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                else:
                    # Add edge
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                
                new_k5, new_ind5 = _count_violations(n, adj)
                new_score = new_k5 + new_ind5
                
                if new_score < current_score or (new_score == current_score and random.random() < 0.1):
                    current_score = new_score
                else:
                    # Revert
                    if (adj[i] >> j) & 1:
                        # It was added, so remove it
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
                    else:
                        # It was removed, so add it back
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                
                if current_score == 0:
                    found_valid = True
                    best_n = n
                    best_adj = adj
                    break
            
            if found_valid:
                break
                
        if found_valid:
            # We found a valid graph for this n. Since we iterate n downwards,
            # this is the maximum n we found so far.
            # We can return immediately if we want to maximize n?
            # The problem asks for "as many vertices as possible".
            # So if we find a valid 42, we return 42.
            # If we fail 42, we try 41. If we find 41, we return 41.
            # So we can return as soon as we find a valid graph for the current n.
            edges = []
            for i in range(n):
                m = adj[i]
                while m:
                    lsb = m & -m
                    j = lsb.bit_length() - 1
                    if i < j:
                        edges.append((i, j))
                    m ^= lsb
            return n, edges

    # If no valid graph found for n > 37, return the Paley graph on 37
    adj = _paley_graph_bitmask(37)
    n = 37
    edges = []
    for i in range(n):
        m = adj[i]
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            if i < j:
                edges.append((i, j))
            m ^= lsb
    return n, edges
