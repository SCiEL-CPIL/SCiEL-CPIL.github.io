import time
import random
import itertools
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 250  # Safety margin
    
    # 1. Initialize with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    edges = set()
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                edges.add((i, j))
    
    best_n = n
    best_edges = edges
    best_adj = [[False] * n for _ in range(n)]
    for u, v in best_edges:
        best_adj[u][v] = True
        best_adj[v][u] = True

    # Fast validation using adjacency matrix
    def is_valid_fast(adj, n):
        """Check for K5 and independent 5-sets using bitmasks."""
        # Convert to bitmask representation
        masks = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if adj[i][j]:
                    mask |= (1 << j)
            masks[i] = mask
        
        # Check for K5
        for a in range(n):
            for b in range(a+1, n):
                if not adj[a][b]:
                    continue
                for c in range(b+1, n):
                    if not adj[a][c] or not adj[b][c]:
                        continue
                    for d in range(c+1, n):
                        if not adj[a][d] or not adj[b][d] or not adj[c][d]:
                            continue
                        for e in range(d+1, n):
                            if (adj[a][e] and adj[b][e] and adj[c][e] and adj[d][e]):
                                return False
        
        # Check for independent 5-set
        for a in range(n):
            for b in range(a+1, n):
                if adj[a][b]:
                    continue
                for c in range(b+1, n):
                    if adj[a][c] or adj[b][c]:
                        continue
                    for d in range(c+1, n):
                        if adj[a][d] or adj[b][d] or adj[c][d]:
                            continue
                        for e in range(d+1, n):
                            if (not adj[a][e] and not adj[b][e] and not adj[c][e] and not adj[d][e]):
                                return False
        return True

    # Verify seed
    if not is_valid_fast(best_adj, best_n):
        return best_n, list(best_edges)

    # Helper to check if adding vertex with given neighbors keeps graph valid
    def can_add_vertex(adj, n, new_vertex, neighbors):
        """
        Check if we can add a new vertex connected to exactly 'neighbors'.
        Only need to check new K5s and new independent 5-sets involving the new vertex.
        """
        # A new K5 must include the new vertex and 4 old vertices that form a K4 in old graph
        # and are all connected to new vertex
        # A new independent 5-set must include new vertex and 4 old vertices that form 
        # an independent 4-set and are all NOT connected to new vertex
        
        # Find all K4s in old graph where all 4 vertices are in neighbors
        # For efficiency, iterate over combinations of 4 from neighbors
        if len(neighbors) >= 4:
            neigh_list = list(neighbors)
            for combo in itertools.combinations(neigh_list, 4):
                a, b, c, d = combo
                # Check if these form K4
                if (adj[a][b] and adj[a][c] and adj[a][d] and 
                    adj[b][c] and adj[b][d] and adj[c][d]):
                    return False  # Would create K5
        
        # Find all independent 4-sets in old graph where all 4 vertices are NOT in neighbors
        non_neighbors = set(range(n)) - neighbors
        if len(non_neighbors) >= 4:
            non_neigh_list = list(non_neighbors)
            for combo in itertools.combinations(non_neigh_list, 4):
                a, b, c, d = combo
                # Check if these form independent 4-set
                if (not adj[a][b] and not adj[a][c] and not adj[a][d] and 
                    not adj[b][c] and not adj[b][d] and not adj[c][d]):
                    return False  # Would create independent 5-set
        
        return True

    # Smart neighbor generation strategies
    def generate_neighbor_candidates(adj, n, target_degree=None):
        """Generate multiple candidate neighbor sets for new vertex."""
        candidates = []
        
        if target_degree is None:
            target_degree = n // 2
        
        # Strategy 1: Random
        for _ in range(3):
            neighbors = set(random.sample(range(n), min(target_degree, n)))
            candidates.append(neighbors)
        
        # Strategy 2: Based on degree balancing - connect to low-degree vertices
        degrees = [sum(adj[i]) for i in range(n)]
        sorted_by_degree = sorted(range(n), key=lambda x: degrees[x])
        for frac in [0.3, 0.5, 0.7]:
            k = int(n * frac)
            neighbors = set(sorted_by_degree[:k])
            candidates.append(neighbors)
        
        # Strategy 3: Based on anti-degree (connect to high-degree in complement)
        for frac in [0.3, 0.5, 0.7]:
            k = int(n * frac)
            neighbors = set(sorted_by_degree[-k:])
            candidates.append(neighbors)
        
        # Strategy 4: Even/odd patterns
        candidates.append(set(range(0, n, 2)))
        candidates.append(set(range(1, n, 2)))
        
        # Strategy 5: Try target degree with some randomness
        for _ in range(5):
            k = max(1, min(n-1, target_degree + random.randint(-3, 3)))
            neighbors = set(random.sample(range(n), k))
            candidates.append(neighbors)
        
        return candidates

    # 2. Try to extend to n+1, n+2, etc.
    current_n = best_n
    current_adj = best_adj
    current_edges = best_edges
    
    for target_n in range(current_n + 1, 43):  # Try up to 42
        if time.time() > deadline:
            break
            
        found = False
        attempts = 0
        max_attempts = 200  # More attempts with smarter candidates
        
        # Generate a pool of candidate neighbor sets
        candidate_pool = generate_neighbor_candidates(current_adj, current_n)
        
        while attempts < max_attempts and not found:
            attempts += 1
            if time.time() > deadline:
                break
                
            # Pick a candidate (cycle through pool, then random)
            if attempts <= len(candidate_pool):
                neighbors = candidate_pool[attempts - 1]
            else:
                # Generate new random candidate
                target_deg = current_n // 2 + random.randint(-5, 5)
                target_deg = max(1, min(current_n - 1, target_deg))
                neighbors = set(random.sample(range(current_n), target_deg))
            
            # Quick check before full validation
            if can_add_vertex(current_adj, current_n, target_n - 1, neighbors):
                # Build new adjacency
                new_adj = [row[:] for row in current_adj]
                new_adj.append([False] * (target_n))
                for i in range(current_n):
                    new_adj[i].append(False)
                
                new_edges = set(current_edges)
                new_vertex = target_n - 1
                for v in neighbors:
                    new_adj[new_vertex][v] = True
                    new_adj[v][new_vertex] = True
                    new_edges.add((min(v, new_vertex), max(v, new_vertex)))
                
                # Full validation as safety
                if is_valid_fast(new_adj, target_n):
                    found = True
                    current_n = target_n
                    current_adj = new_adj
                    current_edges = new_edges
                    best_n = current_n
                    best_edges = current_edges
                    break
                else:
                    # Can happen due to subtle bugs; fall back to no change
                    pass

        if not found:
            # Try local search: flip edges in current graph to enable extension
            if time.time() > deadline - 30:
                break
                
            improved = False
            for _ in range(50):  # Try some edge flips
                if time.time() > deadline:
                    break
                    
                # Pick random edge to flip
                if not current_edges:
                    break
                u, v = random.choice(list(current_edges))
                
                # Try removing edge
                new_adj = [row[:] for row in current_adj]
                new_adj[u][v] = False
                new_adj[v][u] = False
                new_edges = set(current_edges)
                new_edges.discard((u, v))
                
                if is_valid_fast(new_adj, current_n):
                    # Check if this helps with extension
                    test_found = False
                    for _ in range(20):
                        if time.time() > deadline:
                            break
                        target_deg = current_n // 2 + random.randint(-3, 3)
                        target_deg = max(1, min(current_n - 1, target_deg))
                        neighbors = set(random.sample(range(current_n), target_deg))
                        if can_add_vertex(new_adj, current_n, current_n, neighbors):
                            # Build extended graph
                            ext_adj = [row[:] for row in new_adj]
                            ext_adj.append([False] * (current_n + 1))
                            for i in range(current_n):
                                ext_adj[i].append(False)
                            ext_edges = set(new_edges)
                            for vv in neighbors:
                                ext_adj[current_n][vv] = True
                                ext_adj[vv][current_n] = True
                                ext_edges.add((min(vv, current_n), max(vv, current_n)))
                            if is_valid_fast(ext_adj, current_n + 1):
                                test_found = True
                                current_n = current_n + 1
                                current_adj = ext_adj
                                current_edges = ext_edges
                                best_n = current_n
                                best_edges = current_edges
                                improved = True
                                break
                    
                    if test_found:
                        break
                    else:
                        # Accept the flip if it's still valid (might help later)
                        current_adj = new_adj
                        current_edges = new_edges
                        improved = True
                        break
                # If removing edge makes invalid, try adding a non-edge
                # Find a non-edge
                non_edges = [(i, j) for i in range(current_n) for j in range(i+1, current_n) 
                            if (i, j) not in current_edges]
                if non_edges:
                    u2, v2 = random.choice(non_edges)
                    new_adj2 = [row[:] for row in current_adj]
                    new_adj2[u2][v2] = True
                    new_adj2[v2][u2] = True
                    new_edges2 = set(current_edges)
                    new_edges2.add((u2, v2))
                    
                    if is_valid_fast(new_adj2, current_n):
                        current_adj = new_adj2
                        current_edges = new_edges2
                        improved = True
                        break
            
            if not improved:
                break

    # Convert set of edges to list of tuples
    final_edges = [(i, j) for i, j in best_edges if i < j]
    return best_n, final_edges
# EVOLVE-BLOCK-END
