import time
import random
from itertools import combinations

def _build_bitmasks(n, edges):
    """Build adjacency bitmasks for fast clique/independent set checks."""
    masks = [0] * n
    for u, v in edges:
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
    return masks

def _has_k5_fast(n, masks, k=5):
    """Check for K5 using bitmask intersections."""
    for i in range(n):
        for j in range(i+1, n):
            if not (masks[i] >> j) & 1:
                continue
            # Common neighbors of i and j
            common_ij = masks[i] & masks[j]
            for l in range(j+1, n):
                if not (common_ij >> l) & 1:
                    continue
                # Common neighbors of i, j, l
                common_ijl = common_ij & masks[l]
                for m in range(l+1, n):
                    if not (common_ijl >> m) & 1:
                        continue
                    # Common neighbors of i, j, l, m
                    common_ijlm = common_ijl & masks[m]
                    for p in range(m+1, n):
                        if (common_ijlm >> p) & 1:
                            return True
    return False

def _has_ind5_fast(n, masks, k=5):
    """Check for independent set of size 5 using bitmask operations."""
    # Independent set: no two vertices are adjacent
    # For each vertex, its non-neighbors (excluding itself)
    for i in range(n):
        # Non-neighbors of i (excluding i itself)
        non_adj_i = ((1 << n) - 1) ^ masks[i] ^ (1 << i)
        for j in range(i+1, n):
            if not (non_adj_i >> j) & 1:
                continue
            # Common non-neighbors of i and j
            non_adj_j = ((1 << n) - 1) ^ masks[j] ^ (1 << j)
            common_non = non_adj_i & non_adj_j
            for l in range(j+1, n):
                if not (common_non >> l) & 1:
                    continue
                non_adj_l = ((1 << n) - 1) ^ masks[l] ^ (1 << l)
                common_non2 = common_non & non_adj_l
                for m in range(l+1, n):
                    if not (common_non2 >> m) & 1:
                        continue
                    non_adj_m = ((1 << n) - 1) ^ masks[m] ^ (1 << m)
                    common_non3 = common_non2 & non_adj_m
                    for p in range(m+1, n):
                        if (common_non3 >> p) & 1:
                            return True
    return False

def _is_valid_fast(n, edges, clique_size=5, ind_size=5):
    """Fast validity check using bitmasks."""
    masks = _build_bitmasks(n, edges)
    if _has_k5_fast(n, masks, clique_size):
        return False
    if _has_ind5_fast(n, masks, ind_size):
        return False
    return True

def _paley_37_edges():
    """Generate the Paley graph on 37 vertices."""
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                edges.append((i, j))
    return edges

def _try_add_vertex_optimized(base_edges, base_n, new_vertex, deadline):
    """Try to add a new vertex with optimized constraint checking."""
    q = base_n
    masks = _build_bitmasks(q, base_edges)
    attempts = 0
    
    while time.time() < deadline and attempts < 10000:
        attempts += 1
        
        # Randomly decide edges for new vertex
        new_mask = 0
        for i in range(q):
            if random.random() < 0.5:
                new_mask |= (1 << i)
        
        # Check if adding this vertex creates a K5
        # A K5 would involve the new vertex + 4 others that form a K4 in common neighbors
        # Or the new vertex + 3 others + 1 existing edge
        # More precisely: check all 4-subsets of neighbors of new vertex for K4
        # But that's expensive. Instead, check incrementally.
        
        # For each set of 4 vertices that are all neighbors of new_vertex,
        # check if they form a K4 (clique of size 4)
        neighbors = [i for i in range(q) if (new_mask >> i) & 1]
        
        valid = True
        # Check for K5 involving new vertex
        for quad in combinations(neighbors, 4):
            i, j, l, m = quad
            # Check if i,j,l,m form a K4
            if ((masks[i] >> j) & 1 and 
                (masks[i] >> l) & 1 and 
                (masks[i] >> m) & 1 and
                (masks[j] >> l) & 1 and 
                (masks[j] >> m) & 1 and
                (masks[l] >> m) & 1):
                valid = False
                break
        
        if not valid:
            continue
        
        # Check for independent set of size 5 involving new vertex
        # New vertex is not connected to non-neighbors
        non_neighbors = [i for i in range(q) if not (new_mask >> i) & 1]
        
        # Check if any 4 non-neighbors form an independent set of size 4
        for quad in combinations(non_neighbors, 4):
            i, j, l, m = quad
            # Check if i,j,l,m are mutually non-adjacent
            if (not ((masks[i] >> j) & 1) and
                not ((masks[i] >> l) & 1) and
                not ((masks[i] >> m) & 1) and
                not ((masks[j] >> l) & 1) and
                not ((masks[j] >> m) & 1) and
                not ((masks[l] >> m) & 1)):
                valid = False
                break
        
        if valid:
            # Also need to verify no K5 or independent 5-set exists entirely among old vertices
            # But we already know the base graph is valid, so we only need to check
            # configurations involving the new vertex.
            # Build new edge list
            new_edges = list(base_edges)
            for i in neighbors:
                new_edges.append((i, new_vertex))
            
            # Final verification
            if _is_valid_fast(q + 1, new_edges, 5, 5):
                return new_edges
    
    return None

def _local_search_improve(edges, n, deadline, max_iters=5000):
    """Local search: try to improve the graph by flipping edges."""
    current_edges = set(edges)
    current_valid = _is_valid_fast(n, list(current_edges), 5, 5)
    
    if not current_valid:
        return None
    
    best_edges = list(current_edges)
    best_n = n
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try to add a new vertex
        if n < 42:
            new_edges = _try_add_vertex_optimized(list(current_edges), n, n, deadline)
            if new_edges is not None:
                current_edges = set(new_edges)
                n += 1
                best_edges = list(current_edges)
                best_n = n
                continue
        
        # Try edge flips to potentially enable further expansion
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        
        # Flip the edge
        if (u, v) in current_edges:
            current_edges.remove((u, v))
        else:
            current_edges.add((u, v))
        
        if _is_valid_fast(n, list(current_edges), 5, 5):
            # Keep the flip
            best_edges = list(current_edges)
            best_n = n
        else:
            # Revert
            if (u, v) in current_edges:
                current_edges.remove((u, v))
            else:
                current_edges.add((u, v))
    
    return best_edges, best_n

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 330s limit
    
    # 1. Generate Paley 37
    base_edges = _paley_37_edges()
    q = 37
    
    # Verify Paley 37
    if not _is_valid_fast(q, base_edges, 5, 5):
        return q, base_edges

    best_n = q
    best_edges = base_edges
    
    # 2. Try to extend using optimized vertex addition
    new_edges = _try_add_vertex_optimized(base_edges, q, q, deadline)
    if new_edges is not None:
        best_n = q + 1
        best_edges = new_edges
    
    # 3. Try to extend further
    if best_n == q + 1 and time.time() < deadline - 60:
        new_edges_39 = _try_add_vertex_optimized(best_edges, best_n, best_n, deadline)
        if new_edges_39 is not None:
            best_n = q + 2
            best_edges = new_edges_39
    
    # 4. Try local search to improve
    if time.time() < deadline - 30:
        improved_edges, improved_n = _local_search_improve(best_edges, best_n, deadline)
        if improved_edges is not None and improved_n > best_n:
            best_n = improved_n
            best_edges = improved_edges
    
    return best_n, best_edges
