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

def _try_add_vertex(base_edges, base_n, new_vertex, deadline):
    """Try to add a new vertex to the graph with valid connections."""
    q = base_n
    attempts = 0
    while time.time() < deadline and attempts < 5000:
        attempts += 1
        # Randomly decide edges for new vertex
        new_edges = list(base_edges)
        for i in range(q):
            if random.random() < 0.5:
                new_edges.append((i, new_vertex))
        
        if _is_valid_fast(q + 1, new_edges, 5, 5):
            return new_edges
    return None

def _local_search_edges(edges, n, deadline, max_iters=2000):
    """Local search: flip edges to maintain validity, try to find valid config."""
    current_edges = list(edges)
    current_valid = _is_valid_fast(n, current_edges, 5, 5)
    
    if current_valid:
        return current_edges
    
    # Build edge set for fast lookup
    edge_set = set()
    for u, v in current_edges:
        edge_set.add((u, v))
    
    best_edges = current_edges
    best_valid = current_valid
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        # Pick a random pair to flip
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        
        # Flip the edge
        if (u, v) in edge_set:
            edge_set.remove((u, v))
        else:
            edge_set.add((u, v))
        
        # Rebuild edges list
        new_edges = list(edge_set)
        
        if _is_valid_fast(n, new_edges, 5, 5):
            best_edges = new_edges
            best_valid = True
            return new_edges
        else:
            # Revert
            if (u, v) in edge_set:
                edge_set.remove((u, v))
            else:
                edge_set.add((u, v))
    
    return best_edges if best_valid else None

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 330s limit
    
    # 1. Generate Paley 37
    base_edges = _paley_37_edges()
    q = 37
    
    # Verify Paley 37
    if not _is_valid_fast(q, base_edges, 5, 5):
        # Should never happen, but fallback
        return q, base_edges

    best_n = q
    best_edges = base_edges
    
    # 2. Try to extend to 38 vertices
    # Try adding vertex 37
    new_edges = _try_add_vertex(base_edges, q, q, deadline)
    if new_edges is not None:
        best_n = q + 1
        best_edges = new_edges
    
    # 3. Try to extend to 39 vertices if we have 38
    if best_n == q + 1 and time.time() < deadline - 60:
        new_edges_39 = _try_add_vertex(best_edges, best_n, best_n, deadline)
        if new_edges_39 is not None:
            best_n = q + 2
            best_edges = new_edges_39
    
    # 4. If we couldn't extend, try local search on the base graph
    #    to see if we can find a different 37-vertex config that extends
    if best_n == q and time.time() < deadline - 30:
        # Try several random perturbations of the Paley graph
        for _ in range(50):
            if time.time() > deadline - 30:
                break
            perturbed = list(base_edges)
            # Flip a few random edges
            for _ in range(10):
                u = random.randint(0, q-1)
                v = random.randint(0, q-1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                if (u, v) in perturbed:
                    perturbed.remove((u, v))
                else:
                    perturbed.append((u, v))
            
            if _is_valid_fast(q, perturbed, 5, 5):
                # Try to extend this perturbed graph
                new_edges = _try_add_vertex(perturbed, q, q, deadline)
                if new_edges is not None:
                    best_n = q + 1
                    best_edges = new_edges
                    break
    
    return best_n, best_edges
