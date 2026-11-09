"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _build_bitmask(adj, n):
    """Convert adjacency sets to integer bitmasks for fast operations."""
    masks = [0] * n
    for u in range(n):
        mask = 0
        for v in adj[u]:
            mask |= (1 << v)
        masks[u] = mask
    return masks

def _is_not_3_colorable_masks(masks, n):
    """Check if graph is NOT 3-colorable using bitmask-based backtracking with MRV."""
    # Precompute vertex ordering by degree (descending) for better pruning
    degrees = [bin(masks[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda x: -degrees[x])
    
    # Map from position in order to original vertex
    pos_to_vertex = [order[i] for i in range(n)]
    
    # Use iterative deepening or simple backtracking with MRV
    def backtrack(coloring, idx):
        if idx == n:
            return True
        
        # MRV: pick uncolored vertex with fewest available colors
        best_idx = -1
        best_avail = 4
        for i in range(idx, n):
            v = pos_to_vertex[i]
            if coloring[v] != -1:
                continue
            # Count forbidden colors from colored neighbors
            forbidden = [False, False, False]
            m = masks[v]
            for u in range(n):
                if (m >> u) & 1:
                    if coloring[u] != -1:
                        forbidden[coloring[u]] = True
            avail = sum(1 for c in range(3) if not forbidden[c])
            if avail < best_avail:
                best_avail = avail
                best_idx = i
                if avail == 0:
                    break
        
        if best_idx == -1:
            return True  # All colored
        
        v = pos_to_vertex[best_idx]
        
        # Get forbidden colors
        forbidden = [False, False, False]
        m = masks[v]
        for u in range(n):
            if (m >> u) & 1:
                if coloring[u] != -1:
                    forbidden[coloring[u]] = True
        
        for c in range(3):
            if not forbidden[c]:
                coloring[v] = c
                if backtrack(coloring, idx + 1):
                    return True
                coloring[v] = -1
        
        return False
    
    coloring = [-1] * n
    return not backtrack(coloring, 0)

def construct():
    deadline = time.time() + 145  # 145s safety margin
    
    n = 16
    adj = [set() for _ in range(n)]
    
    # Start with Groetzsch graph on vertices 0..10
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    for u, v in cycle:
        adj[u].add(v)
        adj[v].add(u)
    for u, w in cycle:
        adj[u].add(5 + w)
        adj[5 + w].add(u)
        adj[w].add(5 + u)
        adj[5 + u].add(w)
    for u in range(5):
        adj[5 + u].add(10)
        adj[10].add(5 + u)
    
    # Convert to bitmasks
    masks = _build_bitmask(adj, n)
    
    def count_edges(masks):
        total = 0
        for i in range(n):
            total += bin(masks[i]).count('1')
        return total // 2
    
    def get_edges_from_masks(masks):
        edges = []
        for u in range(n):
            m = masks[u]
            v = 0
            while m:
                if m & 1:
                    if u < v:
                        edges.append([u, v])
                m >>= 1
                v += 1
        return edges
    
    best_edges = get_edges_from_masks(masks)
    best_edge_count = len(best_edges)
    best_masks = masks[:]
    
    # Precompute all missing edges that don't form triangles
    random.seed(42)
    
    for iteration in range(500000):
        if time.time() > deadline:
            break
        
        # Try adding a random missing edge
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v or (masks[u] >> v) & 1:
            continue
        
        # Check triangle-freeness using bitmasks: common neighbors of u and v
        common = masks[u] & masks[v]
        if common:
            continue
        
        # Try adding the edge
        new_masks = masks[:]
        new_masks[u] |= (1 << v)
        new_masks[v] |= (1 << u)
        
        # Check if still not 3-colorable
        if _is_not_3_colorable_masks(new_masks, n):
            # Keep the edge
            masks = new_masks
            current_count = count_edges(masks)
            if current_count > best_edge_count:
                best_edge_count = current_count
                best_masks = masks[:]
        # Else: discard (masks unchanged)
    
    return get_edges_from_masks(best_masks)
# EVOLVE-BLOCK-END
