"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    d = 7
    n = 2 ** d
    deadline = time.time() + 138  # safety margin before 150s kill
    
    # Precompute all edges with their bit index
    all_edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y, i))
    
    # Use a 2D numpy array for fast membership: edge_present[x][i] = True if edge (x, x^(1<<i)) is in the set
    # This avoids tuple hashing overhead
    
    def greedy_random(seed):
        """Randomized greedy: process edges in random order, add if no C4 is formed."""
        rng = random.Random(seed)
        order = list(all_edges)
        rng.shuffle(order)
        
        # edge_present[v][i] means edge between v and v^(1<<i) is selected
        edge_present = np.zeros((n, d), dtype=np.bool_)
        out = []
        
        for x, y, i in order:
            # Check if adding (x,y) creates a C4
            # For each j != i, check if the three other edges of the face exist
            ok = True
            for j in range(d):
                if j == i:
                    continue
                c = x ^ (1 << j)
                e = y ^ (1 << j)
                # Need edges (x,c), (y,e), (c,e) all present
                if edge_present[x, j] and edge_present[y, j] and edge_present[c, j]:
                    ok = False
                    break
            if ok:
                edge_present[x, i] = True
                edge_present[y, i] = True
                out.append([x, y])
        
        return out, edge_present
    
    def local_improve(out, edge_present):
        """Try to add back rejected edges by removing conflicting ones, or swap edges."""
        # Simple approach: try adding each currently-missing edge by removing one conflicting edge
        # First, identify which edges are missing
        present_set = set()
        for u, v in out:
            present_set.add((min(u, v), max(u, v)))
        
        missing_edges = []
        for x, y, i in all_edges:
            if (x, y) not in present_set:
                missing_edges.append((x, y, i))
        
        improved = False
        rng = random.Random(12345)
        rng.shuffle(missing_edges)
        
        for x, y, i in missing_edges:
            if time.time() > deadline:
                break
            # Find all j != i such that the C4 is "almost complete" (3 of 4 edges present)
            removable = []
            for j in range(d):
                if j == i:
                    continue
                c = x ^ (1 << j)
                e = y ^ (1 << j)
                # The face has edges: (x,y) [missing], (x,c), (y,e), (c,e)
                # We need to remove one of (x,c), (y,e), (c,e) to add (x,y)
                # But removing one might create issues elsewhere - for now just check
                # Actually, we need ALL THREE to be present for the C4 to be blocked
                # So if any one is missing, we can add (x,y) without removing anything in this face
                if not (edge_present[x, j] and edge_present[y, j] and edge_present[c, j]):
                    # This face doesn't block us
                    continue
                else:
                    # All three are present, we'd need to remove one
                    # For simplicity, skip complex removals in first pass
                    pass
            
            # Try direct addition (no removal needed if no face has all 3 other edges)
            can_add = True
            for j in range(d):
                if j == i:
                    continue
                c = x ^ (1 << j)
                e = y ^ (1 << j)
                if edge_present[x, j] and edge_present[y, j] and edge_present[c, j]:
                    can_add = False
                    break
            
            if can_add:
                edge_present[x, i] = True
                edge_present[y, i] = True
                out.append([x, y])
                improved = True
        
        return out, edge_present, improved
    
    best_out = None
    best_count = 0
    
    # Deterministic greedy baseline
    order_det = list(all_edges)
    edge_present = np.zeros((n, d), dtype=np.bool_)
    out_det = []
    for x, y, i in order_det:
        ok = True
        for j in range(d):
            if j == i:
                continue
            c = x ^ (1 << j)
            e = y ^ (1 << j)
            if edge_present[x, j] and edge_present[y, j] and edge_present[c, j]:
                ok = False
                break
        if ok:
            edge_present[x, i] = True
            edge_present[y, i] = True
            out_det.append([x, y])
    
    # Apply local improvement
    out_det, edge_present, _ = local_improve(out_det, edge_present)
    
    if len(out_det) > best_count:
        best_count = len(out_det)
        best_out = out_det[:]
    
    if time.time() > deadline:
        return best_out
    
    # Randomized greedy with multiple seeds
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        out, ep = greedy_random(attempt * 7 + 13)
        
        # Try local improvement
        out, ep, improved = local_improve(out, ep)
        
        if len(out) > best_count:
            best_count = len(out)
            best_out = out[:]
    
    return best_out
# EVOLVE-BLOCK-END
