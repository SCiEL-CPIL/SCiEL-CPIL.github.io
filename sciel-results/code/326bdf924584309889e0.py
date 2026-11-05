"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin before 120s kill
    
    d = 7
    n = 2 ** d
    
    # Precompute all edges as (u, v, bit) where u < v
    all_edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y, i))
    
    # For each edge, precompute the other 3 edges in the 4-cycle it belongs to
    # Edge (x, y) with bit i: the other 3 edges are (x^j, y^j) for j != i
    # But we need to be careful about orientation
    
    def get_cycle_edges(x, y, bit_i):
        """Get the other 3 edges in the 4-cycle containing edge (x,y) with bit bit_i."""
        other_edges = []
        for j in range(d):
            if j == bit_i:
                continue
            c = x ^ (1 << j)
            e = y ^ (1 << j)
            # Edge x-c
            other_edges.append((min(x, c), max(x, c)))
            # Edge y-e
            other_edges.append((min(y, e), max(y, e)))
            # Edge c-e
            other_edges.append((min(c, e), max(c, e)))
        return other_edges
    
    # Precompute cycle edges for each edge
    edge_cycle = {}
    for (x, y, bit_i) in all_edges:
        edge_cycle[(x, y)] = get_cycle_edges(x, y, bit_i)
    
    def is_valid_add(edge_set, edge):
        """Check if adding edge to edge_set creates a C4."""
        x, y, bit_i = edge
        for other_edge in edge_cycle[(x, y)]:
            if other_edge in edge_set:
                # Need to check if all 4 edges of the cycle are present
                # Actually, we need to check if the other 3 edges are all present
                # But we only know one other edge is present. Let me reconsider.
                pass
        # Actually, let me use a simpler check: for each j != bit_i,
        # check if all 4 edges of the square are present
        for j in range(d):
            if j == bit_i:
                continue
            c = x ^ (1 << j)
            e = y ^ (1 << j)
            e1 = (min(x, c), max(x, c))
            e2 = (min(y, e), max(y, e))
            e3 = (min(c, e), max(c, e))
            if e1 in edge_set and e2 in edge_set and e3 in edge_set:
                return False
        return True
    
    def greedy_randomized(all_edges, seed=None):
        """Run greedy with randomized edge order."""
        if seed is not None:
            random.seed(seed)
        
        edges = list(all_edges)
        random.shuffle(edges)
        
        chosen = set()
        out = []
        
        for edge in edges:
            x, y, bit_i = edge
            if is_valid_add(chosen, edge):
                chosen.add((x, y))
                out.append([x, y])
        
        return out, chosen
    
    def local_search(out, chosen, time_remaining):
        """Try to improve by removing one edge and adding another."""
        if not out:
            return out, chosen
        
        # Try random pairs: remove edge A, add edge B
        best_out = out
        best_chosen = chosen
        
        while time.time() - start_time < time_limit - 5.0:
            # Pick a random edge to try removing
            idx = random.randint(0, len(out) - 1)
            edge_to_remove = out[idx]
            
            # Try to add a random edge that's not currently in the set
            # Sample a few candidate edges to add
            improved = False
            for _ in range(20):
                candidate = random.choice(all_edges)
                cx, cy, cbit = candidate
                if (cx, cy) in chosen:
                    continue
                
                # Remove edge_to_remove, try to add candidate
                new_chosen = set(chosen)
                new_chosen.discard((edge_to_remove[0], edge_to_remove[1]))
                
                if is_valid_add(new_chosen, candidate):
                    new_chosen.add((cx, cy))
                    # This is a valid swap, but we need same count
                    # Actually, we want to try to add an edge without removing
                    # Let me reconsider: try to just add an edge
                    pass
            
            # Try to just add a random edge
            for _ in range(10):
                candidate = random.choice(all_edges)
                cx, cy, cbit = candidate
                if (cx, cy) not in chosen and is_valid_add(chosen, candidate):
                    chosen.add((cx, cy))
                    out.append([cx, cy])
                    best_out = out
                    best_chosen = chosen
                    improved = True
                    break
            
            if not improved:
                # Try removing a random edge to see if we can add more
                if len(out) > 300:  # Only try if we have many edges
                    idx = random.randint(0, len(out) - 1)
                    edge_to_remove = out[idx]
                    ex, ey = edge_to_remove
                    
                    # Remove it
                    chosen.discard((ex, ey))
                    out.pop(idx)
                    
                    # Try to add several edges
                    added_count = 0
                    for _ in range(50):
                        candidate = random.choice(all_edges)
                        cx, cy, cbit = candidate
                        if (cx, cy) not in chosen and is_valid_add(chosen, candidate):
                            chosen.add((cx, cy))
                            out.append([cx, cy])
                            added_count += 1
                    
                    # If we didn't add back at least 1, revert
                    if added_count == 0:
                        chosen.add((ex, ey))
                        out.append([ex, ey])
        
        return out, chosen
    
    # Run multiple randomized greedy restarts
    best_result = None
    best_count = 0
    
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 10.0:
            break
        
        out, chosen = greedy_randomized(all_edges, seed=restart)
        
        if len(out) > best_count:
            best_count = len(out)
            best_result = out
        
        # Local search improvement
        if time.time() - start_time < time_limit - 10.0:
            out, chosen = local_search(out, chosen, time_limit - time.time() - start_time)
            if len(out) > best_count:
                best_count = len(out)
                best_result = out
    
    if best_result is None:
        # Fallback to deterministic greedy
        chosen = set()
        best_result = []
        for edge in all_edges:
            x, y, bit_i = edge
            if is_valid_add(chosen, edge):
                chosen.add((x, y))
                best_result.append([x, y])
    
    return best_result
# EVOLVE-BLOCK-END
