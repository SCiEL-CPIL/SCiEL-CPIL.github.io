import itertools
import random
import time

def _is_not_2_colorable_bitset(n, edges):
    """
    Fast bitset-based DPLL with unit propagation for 5-uniform hypergraph 2-colorability.
    Returns True if NOT 2-colorable (i.e., no proper 2-coloring exists).
    """
    if not edges:
        return False
    
    m = len(edges)
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    # For very small n, brute force
    if n <= 10:
        for c in range(1 << n):
            valid = True
            for em in edge_masks:
                if (em & c) == 0 or (em & c) == em:
                    valid = False
                    break
            if valid:
                return False
        return True
    
    # Precompute vertex-to-edges
    vertex_edges = [[] for _ in range(n)]
    for i, em in enumerate(edge_masks):
        for v in range(n):
            if em & (1 << v):
                vertex_edges[v].append(i)
    
    # State arrays
    color = [-1] * n
    c0 = [0] * m
    c1 = [0] * m
    
    # Use iterative deepening / simple recursion with memoization-like pruning
    # We'll use a stack-based approach to avoid recursion depth issues
    
    def propagate():
        changed = True
        while changed:
            changed = False
            for i in range(m):
                if c0[i] == 5 or c1[i] == 5:
                    return False
                if c0[i] == 4 and c1[i] == 0:
                    for v in range(n):
                        if color[v] == -1 and (edge_masks[i] & (1 << v)):
                            color[v] = 1
                            c1[i] += 1
                            for oe in vertex_edges[v]:
                                c1[oe] += 1
                            changed = True
                            if c1[i] == 5:
                                return False
                            break
                elif c1[i] == 4 and c0[i] == 0:
                    for v in range(n):
                        if color[v] == -1 and (edge_masks[i] & (1 << v)):
                            color[v] = 0
                            c0[i] += 1
                            for oe in vertex_edges[v]:
                                c0[oe] += 1
                            changed = True
                            if c0[i] == 5:
                                return False
                            break
        return True
    
    def backtrack():
        # Conflict check
        for i in range(m):
            if c0[i] == 5 or c1[i] == 5:
                return False
        
        # Find most constrained edge
        best_i = -1
        best_unassigned = 6
        best_imbalance = 0
        for i in range(m):
            un = 5 - c0[i] - c1[i]
            imb = max(c0[i], c1[i])
            if un < best_unassigned or (un == best_unassigned and imb > best_imbalance):
                best_unassigned = un
                best_imbalance = imb
                best_i = i
        
        if best_i == -1 or best_unassigned == 0:
            return True  # All edges satisfied or no unassigned
        
        # Find unassigned vertex in best edge
        candidates = []
        for v in range(n):
            if color[v] == -1 and (edge_masks[best_i] & (1 << v)):
                candidates.append(v)
        
        if not candidates:
            return True
        
        # Pick vertex with highest degree for better pruning
        v = max(candidates, key=lambda x: len(vertex_edges[x]))
        
        # Try color 0
        sc0 = c0[:]
        sc1 = c1[:]
        color[v] = 0
        ok = True
        for ei in vertex_edges[v]:
            c0[ei] += 1
            if c0[ei] == 5:
                ok = False
                break
        if ok and propagate() and backtrack():
            return True
        color[v] = -1
        c0[:] = sc0
        c1[:] = sc1
        
        # Try color 1
        sc0 = c0[:]
        sc1 = c1[:]
        color[v] = 1
        ok = True
        for ei in vertex_edges[v]:
            c1[ei] += 1
            if c1[ei] == 5:
                ok = False
                break
        if ok and propagate() and backtrack():
            return True
        color[v] = -1
        c0[:] = sc0
        c1[:] = sc1
        
        return False
    
    # If backtrack finds a valid coloring, it's 2-colorable
    return not backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    best_edges = None
    best_n = 9
    best_m = 10**9
    
    # Strategy: Random constraint accumulation
    # Start sparse, add edges, check for non-2-colorability
    # Also try local search (removal + swap) from found solutions
    
    # Precompute all possible edges for various n
    all_edges_by_n = {}
    for n in range(9, 17):
        all_edges_by_n[n] = [list(e) for e in itertools.combinations(range(n), 5)]
    
    # Phase 1: Random accumulation for multiple n values
    for n in range(9, 15):
        if time.time() > deadline - 30:
            break
        
        all_edges = all_edges_by_n[n]
        num_all = len(all_edges)
        
        # Multiple restarts of random accumulation
        num_restarts = 20
        for restart in range(num_restarts):
            if time.time() > deadline - 25:
                break
            
            current_edges = []
            current_set = set()
            
            # Randomly add edges
            shuffled_indices = list(range(num_all))
            random.shuffle(shuffled_indices)
            
            for idx in shuffled_indices:
                if time.time() > deadline - 25:
                    break
                
                e = all_edges[idx]
                e_key = tuple(e)
                if e_key in current_set:
                    continue
                
                # Only check colorability when we have enough edges
                # Below ~20 edges, almost certainly 2-colorable
                if len(current_edges) < 20:
                    current_edges.append(e)
                    current_set.add(e_key)
                    continue
                
                # Check if adding this edge makes it non-2-colorable
                test_edges = current_edges + [e]
                if _is_not_2_colorable_bitset(n, test_edges):
                    current_edges.append(e)
                    current_set.add(e_key)
                    
                    # Found a non-2-colorable hypergraph!
                    if len(current_edges) < best_m:
                        best_m = len(current_edges)
                        best_edges = current_edges[:]
                        best_n = n
                    
                    # Now try to minimize via removal
                    # Greedy removal
                    for _ in range(100):  # Max removal attempts
                        if time.time() > deadline - 20:
                            break
                        
                        removed = False
                        rem_indices = list(range(len(current_edges)))
                        random.shuffle(rem_indices)
                        
                        for ri in rem_indices:
                            if time.time() > deadline - 20:
                                break
                            if ri >= len(current_edges):
                                continue
                            
                            new_edges = current_edges[:ri] + current_edges[ri+1:]
                            if _is_not_2_colorable_bitset(n, new_edges):
                                current_edges = new_edges
                                removed = True
                                if len(current_edges) < best_m:
                                    best_m = len(current_edges)
                                    best_edges = current_edges[:]
                                    best_n = n
                                break
                        
                        if not removed:
                            # Try swaps
                            if len(current_edges) > 0:
                                swap_done = False
                                for _swap in range(50):
                                    if time.time() > deadline - 20:
                                        break
                                    
                                    e1_idx = random.randint(0, len(current_edges) - 1)
                                    e1 = current_edges[e1_idx]
                                    
                                    # Find random edge not in current
                                    for _ in range(100):
                                        e2_idx = random.randint(0, num_all - 1)
                                        e2 = all_edges[e2_idx]
                                        if tuple(e2) not in current_set:
                                            new_edges = current_edges[:]
                                            new_edges[e1_idx] = e2
                                            if _is_not_2_colorable_bitset(n, new_edges):
                                                current_edges = new_edges
                                                current_set.discard(tuple(e1))
                                                current_set.add(tuple(e2))
                                                swap_done = True
                                                break
                                    
                                    if swap_done:
                                        break
                                if not swap_done:
                                    break
                            else:
                                break
                    break  # Break out of edge-adding loop for this restart
            
            # Continue with more restarts
    
    # Phase 2: If we found something, try to improve further with more local search
    if best_edges is not None and time.time() < deadline - 15:
        n = best_n
        all_edges = all_edges_by_n[n]
        num_all = len(all_edges)
        current_edges = best_edges[:]
        current_set = set(tuple(e) for e in current_edges)
        
        for _iter in range(50):
            if time.time() > deadline - 10:
                break
            
            # Greedy removal
            removed = False
            rem_indices = list(range(len(current_edges)))
            random.shuffle(rem_indices)
            
            for ri in rem_indices:
                if time.time() > deadline - 10:
                    break
                if ri >= len(current_edges):
                    continue
                
                new_edges = current_edges[:ri] + current_edges[ri+1:]
                if _is_not_2_colorable_bitset(n, new_edges):
                    current_edges = new_edges
                    removed = True
                    if len(current_edges) < best_m:
                        best_m = len(current_edges)
                        best_edges = current_edges[:]
                        best_n = n
                    break
            
            if not removed:
                # Try swaps
                swap_done = False
                for _swap in range(100):
                    if time.time() > deadline - 10:
                        break
                    
                    if len(current_edges) == 0:
                        break
                    
                    e1_idx = random.randint(0, len(current_edges) - 1)
                    e1 = current_edges[e1_idx]
                    
                    for _ in range(200):
                        e2_idx = random.randint(0, num_all - 1)
                        e2 = all_edges[e2_idx]
                        if tuple(e2) not in current_set:
                            new_edges = current_edges[:]
                            new_edges[e1_idx] = e2
                            if _is_not_2_colorable_bitset(n, new_edges):
                                current_edges = new_edges
                                current_set.discard(tuple(e1))
                                current_set.add(tuple(e2))
                                swap_done = True
                                break
                    
                    if swap_done:
                        break
                if not swap_done:
                    break
    
    # Fallback: if nothing found, use K9^5
    if best_edges is None:
        n = 9
        best_edges = all_edges_by_n[9]
        best_n = n
    
    return {'n': best_n, 'edges': best_edges}

# Optional: Allow running as a script for testing
if __name__ == "__main__":
    import sys
    start = time.time()
    result = construct()
    end = time.time()
    print(f"Time taken: {end - start:.2f}s")
    print(f"n: {result['n']}")
    print(f"Edges: {len(result['edges'])}")
    # Verify
    if _is_not_2_colorable_bitset(result['n'], result['edges']):
        print("Valid: Non-2-colorable")
    else:
        print("INVALID: 2-colorable")
