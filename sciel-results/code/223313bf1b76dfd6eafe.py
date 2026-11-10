import time
import random
from itertools import combinations

def _is_2_colorable(n, edges):
    """
    Returns True if the hypergraph is 2-colorable, False otherwise.
    Uses backtracking with unit propagation and bitmask optimization.
    """
    if not edges:
        return True
        
    m = len(edges)
    
    # Precompute bitmasks for edges
    edge_masks = [0] * m
    for i, e in enumerate(edges):
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks[i] = mask
        
    import sys
    sys.setrecursionlimit(10000)
    
    # We need to track the number of free vertices in each edge for MRV?
    # Or just calculate it on the fly.
    
    def backtrack(pos, neg, free):
        # Propagation loop
        changed = True
        while changed:
            changed = False
            for e_mask in edge_masks:
                e_pos = e_mask & pos
                e_neg = e_mask & neg
                e_free = e_mask & free
                
                if e_pos and e_neg:
                    continue
                
                if e_pos:
                    if e_free == 0:
                        return False
                    # Force free to neg
                    new_neg = neg | e_free
                    new_free = free & ~e_free
                    neg = new_neg
                    free = new_free
                    changed = True
                elif e_neg:
                    if e_free == 0:
                        return False
                    # Force free to pos
                    new_pos = pos | e_free
                    new_free = free & ~e_free
                    pos = new_pos
                    free = new_free
                    changed = True
        
        if free == 0:
            return True
        
        # MRV Heuristic: Find edge with min free vertices > 0
        min_free_count = n + 1
        target_vertex = -1
        
        for e_mask in edge_masks:
            e_free = e_mask & free
            if e_free:
                count = bin(e_free).count('1')
                if count < min_free_count:
                    min_free_count = count
                    # Pick first vertex in this edge
                    target_vertex = (e_free & -e_free).bit_length() - 1
                    if count == 1:
                        break
        
        if target_vertex == -1:
            # Should not happen if free != 0
            return False
        
        v = target_vertex
        bit = 1 << v
        
        # Try 1
        if backtrack(pos | bit, neg, free & ~bit):
            return True
        
        # Try 0
        if backtrack(pos, neg | bit, free & ~bit):
            return True
        
        return False
    
    all_mask = (1 << n) - 1
    return backtrack(0, 0, all_mask)

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_edges = None
    best_count = float('inf')
    best_n = 0
    
    rng = random.Random(42)
    
    for trial in range(1000):
        if time.time() > deadline:
            break
            
        n = rng.randint(14, 20)
        m_start = rng.randint(48, 60)
        
        all_possible = list(combinations(range(n), 5))
        rng.shuffle(all_possible)
        
        edges = [all_possible[i] for i in range(m_start)]
        edges = [list(e) for e in edges]
        
        # Ensure distinct
        unique_edges = []
        seen = set()
        for e in edges:
            t = tuple(e)
            if t not in seen:
                seen.add(t)
                unique_edges.append(e)
        edges = unique_edges
        
        if not _is_2_colorable(n, edges):
            current_edges = edges[:]
            current_n = n
            
            # Greedy removal
            removed_any = True
            while removed_any and time.time() < deadline:
                removed_any = False
                for i in range(len(current_edges) - 1, -1, -1):
                    if time.time() > deadline:
                        break
                    test_edges = current_edges[:i] + current_edges[i+1:]
                    if not _is_2_colorable(current_n, test_edges):
                        current_edges = test_edges
                        removed_any = True
                        break
            
            if len(current_edges) < best_count:
                best_count = len(current_edges)
                best_edges = current_edges[:]
                best_n = current_n
        else:
            # Repair
            if len(edges) < 100:
                added = 0
                while time.time() < deadline and added < 20:
                    e = rng.choice(all_possible)
                    e_list = list(e)
                    if e not in seen:
                        seen.add(e)
                        edges.append(e_list)
                        added += 1
                        if not _is_2_colorable(n, edges):
                            current_edges = edges[:]
                            current_n = n
                            
                            removed_any = True
                            while removed_any and time.time() < deadline:
                                removed_any = False
                                for i in range(len(current_edges) - 1, -1, -1):
                                    if time.time() > deadline:
                                        break
                                    test_edges = current_edges[:i] + current_edges[i+1:]
                                    if not _is_2_colorable(current_n, test_edges):
                                        current_edges = test_edges
                                        removed_any = True
                                        break
                            
                            if len(current_edges) < best_count:
                                best_count = len(current_edges)
                                best_edges = current_edges[:]
                                best_n = current_n
                            break
    
    if best_edges is None:
        n = 9
        best_edges = [list(e) for e in combinations(range(n), 5)]
        best_n = n
    
    return {'n': best_n, 'edges': best_edges}
