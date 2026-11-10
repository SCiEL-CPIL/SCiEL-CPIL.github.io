import itertools
import random
import time

def _propagate_and_check(n, edge_masks, c1, unassigned, full_mask):
    """
    Apply unit propagation. Return (c1, unassigned, ok).
    ok=False means a conflict was found.
    """
    changed = True
    while changed:
        changed = False
        for mask in edge_masks:
            assigned = mask & ~unassigned
            c1_count = bin(assigned & c1).count('1')
            c0_count = bin(assigned & ~c1).count('1')
            
            if c1_count == 5 or c0_count == 5:
                return c1, unassigned, False
            
            if unassigned & mask:
                vu = bin(mask & unassigned).count('1')
                if vu == 1:
                    # Exactly one unassigned in this edge
                    if c1_count == 4:
                        # Must be color 0
                        # Just remove from unassigned (it's already not in c1)
                        unassigned ^= (mask & unassigned)
                        changed = True
                    elif c0_count == 4:
                        # Must be color 1
                        c1 |= (mask & unassigned)
                        unassigned ^= (mask & unassigned)
                        changed = True
    return c1, unassigned, True

def _is_non_2_colorable(n, edges):
    if not edges:
        return False
    
    full_mask = (1 << n) - 1
    edge_masks = []
    for e in edges:
        m = 0
        for v in e:
            m |= (1 << v)
        edge_masks.append(m)
    
    def bt(c1, unassigned):
        c1, unassigned, ok = _propagate_and_check(n, edge_masks, c1, unassigned, full_mask)
        if not ok:
            return False
        
        if unassigned == 0:
            # All assigned. Check for mono edges (should have been caught by prop, but be safe)
            for m in edge_masks:
                if (m & ~c1) == 0 or (m & c1) == 0:
                    return False
            return True
            
        # Heuristic: pick vertex with highest degree in unassigned edges?
        # For simplicity and speed with small n, just pick lowest bit.
        # However, picking a vertex in an edge with minimal unassigned count is good.
        # Let's try picking a vertex from the edge with fewest unassigned vertices.
        
        min_unassigned_count = 10
        best_bit = 0
        for m in edge_masks:
            cnt = bin(m & unassigned).count('1')
            if 1 <= cnt < min_unassigned_count:
                min_unassigned_count = cnt
                best_bit = m & unassigned
                if best_bit:
                    best_bit &= -best_bit # lowest bit of that edge
                    break # Or keep searching? Just take the first good one.
        
        if not best_bit:
            best_bit = unassigned & -unassigned
        
        # Try 0 (not in c1)
        if bt(c1, unassigned ^ best_bit):
            return True
        
        # Try 1 (in c1)
        if bt(c1 | best_bit, unassigned ^ best_bit):
            return True
        
        return False

    # Symmetry breaking: fix vertex 0 to color 0
    # c1 = 0, unassigned = all except 0
    init_unassigned = full_mask ^ 1
    init_c1 = 0
    
    # Initial propagation
    init_c1, init_unassigned, ok = _propagate_and_check(n, edge_masks, init_c1, init_unassigned, full_mask)
    if not ok:
        return True
    
    if init_unassigned == 0:
        # Check validity
        for m in edge_masks:
            if (m & ~init_c1) == 0 or (m & init_c1) == 0:
                return False
        return False # Valid coloring found
        
    return not bt(init_c1, init_unassigned)

def construct():
    start_time = time.time()
    deadline = start_time + 115 # Safety margin for 120s limit
    
    # Hardcoded known good construction: 51 edges on 14 vertices (or similar)
    # Actually, let's try to find a small one quickly.
    # Known best is 51 edges.
    # We can try to construct one by adding edges to a base or removing from a large one.
    # Or just hardcode a known minimal set if we have one.
    # Since I don't have a specific 51-edge set memorized, I will use a fast heuristic search.
    
    # Strategy:
    # 1. Start with K9^(5) (126 edges).
    # 2. Try to reduce to fewer vertices if possible? No, n must be >= 5.
    #    Actually, increasing n allows fewer edges.
    #    Let's try n=10, 11, 12...
    #    For n=10, C(10,5)=252.
    #    For n=12, C(12,5)=792.
    #    
    #    Random hypergraphs become non-2-colorable with high probability when m/n is around 2 (for r=5).
    #    So m ~ 2n.
    #    For n=10, m~20? But we need to be sure.
    #    For n=26, m~52. This matches the 51 edge bound.
    
    # Let's try to find a non-2-colorable hypergraph with minimal edges on n=10..20.
    # We will iterate n from 8 to 20.
    # For each n, we try to find a minimal non-2-colorable set.
    # 
    # Algorithm for fixed n:
    # 1. Generate random edges.
    # 2. Check if non-2-colorable.
    # 3. If yes, try to remove edges greedily.
    # 4. Keep the best (smallest) one found.
    
    best_edges = None
    best_n = 0
    
    # To ensure we have a valid output quickly, let's start with a known large one?
    # No, let's just search.
    
    rng = random.Random(42)
    
    for n in range(10, 21):
        if time.time() > deadline - 5:
            break
            
        # We want to find a small m.
        # Estimate m_needed ~ 2*n + 10?
        # Let's try m from 20 to 60.
        
        # Actually, let's just generate random sets of size m and check.
        # Or better: start with a large random set, check, then reduce.
        
        # Let's try to find one for this n.
        found = None
        
        # Heuristic: Random set of size m
        # For n=10, m=25 might work?
        # For n=20, m=50?
        
        # Let's do a few random trials for each n.
        # Trial 1: Random m=30..50
        for _ in range(50): # 50 trials per n
            if time.time() > deadline - 5:
                break
            
            m = rng.randint(30, 60)
            all_edges = list(itertools.combinations(range(n), 5))
            rng.shuffle(all_edges)
            edges = all_edges[:m]
            
            if _is_non_2_colorable(n, edges):
                # Found a non-2-colorable set of size m.
                # Try to reduce it.
                current_edges = [list(e) for e in edges]
                # Greedy reduction
                changed = True
                while changed:
                    changed = False
                    rng.shuffle(current_edges)
                    for i in range(len(current_edges) - 1, -1, -1):
                        if time.time() > deadline - 2:
                            break
                        new_edges = current_edges[:i] + current_edges[i+1:]
                        if _is_non_2_colorable(n, new_edges):
                            current_edges = new_edges
                            changed = True
                    if time.time() > deadline - 2:
                        break
                
                if not found or len(current_edges) < len(found):
                    found = current_edges
        
        if found:
            if not best_edges or len(found) < len(best_edges):
                best_edges = found
                best_n = n
    
    if not best_edges:
        # Fallback: K9^(5)
        n = 9
        best_edges = [list(e) for e in itertools.combinations(range(n), 5)]
        best_n = n
        # Verify fallback (should be non-2-colorable)
        # If not, just return it, hoping for the best (but K9^5 is definitely non-2-colorable)
    
    return {'n': best_n, 'edges': best_edges}

if __name__ == '__main__':
    # Quick self test?
    # res = construct()
    # print(len(res['edges']))
    pass
