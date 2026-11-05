"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _count_regular_6(adj_masks, subsets_6):
    """Count number of 6-subsets that induce a regular graph."""
    bad_count = 0
    for idx, mask in subsets_6:
        # Calculate degrees for the 6 vertices within the subset
        degs = []
        for i in idx:
            # Count neighbors in the subset
            d = bin(adj_masks[i] & mask).count('1')
            degs.append(d)
        # Check if all degrees are equal
        if len(set(degs)) == 1:
            bad_count += 1
    return bad_count

def _count_regular_7(adj_masks, subsets_7):
    """Count number of 7-subsets that induce a regular graph."""
    bad_count = 0
    for idx, mask in subsets_7:
        degs = []
        for i in idx:
            d = bin(adj_masks[i] & mask).count('1')
            degs.append(d)
        if len(set(degs)) == 1:
            bad_count += 1
    return bad_count

def _construct_graph(n):
    """Construct a graph on n vertices with no regular induced subgraph on 6+ vertices."""
    start_time = time.time()
    deadline = start_time + 115
    
    # Precompute subsets
    subsets_6 = []
    for combo in combinations(range(n), 6):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        subsets_6.append((list(combo), mask))
        
    subsets_7 = []
    for combo in combinations(range(n), 7):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        subsets_7.append((list(combo), mask))
        
    best_adj_masks = None
    best_score = float('inf')
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        # Initialize random graph
        adj_masks = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
                    
        current_bad_6 = _count_regular_6(adj_masks, subsets_6)
        
        # Local search to minimize 6-regular subgraphs
        stuck_counter = 0
        for step in range(1000):
            if time.time() > deadline:
                break
                
            if current_bad_6 == 0:
                break
                
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
                
            # Flip edge (u, v)
            adj_masks[u] ^= (1 << v)
            adj_masks[v] ^= (1 << u)
            
            new_bad_6 = _count_regular_6(adj_masks, subsets_6)
            
            if new_bad_6 < current_bad_6:
                current_bad_6 = new_bad_6
                stuck_counter = 0
            else:
                # Revert
                adj_masks[u] ^= (1 << v)
                adj_masks[v] ^= (1 << u)
                stuck_counter += 1
                if stuck_counter > 50:
                    break
        
        if current_bad_6 == 0:
            # Check 7-subgraphs
            bad_7 = _count_regular_7(adj_masks, subsets_7)
            if bad_7 == 0:
                # Valid graph found!
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj_masks[u] & (1 << v):
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
            else:
                # Try to fix 7-subgraphs? For now, if 6 is clean, it's a good candidate.
                # But the requirement is strict.
                # We might need a more complex search.
                # However, finding a graph with no 6-regular subgraphs is already hard.
                # If we find one, we can try to tweak it to remove 7-regular ones.
                pass
        else:
            if current_bad_6 < best_score:
                best_score = current_bad_6
                best_adj_masks = adj_masks[:]
                
    # If we didn't find a perfect one, return the best effort or a smaller n?
    # The task asks to maximize n.
    # If we can't find n=20, try n=19, etc.
    # But usually, if we get stuck at n=20, smaller n is easier.
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    # Try n from 20 down to 6
    for n in range(20, 5, -1):
        if time.time() > deadline - 10:
            break
            
        result = _construct_graph(n)
        if result is not None:
            return result
            
    # Fallback: return a small valid graph (e.g., n=5 is always valid)
    # Or n=6 with no regular 6-subgraph (which is just not K6, Empty, etc)
    # Actually, any graph on 5 vertices is valid.
    # Let's return n=5 to be safe, or try a quick random search for n=6.
    
    # Quick search for n=6
    for _ in range(1000):
        if time.time() > deadline - 2:
            break
        adj_masks = [0] * 6
        for u in range(6):
            for v in range(u + 1, 6):
                if random.random() < 0.5:
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
        
        # Check if the whole graph is regular
        degs = [bin(adj_masks[i]).count('1') for i in range(6)]
        if len(set(degs)) != 1:
            edges = []
            for u in range(6):
                for v in range(u + 1, 6):
                    if adj_masks[u] & (1 << v):
                        edges.append([u, v])
            return {'n': 6, 'edges': edges}
            
    return {'n': 5, 'edges': []}
# EVOLVE-BLOCK-END
