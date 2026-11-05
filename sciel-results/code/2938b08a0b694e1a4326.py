import time
import random

def construct():
    N = 20
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin before 150s limit
    
    # Bitmask representation
    masks = [0] * N
    
    def popcount(x):
        return bin(x).count('1')
        
    def is_valid_add(u, v):
        # Check if adding edge (u, v) creates a K3,3
        # For triples (u, x, y): need |N(u) ∪ {v} ∩ N(x) ∩ N(y)| <= 2
        # For triples (x, y, v): need |N(x) ∩ N(y) ∩ (N(v) ∪ {u})| <= 2
        
        for x in range(N):
            if x == u: continue
            for y in range(N):
                if y == u or y == x: continue
                common = (masks[u] | (1 << v)) & masks[x] & masks[y]
                if popcount(common) >= 3:
                    return False
        
        for x in range(N):
            if x == v or x == u: continue
            for y in range(N):
                if y == v or y == u or y == x: continue
                common = masks[x] & masks[y] & (masks[v] | (1 << u))
                if popcount(common) >= 3:
                    return False
        return True

    def try_add_edge(u, v):
        if u == v: return False
        if (masks[u] >> v) & 1: return False
        if is_valid_add(u, v):
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
            return True
        return False
    
    def remove_edge(u, v):
        masks[u] &= ~(1 << v)
        masks[v] &= ~(1 << u)
    
    def count_edges():
        return sum(popcount(m) for m in masks) // 2
    
    def get_non_edges():
        return [(u, v) for u in range(N) for v in range(u+1, N) if not ((masks[u] >> v) & 1)]
    
    def greedy_fill():
        non_edges = get_non_edges()
        random.shuffle(non_edges)
        added = 0
        for u, v in non_edges:
            if try_add_edge(u, v):
                added += 1
        return added
    
    def swap_escape():
        """Remove a random edge, then try to add multiple edges to compensate"""
        all_edges = [(u, v) for u in range(N) for v in range(u+1, N) if (masks[u] >> v) & 1]
        if not all_edges:
            return 0
        
        u1, v1 = random.choice(all_edges)
        remove_edge(u1, v1)
        
        # Try to add edges
        added = greedy_fill()
        return added
    
    def local_search():
        current_edges = count_edges()
        best_edges = current_edges
        best_masks = masks[:]
        
        # Iterative improvement
        while time.time() < deadline:
            # Try to add more edges
            greedy_fill()
            current_edges = count_edges()
            
            if current_edges > best_edges:
                best_edges = current_edges
                best_masks = masks[:]
            
            # Check if we're stuck
            non_edges = get_non_edges()
            can_add = False
            for u, v in non_edges[:20]:  # Check a subset
                if is_valid_add(u, v):
                    can_add = True
                    break
            
            if not can_add and current_edges > 0:
                # Try swap escape
                added = swap_escape()
                current_edges = count_edges()
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_masks = masks[:]
        
        return best_edges, best_masks
    
    # Multiple restarts with different seeds
    best_edges = 0
    best_masks = [0] * N
    
    for seed in [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:
        if time.time() > deadline:
            break
        
        random.seed(seed)
        masks = [0] * N
        
        # Greedy initialization
        edges_to_try = [(u, v) for u in range(N) for v in range(u+1, N)]
        random.shuffle(edges_to_try)
        for u, v in edges_to_try:
            try_add_edge(u, v)
        
        # Local search
        current_best, current_masks = local_search()
        
        if current_best > best_edges:
            best_edges = current_best
            best_masks = current_masks[:]
    
    # Final greedy fill
    masks = best_masks[:]
    greedy_fill()
    final_edges = count_edges()
    
    if final_edges > best_edges:
        best_edges = final_edges
        best_masks = masks[:]
    
    # Reconstruct edge list
    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if (best_masks[u] >> v) & 1:
                edges.append([u, v])
    return edges
