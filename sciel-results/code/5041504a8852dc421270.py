import time
import random

def construct():
    N = 20
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Bitmask representation
    # mask[i] is a bitmask of neighbors of i
    masks = [0] * N
    
    def popcount(x):
        return bin(x).count('1')
        
    def is_valid_add(u, v):
        # Check if adding edge (u, v) creates a K3,3
        # The condition is: For all distinct a, b, c, |N(a) ∩ N(b) ∩ N(c)| <= 2.
        # Adding edge (u, v) changes N(u) to N(u) ∪ {v} and N(v) to N(v) ∪ {u}.
        # We only need to check triples involving u or v.
        
        # Triples (u, x, y) with x != y, x,y != u
        for x in range(N):
            if x == u: continue
            for y in range(N):
                if y == u or y == x: continue
                # Check triple (u, x, y)
                # New common neighbors: (masks[u] | (1 << v)) & masks[x] & masks[y]
                common = (masks[u] | (1 << v)) & masks[x] & masks[y]
                if popcount(common) >= 3:
                    return False
        
        # Triples (x, y, v) with x != y, x,y != v
        for x in range(N):
            if x == v or x == u: continue
            for y in range(N):
                if y == v or y == u or y == x: continue
                # Check triple (x, y, v)
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

    def add_all_possible():
        """Try to add all possible edges in multiple passes until no more can be added."""
        non_edges = [(u, v) for u in range(N) for v in range(u+1, N) if not ((masks[u] >> v) & 1)]
        while non_edges:
            random.shuffle(non_edges)
            added = False
            new_non_edges = []
            for u, v in non_edges:
                if try_add_edge(u, v):
                    added = True
                    # Don't add to new_non_edges since it's now an edge
                else:
                    new_non_edges.append((u, v))
            if not added:
                break
            non_edges = new_non_edges
        return non_edges

    # Greedy initialization with multiple random restarts
    random.seed(42)
    best_edges = 0
    best_masks = [0] * N
    
    # Try several different random orderings for initial construction
    for seed in range(50):
        random.seed(seed)
        masks = [0] * N
        edges_to_try = [(u, v) for u in range(N) for v in range(u+1, N)]
        random.shuffle(edges_to_try)
        for u, v in edges_to_try:
            try_add_edge(u, v)
        
        current_edges = sum(popcount(m) for m in masks) // 2
        if current_edges > best_edges:
            best_edges = current_edges
            best_masks = masks[:]
    
    # Restore best state
    masks = best_masks[:]
    current_edges = best_edges
    
    # Local Search: Hill climbing with random restarts
    # Use simulated annealing-like approach with temperature-based acceptance
    while time.time() < deadline:
        # Try to add edges
        non_edges = add_all_possible()
        
        if not non_edges:
            # Completely stuck - no edges can be added
            break
        
        # Check if we found a new best
        if current_edges > best_edges:
            best_edges = current_edges
            best_masks = masks[:]
        
        # If stuck (can't add more edges), perturb by removing multiple edges
        if non_edges:
            # Count how many edges we can potentially add
            addable = 0
            for u, v in non_edges[:min(20, len(non_edges))]:
                if is_valid_add(u, v):
                    addable += 1
            
            # If we can still add edges, do so
            if addable > 0:
                for u, v in non_edges:
                    if try_add_edge(u, v):
                        current_edges += 1
                # Continue loop to try more
                continue
            
            # Otherwise, perturb: remove 2-5 random edges to escape local optimum
            if current_edges < 2:
                break
            all_edges = [(u, v) for u in range(N) for v in range(u+1, N) if (masks[u] >> v) & 1]
            if not all_edges:
                break
            
            # Remove multiple edges for stronger perturbation
            num_remove = min(random.randint(2, 5), current_edges)
            random.shuffle(all_edges)
            for u, v in all_edges[:num_remove]:
                masks[u] &= ~(1 << v)
                masks[v] &= ~(1 << u)
                current_edges -= 1
            
            # Try to add edges again after perturbation
            non_edges = add_all_possible()
            if current_edges > best_edges:
                best_edges = current_edges
                best_masks = masks[:]

    # Reconstruct edge list from best_masks
    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if (best_masks[u] >> v) & 1:
                edges.append([u, v])
    return edges
