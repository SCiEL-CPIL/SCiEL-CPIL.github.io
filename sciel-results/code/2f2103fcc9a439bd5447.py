import time
import random

def construct():
    N = 20
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    # Precompute popcount for 20-bit integers
    popcount = [0] * (1 << N)
    for i in range(1, 1 << N):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Precompute all pairs
    all_pairs = [(u, v) for u in range(N) for v in range(u+1, N)]
    
    def is_valid_add(masks, u, v):
        # Check if adding edge (u, v) creates a K3,3
        # Need to check all triples involving u or v
        mu = masks[u]
        mv = masks[v]
        
        # Triples (u, x, y): new common neighbors = (mu | (1<<v)) & masks[x] & masks[y]
        # This equals (mu & masks[x] & masks[y]) | ((1<<v) & masks[x] & masks[y])
        # Size increases by 1 iff v is in N(x) ∩ N(y) and not in N(u)
        # So we need: for all x,y, if v in N(x)∩N(y) then |N(u)∩N(x)∩N(y)| <= 1
        # Equivalently: for all x,y with v in N(x)∩N(y), popcount(mu & masks[x] & masks[y]) <= 1
        
        # Triples (x, y, v): new common neighbors = masks[x] & masks[y] & (mv | (1<<u))
        # Size increases by 1 iff u is in N(x)∩N(y) and not in N(v)
        # So we need: for all x,y with u in N(x)∩N(y), popcount(mv & masks[x] & masks[y]) <= 1
        
        # Check triples (u, x, y) where x, y != u
        for x in range(N):
            if x == u:
                continue
            mx = masks[x]
            # v must be in N(x) for the triple to potentially violate
            if not (mx >> v) & 1:
                continue
            for y in range(N):
                if y == u or y == x:
                    continue
                my = masks[y]
                if not (my >> v) & 1:
                    continue
                # v is in N(x) ∩ N(y), check if |N(u) ∩ N(x) ∩ N(y)| >= 2
                if popcount(mu & mx & my) >= 2:
                    return False
        
        # Check triples (x, y, v) where x, y != v, u
        for x in range(N):
            if x == v or x == u:
                continue
            mx = masks[x]
            # u must be in N(x) for the triple to potentially violate
            if not (mx >> u) & 1:
                continue
            for y in range(N):
                if y == v or y == u or y == x:
                    continue
                my = masks[y]
                if not (my >> u) & 1:
                    continue
                # u is in N(x) ∩ N(y), check if |N(v) ∩ N(x) ∩ N(y)| >= 2
                if popcount(mv & mx & my) >= 2:
                    return False
        
        return True
    
    def try_add_edge(masks, u, v):
        if u == v:
            return False
        if (masks[u] >> v) & 1:
            return False
        if is_valid_add(masks, u, v):
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
            return True
        return False
    
    def count_edges(masks):
        return sum(popcount(m) for m in masks) // 2
    
    def greedy_init(seed_val):
        rng = random.Random(seed_val)
        masks = [0] * N
        edges_to_try = all_pairs[:]
        rng.shuffle(edges_to_try)
        for u, v in edges_to_try:
            try_add_edge(masks, u, v)
        return masks
    
    def local_search(masks, rng, time_limit):
        current_edges = count_edges(masks)
        best_edges = current_edges
        best_masks = masks[:]
        
        while time.time() < time_limit:
            # Try to add edges
            improved = False
            non_edges = [(u, v) for u in range(N) for v in range(u+1, N) 
                        if not ((masks[u] >> v) & 1)]
            rng.shuffle(non_edges)
            
            for u, v in non_edges:
                if try_add_edge(masks, u, v):
                    improved = True
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_masks = masks[:]
            
            if not improved:
                # Try edge swaps: remove one edge, add another
                all_edges = [(u, v) for u in range(N) for v in range(u+1, N) 
                           if (masks[u] >> v) & 1]
                if not all_edges:
                    break
                
                swapped = False
                for _ in range(10):  # Try a few swaps
                    u, v = rng.choice(all_edges)
                    # Remove edge
                    masks[u] &= ~(1 << v)
                    masks[v] &= ~(1 << u)
                    current_edges -= 1
                    
                    # Try to add edges
                    non_edges = [(a, b) for a in range(N) for b in range(a+1, N) 
                                if not ((masks[a] >> b) & 1)]
                    rng.shuffle(non_edges)
                    
                    added = 0
                    for a, b in non_edges:
                        if try_add_edge(masks, a, b):
                            added += 1
                            current_edges += 1
                    
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_masks = masks[:]
                        swapped = True
                        break
                    else:
                        # Restore the removed edge
                        masks[u] |= (1 << v)
                        masks[v] |= (1 << u)
                        current_edges += 1
                        # Remove the added edges
                        for a, b in non_edges:
                            if (masks[a] >> b) & 1 and not ((best_masks[a] >> b) & 1):
                                masks[a] &= ~(1 << b)
                                masks[b] &= ~(1 << a)
                                current_edges -= 1
                
                if not swapped:
                    # Remove a random edge to escape
                    if current_edges == 0:
                        break
                    u, v = rng.choice(all_edges)
                    masks[u] &= ~(1 << v)
                    masks[v] &= ~(1 << u)
                    current_edges -= 1
    
        return best_masks, best_edges
    
    best_overall = 0
    best_overall_masks = [0] * N
    
    # Multiple restarts
    for seed_val in range(50):
        if time.time() >= deadline - 5:
            break
        
        masks = greedy_init(seed_val)
        best_masks, best_edges = local_search(masks, random.Random(seed_val + 1000), deadline - 5)
        
        if best_edges > best_overall:
            best_overall = best_edges
            best_overall_masks = best_masks[:]
    
    # Reconstruct edge list
    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if (best_overall_masks[u] >> v) & 1:
                edges.append([u, v])
    return edges
