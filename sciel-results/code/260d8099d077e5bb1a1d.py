"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    
    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)
    
    def build_conflict_bitmask(edges, adj):
        """Build conflict graph as bitmasks for efficient operations."""
        m = len(edges)
        edge_idx = {e: i for i, e in enumerate(edges)}
        conflict = [0] * m  # bitmask of conflicting edges
        
        for i, (u, v) in enumerate(edges):
            # Edges sharing a vertex
            for w in adj.get(u, ()):
                e2 = (min(u, w), max(u, w))
                if e2 in edge_idx and e2 != (u, v):
                    j = edge_idx[e2]
                    conflict[i] |= (1 << j)
            for w in adj.get(v, ()):
                e2 = (min(v, w), max(v, w))
                if e2 in edge_idx and e2 != (u, v):
                    j = edge_idx[e2]
                    conflict[i] |= (1 << j)
            # Edges at distance 2 (connected by an edge)
            for x in adj.get(u, ()):
                for y in adj.get(x, ()):
                    if y == u:
                        continue
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_idx and e2 != (u, v):
                        j = edge_idx[e2]
                        conflict[i] |= (1 << j)
            for x in adj.get(v, ()):
                for y in adj.get(x, ()):
                    if y == v:
                        continue
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_idx and e2 != (u, v):
                        j = edge_idx[e2]
                        conflict[i] |= (1 << j)
        
        return conflict
    
    def greedy_colour(conflict, m, order):
        """Greedy colouring with bitmask conflict checks."""
        colour = [-1] * m
        colour_count = 0
        # For each colour, track which edges use it (bitmask)
        colour_masks = []
        
        for i in order:
            # Check which colours are forbidden
            forbidden = 0
            for c, mask in enumerate(colour_masks):
                if conflict[i] & mask:
                    forbidden |= (1 << c)
            
            # Find smallest available colour
            c = 0
            while (forbidden >> c) & 1:
                c += 1
            
            colour[i] = c
            if c >= len(colour_masks):
                colour_masks.append(1 << i)
            else:
                colour_masks[c] |= (1 << i)
            if c + 1 > colour_count:
                colour_count = c + 1
        
        return colour, colour_count, colour_masks
    
    def can_move_edge(conflict, m, colour, colour_masks, edge, new_colour):
        """Check if edge can be moved to new_colour."""
        if new_colour >= len(colour_masks):
            return True
        # Check if edge conflicts with any edge in new_colour's class
        return (conflict[edge] & colour_masks[new_colour]) == 0
    
    def local_search(conflict, m, colour, colour_count, colour_masks, max_iterations=500):
        """Local search: try to eliminate highest colour, then do 2-swaps."""
        colour = list(colour)
        colour_masks = list(colour_masks)
        
        # Phase 1: Try to eliminate the highest colour
        for attempt in range(200):
            if colour_count <= 1:
                break
            max_c = colour_count - 1
            if max_c >= len(colour_masks):
                break
            
            to_reassign = []
            mask = colour_masks[max_c]
            for i in range(m):
                if (mask >> i) & 1:
                    to_reassign.append(i)
            
            if not to_reassign:
                colour_count -= 1
                if max_c < len(colour_masks):
                    del colour_masks[max_c]
                    # Renumber colours
                    for i in range(m):
                        if colour[i] > max_c:
                            colour[i] -= 1
                continue
            
            # Try to reassign each edge
            success = True
            new_colour = list(colour)
            new_masks = list(colour_masks)
            
            for i in to_reassign:
                found = False
                for c in range(max_c):
                    if c < len(new_masks) and can_move_edge(conflict, m, new_colour, new_masks, i, c):
                        # Remove from old colour
                        new_masks[max_c] &= ~(1 << i)
                        # Add to new colour
                        new_masks[c] |= (1 << i)
                        new_colour[i] = c
                        found = True
                        break
                if not found:
                    success = False
                    break
            
            if success:
                colour = new_colour
                colour_masks = new_masks
                colour_count -= 1
                if max_c < len(colour_masks):
                    del colour_masks[max_c]
                    for i in range(m):
                        if colour[i] > max_c:
                            colour[i] -= 1
            else:
                break
        
        # Phase 2: 2-swap local search
        for iteration in range(max_iterations):
            improved = False
            for i in range(m):
                for j in range(i + 1, m):
                    if colour[i] == colour[j]:
                        continue
                    ci, cj = colour[i], colour[j]
                    # Check if swapping i and j is valid
                    # i needs to be able to go to cj, j needs to be able to go to ci
                    # Remove i from ci, j from cj temporarily
                    new_ci_mask = colour_masks[ci] & ~(1 << i) if ci < len(colour_masks) else 0
                    new_cj_mask = colour_masks[cj] & ~(1 << j) if cj < len(colour_masks) else 0
                    
                    # Check if i can go to cj
                    if (conflict[i] & new_cj_mask) != 0:
                        continue
                    # Check if j can go to ci
                    if (conflict[j] & new_ci_mask) != 0:
                        continue
                    
                    # Perform swap
                    colour[i], colour[j] = cj, ci
                    colour_masks[ci] = new_ci_mask
                    colour_masks[cj] = new_cj_mask
                    colour_masks[ci] |= (1 << j)
                    colour_masks[cj] |= (1 << i)
                    improved = True
            
            if not improved:
                break
        
        return colour, colour_count, colour_masks
    
    def class_merge(conflict, m, colour, colour_count, colour_masks):
        """Try to merge two colour classes if possible."""
        colour = list(colour)
        colour_masks = list(colour_masks)
        
        # Try all pairs of colours
        for c1 in range(colour_count):
            if c1 >= len(colour_masks):
                continue
            for c2 in range(c1 + 1, colour_count):
                if c2 >= len(colour_masks):
                    continue
                
                # Check if we can merge c1 and c2
                # An edge in c1 can move to c2 if it doesn't conflict with c2's edges
                # and vice versa
                mask1 = colour_masks[c1]
                mask2 = colour_masks[c2]
                
                # Get edges in c1
                edges_c1 = [i for i in range(m) if (mask1 >> i) & 1]
                edges_c2 = [i for i in range(m) if (mask2 >> i) & 1]
                
                # Try to move all edges from c2 to c1
                success = True
                new_colour = list(colour)
                new_mask1 = mask1
                new_mask2 = 0
                
                for i in edges_c2:
                    # Check if i can move to c1 (considering current state)
                    if (conflict[i] & new_mask1) != 0:
                        success = False
                        break
                    new_mask1 |= (1 << i)
                    new_mask2 &= ~(1 << i)
                    new_colour[i] = c1
                
                if success and new_mask2 == 0:
                    colour = new_colour
                    colour_masks[c1] = new_mask1
                    # Remove c2
                    del colour_masks[c2]
                    # Renumber
                    for i in range(m):
                        if colour[i] > c2:
                            colour[i] -= 1
                    colour_count -= 1
                    return colour, colour_count, colour_masks
        
        return colour, colour_count, colour_masks
    
    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > deadline:
            out.append([])
            continue
        
        edges = instance(*spec)
        m = len(edges)
        if m == 0:
            out.append([])
            continue
        
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        conflict = build_conflict_bitmask(edges, adj)
        
        # Try multiple random orderings with greedy + local search
        rng = random.Random(42)
        best_colour = None
        best_cc = float('inf')
        best_masks = None
        
        # First, try DSATUR-like ordering (by degree)
        order = list(range(m))
        order.sort(key=lambda i: -bin(conflict[i]).count('1'))
        colour, cc, masks = greedy_colour(conflict, m, order)
        colour, cc, masks = local_search(conflict, m, colour, cc, masks)
        colour, cc, masks = class_merge(conflict, m, colour, cc, masks)
        
        if cc < best_cc:
            best_cc = cc
            best_colour = colour
            best_masks = masks
        
        # Try random orderings
        for trial in range(30):
            if time.time() > deadline:
                break
            
            order = list(range(m))
            rng.shuffle(order)
            # Bias towards high-degree vertices
            order.sort(key=lambda i: (-bin(conflict[i]).count('1'), rng.random()))
            
            colour, cc, masks = greedy_colour(conflict, m, order)
            colour, cc, masks = local_search(conflict, m, colour, cc, masks)
            colour, cc, masks = class_merge(conflict, m, colour, cc, masks)
            
            if cc < best_cc:
                best_cc = cc
                best_colour = colour
                best_masks = masks
        
        out.append([best_colour[i] for i in range(m)])
    
    return out
# EVOLVE-BLOCK-END
