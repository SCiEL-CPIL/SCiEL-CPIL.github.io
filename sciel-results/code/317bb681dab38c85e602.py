import time
import random
import math

def _is_triangle_free(adj_masks):
    n = 16
    for u in range(n):
        v_mask = adj_masks[u]
        while v_mask:
            v_bit = v_mask & -v_mask
            v = v_bit.bit_length() - 1
            v_mask ^= v_bit
            if u < v:
                if adj_masks[u] & adj_masks[v]:
                    return False
    return True

def _is_3_colorable(adj_masks):
    n = 16
    degrees = [bin(mask).count('1') for mask in adj_masks]
    order = sorted(range(n), key=lambda v: degrees[v], reverse=True)
    
    color = [-1] * n
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        forbidden = 0
        neighbors = adj_masks[v]
        while neighbors:
            u_bit = neighbors & -neighbors
            u = u_bit.bit_length() - 1
            neighbors ^= u_bit
            if color[u] != -1:
                forbidden |= (1 << color[u])
        
        for c in range(3):
            if not (forbidden & (1 << c)):
                color[v] = c
                if backtrack(idx + 1):
                    return True
                color[v] = -1
        return False
    
    return backtrack(0)

def construct():
    deadline = time.time() + 145
    n = 16
    
    def get_edge_count(adj):
        return sum(bin(m).count('1') for m in adj) // 2
    
    def build_grotzsch():
        """Build Grötzsch graph (Mycielski of C5) on vertices 0-10."""
        adj = [0] * 16
        # C5: 0-1-2-3-4-0
        for u, v in [(0,1),(1,2),(2,3),(3,4),(4,0)]:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        # Copies 5-9
        for u, v in [(5,1),(5,4),(6,0),(6,2),(7,1),(7,3),(8,2),(8,4),(9,3),(9,0)]:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        # Apex 10
        for v in range(5, 10):
            adj[10] |= (1 << v)
            adj[v] |= (1 << 10)
        return adj
    
    def get_addable_edges(adj):
        """Return list of (u,v) edges that can be added without creating triangles."""
        candidates = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & (1 << v):
                    continue
                if adj[u] & adj[v]:
                    continue
                candidates.append((u, v))
        return candidates
    
    # Start with Grötzsch graph
    current_adj = build_grotzsch()
    current_edges = get_edge_count(current_adj)
    best_adj = [m for m in current_adj]
    best_edges = current_edges
    
    # Phase 1: Deterministic exhaustive search - try all combinations of addable edges
    all_addable = get_addable_edges(current_adj)
    all_addable.sort()
    
    # Try adding edges greedily in different orders
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Create a shuffled copy
        shuffled = all_addable[:]
        random.shuffle(shuffled)
        
        test_adj = [m for m in current_adj]
        test_edges = current_edges
        
        for u, v in shuffled:
            if test_adj[u] & (1 << v):
                continue
            if test_adj[u] & test_adj[v]:
                continue
            
            test_adj[u] |= (1 << v)
            test_adj[v] |= (1 << u)
            test_edges += 1
            
            if not _is_3_colorable(test_adj):
                if test_edges > best_edges:
                    best_edges = test_edges
                    best_adj = [m for m in test_adj]
    
    # Phase 2: Local search with multiple restarts
    for restart in range(30):
        if time.time() > deadline:
            break
        
        # Start from best found so far or a random valid graph
        if restart == 0:
            current_adj = [m for m in best_adj]
            current_edges = best_edges
        else:
            # Perturb the best solution
            current_adj = [m for m in best_adj]
            current_edges = best_edges
            
            # Remove some random edges
            for _ in range(random.randint(2, 5)):
                edges = []
                for u in range(n):
                    v_mask = current_adj[u]
                    while v_mask:
                        v_bit = v_mask & -v_mask
                        v = v_bit.bit_length() - 1
                        v_mask ^= v_bit
                        if u < v:
                            edges.append((u, v))
                if not edges:
                    break
                u, v = random.choice(edges)
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
                current_edges -= 1
        
        # Local search iterations
        for step in range(500):
            if time.time() > deadline:
                break
            
            move_type = random.random()
            
            if move_type < 0.6:
                # Add edge
                addable = get_addable_edges(current_adj)
                if not addable:
                    continue
                
                u, v = random.choice(addable)
                new_adj = [m for m in current_adj]
                new_adj[u] |= (1 << v)
                new_adj[v] |= (1 << u)
                
                if _is_3_colorable(new_adj):
                    continue
                
                current_adj = new_adj
                current_edges += 1
                
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = [m for m in current_adj]
                    
            elif move_type < 0.8:
                # Remove edge
                edges = []
                for u in range(n):
                    v_mask = current_adj[u]
                    while v_mask:
                        v_bit = v_mask & -v_mask
                        v = v_bit.bit_length() - 1
                        v_mask ^= v_bit
                        if u < v:
                            edges.append((u, v))
                if not edges:
                    continue
                u, v = random.choice(edges)
                new_adj = [m for m in current_adj]
                new_adj[u] &= ~(1 << v)
                new_adj[v] &= ~(1 << u)
                
                if _is_3_colorable(new_adj):
                    continue
                
                current_adj = new_adj
                current_edges -= 1
                
            else:
                # Swap: remove one edge, add another
                edges = []
                for u in range(n):
                    v_mask = current_adj[u]
                    while v_mask:
                        v_bit = v_mask & -v_mask
                        v = v_bit.bit_length() - 1
                        v_mask ^= v_bit
                        if u < v:
                            edges.append((u, v))
                if not edges:
                    continue
                u1, v1 = random.choice(edges)
                
                new_adj = [m for m in current_adj]
                new_adj[u1] &= ~(1 << v1)
                new_adj[v1] &= ~(1 << u1)
                
                addable = get_addable_edges(new_adj)
                if not addable:
                    continue
                
                u2, v2 = random.choice(addable)
                new_adj[u2] |= (1 << v2)
                new_adj[v2] |= (1 << u2)
                
                if _is_3_colorable(new_adj):
                    continue
                
                current_adj = new_adj
    
    # Phase 3: Final exhaustive polish - try all possible single edge additions
    for u in range(n):
        if time.time() > deadline:
            break
        for v in range(u+1, n):
            if time.time() > deadline:
                break
            
            if best_adj[u] & (1 << v):
                continue
            if best_adj[u] & best_adj[v]:
                continue
            
            new_adj = [m for m in best_adj]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            
            if not _is_3_colorable(new_adj):
                best_adj = new_adj
                best_edges += 1
    
    # Phase 4: Try to add multiple edges at once
    for _ in range(100):
        if time.time() > deadline:
            break
        
        addable = get_addable_edges(best_adj)
        if not addable:
            break
        
        random.shuffle(addable)
        new_adj = [m for m in best_adj]
        new_edges = best_edges
        
        for u, v in addable:
            if new_adj[u] & (1 << v):
                continue
            if new_adj[u] & new_adj[v]:
                continue
            
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            new_edges += 1
            
            if _is_3_colorable(new_adj):
                new_adj[u] &= ~(1 << v)
                new_adj[v] &= ~(1 << u)
                new_edges -= 1
                continue
        
        if new_edges > best_edges:
            best_edges = new_edges
            best_adj = [m for m in new_adj]
    
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges
