import time
import random
import sys

def construct():
    n = 16
    deadline = time.time() + 110
    
    best_edges = []
    best_count = 0
    
    def is_triangle_free_adj(adj, u, v):
        return (adj[u] & adj[v]) == 0

    def is_3_colorable(adj):
        # Fast backtracking with bitmasks and forward checking
        color = [-1] * n
        avail = [7] * n  # 0b111 for each vertex
        
        # Order vertices by degree descending for faster pruning
        order = list(range(n))
        order.sort(key=lambda i: bin(adj[i]).count('1'), reverse=True)
        
        def backtrack(pos):
            if pos == n:
                return True
            
            u = order[pos]
            mask = avail[u]
            while mask:
                c_bit = mask & (-mask)
                c = c_bit.bit_length() - 1
                mask ^= c_bit
                
                color[u] = c
                neighbors = adj[u]
                
                # Forward checking: update avail for uncolored neighbors
                undo = []
                feasible = True
                m = neighbors
                while m:
                    v_bit = m & (-m)
                    v = v_bit.bit_length() - 1
                    m ^= v_bit
                    if color[v] == -1:
                        old = avail[v]
                        new = old & ~c_bit
                        if new == 0:
                            feasible = False
                            break
                        avail[v] = new
                        undo.append((v, old))
                
                if feasible and backtrack(pos + 1):
                    return True
                
                # Undo
                for v, old in undo:
                    avail[v] = old
                color[u] = -1
            
            return False
        
        return backtrack(0)

    def get_edges_from_adj(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges

    def greedy_add_edges(adj):
        added = 0
        possible = [(u, v) for u in range(n) for v in range(u + 1, n) if not (adj[u] & (1 << v))]
        random.shuffle(possible)
        
        for u, v in possible:
            if is_triangle_free_adj(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                added += 1
        return added

    def make_grotzsch():
        adj = [0] * n
        cycle = [(0,1), (1,2), (2,3), (3,4), (4,0)]
        for u, v in cycle:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        for u, v in cycle:
            adj[u] |= (1 << (5+v))
            adj[5+v] |= (1 << u)
            adj[v] |= (1 << (5+u))
            adj[5+u] |= (1 << v)
            
        for i in range(5):
            adj[10] |= (1 << (5+i))
            adj[5+i] |= (1 << 10)
            
        return adj

    def make_random_tf():
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    if is_triangle_free_adj(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        return adj

    def local_search(adj, max_iters=200):
        # Try remove-one-add-two moves to increase edge count while maintaining properties
        cur_edges = sum(bin(adj[i]).count('1') for i in range(n)) // 2
        best_adj = adj[:]
        best_count = cur_edges
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Find all possible remove-one-add-two moves
            removed = [(u, v) for u in range(n) for v in range(u+1, n) if (adj[u] & (1 << v))]
            random.shuffle(removed)
            
            found_improvement = False
            for u, v in removed:
                if time.time() > deadline:
                    break
                
                # Remove edge (u, v)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
                # Try to add two edges
                candidates = [(a, b) for a in range(n) for b in range(a+1, n) 
                             if not (adj[a] & (1 << b)) and is_triangle_free_adj(adj, a, b)]
                random.shuffle(candidates)
                
                added = 0
                new_edges = []
                for a, b in candidates:
                    if added >= 2:
                        break
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
                    added += 1
                    new_edges.append((a, b))
                
                new_count = sum(bin(adj[i]).count('1') for i in range(n)) // 2
                
                if new_count > best_count and not is_3_colorable(adj):
                    best_count = new_count
                    best_adj = adj[:]
                    found_improvement = True
                else:
                    # Undo
                    for a, b in new_edges:
                        adj[a] &= ~(1 << b)
                        adj[b] &= ~(1 << a)
                
                # Re-add the removed edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                if found_improvement:
                    break
            
            if not found_improvement:
                break
        
        return best_adj, best_count

    while time.time() < deadline:
        # We need a starting graph that is NOT 3-colorable.
        # Grötzsch is not 3-colorable.
        # Random TF graphs are likely 3-colorable (most are bipartite or close).
        # So we should mostly start with Grötzsch or derivatives.
        
        if random.random() < 0.8:
            adj = make_grotzsch()
        else:
            adj = make_random_tf()
        
        # Verify it's not 3-colorable
        if not is_3_colorable(adj):
            greedy_add_edges(adj)
            
            # Try local search to improve
            improved_adj, improved_count = local_search(adj)
            
            if improved_count > best_count:
                best_count = improved_count
                best_edges = get_edges_from_adj(improved_adj)
            else:
                edges = get_edges_from_adj(adj)
                if len(edges) > best_count:
                    best_count = len(edges)
                    best_edges = edges
        else:
            # If random TF is 3-colorable, we can try to add edges to make it not?
            # No, adding edges doesn't help with 3-colorability if it's already 3-colorable?
            # Wait. If G is 3-colorable, G+e might be not 3-colorable.
            # Example: C5 is not 3-colorable? No, C5 is not 3-colorable?
            # C5 has odd cycle. Chromatic number 3.
            # C3 is not 3-colorable? No, C3 is 3-colorable.
            # K4 is not 3-colorable.
            # If we have a 3-colorable graph, we can add edges to make it not 3-colorable?
            # Yes, if we add enough edges to force a 4-chromatic structure.
            # But we are constrained by triangle-freeness.
            # So we can't add K4.
            # We can add edges to create a Grötzsch-like structure?
            # This is hard to do greedily.
            pass

    return best_edges
