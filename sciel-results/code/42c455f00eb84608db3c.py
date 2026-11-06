import time
import random

def construct():
    n = 16
    deadline = time.time() + 110  # Safety margin
    
    best_edges = []
    best_count = 0
    
    # --- Helpers ---
    
    def is_triangle_free_add(adj, u, v):
        return (adj[u] & adj[v]) == 0

    def is_3_colorable(adj):
        color = [-1] * n
        
        def backtrack(i):
            if i == n:
                return True
            
            mask = adj[i]
            # Only consider neighbors j < i for coloring constraint
            mask_lower = mask & ((1 << i) - 1)
            
            used_colors = 0
            m = mask_lower
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if color[j] != -1:
                    used_colors |= (1 << color[j])
                m ^= lsb
            
            for c in range(3):
                if not (used_colors & (1 << c)):
                    color[i] = c
                    if backtrack(i + 1):
                        return True
            color[i] = -1
            return False
        
        return backtrack(0)

    def get_edges_from_adj(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            mask_higher = mask & ~((1 << (u + 1)) - 1)
            m = mask_higher
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                edges.append([u, v])
                m ^= lsb
        return edges

    def make_grotzsch_adj():
        adj = [0] * n
        # C5: 0-1-2-3-4-0
        cycle = [(0,1), (1,2), (2,3), (3,4), (4,0)]
        for u, v in cycle:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Copies 5-9 for 0-4
        for u, v in cycle:
            adj[u] |= (1 << (5+v))
            adj[5+v] |= (1 << u)
            adj[v] |= (1 << (5+u))
            adj[5+u] |= (1 << v)
            
        # w=10 connected to 5-9
        for i in range(5):
            adj[10] |= (1 << (5+i))
            adj[5+i] |= (1 << 10)
            
        return adj

    def make_random_tf_adj():
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    if is_triangle_free_add(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        return adj

    def greedy_expand(adj):
        """Greedily add all possible triangle-free edges."""
        possible = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    if is_triangle_free_add(adj, u, v):
                        possible.append((u, v))
        
        random.shuffle(possible)
        
        for u, v in possible:
            if is_triangle_free_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)

    # --- Main Search Loop ---
    
    while time.time() < deadline:
        # Choose starting point: 50% Grötzsch, 50% random
        if random.random() < 0.5:
            adj = make_grotzsch_adj()
        else:
            adj = make_random_tf_adj()
        
        # Check if current graph is not 3-colorable
        if is_3_colorable(adj):
            # For random starts, try to make it 4-chromatic by adding edges
            # This is less reliable, so just skip
            continue
        
        # Greedily expand to add all possible triangle-free edges
        greedy_expand(adj)
        
        # Verify still not 3-colorable (greedy expansion preserves non-3-colorability)
        if is_3_colorable(adj):
            continue
        
        current_edges = get_edges_from_adj(adj)
        current_count = len(current_edges)
        
        if current_count > best_count:
            best_count = current_count
            best_edges = current_edges
        
        # Local Search: try to add more edges
        while time.time() < deadline:
            improved = False
            
            # 1. Try to add edges
            possible_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        if is_triangle_free_add(adj, u, v):
                            possible_edges.append((u, v))
            
            random.shuffle(possible_edges)
            
            for u, v in possible_edges:
                if time.time() > deadline:
                    break
                
                # Tentatively add
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                if not is_3_colorable(adj):
                    # Accept
                    improved = True
                    current_count += 1
                    if current_count > best_count:
                        best_count = current_count
                        best_edges = get_edges_from_adj(adj)
                    break  # Restart scan after adding one edge
                else:
                    # Revert
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            
            if improved:
                continue
            
            # 2. If no edge could be added, try to remove one to escape local optimum
            existing_edges = []
            for u in range(n):
                mask = adj[u]
                mask_higher = mask & ~((1 << (u + 1)) - 1)
                m = mask_higher
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    existing_edges.append((u, v))
                    m ^= lsb
            
            if not existing_edges:
                break
                
            random.shuffle(existing_edges)
            
            removed = False
            for u, v in existing_edges:
                if time.time() > deadline:
                    break
                
                # Tentatively remove
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
                # Check if still non-3-colorable
                if not is_3_colorable(adj):
                    # Accept removal
                    current_count -= 1
                    removed = True
                    break
                else:
                    # Revert
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            if not removed:
                # Stuck: cannot add or remove any edge
                break
                
    return best_edges
