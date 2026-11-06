"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0  # safety margin before 150s kill
    
    def is_triangle_free(adj):
        """Check if graph is triangle-free using adjacency bitmasks."""
        for i in range(16):
            for j in range(i + 1, 16):
                if adj[i] & (1 << j):
                    if adj[i] & adj[j]:
                        return False
        return True
    
    def is_3_colorable(adj):
        """Check if graph is 3-colorable using fast backtracking with bitmasks.
        Returns True if 3-colorable, False if not (i.e., 4-chromatic or higher).
        """
        neighbors = [adj[i] for i in range(16)]
        degrees = [bin(adj[i]).count('1') for i in range(16)]
        order = sorted(range(16), key=lambda v: -degrees[v])
        
        colors = [-1] * 16
        color_masks = [0, 0, 0]
        
        def backtrack(pos):
            if pos == 16:
                return True
            
            v = order[pos]
            nmask = neighbors[v]
            
            # Determine available colors using bitmask operations
            forbidden = 0
            for c in range(3):
                if nmask & color_masks[c]:
                    forbidden |= (1 << c)
            
            available = [c for c in range(3) if not (forbidden >> c) & 1]
            
            if not available:
                return False
            
            # Try each available color - order by least constraining
            for c in available:
                colors[v] = c
                color_masks[c] |= (1 << v)
                
                if backtrack(pos + 1):
                    return True
                
                color_masks[c] &= ~(1 << v)
                colors[v] = -1
            
            return False
        
        return backtrack(0)
    
    def count_edges(adj):
        count = 0
        for i in range(16):
            count += bin(adj[i]).count('1')
        return count // 2
    
    def adj_to_edges(adj):
        edges = []
        for i in range(16):
            for j in range(i + 1, 16):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges
    
    def edges_to_adj(edges):
        adj = [0] * 16
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def build_grotzsch():
        """Build the Grötzsch graph (Mycielski of C5) on vertices 0-10."""
        adj = [0] * 16
        # C5 on vertices 0-4
        for i in range(5):
            adj[i] |= (1 << ((i + 1) % 5))
            adj[(i + 1) % 5] |= (1 << i)
        
        # Mycielski: add vertices 5-9 as copies of 0-4, and vertex 10 as universal to copies
        for i in range(5):
            for j in range(5):
                if adj[i] & (1 << j):
                    adj[5 + i] |= (1 << j)
                    adj[j] |= (1 << (5 + i))
            adj[5 + i] |= (1 << 10)
            adj[10] |= (1 << (5 + i))
        
        return adj
    
    def try_add_edge(adj, u, v):
        """Try to add edge (u,v) if it doesn't create a triangle. Returns True if added."""
        if adj[u] & (1 << v):
            return False
        common = adj[u] & adj[v]
        if common:
            return False
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        return True
    
    def try_remove_edge(adj, u, v):
        """Remove edge (u,v). Returns True if removed."""
        if adj[u] & (1 << v):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            return True
        return False
    
    def systematic_maximize(adj, deadline):
        """Systematically try to add all possible edges, maintaining 4-chromaticity."""
        current = adj[:]
        current_count = count_edges(current)
        
        # Get all possible edges not in current
        possible = []
        for i in range(16):
            for j in range(i + 1, 16):
                if not (current[i] & (1 << j)):
                    possible.append((i, j))
        
        # Sort by "difficulty" - edges between high-degree vertices first
        # These are most likely to be in the maximum graph
        def edge_difficulty(e):
            u, v = e
            return (bin(current[u]).count('1') + bin(current[v]).count('1'), u, v)
        
        possible.sort(key=edge_difficulty, reverse=True)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u, v in possible:
                if time.time() > deadline:
                    break
                # Check if edge still possible
                if current[u] & (1 << v):
                    continue
                common = current[u] & current[v]
                if common:
                    continue
                
                # Try adding
                current[u] |= (1 << v)
                current[v] |= (1 << u)
                
                if not is_3_colorable(current):
                    current_count += 1
                    improved = True
                else:
                    current[u] &= ~(1 << v)
                    current[v] &= ~(1 << u)
        
        return current, current_count
    
    def random_triangle_free_graph():
        """Generate a random triangle-free graph."""
        adj = [0] * 16
        all_edges = [(i, j) for i in range(16) for j in range(i + 1, 16)]
        random.shuffle(all_edges)
        for u, v in all_edges:
            common = adj[u] & adj[v]
            if common == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj
    
    best_adj = None
    best_count = 0
    
    # Phase 1: Start from Grötzsch graph and maximize
    grotzsch = build_grotzsch()
    if is_triangle_free(grotzsch) and not is_3_colorable(grotzsch):
        maxed, cnt = systematic_maximize(grotzsch, deadline)
        if cnt > best_count:
            best_count = cnt
            best_adj = maxed[:]
    
    # Phase 2: Try multiple random triangle-free graphs that are 4-chromatic
    for _ in range(100):
        if time.time() > deadline:
            break
        adj = random_triangle_free_graph()
        cnt = count_edges(adj)
        if cnt >= best_count and not is_3_colorable(adj):
            maxed, new_cnt = systematic_maximize(adj, deadline)
            if new_cnt > best_count:
                best_count = new_cnt
                best_adj = maxed[:]
    
    # Phase 3: If we still have time, try more random starts
    if time.time() < deadline - 10:
        for _ in range(200):
            if time.time() > deadline:
                break
            adj = random_triangle_free_graph()
            cnt = count_edges(adj)
            if cnt >= best_count - 5 and not is_3_colorable(adj):
                maxed, new_cnt = systematic_maximize(adj, deadline)
                if new_cnt > best_count:
                    best_count = new_cnt
                    best_adj = maxed[:]
    
    # Phase 4: Final local search from best
    if best_adj is not None and time.time() < deadline - 5:
        current = best_adj[:]
        current_count = best_count
        
        # Try all single-edge additions
        for i in range(16):
            for j in range(i + 1, 16):
                if time.time() > deadline:
                    break
                if current[i] & (1 << j):
                    continue
                common = current[i] & current[j]
                if common:
                    continue
                
                current[i] |= (1 << j)
                current[j] |= (1 << i)
                
                if not is_3_colorable(current):
                    current_count += 1
                    best_count = current_count
                    best_adj = current[:]
                else:
                    current[i] &= ~(1 << j)
                    current[j] &= ~(1 << i)
    
    if best_adj is None:
        # Absolute fallback: return the Grötzsch graph
        grotzsch = build_grotzsch()
        return adj_to_edges(grotzsch)
    
    return adj_to_edges(best_adj)
# EVOLVE-BLOCK-END
