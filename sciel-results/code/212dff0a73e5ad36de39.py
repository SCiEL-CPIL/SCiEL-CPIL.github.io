import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s limit
    
    # 255 = 3 * 5 * 17
    # Candidates must share a factor with 255
    candidates = []
    for i in range(1, 256):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)
    
    # Ensure 255 is in candidates (it is, 255%3==0)
    if 255 not in candidates:
        candidates.append(255)
    
    n = len(candidates)
    idx_255 = candidates.index(255)
    
    # Build conflict graph: edge between i,j if gcd(candidates[i], candidates[j]) == 1
    # We want maximum independent set in this conflict graph
    # Equivalent to maximum clique in the complement (where edge = gcd > 1)
    conflict = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) == 1:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    # Greedy coloring for upper bound (Tomita's method)
    # Returns colors array where colors[i] is the color class index for vertex i
    # The number of colors gives an upper bound on independent set size in any subgraph
    def color_sort(vertices, adj):
        """
        vertices: list of vertex indices
        adj: adjacency list (conflict graph)
        Returns: (ordered_vertices, colors) where colors[i] is the color of ordered_vertices[i]
        """
        if not vertices:
            return [], []
        
        # Greedy sequential coloring
        colors = [0] * len(vertices)
        color_assign = {}  # vertex -> color
        next_color = 1
        
        for v in vertices:
            # Find the smallest color not used by neighbors
            used_colors = set()
            for u in range(n):
                if (adj[v] >> u) & 1 and u in color_assign:
                    used_colors.add(color_assign[u])
            
            c = 1
            while c in used_colors:
                c += 1
            color_assign[v] = c
            next_color = max(next_color, c)
        
        # Sort vertices by color (ascending), then by degree (descending)
        # This helps with better pruning
        sorted_vertices = sorted(vertices, key=lambda v: (color_assign[v], -bin(adj[v]).count('1')))
        sorted_colors = [color_assign[v] for v in sorted_vertices]
        
        return sorted_vertices, sorted_colors
    
    # Branch and bound for maximum independent set
    # P: bitmask of available vertices
    # best_size: current best size found
    # best_set: current best independent set (as bitmask)
    best_size = 0
    best_set = 0
    
    # We must include 255, so we start with 255 in the independent set
    # In the conflict graph, we need 255 to have no conflicts with other selected vertices
    # So we can only select vertices that are NOT adjacent to 255 in the conflict graph
    # i.e., vertices that share a factor with 255 (gcd > 1)
    
    # Start: include 255, then find max independent set among non-neighbors of 255
    # Non-neighbors of 255 in conflict graph = vertices with gcd(255, v) > 1
    # But we already restricted to candidates divisible by 3, 5, or 17, so all share a factor with 255
    # Wait, that's not right. Two candidates can both be divisible by 3 or 5 or 17 but still be coprime to each other.
    # Example: 3 and 5 are both candidates, gcd(3,5)=1, so they conflict.
    
    # So we need to find the maximum independent set in the conflict graph that includes 255.
    # This is equivalent to: 1 + (max independent set in the subgraph induced by non-neighbors of 255)
    # Because if we include 255, we can't include any of its neighbors in the conflict graph.
    
    # Non-neighbors of 255 in conflict graph:
    non_neighbors_255 = 0
    for i in range(n):
        if i != idx_255 and not ((conflict[idx_255] >> i) & 1):
            non_neighbors_255 |= (1 << i)
    
    # Now find max independent set in the subgraph induced by non_neighbors_255
    # Using branch and bound with coloring
    
    def branch_bound(P, current_size, current_set):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
        
        # Update best if current is better
        if current_size > best_size:
            best_size = current_size
            best_set = current_set
        
        if P == 0:
            return
        
        # Get list of vertices in P
        vertices = []
        temp = P
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            vertices.append(v)
            temp ^= lsb
        
        # Color sort for upper bound
        sorted_vertices, sorted_colors = color_sort(vertices, conflict)
        max_color = max(sorted_colors) if sorted_colors else 0
        
        # Upper bound: current_size + max_color
        if current_size + max_color <= best_size:
            return
        
        # Process vertices in order (from last to first for better pruning)
        # We'll iterate through sorted_vertices in reverse order
        for i in range(len(sorted_vertices) - 1, -1, -1):
            if time.time() > deadline:
                return
            
            v = sorted_vertices[i]
            
            # If we can't beat best even with this vertex and all remaining, skip
            if current_size + i + 1 <= best_size:
                break
            
            # Include v in the independent set
            # Remove v and all its neighbors from P
            new_P = P & ~(conflict[v] | (1 << v))
            
            branch_bound(new_P, current_size + 1, current_set | (1 << v))
            
            # Remove v from P for subsequent iterations
            P &= ~(1 << v)
            
            # Recolor remaining vertices for tighter bound
            if P == 0:
                break
            
            # Quick check: if remaining vertices + current can't beat best, stop
            remaining_count = bin(P).count('1')
            if current_size + remaining_count <= best_size:
                break
    
    # Start the search
    # We include 255, so base size is 1
    base_set = 1 << idx_255
    branch_bound(non_neighbors_255, 1, base_set)
    
    # Convert best_set back to values
    result = []
    for i in range(n):
        if (best_set >> i) & 1:
            result.append(candidates[i])
    result.sort()
    
    # Ensure 255 is in the result (it should be, but just in case)
    if 255 not in result:
        result.append(255)
        result.sort()
    
    return result
