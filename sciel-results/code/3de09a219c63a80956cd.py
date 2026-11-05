"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n = 18
    # Start with complete tripartite K6,6,6
    # Parts: 0-5, 6-11, 12-17
    # Edges: all between different parts, none within parts
    
    # Represent adjacency as set of frozensets for O(1) lookup
    adj = [set() for _ in range(n)]
    
    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)
    
    def has_edge(u, v):
        return v in adj[u]
    
    def is_k4_free():
        """Check if graph is K4-free."""
        for combo in itertools.combinations(range(n), 4):
            a, b, c, d = combo
            if (has_edge(a,b) and has_edge(a,c) and has_edge(a,d) and
                has_edge(b,c) and has_edge(b,d) and has_edge(c,d)):
                return False
        return True
    
    def max_triangle_free_size():
        """Find the largest triangle-free induced subgraph size."""
        # Branch and bound
        best = [0]
        
        def is_triangle_free(vertices):
            """Check if the set of vertices induces a triangle-free graph."""
            for i in range(len(vertices)):
                for j in range(i+1, len(vertices)):
                    for k in range(j+1, len(vertices)):
                        if (has_edge(vertices[i], vertices[j]) and 
                            has_edge(vertices[i], vertices[k]) and 
                            has_edge(vertices[j], vertices[k])):
                            return False
            return True
        
        def branch(vertices, current_set, depth):
            if time.time() > deadline:
                return
            if len(current_set) + len(vertices) <= best[0]:
                return
            if not vertices:
                if len(current_set) > best[0]:
                    best[0] = len(current_set)
                return
            
            v = vertices[0]
            remaining = vertices[1:]
            
            # Option 1: include v
            if is_triangle_free(current_set + [v]):
                branch(remaining, current_set + [v], depth+1)
            
            # Option 2: exclude v
            branch(remaining, current_set, depth+1)
        
        branch(list(range(n)), [], 0)
        return best[0]
    
    # Initialize with complete tripartite
    for i in range(n):
        for j in range(i+1, n):
            part_i = i // 6
            part_j = j // 6
            if part_i != part_j:
                add_edge(i, j)
    
    # Verify initial is K4-free
    assert is_k4_free()
    
    current_score = max_triangle_free_size()
    best_graph = [list(adj[i]) for i in range(n)]
    best_score = current_score
    
    # Local search: try adding edges within parts
    # For each pair within the same part, try adding the edge
    # Check K4-free, then measure score
    
    # Precompute pairs within each part
    within_part_pairs = []
    for part in range(3):
        part_vertices = list(range(part*6, (part+1)*6))
        for i in range(len(part_vertices)):
            for j in range(i+1, len(part_vertices)):
                within_part_pairs.append((part_vertices[i], part_vertices[j]))
    
    random.seed(42)
    random.shuffle(within_part_pairs)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for u, v in within_part_pairs:
            if time.time() > deadline:
                break
            if has_edge(u, v):
                continue  # already added
            
            # Try adding edge (u, v)
            add_edge(u, v)
            
            if not is_k4_free():
                adj[u].discard(v)
                adj[v].discard(u)
                continue
            
            new_score = max_triangle_free_size()
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_graph = [list(adj[i]) for i in range(n)]
                improved = True
            else:
                # Remove the edge
                adj[u].discard(v)
                adj[v].discard(u)
    
    # Also try removing edges between parts to see if it helps
    # Actually, removing edges between parts would make it easier to have triangle-free sets
    # So we probably don't want to remove inter-part edges
    
    # Build the edge list from best_graph
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if j in best_graph[i]:
                edges.append([i, j])
    
    return edges

# EVOLVE-BLOCK-END
