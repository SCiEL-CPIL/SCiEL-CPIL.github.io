import random

def construct():
    n = 16
    # Use a fixed seed for reproducibility in the search
    random.seed(42)
    
    # 1. Base Graph: Grötzsch Graph (Mycielski of C5)
    # Vertices 0-4: C5 cycle
    # Vertices 5-9: Copies of 0-4
    # Vertex 10: Root
    edges = []
    # C5
    for i in range(5):
        edges.append([i, (i + 1) % 5])
    # Mycielski step
    for i in range(5):
        u = i
        v = (i + 1) % 5
        edges.append([u, 5 + v])
        edges.append([v, 5 + u])
    for i in range(5):
        edges.append([5 + i, 10])
        
    # Current vertex count: 11
    # Remaining vertices: 11, 12, 13, 14, 15
    
    # Initialize adjacency
    adj = {i: set() for i in range(n)}
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add(frozenset((u, v)))
        
    current_edges = len(edge_set)
    
    # Vertices to add: 11, 12, 13, 14, 15
    new_vertices = [11, 12, 13, 14, 15]
    
    # Exact Maximum Independent Set finder using backtracking
    def max_independent_set(nodes, adj):
        nodes = list(nodes)
        best_set = set()
        
        def backtrack(current_set, remaining_nodes):
            nonlocal best_set
            # Pruning: if current size + remaining count <= best size, stop
            if len(current_set) + len(remaining_nodes) <= len(best_set):
                return
            
            if not remaining_nodes:
                if len(current_set) > len(best_set):
                    best_set = set(current_set)
                return
            
            # Pick a node
            node = remaining_nodes[0]
            rest_nodes = remaining_nodes[1:]
            
            # Option 1: Include node
            # Neighbors of node in remaining_nodes
            neighbors = adj[node]
            new_remaining = [u for u in rest_nodes if u not in neighbors]
            current_set.add(node)
            backtrack(current_set, new_remaining)
            current_set.remove(node)
            
            # Option 2: Exclude node
            backtrack(current_set, rest_nodes)
            
        backtrack(set(), nodes)
        return best_set

    for v in new_vertices:
        # Determine which existing vertices can connect to v
        # S must be an independent set in the subgraph induced by existing vertices (0 to v-1)
        available_nodes = list(range(v))
        
        # Find the Maximum Independent Set
        iset = max_independent_set(available_nodes, adj)
        
        # Connect v to all in iset
        for u in iset:
            edges.append([v, u])
            adj[v].add(u)
            adj[u].add(v)
            edge_set.add(frozenset((v, u)))
            
        current_edges = len(edge_set)
        
    return edges

# EVOLVE-BLOCK-START
# (The code above is the implementation)
# EVOLVE-BLOCK-END
