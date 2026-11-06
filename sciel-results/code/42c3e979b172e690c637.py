import time
import random

N = 20
C = 20
DEADLINE = 148.0

def construct():
    start = time.time()
    rng = random.Random(42)
    
    best_edges = []
    
    # We focus on the bipartite graph K_{10,10} between U={0..9} and V={10..19}
    U = list(range(10))
    V = list(range(10, 20))
    
    # List of all potential bipartite edges
    all_edges = [(u, v) for u in U for v in V]
    
    def is_valid_addition(u, v, c, graph):
        # Check if adding (u, v) with color c creates a non-rainbow 4-cycle
        # u is in U, v is in V (or vice versa, but we enforce U/V structure)
        # We assume graph is a dict {(u,v): c}
        
        # Determine which is in U and which is in V for the new edge
        if u < 10 and v >= 10:
            u_U = u
            v_V = v
        elif u >= 10 and v < 10:
            u_U = v
            v_V = u
        else:
            # Internal edge, not handled in this simple bipartite check
            # For now, return False to avoid adding internal edges in this phase
            return False
            
        # Iterate over all other u' in U and v' in V
        for u_prime in U:
            if u_prime == u_U:
                continue
            # Edge (u_prime, v_V) must exist? 
            # The cycle is u_U - v_V - u_prime - v_prime - u_U
            # Edges: (u_U, v_V) [new], (v_V, u_prime), (u_prime, v_prime), (v_prime, u_U)
            
            # Check if (u_prime, v_V) is in graph
            if (u_prime, v_V) not in graph and (v_V, u_prime) not in graph:
                continue
            c1 = graph.get((u_prime, v_V), graph.get((v_V, u_prime)))
            
            for v_prime in V:
                if v_prime == v_V:
                    continue
                
                # Check if (u_prime, v_prime) is in graph
                if (u_prime, v_prime) not in graph and (v_prime, u_prime) not in graph:
                    continue
                c2 = graph.get((u_prime, v_prime), graph.get((v_prime, u_prime)))
                
                # Check if (v_prime, u_U) is in graph
                if (v_prime, u_U) not in graph and (u_U, v_prime) not in graph:
                    continue
                c3 = graph.get((v_prime, u_U), graph.get((u_U, v_prime)))
                
                # We have a potential 4-cycle
                # Colors: c, c1, c2, c3
                # They must be distinct
                if len({c, c1, c2, c3}) != 4:
                    return False
                    
        return True

    # Greedy construction
    for iteration in range(100):
        if time.time() > DEADLINE - 5:
            break
            
        # Shuffle edges for different structures
        edges_to_try = all_edges[:]
        rng.shuffle(edges_to_try)
        
        graph = {}
        current_edges = []
        
        for u, v in edges_to_try:
            # Try each color
            colors = list(range(C))
            rng.shuffle(colors)
            
            for c in colors:
                if is_valid_addition(u, v, c, graph):
                    # Add edge
                    graph[(u, v)] = c
                    graph[(v, u)] = c
                    current_edges.append([u, v, c])
                    break
        
        if len(current_edges) > len(best_edges):
            best_edges = current_edges[:]
            
    return best_edges
