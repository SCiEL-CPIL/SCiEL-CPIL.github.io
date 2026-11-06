import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    n = 40
    target_edges = 401
    
    # Initialize with K20,20 + 1 random intra edge
    edges = set()
    for u in range(20):
        for v in range(20, 40):
            edges.add((u, v))
            
    a1, a2 = random.sample(range(20), 2)
    edges.add((a1, a2))
    
    current_edges = list(edges)
    
    def evaluate(edge_list):
        adj = [set() for _ in range(n)]
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
        
        max_book = 0
        for u, v in edge_list:
            # Common neighbors
            # Since n is small, intersection is fast
            common = adj[u].intersection(adj[v])
            if len(common) > max_book:
                max_book = len(common)
        return max_book

    current_score = evaluate(current_edges)
    best_edges = current_edges[:]
    best_score = current_score
    
    T = 10.0
    T_min = 0.01
    alpha = 0.9995
    
    # Pre-compute lists of valid cross and intra pairs for faster random selection?
    # No, generating random pairs is O(1) with rejection sampling.
    
    while time.time() < deadline:
        move_type = random.random()
        new_edges = current_edges[:]
        edge_set = set(current_edges)
        valid_move = False
        
        if move_type < 0.4:
            # Swap cross for cross
            # Find a random cross edge
            # To avoid scanning all edges, we can pick a random vertex and look at its cross neighbors
            # But scanning 401 edges is fast.
            cross_edges = [(u, v) for u, v in current_edges if (u < 20 and v >= 20) or (u >= 20 and v < 20)]
            if cross_edges:
                e1 = random.choice(cross_edges)
                # Find a random non-edge cross pair
                while True:
                    a = random.randint(0, 19)
                    b = random.randint(20, 39)
                    if (a, b) not in edge_set and (b, a) not in edge_set:
                        e2 = (a, b)
                        break
                
                edge_set.remove(e1)
                edge_set.add(e2)
                new_edges = list(edge_set)
                valid_move = True
                
        elif move_type < 0.8:
            # Swap intra for intra
            intra_edges = [(u, v) for u, v in current_edges if (u < 20 and v < 20) or (u >= 20 and v >= 20)]
            if intra_edges:
                e1 = random.choice(intra_edges)
                if e1[0] < 20:
                    part_range = range(20)
                else:
                    part_range = range(20, 40)
                
                while True:
                    a, b = random.sample(part_range, 2)
                    if (a, b) not in edge_set and (b, a) not in edge_set:
                        e2 = (a, b)
                        break
                        
                edge_set.remove(e1)
                edge_set.add(e2)
                new_edges = list(edge_set)
                valid_move = True
            else:
                # If no intra edges, force a cross->intra swap to create one
                # This falls through to the else case logic effectively
                pass
        else:
            # Swap cross for intra
            cross_edges = [(u, v) for u, v in current_edges if (u < 20 and v >= 20) or (u >= 20 and v < 20)]
            if cross_edges:
                e1 = random.choice(cross_edges)
                if random.random() < 0.5:
                    part_range = range(20)
                else:
                    part_range = range(20, 40)
                
                while True:
                    a, b = random.sample(part_range, 2)
                    if (a, b) not in edge_set and (b, a) not in edge_set:
                        e2 = (a, b)
                        break
                        
                edge_set.remove(e1)
                edge_set.add(e2)
                new_edges = list(edge_set)
                valid_move = True

        if not valid_move:
            T *= alpha
            continue
            
        new_score = evaluate(new_edges)
        
        if new_score < current_score:
            current_edges = new_edges
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_edges = current_edges[:]
        else:
            delta = new_score - current_score
            if delta < 0:
                accept = True
            else:
                if T > 0:
                    prob = math.exp(-delta / T)
                    accept = random.random() < prob
                else:
                    accept = False
            
            if accept:
                current_edges = new_edges
                current_score = new_score
        
        T *= alpha
        if T < T_min:
            T = T_min

    return [[u, v] for u, v in best_edges]
