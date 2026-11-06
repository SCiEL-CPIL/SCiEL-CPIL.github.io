import time
import random

def construct():
    n = 40
    target_edges = 401
    start_time = time.time()
    timeout = 110  # Safety margin
    
    def init_graph():
        # Start with K20,20 + 1 internal edge
        edges = set()
        for a in range(20):
            for b in range(20, 40):
                edges.add((a, b))
        edges.add((0, 1))
        return edges

    def compute_max_book(edges, neighbors):
        max_b = 0
        for u, v in edges:
            b = (neighbors[u] & neighbors[v]).bit_count()
            if b > max_b:
                max_b = b
        return max_b

    def update_neighbors(edges, neighbors):
        # Rebuild neighbors from edges
        for i in range(n):
            neighbors[i] = 0
        for u, v in edges:
            neighbors[u] |= (1 << v)
            neighbors[v] |= (1 << u)

    best_edges = None
    best_score = float('inf')
    
    # Run for a few seconds or until timeout
    while time.time() - start_time < timeout:
        edges = init_graph()
        neighbors = [0] * n
        update_neighbors(edges, neighbors)
        current_score = compute_max_book(edges, neighbors)
        
        if current_score < best_score:
            best_score = current_score
            best_edges = set(edges)
            
        # Local search
        no_improve_count = 0
        while time.time() - start_time < timeout and no_improve_count < 1000:
            # Choose a move
            # 1. Random swap
            # 2. Targeted swap
            
            if random.random() < 0.5:
                # Random swap
                # Pick random edge to remove
                u, v = random.choice(list(edges))
                # Pick random non-edge to add
                while True:
                    x = random.randint(0, n-1)
                    y = random.randint(0, n-1)
                    if x == y: continue
                    if (x, y) in edges or (y, x) in edges: continue
                    break
                
                # Remove (u,v)
                edges.remove((u, v))
                # Add (x,y)
                if x > y: x, y = y, x
                edges.add((x, y))
            else:
                # Targeted swap
                # Find worst edge
                worst_edge = None
                worst_book = -1
                for e in edges:
                    b = (neighbors[e[0]] & neighbors[e[1]]).bit_count()
                    if b > worst_book:
                        worst_book = b
                        worst_edge = e
                
                if worst_edge is None:
                    break
                    
                u, v = worst_edge
                # Common neighbors
                common = neighbors[u] & neighbors[v]
                if common == 0:
                    # No triangles, just random swap
                    continue
                
                # Pick a random common neighbor w
                w_list = []
                mask = common
                while mask:
                    lsb = mask & -mask
                    w = lsb.bit_length() - 1
                    w_list.append(w)
                    mask ^= lsb
                
                w = random.choice(w_list)
                
                # Try to remove (u,w) or (v,w)
                if random.random() < 0.5:
                    rem_edge = (u, w) if u < w else (w, u)
                else:
                    rem_edge = (v, w) if v < w else (w, v)
                    
                if rem_edge not in edges:
                    continue
                    
                # Pick random non-edge to add
                while True:
                    x = random.randint(0, n-1)
                    y = random.randint(0, n-1)
                    if x == y: continue
                    if (x, y) in edges or (y, x) in edges: continue
                    break
                
                edges.remove(rem_edge)
                if x > y: x, y = y, x
                edges.add((x, y))
                
            # Update neighbors and score
            update_neighbors(edges, neighbors)
            new_score = compute_max_book(edges, neighbors)
            
            if new_score <= current_score:
                current_score = new_score
                no_improve_count = 0
                if current_score < best_score:
                    best_score = current_score
                    best_edges = set(edges)
            else:
                no_improve_count += 1
                # Revert move
                # This is tricky because we don't store the old state easily.
                # Better to store old edges and neighbors?
                # Or just accept the move with low probability (Simulated Annealing)?
                # For now, let's just accept if it's not too bad, or revert.
                # Reverting requires storing old state.
                # Let's store old state before move.
                
    # Convert best_edges set to list of lists
    return [[u, v] for u, v in best_edges]
