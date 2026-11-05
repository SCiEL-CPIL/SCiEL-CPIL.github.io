import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Edge indexing
    edge_map = {}
    idx = 0
    for i in range(16):
        for j in range(i+1, 16):
            edge_map[(i, j)] = idx
            idx += 1
            
    # Precompute K4s
    k4_edges = []
    for a in range(16):
        for b in range(a+1, 16):
            for c in range(b+1, 16):
                for d in range(c+1, 16):
                    edges = [
                        edge_map[(a, b)], edge_map[(a, c)], edge_map[(a, d)],
                        edge_map[(b, c)], edge_map[(b, d)],
                        edge_map[(c, d)]
                    ]
                    k4_edges.append(tuple(edges))
                    
    # Precompute which K4s contain which edge
    k4s_containing_edge = [[] for _ in range(120)]
    for k4_idx, edges in enumerate(k4_edges):
        for e in edges:
            k4s_containing_edge[e].append(k4_idx)
            
    def is_valid_change(e, new_color, colors):
        if colors[e] == new_color:
            return True
        for k4_idx in k4s_containing_edge[e]:
            edges = k4_edges[k4_idx]
            c_set = set()
            for e_idx in edges:
                if e_idx == e:
                    c_set.add(new_color)
                else:
                    c_set.add(colors[e_idx])
            if len(c_set) < 5:
                return False
        return True

    # Initialize colors
    # Start with 17-coloring
    colors = [0] * 120
    for i in range(16):
        for j in range(i+1, 16):
            colors[edge_map[(i, j)]] = (i + j) % 17
            
    best_colors = colors.copy()
    best_count = 17
    
    def update_best():
        nonlocal best_colors, best_count
        active = set(colors)
        if len(active) < best_count:
            best_count = len(active)
            best_colors = colors.copy()

    update_best()
    
    def try_merge(c1, c2):
        edges_to_change = [e for e in range(120) if colors[e] == c2]
        if not edges_to_change:
            return False
            
        affected_k4s = set()
        for e in edges_to_change:
            for k4_idx in k4s_containing_edge[e]:
                affected_k4s.add(k4_idx)
                
        for k4_idx in affected_k4s:
            edges = k4_edges[k4_idx]
            c_set = set()
            for e_idx in edges:
                c = colors[e_idx]
                if c == c2:
                    c_set.add(c1)
                else:
                    c_set.add(c)
            if len(c_set) < 5:
                return False
                
        for e in edges_to_change:
            colors[e] = c1
        return True

    def random_valid_move():
        e = random.randint(0, 119)
        max_c = 20
        new_c = random.randint(0, max_c)
        
        if is_valid_change(e, new_c, colors):
            colors[e] = new_c
            return True
        return False

    stuck_count = 0
    last_count = best_count
    
    while time.time() < deadline:
        if time.time() > deadline - 5:
            break
            
        active_colors = set(colors)
        cur_count = len(active_colors)
        
        if cur_count < best_count:
            update_best()
            stuck_count = 0
        elif cur_count == last_count:
            stuck_count += 1
        else:
            last_count = cur_count
            stuck_count = 0
            
        merged = False
        cols = list(active_colors)
        random.shuffle(cols)
        
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                c1 = cols[i]
                c2 = cols[j]
                if try_merge(c1, c2):
                    merged = True
                    break
            if merged:
                break
                
        if not merged:
            moves_to_try = 20 + stuck_count * 5
            for _ in range(moves_to_try):
                random_valid_move()
                
    return best_colors
