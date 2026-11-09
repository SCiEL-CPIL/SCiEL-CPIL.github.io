import random
import time

def construct():
    n = 20
    u_nodes = list(range(10))
    v_nodes = list(range(10, 20))
    
    edges = []
    for u in u_nodes:
        for v in v_nodes:
            edges.append((u, v))
            
    cycles = []
    for i in range(10):
        for j in range(i+1, 10):
            u1, u2 = i, j
            for k in range(10):
                for l in range(k+1, 10):
                    v1, v2 = 10+k, 10+l
                    cycles.append(((u1, v1), (u2, v1), (u2, v2), (u1, v2)))
                    
    def get_bad_count(colors):
        bad = 0
        for cyc in cycles:
            c1, c2, c3, c4 = cyc
            if colors[c1] == colors[c2] or colors[c1] == colors[c3] or colors[c1] == colors[c4] or \
               colors[c2] == colors[c3] or colors[c2] == colors[c4] or \
               colors[c3] == colors[c4]:
                bad += 1
        return bad

    start_time = time.time()
    timeout = 115
    
    best_colors = None
    best_bad = float('inf')
    
    while time.time() - start_time < timeout:
        colors = {e: random.randint(0, 19) for e in edges}
        current_bad = get_bad_count(colors)
        
        if current_bad < best_bad:
            best_bad = current_bad
            best_colors = colors.copy()
            
        if best_bad == 0:
            break
            
        stuck_count = 0
        while time.time() - start_time < timeout:
            if current_bad == 0:
                break
                
            e = random.choice(edges)
            new_c = random.randint(0, 19)
            old_c = colors[e]
            
            if new_c == old_c:
                continue
                
            colors[e] = new_c
            new_bad = get_bad_count(colors)
            
            if new_bad < current_bad:
                current_bad = new_bad
                stuck_count = 0
                if current_bad < best_bad:
                    best_bad = current_bad
                    best_colors = colors.copy()
            else:
                colors[e] = old_c
                stuck_count += 1
                
            if stuck_count > 1000:
                break
                
    if best_colors is None:
        return [[0, v, 0] for v in range(1, 20)]
        
    result = []
    for e in edges:
        u, v = e
        c = best_colors[e]
        result.append([u, v, c])
        
    return result
