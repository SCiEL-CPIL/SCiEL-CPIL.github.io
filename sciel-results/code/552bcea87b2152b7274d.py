import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    k4s = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    e1 = edge_index[(i, j)]
                    e2 = edge_index[(i, k)]
                    e3 = edge_index[(i, l)]
                    e4 = edge_index[(j, k)]
                    e5 = edge_index[(j, l)]
                    e6 = edge_index[(k, l)]
                    k4s.append([e1, e2, e3, e4, e5, e6])
    
    num_k4s = len(k4s)
    
    edge_to_k4s = [[] for _ in range(120)]
    for idx, k4 in enumerate(k4s):
        for e in k4:
            edge_to_k4s[e].append(idx)

    def is_k4_valid(colors, k4):
        c1, c2, c3, c4, c5, c6 = colors[k4[0]], colors[k4[1]], colors[k4[2]], colors[k4[3]], colors[k4[4]], colors[k4[5]]
        # Unrolled check for speed
        # Count distinct
        # Simple set is fine for 6 elements
        return len({c1, c2, c3, c4, c5, c6}) >= 5

    def is_k4_valid_with_color(colors, k4, e_idx, new_color):
        # Find position of e_idx in k4
        # k4 has 6 elements.
        c_list = []
        for e in k4:
            if e == e_idx:
                c_list.append(new_color)
            else:
                c_list.append(colors[e])
        return len(set(c_list)) >= 5

    best_colors = np.array([(i + j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=int)
    best_k = 17

    for k in range(13, 17):
        if time.time() > deadline:
            break
            
        restarts = 0
        found_valid = False
        while time.time() < deadline and restarts < 200 and not found_valid:
            restarts += 1
            colors = np.random.randint(0, k, size=120)
            
            k4_status = np.zeros(num_k4s, dtype=int)
            total_violations = 0
            for idx in range(num_k4s):
                if is_k4_valid(colors, k4s[idx]):
                    k4_status[idx] = 1
                else:
                    total_violations += 1
            
            if total_violations == 0:
                if k < best_k:
                    best_k = k
                    best_colors = colors.copy()
                found_valid = True
                break
            
            # Local Search
            # Try to reduce violations
            for step in range(1000):
                if time.time() > deadline:
                    break
                if total_violations == 0:
                    if k < best_k:
                        best_k = k
                        best_colors = colors.copy()
                    found_valid = True
                    break
                
                # Try 5 random edges
                moved = False
                for _ in range(5):
                    e_idx = np.random.randint(120)
                    old_color = colors[e_idx]
                    
                    best_new_color = -1
                    best_delta = 0
                    
                    for new_color in range(k):
                        if new_color == old_color:
                            continue
                        
                        delta = 0
                        for k4_idx in edge_to_k4s[e_idx]:
                            old_valid = k4_status[k4_idx]
                            new_valid = 1 if is_k4_valid_with_color(colors, k4s[k4_idx], e_idx, new_color) else 0
                            if new_valid != old_valid:
                                delta += old_valid - new_valid
                            
                        if delta < best_delta:
                            best_delta = delta
                            best_new_color = new_color
                        elif delta == best_delta and best_new_color == -1:
                            # Keep old color if no improvement? 
                            # We want delta < 0.
                            # If delta == 0, it's a neutral move.
                            # Let's only accept delta < 0 for now to ensure progress.
                            pass

                    if best_new_color != -1 and best_delta < 0:
                        colors[e_idx] = best_new_color
                        for k4_idx in edge_to_k4s[e_idx]:
                            old_valid = k4_status[k4_idx]
                            new_valid = 1 if is_k4_valid(colors, k4s[k4_idx]) else 0
                            if new_valid != old_valid:
                                k4_status[k4_idx] = new_valid
                                total_violations += old_valid - new_valid
                        moved = True
                        break # Restart edge picking loop if we moved

                if not moved:
                    # Local minimum reached
                    break

    return list(best_colors)
