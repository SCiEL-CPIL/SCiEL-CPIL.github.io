import random
import time
import math

def construct():
    n = 16
    num_edges = n * (n - 1) // 2
    edges = []
    edge_to_idx = {}
    for i in range(n):
        for j in range(i + 1, n):
            edge_to_idx[(i, j)] = len(edges)
            edges.append((i, j))
    
    k4s = []
    k4_edges = []
    edge_to_k4s = [[] for _ in range(num_edges)]
    
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    k4_idx = len(k4s)
                    k4s.append((a, b, c, d))
                    e1 = edge_to_idx[(a, b)]
                    e2 = edge_to_idx[(a, c)]
                    e3 = edge_to_idx[(a, d)]
                    e4 = edge_to_idx[(b, c)]
                    e5 = edge_to_idx[(b, d)]
                    e6 = edge_to_idx[(c, d)]
                    k4_edges.append([e1, e2, e3, e4, e5, e6])
                    for e in [e1, e2, e3, e4, e5, e6]:
                        edge_to_k4s[e].append(k4_idx)

    def get_initial_colors():
        colors = [0] * num_edges
        for i in range(n):
            for j in range(i + 1, n):
                idx = edge_to_idx[(i, j)]
                colors[idx] = (i + j) % 17
        return colors

    def init_state(colors):
        bad_count = 0
        k4_bad = [False] * len(k4s)
        for k4_idx, e_list in enumerate(k4_edges):
            c_set = set(colors[e] for e in e_list)
            if len(c_set) < 5:
                k4_bad[k4_idx] = True
                bad_count += 1
        colors_used = len(set(colors))
        cost = 1000 * bad_count + colors_used
        color_freq = {}
        for c in colors:
            color_freq[c] = color_freq.get(c, 0) + 1
        return bad_count, k4_bad, colors_used, cost, color_freq

    best_colors = get_initial_colors()
    best_bad_count, best_k4_bad, best_colors_used, best_cost, best_color_freq = init_state(best_colors)
    
    current_colors = best_colors[:]
    current_bad_count = best_bad_count
    current_k4_bad = best_k4_bad[:]
    current_colors_used = best_colors_used
    current_cost = best_cost
    current_color_freq = best_color_freq.copy()
    
    start_time = time.time()
    time_limit = 110.0
    
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    while T > min_T and time.time() - start_time < time_limit:
        e_idx = random.randint(0, num_edges - 1)
        old_color = current_colors[e_idx]
        
        used_colors = [c for c, cnt in current_color_freq.items() if cnt > 0]
        if used_colors:
            if random.random() < 0.8:
                new_color = random.choice(used_colors)
            else:
                new_color = random.randint(0, 20)
        else:
            new_color = random.randint(0, 20)
            
        if new_color == old_color:
            T *= cooling
            continue
            
        # Save state of affected K4s
        affected_k4s = edge_to_k4s[e_idx]
        old_k4_bad_states = [current_k4_bad[k4_idx] for k4_idx in affected_k4s]
        
        # Apply move
        current_colors[e_idx] = new_color
        
        # Update color frequencies
        current_color_freq[old_color] -= 1
        if current_color_freq[old_color] == 0:
            del current_color_freq[old_color]
        current_color_freq[new_color] = current_color_freq.get(new_color, 0) + 1
        
        new_colors_used = len(current_color_freq)
        
        # Update K4 bad counts
        new_bad_count = current_bad_count
        for k4_idx in affected_k4s:
            e_list = k4_edges[k4_idx]
            c_set = set(current_colors[e] for e in e_list)
            is_bad = len(c_set) < 5
            if current_k4_bad[k4_idx] and not is_bad:
                new_bad_count -= 1
            elif not current_k4_bad[k4_idx] and is_bad:
                new_bad_count += 1
            current_k4_bad[k4_idx] = is_bad
            
        new_cost = 1000 * new_bad_count + new_colors_used
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_bad_count = new_bad_count
            current_colors_used = new_colors_used
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_colors = current_colors[:]
                best_bad_count = new_bad_count
                best_k4_bad = current_k4_bad[:]
                best_color_freq = current_color_freq.copy()
                best_colors_used = new_colors_used
        else:
            # Revert
            current_colors[e_idx] = old_color
            current_color_freq[old_color] += 1
            current_color_freq[new_color] -= 1
            if current_color_freq[new_color] == 0:
                del current_color_freq[new_color]
                
            for i, k4_idx in enumerate(affected_k4s):
                current_k4_bad[k4_idx] = old_k4_bad_states[i]
                
        T *= cooling
        
    return best_colors

# EVOLVE-BLOCK-END
