import time
import random
import itertools
import math

def construct():
    deadline = time.time() + 145  # Safety margin for 150s limit
    
    n = 16
    vertices = list(range(n))
    
    # Precompute edges
    edge_list = []
    edge_index = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[(i, j)] = idx
            edge_list.append((i, j))
            idx += 1
    num_edges = len(edge_list) # 120
    
    # Precompute K4s
    k4s = []
    k4_edges = []
    for combo in itertools.combinations(vertices, 4):
        a, b, c, d = combo
        edges = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
        k4s.append(combo)
        k4_edges.append([edge_index[e] for e in edges])
    num_k4s = len(k4_edges) # 1820
    
    # Precompute edge -> k4s mapping
    edge_k4s = [[] for _ in range(num_edges)]
    for k4_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_k4s[e].append(k4_idx)
            
    # Max colors to consider for search
    MAX_COLORS = 20
    
    def init_state(colors):
        """Initialize k4_counts, k4_distinct, total_violations, global_counts, colors_used_count."""
        k4_counts = [[0] * MAX_COLORS for _ in range(num_k4s)]
        k4_distinct = [0] * num_k4s
        global_counts = [0] * MAX_COLORS
        colors_used_count = 0
        
        for e in range(num_edges):
            c = colors[e]
            if c >= MAX_COLORS:
                c = c % MAX_COLORS # Should not happen if we control it
            global_counts[c] += 1
            
        for c in range(MAX_COLORS):
            if global_counts[c] > 0:
                colors_used_count += 1
                
        for k4_idx in range(num_k4s):
            for e in k4_edges[k4_idx]:
                c = colors[e]
                if c >= MAX_COLORS:
                    c = c % MAX_COLORS
                k4_counts[k4_idx][c] += 1
                
            distinct = 0
            for c in range(MAX_COLORS):
                if k4_counts[k4_idx][c] > 0:
                    distinct += 1
            k4_distinct[k4_idx] = distinct
            
        total_violations = 0
        for k4_idx in range(num_k4s):
            if k4_distinct[k4_idx] < 5:
                total_violations += 1
                
        return k4_counts, k4_distinct, total_violations, global_counts, colors_used_count

    def get_cost(total_violations, colors_used_count):
        return total_violations * 1000 + colors_used_count

    def random_coloring():
        return [random.randint(0, MAX_COLORS - 1) for _ in range(num_edges)]

    def seed_coloring():
        # (i+j) % 17
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append((i + j) % 17)
        # Ensure within MAX_COLORS
        return [c % MAX_COLORS for c in colors]

    best_score = float('inf')
    best_colors = None
    best_valid_colors = None # Store best valid coloring specifically
    
    # We will run multiple restarts
    num_restarts = 10
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Initialize
        if restart == 0:
            colors = seed_coloring()
        else:
            colors = random_coloring()
            
        k4_counts, k4_distinct, total_violations, global_counts, colors_used_count = init_state(colors)
        current_cost = get_cost(total_violations, colors_used_count)
        
        if total_violations == 0 and colors_used_count < best_score:
            best_score = colors_used_count
            best_colors = colors[:]
            best_valid_colors = colors[:]
            
        # Simulated Annealing
        T = 10.0
        T_min = 0.1
        cooling_rate = 0.9999
        max_iters = 200000 # Adjust based on time
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Pick random edge
            e = random.randint(0, num_edges - 1)
            c_old = colors[e]
            
            # Pick new color
            # Bias towards existing colors to reduce color count
            # 80% chance pick from used colors, 20% random
            if random.random() < 0.8:
                used = [c for c in range(MAX_COLORS) if global_counts[c] > 0]
                if used:
                    c_new = random.choice(used)
                else:
                    c_new = random.randint(0, MAX_COLORS - 1)
            else:
                c_new = random.randint(0, MAX_COLORS - 1)
                
            if c_new == c_old:
                continue
                
            # Calculate delta
            delta_viol = 0
            affected_k4s = edge_k4s[e]
            
            # Store old distinct counts for rollback if needed? 
            # Or just apply and if rejected, rollback.
            # Since it's fast, apply, check, rollback if needed.
            
            old_distincts = [0] * len(affected_k4s)
            
            for i, k4_idx in enumerate(affected_k4s):
                old_distincts[i] = k4_distinct[k4_idx]
                
                cnt = k4_counts[k4_idx]
                d = k4_distinct[k4_idx]
                
                # Remove c_old
                cnt[c_old] -= 1
                if cnt[c_old] == 0:
                    d -= 1
                    
                # Add c_new
                cnt[c_new] += 1
                if cnt[c_new] == 1:
                    d += 1
                    
                k4_distinct[k4_idx] = d
                
                old_viol = 1 if old_distincts[i] < 5 else 0
                new_viol = 1 if d < 5 else 0
                delta_viol += new_viol - old_viol
                
            # Update global counts
            gc_old = global_counts[c_old]
            gc_new = global_counts[c_new]
            
            global_counts[c_old] -= 1
            global_counts[c_new] += 1
            
            delta_color_count = 0
            if gc_old == 1 and gc_new == 0: # c_old removed, c_new was 0? No, c_new increased.
                # c_old count became 0 -> colors_used_count decreases by 1
                delta_color_count -= 1
            if gc_new == 1: # c_new count became 1 -> colors_used_count increases by 1
                delta_color_count += 1
            # If c_old was >1, no change. If c_new was >0, no change (unless it was 0, handled above).
            # Wait, if c_old was 1 and c_new was 0 (before increment), then:
            # global_counts[c_old] becomes 0.
            # global_counts[c_new] becomes 1.
            # colors_used_count: -1 (for c_old) + 1 (for c_new) = 0.
            # My logic:
            # if gc_old == 1: delta -= 1
            # if gc_new == 0 (before increment, so ==1 after): delta += 1
            # This is correct.
            
            total_violations += delta_viol
            colors_used_count += delta_color_count
            
            colors[e] = c_new
            
            new_cost = get_cost(total_violations, colors_used_count)
            delta_cost = new_cost - current_cost
            
            if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                current_cost = new_cost
                # Accept move
                if total_violations == 0 and colors_used_count < best_score:
                    best_score = colors_used_count
                    best_colors = colors[:]
                    best_valid_colors = colors[:]
            else:
                # Reject move: Rollback
                colors[e] = c_old
                
                for i, k4_idx in enumerate(affected_k4s):
                    cnt = k4_counts[k4_idx]
                    d = k4_distinct[k4_idx]
                    
                    # Undo add c_new
                    cnt[c_new] -= 1
                    if cnt[c_new] == 0:
                        d -= 1
                        
                    # Undo remove c_old
                    cnt[c_old] += 1
                    if cnt[c_old] == 1:
                        d += 1
                        
                    k4_distinct[k4_idx] = d
                    
                global_counts[c_old] += 1
                global_counts[c_new] -= 1
                total_violations -= delta_viol
                colors_used_count -= delta_color_count
                
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
        # End SA loop for this restart
        
    # Final check
    if best_valid_colors is not None:
        # Verify best_valid_colors
        # Quick check
        k4_counts, k4_distinct, total_violations, global_counts, colors_used_count = init_state(best_valid_colors)
        if total_violations == 0:
            return best_valid_colors
        else:
            # Fallback to seed if best is invalid (should not happen)
            return seed_coloring()
    else:
        # If no valid coloring found, return seed (which is valid)
        return seed_coloring()
