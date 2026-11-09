import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    n = 16
    edges = []
    edge_index = {}
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[(i, j)] = len(edges)
            edges.append((i, j))
    
    num_edges = len(edges) # 120
    
    k4_edges_list = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    e1 = edge_index[(a, b)]
                    e2 = edge_index[(a, c)]
                    e3 = edge_index[(a, d)]
                    e4 = edge_index[(b, c)]
                    e5 = edge_index[(b, d)]
                    e6 = edge_index[(c, d)]
                    k4_edges_list.append([e1, e2, e3, e4, e5, e6])
    
    num_k4s = len(k4_edges_list) # 1820
    
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_list in enumerate(k4_edges_list):
        for e in e_list:
            edge_to_k4s[e].append(k4_idx)

    def get_seed_17():
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

    def compress_colors(colors_17, k):
        return [c % k for c in colors_17]

    def solve_fast(k, time_limit, init_colors=None):
        end_time = time.time() + time_limit
        
        if init_colors is None:
            init_colors = [random.randint(0, k - 1) for _ in range(num_edges)]
        
        colours = init_colors[:]
        
        # Initialize state
        # k4_counts[k4_idx][color] = count
        k4_counts = [[0]*k for _ in range(num_k4s)]
        k4_distinct = [0] * num_k4s
        is_bad = [False] * num_k4s
        bad_k4_indices = []
        bad_k4_pos = [-1] * num_k4s
        
        for k4_idx, e_list in enumerate(k4_edges_list):
            for e in e_list:
                k4_counts[k4_idx][colours[e]] += 1
            
            distinct = 0
            for c in range(k):
                if k4_counts[k4_idx][c] > 0:
                    distinct += 1
            k4_distinct[k4_idx] = distinct
            if distinct < 5:
                is_bad[k4_idx] = True
                bad_k4_indices.append(k4_idx)
                bad_k4_pos[k4_idx] = len(bad_k4_indices) - 1
        
        if not bad_k4_indices:
            return colours

        T = 1.0
        cool_rate = 0.999
        min_T = 0.001
        no_improve = 0
        max_no_improve = 10000
        best_bad_count = len(bad_k4_indices)
        best_colours = colours[:]

        while time.time() < end_time and bad_k4_indices:
            if T < min_T:
                break
            
            # Pick a random bad K4
            k4_idx = random.choice(bad_k4_indices)
            e_list = k4_edges_list[k4_idx]
            e_idx = random.choice(e_list)
            
            old_c = colours[e_idx]
            new_c = random.randint(0, k - 1)
            if new_c == old_c:
                continue
            
            affected = edge_to_k4s[e_idx]
            delta = 0
            changes = [] # (ak, old_distinct, new_distinct)
            
            # Simulate changes
            for ak in affected:
                cnts = k4_counts[ak]
                old_dist = k4_distinct[ak]
                
                # Apply
                cnts[old_c] -= 1
                cnts[new_c] += 1
                
                new_dist = old_dist
                if cnts[old_c] == 0:
                    new_dist -= 1
                if cnts[new_c] == 1:
                    new_dist += 1
                
                changes.append((ak, old_dist, new_dist))
                
                # Revert for now
                cnts[old_c] += 1
                cnts[new_c] -= 1
                
                # Update delta
                was_bad = is_bad[ak]
                is_bad_new = (new_dist < 5)
                
                if was_bad and not is_bad_new:
                    delta -= 1
                elif not was_bad and is_bad_new:
                    delta += 1
            
            # Acceptance
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = random.random() < 0.5
            else:
                if random.random() < math.exp(-delta / T):
                    accept = True
            
            if accept:
                colours[e_idx] = new_c
                for ak, old_dist, new_dist in changes:
                    cnts = k4_counts[ak]
                    cnts[old_c] -= 1
                    cnts[new_c] += 1
                    
                    k4_distinct[ak] = new_dist
                    
                    was_bad = is_bad[ak]
                    is_bad_new = (new_dist < 5)
                    
                    if was_bad and not is_bad_new:
                        # Remove from bad list
                        is_bad[ak] = False
                        pos = bad_k4_pos[ak]
                        last_idx = bad_k4_indices[-1]
                        bad_k4_indices[pos] = last_idx
                        bad_k4_pos[last_idx] = pos
                        bad_k4_indices.pop()
                        bad_k4_pos[ak] = -1
                    elif not was_bad and is_bad_new:
                        # Add to bad list
                        is_bad[ak] = True
                        bad_k4_indices.append(ak)
                        bad_k4_pos[ak] = len(bad_k4_indices) - 1
                
                if delta < 0:
                    no_improve = 0
                    if len(bad_k4_indices) < best_bad_count:
                        best_bad_count = len(bad_k4_indices)
                        best_colours = colours[:]
                else:
                    no_improve += 1
            else:
                no_improve += 1
            
            T *= cool_rate
            
            if no_improve > max_no_improve:
                break
        
        if not bad_k4_indices:
            return colours
        return None

    # Strategy:
    # 1. Try k=13.
    # 2. Try k=14.
    # 3. Try k=15.
    # 4. Fallback to 17 (known valid).
    
    seed_17 = get_seed_17()
    
    # Try 13
    if time.time() < deadline - 40:
        # Random init for 13 might be better than compressed 17?
        # Compressed 17 to 13: 0..12, 13->0, 14->1, 15->2, 16->3.
        init_13 = compress_colors(seed_17, 13)
        res = solve_fast(13, 20, init_13)
        if res is not None:
            return res

    # Try 14
    if time.time() < deadline - 20:
        init_14 = compress_colors(seed_17, 14)
        res = solve_fast(14, 20, init_14)
        if res is not None:
            return res

    # Try 15
    if time.time() < deadline - 10:
        init_15 = compress_colors(seed_17, 15)
        res = solve_fast(15, 10, init_15)
        if res is not None:
            return res

    # Fallback
    return seed_17
