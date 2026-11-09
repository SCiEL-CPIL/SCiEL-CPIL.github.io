import random
import time
import sys

def construct():
    n = 16
    num_edges = 120
    
    # Map (i, j) to index
    edges = []
    edge_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
            edge_to_idx[(i, j)] = idx
            idx += 1
            
    # Precompute K4s
    k4s = []
    k4_edge_indices = []
    
    # Generate all combinations of 4 vertices
    for v0 in range(n):
        for v1 in range(v0 + 1, n):
            for v2 in range(v1 + 1, n):
                for v3 in range(v2 + 1, n):
                    verts = [v0, v1, v2, v3]
                    e_indices = []
                    for a in range(4):
                        for b in range(a + 1, 4):
                            e = (verts[a], verts[b])
                            e_indices.append(edge_to_idx[e])
                    k4s.append(verts)
                    k4_edge_indices.append(e_indices)
                    
    num_k4s = len(k4s)
    
    # Map edge index to list of k4 indices containing it
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx in range(num_k4s):
        for e_idx in k4_edge_indices[k4_idx]:
            edge_to_k4s[e_idx].append(k4_idx)

    def is_valid_k4_colors(colors_list):
        return len(set(colors_list)) >= 5

    def get_k4_colors(k4_idx, colors):
        e_indices = k4_edge_indices[k4_idx]
        return [colors[e] for e in e_indices]

    def count_bad_k4s(colors):
        bad = 0
        for k4_idx in range(num_k4s):
            if not is_valid_k4_colors(get_k4_colors(k4_idx, colors)):
                bad += 1
        return bad

    def walk_sat(K, time_limit):
        # Initialize random colors
        colors = [random.randint(0, K - 1) for _ in range(num_edges)]
        
        # Precompute initial bad status for each K4
        bad_status = [False] * num_k4s
        total_bad = 0
        for k4_idx in range(num_k4s):
            if not is_valid_k4_colors(get_k4_colors(k4_idx, colors)):
                bad_status[k4_idx] = True
                total_bad += 1
                
        if total_bad == 0:
            return colors

        start_time = time.time()
        bad_k4_list = [i for i in range(num_k4s) if bad_status[i]]
        steps = 0
        
        while time.time() - start_time < time_limit:
            if total_bad == 0:
                return colors
            
            steps += 1
            
            # Filter list every 100 steps to remove stale entries
            if steps % 100 == 0:
                bad_k4_list = [i for i in range(num_k4s) if bad_status[i]]
                if not bad_k4_list:
                    return colors
            
            if not bad_k4_list:
                return colors
                
            k4_idx = bad_k4_list[random.randint(0, len(bad_k4_list) - 1)]
            
            # If it became good since last filter, skip
            if not bad_status[k4_idx]:
                continue
            
            e_indices = k4_edge_indices[k4_idx]
            e_idx = e_indices[random.randint(0, 5)]
            
            old_color = colors[e_idx]
            
            min_delta = float('inf')
            candidates = []
            
            for new_color in range(K):
                if new_color == old_color:
                    continue
                    
                delta = 0
                for k4_j in edge_to_k4s[e_idx]:
                    was_bad = bad_status[k4_j]
                    
                    # Check if new color makes it valid/invalid
                    c_list = []
                    for e in k4_edge_indices[k4_j]:
                        if e == e_idx:
                            c_list.append(new_color)
                        else:
                            c_list.append(colors[e])
                            
                    is_now_bad = not is_valid_k4_colors(c_list)
                    
                    if was_bad and not is_now_bad:
                        delta -= 1
                    elif not was_bad and is_now_bad:
                        delta += 1
                        
                if delta < min_delta:
                    min_delta = delta
                    candidates = [new_color]
                elif delta == min_delta:
                    candidates.append(new_color)
            
            new_color = candidates[random.randint(0, len(candidates) - 1)]
            
            # Apply the move
            colors[e_idx] = new_color
            
            # Update bad_status and total_bad
            for k4_j in edge_to_k4s[e_idx]:
                c_list = get_k4_colors(k4_j, colors)
                new_is_bad = not is_valid_k4_colors(c_list)
                old_is_bad = bad_status[k4_j]
                
                if old_is_bad and not new_is_bad:
                    total_bad -= 1
                elif not old_is_bad and new_is_bad:
                    total_bad += 1
                else:
                    pass
                bad_status[k4_j] = new_is_bad
            
        return None

    # Fallback solution (17 colors)
    fallback = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    result = None
    # Try K=13, 14, 15
    # 13 is the target.
    # WalkSAT is fast, so we can try 13 for a while.
    
    for K in [13, 14, 15]:
        t_limit = 40 if K == 13 else 20
        result = walk_sat(K, t_limit)
        if result:
            # Verify
            if count_bad_k4s(result) == 0:
                return result
            
    return fallback
