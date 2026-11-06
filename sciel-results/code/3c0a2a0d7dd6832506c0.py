import itertools
import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 16
    vertices = list(range(n))
    subsets_4 = list(itertools.combinations(vertices, 4))
    num_subsets = len(subsets_4)
    
    edges = [(i,j) for i in range(n) for j in range(i+1, n)]
    num_edges = len(edges)
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    subset_edges = []
    edge_to_subsets = [[] for _ in range(num_edges)]
    
    for s_idx, s in enumerate(subsets_4):
        a,b,c,d = s
        e_indices = [
            edge_index[(a,b)], edge_index[(a,c)], edge_index[(a,d)],
            edge_index[(b,c)], edge_index[(b,d)], edge_index[(c,d)]
        ]
        subset_edges.append(e_indices)
        for e_idx in e_indices:
            edge_to_subsets[e_idx].append(s_idx)
            
    def count_colors(colours, e_indices):
        # Returns number of distinct colors in the 6 edges
        c1 = colours[e_indices[0]]
        c2 = colours[e_indices[1]]
        c3 = colours[e_indices[2]]
        c4 = colours[e_indices[3]]
        c5 = colours[e_indices[4]]
        c6 = colours[e_indices[5]]
        return len({c1,c2,c3,c4,c5,c6})

    def is_valid(colours):
        for s_idx in range(num_subsets):
            if count_colors(colours, subset_edges[s_idx]) < 5:
                return False
        return True

    # 1. Try simple additive structures
    for m in range(13, 17):
        if time.time() > deadline:
            break
        colours = [(i+j) % m for i in range(n) for j in range(i+1, n)]
        if is_valid(colours):
            return colours

    # 2. Try Geometry-based structure (Affine Plane AG(2,4))
    # Map 16 vertices to F_4^2
    # F_4 elements: 0, 1, w, w+1 (where w^2 + w + 1 = 0)
    # We can represent F_4 as {0, 1, 2, 3} with custom addition/multiplication
    # Or just use integers 0..3 and define operations.
    # Let's use a lookup table for F_4 arithmetic.
    # 0 is 0. 1 is 1. 2 is w. 3 is w+1.
    # w^2 = w+1 (3). w^3 = 1.
    # Add:
    # 0+x=x
    # 1+1=0
    # 1+2=3
    # 1+3=2
    # 2+2=0
    # 2+3=1
    # 3+3=0
    
    add_table = np.array([
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0]
    ])
    
    # Mult:
    # 0*x=0
    # 1*x=x
    # 2*1=2
    # 2*2=3
    # 2*3=2* (w+1) = w^2+w = (w+1)+w = 1
    # 3*1=3
    # 3*2=1
    # 3*3=(w+1)^2 = w^2+2w+1 = (w+1)+0+1 = w = 2
    
    mult_table = np.array([
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2]
    ])
    
    def f4_add(a, b):
        return add_table[a, b]
        
    def f4_sub(a, b):
        # In F_4, -1 = 1 (char 2), so a-b = a+b
        return add_table[a, b]
        
    def f4_mul(a, b):
        return mult_table[a, b]
        
    def f4_inv(a):
        if a == 0: return 0
        if a == 1: return 1
        if a == 2: return 3 # 2*3 = 1
        if a == 3: return 2 # 3*2 = 1
        return 0

    # Vertices are pairs (x, y) in F_4^2.
    # We map index i (0..15) to (i // 4, i % 4).
    # This is a bijection.
    
    # Slopes:
    # For two distinct points P1=(x1,y1), P2=(x2,y2) in F_4^2:
    # If x1 != x2, slope m = (y2-y1) / (x2-x1) in F_4. (4 values)
    # If x1 == x2, slope is "infinity" (1 value).
    # Total 5 slopes.
    
    # But we need to minimize colors. 5 colors is the lower bound?
    # Wait, the problem says "Minimize the number of colours used".
    # If we use 5 colors, does every K4 see 5 colors?
    # Let's check a K4 in AG(2,4).
    # If the 4 points form a parallelogram (affine plane structure), the 6 edges have slopes:
    # P1=(0,0), P2=(1,0), P3=(0,1), P4=(1,1)
    # (0,0)-(1,0): dy=0, dx=1 -> m=0
    # (0,0)-(0,1): dx=0 -> inf
    # (0,0)-(1,1): dy=1, dx=1 -> m=1
    # (1,0)-(0,1): dy=1, dx=1 (sub: 0-1=1, 1-0=1) -> m=1
    # (1,0)-(1,1): dx=0 -> inf
    # (0,1)-(1,1): dy=0, dx=1 -> m=0
    # Colors used: {0, inf, 1}. Only 3 colors!
    # So 5 colors is NOT enough if we just use slope.
    # We need more colors or a different mapping.
    
    # However, the task says "Minimize the number of colours used".
    # The lower bound is 13.
    # So 5 colors is invalid.
    
    # Let's stick to the search approach but maybe use the geometry to guide initial states?
    # Or just improve the local search.
    
    # Let's try a better local search with Simulated Annealing or just more restarts.
    # The current local search is pure hill climbing with random restarts.
    
    def local_search(m, max_time=5.0):
        start_ls = time.time()
        best_viol = float('inf')
        best_colours = None
        
        # Increase restarts if time permits
        max_restarts = 1000
        for restart in range(max_restarts):
            if time.time() - start_ls > max_time:
                break
            if time.time() > deadline:
                break
                
            colours = [random.randint(0, m-1) for _ in range(num_edges)]
            
            # Calculate initial violations
            viol = [0] * num_subsets
            total_viol = 0
            for s_idx in range(num_subsets):
                e_idx_list = subset_edges[s_idx]
                if count_colors(colours, e_idx_list) < 5:
                    viol[s_idx] = 1
                    total_viol += 1
                    
            if total_viol == 0:
                return colours
                
            if total_viol < best_viol:
                best_viol = total_viol
                best_colours = colours[:]
                
            # Simple Hill Climbing
            improved = True
            stall_counter = 0
            # Allow some iterations per restart
            max_iter_per_restart = 1000 
            while improved and stall_counter < max_iter_per_restart:
                improved = False
                stall_counter += 1
                if time.time() - start_ls > max_time:
                    break
                    
                idx = random.randint(0, num_edges-1)
                old_c = colours[idx]
                
                affected_subsets = edge_to_subsets[idx]
                
                min_new_total = total_viol
                best_new_c = old_c
                
                # Try all m colors
                for new_c in range(m):
                    if new_c == old_c:
                        continue
                    
                    colours[idx] = new_c
                    
                    new_total = total_viol
                    for s_idx in affected_subsets:
                        was_viol = viol[s_idx]
                        e_idx_list = subset_edges[s_idx]
                        is_viol = 1 if count_colors(colours, e_idx_list) < 5 else 0
                        new_total += (is_viol - was_viol)
                        
                    if new_total < min_new_total:
                        min_new_total = new_total
                        best_new_c = new_c
                        
                colours[idx] = old_c
                
                if min_new_total < total_viol:
                    colours[idx] = best_new_c
                    for s_idx in affected_subsets:
                        was_viol = viol[s_idx]
                        e_idx_list = subset_edges[s_idx]
                        is_viol = 1 if count_colors(colours, e_idx_list) < 5 else 0
                        viol[s_idx] = is_viol
                        total_viol += (is_viol - was_viol)
                    
                    improved = True
                    if total_viol < best_viol:
                        best_viol = total_viol
                        best_colours = colours[:]
                    if total_viol == 0:
                        return colours
                else:
                    # If no improvement, break this restart's hill climbing
                    # This is a "first improvement" or "best improvement" that stops on stall.
                    # Actually, the original code broke immediately on no improvement.
                    # Let's try a bit more randomization: if no improvement, accept a random move with small prob?
                    # Or just restart.
                    break

        return best_colours if best_viol == 0 else None

    # Try 13, 14, 15, 16 with local search
    for m in [13, 14, 15, 16]:
        if time.time() > deadline:
            break
        # Give more time to lower m?
        # 13 is harder.
        time_alloc = 10.0 if m <= 14 else 5.0
        res = local_search(m, max_time=time_alloc)
        if res is not None:
            return res

    # Fallback to 17
    return [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
