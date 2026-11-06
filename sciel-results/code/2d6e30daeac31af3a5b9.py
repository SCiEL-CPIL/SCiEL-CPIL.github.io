import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 16
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    # Precompute K4s
    k4s = list(combinations(range(n), 4))
    k4_edge_indices = []
    for quad in k4s:
        e_indices = []
        for i in range(4):
            for j in range(i + 1, 4):
                e_indices.append(edge_index[(quad[i], quad[j])])
        k4_edge_indices.append(e_indices)
        
    num_k4s = len(k4s)
    
    # Map edge to K4s
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_indices in enumerate(k4_edge_indices):
        for e_idx in e_indices:
            edge_to_k4s[e_idx].append(k4_idx)
            
    def get_coloring_15():
        """
        Construct a valid 15-coloring using the group Z_3 x Z_3 x Z_3 x Z_3.
        Vertices are (a,b,c,d) in {0,1,2}^4.
        Edge color is (a+b, c+d) mod 3.
        This uses 3*3=9 colors. Wait, 9 colors?
        Let's check the property.
        If vertices are v1=(a1,b1,c1,d1), etc.
        Colors are (a_i+a_j, c_i+c_j).
        In a K4, can we have < 5 colors?
        If we take a rectangle in the (a,c) plane, e.g. (0,0), (1,0), (0,1), (1,1).
        Colors: (1,1), (0,1), (1,0), (2,2), (1,1), (0,1).
        Distinct: {(1,1), (0,1), (1,0), (2,2)}. Only 4 colors!
        So this construction fails.
        
        We need a construction that guarantees >= 5 colors.
        The seed uses (i+j) % 17.
        Let's try to find a known valid 15-coloring or use a different group.
        
        Actually, let's just use the seed's fallback if it's valid, or find a better one.
        The seed's fallback is (i+j) % 17.
        Let's check if (i+j) % 17 is valid.
        Vertices 0..15.
        Edge (i,j) color (i+j) % 17.
        K4: {0, 1, 2, 3}.
        Edges: (0,1)->1, (0,2)->2, (0,3)->3, (1,2)->3, (1,3)->4, (2,3)->5.
        Colors: {1, 2, 3, 4, 5}. 5 distinct.
        
        Try {0, 1, 2, 4}.
        (0,1)->1, (0,2)->2, (0,4)->4, (1,2)->3, (1,4)->5, (2,4)->6.
        Colors: {1, 2, 3, 4, 5, 6}. 6 distinct.
        
        Try {0, 1, 3, 4}.
        (0,1)->1, (0,3)->3, (0,4)->4, (1,3)->4, (1,4)->5, (3,4)->7.
        Colors: {1, 3, 4, 5, 7}. 5 distinct.
        
        Try {0, 2, 4, 6}.
        (0,2)->2, (0,4)->4, (0,6)->6, (2,4)->6, (2,6)->8, (4,6)->10.
        Colors: {2, 4, 6, 8, 10}. 5 distinct.
        
        It seems (i+j) % 17 is a valid 17-coloring.
        We want to minimize colors. 13 <= f(16) <= 15.
        If we can find a 15-coloring, we can try to reduce it.
        
        Let's use a random 15-coloring initialized by the seed's logic but mapped to 15 colors,
        or just use a random valid 15-coloring if we can find one quickly.
        
        Actually, the seed program's fallback is valid for 17 colors.
        Can we map 17 colors to 15?
        If we merge colors, we might violate the condition.
        
        Let's try a different constructive approach.
        Use Z_5 x Z_3? 15 colors.
        Vertices in Z_5 x Z_3. 15 vertices. We have 16.
        
        Let's just use a random 15-coloring and repair it.
        Or, use the 17-coloring and try to merge colors.
        """
        # Let's start with the seed's 17-coloring and merge colors 15, 16 into 0..14?
        # No, that's risky.
        
        # Let's try to generate a valid 15-coloring via random restarts with a very tight constraint.
        # Or, use the fact that (i+j) % 17 is valid.
        # If we take (i+j) % 15, is it valid?
        # {0, 1, 2, 3} -> {1, 2, 3, 3, 4, 5} -> {1,2,3,4,5}. Valid.
        # {0, 1, 3, 4} -> {1, 3, 4, 4, 5, 7} -> {1,3,4,5,7}. Valid.
        # {0, 2, 4, 6} -> {2, 4, 6, 6, 8, 10} -> {2,4,6,8,10}. Valid.
        
        # Let's check if (i+j) % 15 is always valid.
        # If yes, we have a 15-coloring.
        # Then we can try to reduce it to 14 or 13.
        
        colors = [(i + j) % 15 for i in range(16) for j in range(i + 1, 16)]
        # Verify validity
        for k4_idx in range(num_k4s):
            e_indices = k4_edge_indices[k4_idx]
            distinct = len(set(colors[ei] for ei in e_indices))
            if distinct < 5:
                return None # Fallback if not valid
        
        return colors

    def run_palette_reduction(initial_colors, K_start, time_limit):
        """
        Start with a valid K_start coloring.
        Try to merge color classes to reduce K.
        """
        colors = initial_colors[:]
        current_K = K_start
        
        # State for validity check
        k4_counts = [[0] * current_K for _ in range(num_k4s)]
        k4_distinct = [0] * num_k4s
        
        for k4_idx in range(num_k4s):
            e_indices = k4_edge_indices[k4_idx]
            for ei in e_indices:
                c = colors[ei]
                k4_counts[k4_idx][c] += 1
            k4_distinct[k4_idx] = sum(1 for c in range(current_K) if k4_counts[k4_idx][c] > 0)
            
        # Check if valid
        if any(d < 5 for d in k4_distinct):
            return None, current_K
            
        best_colors = colors[:]
        best_K = current_K
        
        while time.time() < time_limit and current_K > 13:
            # Try to merge a color into another
            # Pick a random color c1 to eliminate
            c1 = random.randint(0, current_K - 1)
            # Pick a random target color c2 != c1
            c2 = random.randint(0, current_K - 1)
            while c2 == c1:
                c2 = random.randint(0, current_K - 1)
                
            # Find all edges with color c1
            edges_c1 = [ei for ei in range(num_edges) if colors[ei] == c1]
            if not edges_c1:
                # Color c1 is not used. Just decrement K.
                # This shouldn't happen if we manage K correctly, but just in case.
                current_K -= 1
                continue
                
            # Check if merging c1 into c2 keeps validity
            # We need to check if any K4 that has an edge of color c1 will drop below 5 distinct colors.
            # A K4 is affected if it has at least one edge of color c1.
            # For such a K4, if the number of edges with color c1 is equal to the number of edges with color c2?
            # No, if c1 is the only edge of its color, and c2 is the only edge of its color, merging them reduces distinct count by 1.
            
            valid_merge = True
            # To be fast, only check K4s that contain an edge of color c1.
            affected_k4s = set()
            for ei in edges_c1:
                for k4_idx in edge_to_k4s[ei]:
                    affected_k4s.add(k4_idx)
                    
            for k4_idx in affected_k4s:
                # Count distinct colors in this K4
                # If c1 count > 0 and c2 count > 0, distinct might drop by 1.
                # If c1 count > 0 and c2 count == 0, distinct stays same.
                # If c1 count == 0, not affected.
                
                # We know c1 count > 0 for these K4s.
                if k4_counts[k4_idx][c2] > 0:
                    # If we merge, c1 and c2 become one color.
                    # Distinct count decreases by 1.
                    if k4_distinct[k4_idx] <= 5:
                        # If it was 5, it becomes 4. Invalid.
                        valid_merge = False
                        break
                        
            if valid_merge:
                # Apply merge
                for ei in edges_c1:
                    colors[ei] = c2
                    for k4_idx in edge_to_k4s[ei]:
                        k4_counts[k4_idx][c1] -= 1
                        k4_counts[k4_idx][c2] += 1
                        if k4_counts[k4_idx][c1] == 0:
                            k4_distinct[k4_idx] -= 1
                            
                # Remove color c1 from the palette
                # We don't need to physically remove it from k4_counts if we just ignore it,
                # but for simplicity, we can just decrement current_K and remap?
                # No, remapping is expensive.
                # Let's just keep the array size and mark c1 as "dead".
                # But then we might pick a dead color again.
                # Better: remap colors.
                
                # Remap: all colors > c1 shift down by 1.
                # This is O(E).
                for ei in range(num_edges):
                    if colors[ei] > c1:
                        colors[ei] -= 1
                        
                # Update k4_counts
                new_k4_counts = [[0] * (current_K - 1) for _ in range(num_k4s)]
                new_k4_distinct = [0] * num_k4s
                for k4_idx in range(num_k4s):
                    for c in range(current_K - 1):
                        if c < c1:
                            new_k4_counts[k4_idx][c] = k4_counts[k4_idx][c]
                        else:
                            new_k4_counts[k4_idx][c] = k4_counts[k4_idx][c + 1]
                    new_k4_distinct[k4_idx] = sum(1 for c in range(current_K - 1) if new_k4_counts[k4_idx][c] > 0)
                    
                k4_counts = new_k4_counts
                k4_distinct = new_k4_distinct
                current_K -= 1
                
                if current_K < best_K:
                    best_K = current_K
                    best_colors = colors[:]
                    
            else:
                # Try a different pair
                pass # Loop continues

        return best_colors, best_K

    # Strategy:
    # 1. Get a valid 15-coloring.
    # 2. Try to reduce it to 14, then 13.
    
    # First, check if (i+j) % 15 is valid.
    colors_15 = get_coloring_15()
    
    if colors_15:
        # Try to reduce from 15
        time_left = deadline - time.time() - 5
        if time_left > 5:
            # Split time between trying to reduce 15->14 and 14->13
            # Or just run the reduction loop which handles both.
            res_colors, res_K = run_palette_reduction(colors_15, 15, time_left)
            if res_colors:
                return res_colors
    else:
        # Fallback to seed's 17-coloring logic if 15 fails?
        # But we want to minimize.
        # If 15 is not valid, we need another 15-coloring.
        # Let's try a random 15-coloring with local search to fix violations.
        # This is the original strategy but starting from 15.
        
        def run_search(K, time_limit):
            colors = [random.randint(0, K - 1) for _ in range(num_edges)]
            k4_counts = [[0] * K for _ in range(num_k4s)]
            k4_distinct = [0] * num_k4s
            cost = 0
            
            for k4_idx in range(num_k4s):
                e_indices = k4_edge_indices[k4_idx]
                for ei in e_indices:
                    c = colors[ei]
                    k4_counts[k4_idx][c] += 1
                distinct = 0
                for c in range(K):
                    if k4_counts[k4_idx][c] > 0:
                        distinct += 1
                k4_distinct[k4_idx] = distinct
                if distinct < 5:
                    cost += 1
                    
            best_cost = cost
            best_colors = colors[:]
            
            no_improve_count = 0
            max_no_improve = 5000
            
            while time.time() < time_limit:
                if cost == 0:
                    return colors[:]
                    
                violating = [k4_idx for k4_idx in range(num_k4s) if k4_distinct[k4_idx] < 5]
                if not violating:
                    return colors[:]
                    
                k4_idx = random.choice(violating)
                e_indices = k4_edge_indices[k4_idx]
                
                best_delta = 1
                best_e = -1
                best_c = -1
                
                for ei in e_indices:
                    old_c = colors[ei]
                    for new_c in range(K):
                        if new_c == old_c:
                            continue
                        delta = 0
                        for k4_i in edge_to_k4s[ei]:
                            old_dist = k4_distinct[k4_i]
                            old_count = k4_counts[k4_i][old_c]
                            new_count = k4_counts[k4_i][new_c]
                            
                            new_dist = old_dist
                            if old_count == 1:
                                new_dist -= 1
                            if new_count == 0:
                                new_dist += 1
                            
                            if old_dist < 5 and new_dist >= 5:
                                delta -= 1
                            elif old_dist >= 5 and new_dist < 5:
                                delta += 1
                                
                        if delta < best_delta:
                            best_delta = delta
                            best_e = ei
                            best_c = new_c
                            if delta <= -2:
                                break
                    if best_delta <= -2:
                        break
                        
                if best_e >= 0 and best_delta < 0:
                    old_c = colors[best_e]
                    for k4_i in edge_to_k4s[best_e]:
                        k4_counts[k4_i][old_c] -= 1
                        k4_counts[k4_i][best_c] += 1
                        old_dist = k4_distinct[k4_i]
                        new_dist = old_dist
                        if k4_counts[k4_i][old_c] == 0:
                            new_dist -= 1
                        if k4_counts[k4_i][best_c] == 1:
                            new_dist += 1
                        k4_distinct[k4_i] = new_dist
                        
                    colors[best_e] = best_c
                    cost += best_delta
                    no_improve_count = 0
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_colors = colors[:]
                else:
                    no_improve_count += 1
                    if no_improve_count % 100 == 0:
                        for _ in range(3):
                            e_idx = random.randint(0, num_edges - 1)
                            new_color = random.randint(0, K - 1)
                            if new_color != colors[e_idx]:
                                old_color = colors[e_idx]
                                for k4_idx2 in edge_to_k4s[e_idx]:
                                    k4_counts[k4_idx2][old_color] -= 1
                                    k4_counts[k4_idx2][new_color] += 1
                                    old_dist = k4_distinct[k4_idx2]
                                    new_dist = old_dist
                                    if k4_counts[k4_idx2][old_color] == 0:
                                        new_dist -= 1
                                    if k4_counts[k4_idx2][new_color] == 1:
                                        new_dist += 1
                                    k4_distinct[k4_idx2] = new_dist
                                colors[e_idx] = new_color
                        cost = sum(1 for k4_idx2 in range(num_k4s) if k4_distinct[k4_idx2] < 5)
                        
                    if no_improve_count > max_no_improve:
                        break
                        
            return best_colors if best_cost == 0 else None

        # Try to find a valid 15-coloring
        limit = deadline - time.time() - 5
        if limit > 5:
            res = run_search(15, limit)
            if res:
                # Now try to reduce it
                time_left = deadline - time.time() - 5
                if time_left > 5:
                    res_colors, res_K = run_palette_reduction(res, 15, time_left)
                    if res_colors:
                        return res_colors
                return res
                
    # Fallback
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
