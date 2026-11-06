import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit (10s buffer)
    
    n = 16
    edges = []
    edge_index = {}
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[(i, j)] = len(edges)
            edges.append((i, j))
            
    num_edges = len(edges)
    
    # Pre-compute K4s and their edges
    k4_edges = [] # list of list of edge indices
    for comb in itertools.combinations(range(n), 4):
        v0, v1, v2, v3 = comb
        e_indices = [
            edge_index[(v0, v1)],
            edge_index[(v0, v2)],
            edge_index[(v0, v3)],
            edge_index[(v1, v2)],
            edge_index[(v1, v3)],
            edge_index[(v2, v3)]
        ]
        k4_edges.append(e_indices)
        
    num_k4s = len(k4_edges)
    
    # For each edge, list of K4s it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_list in enumerate(k4_edges):
        for e_idx in e_list:
            edge_to_k4s[e_idx].append(k4_idx)
            
    # Initial colors: Seed (17 colors)
    # (u + v) % 17 is a known valid construction for this problem
    colors = [0] * num_edges
    for i, (u, v) in enumerate(edges):
        colors[i] = (u + v) % 17
        
    # Verify seed is valid (sanity check)
    # for k4_idx in range(num_k4s):
    #     e_list = k4_edges[k4_idx]
    #     if len({colors[e] for e in e_list}) < 5:
    #         raise ValueError("Seed construction invalid")

    best_colors = colors[:]
    best_num_colors = max(colors) + 1
    
    # Helper to compact colors to 0..k-1
    def compact_colors(cols):
        unique_c = sorted(set(cols))
        if len(unique_c) <= 1:
            return cols[:]
        mapping = {old: new for new, old in enumerate(unique_c)}
        return [mapping[c] for c in cols]
    
    # Fast validity check for changing one edge
    def can_change_edge(e_idx, new_color, colors):
        old_color = colors[e_idx]
        if old_color == new_color:
            return True
        
        for k4_idx in edge_to_k4s[e_idx]:
            e_list = k4_edges[k4_idx]
            # Calculate colors of other 5 edges
            other_colors = set()
            for e in e_list:
                if e != e_idx:
                    other_colors.add(colors[e])
            
            # The set of colors in this K4 will be other_colors U {new_color}
            # It is valid if len(other_colors U {new_color}) >= 5
            
            # Optimization:
            # If new_color in other_colors: size = len(other_colors)
            # Else: size = len(other_colors) + 1
            
            if new_color in other_colors:
                if len(other_colors) < 5:
                    return False
            else:
                if len(other_colors) < 4:
                    return False
        return True

    # Local Search Loop
    # We use a greedy descent with random kicks to escape local minima.
    # The objective is to minimize the number of distinct colors.
    
    no_improve_count = 0
    current_best = best_num_colors
    
    # Pre-compute for each edge the count of K4s it belongs to (for targeted kicks)
    edge_k4_count = [len(k4s) for k4s in edge_to_k4s]
    
    # Helper: try a 2-opt move (swap colors of two edges)
    def try_two_edge_swap(e1, e2, colors):
        """Try swapping colors of edges e1 and e2. Returns True if valid and applied."""
        c1 = colors[e1]
        c2 = colors[e2]
        if c1 == c2:
            return False
        
        # Check if swapping is valid
        # For each K4 containing e1 (but not e2), check if replacing c1 with c2 keeps >= 5 colors
        # For each K4 containing e2 (but not e1), check if replacing c2 with c1 keeps >= 5 colors
        # For K4s containing both e1 and e2, the colors are the same after swap
        
        for k4_idx in edge_to_k4s[e1]:
            e_list = k4_edges[k4_idx]
            other_colors = set()
            for e in e_list:
                if e == e1:
                    other_colors.add(c2)  # e1 gets c2 after swap
                elif e == e2:
                    other_colors.add(c1)  # e2 gets c1 after swap
                else:
                    other_colors.add(colors[e])
            if len(other_colors) < 5:
                return False
        
        for k4_idx in edge_to_k4s[e2]:
            e_list = k4_edges[k4_idx]
            other_colors = set()
            for e in e_list:
                if e == e1:
                    other_colors.add(c2)
                elif e == e2:
                    other_colors.add(c1)
                else:
                    other_colors.add(colors[e])
            if len(other_colors) < 5:
                return False
        
        # Swap is valid
        colors[e1], colors[e2] = c2, c1
        return True
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Single-edge descent - try to lower colors
        for _ in range(300):
            if time.time() > deadline:
                break
            e_idx = random.randint(0, num_edges - 1)
            current_c = colors[e_idx]
            
            min_valid = -1
            for c in range(0, current_c):
                if can_change_edge(e_idx, c, colors):
                    min_valid = c
                    break
            
            if min_valid != -1:
                colors[e_idx] = min_valid
                improved = True
        
        # Phase 2: Two-edge swap moves - can reduce distinct colors
        for _ in range(200):
            if time.time() > deadline:
                break
            e1 = random.randint(0, num_edges - 1)
            e2 = random.randint(0, num_edges - 1)
            if e1 == e2:
                continue
            if try_two_edge_swap(e1, e2, colors):
                improved = True
        
        if improved:
            no_improve_count = 0
            # Check if we have a new best
            if max(colors) < current_best:
                compacted = compact_colors(colors)
                num_unique = max(compacted) + 1
                if num_unique < current_best:
                    current_best = num_unique
                    best_num_colors = num_unique
                    best_colors = compacted[:]
                    colors = compacted
        else:
            no_improve_count += 1
            
            # Kick strategy: target edges with high K4 count, try multiple random changes
            # More aggressive kicks when stuck longer
            kick_strength = min(10 + no_improve_count // 5, 50)
            
            for _ in range(kick_strength):
                if time.time() > deadline:
                    break
                
                # Target edges with more constraints (higher K4 count)
                # Weighted random selection
                e_idx = random.randint(0, num_edges - 1)
                current_c = colors[e_idx]
                
                # Try to change to a different color
                max_kick_color = current_best + 8
                for _ in range(15):
                    new_c = random.randint(0, max_kick_color)
                    if new_c == current_c:
                        continue
                    if can_change_edge(e_idx, new_c, colors):
                        colors[e_idx] = new_c
                        break

    # Final: try to further reduce by compacting and one more descent pass
    compacted = compact_colors(best_colors)
    best_colors = compacted
    final_num = max(best_colors) + 1
    
    # One more quick descent pass on the best solution
    if time.time() < deadline - 5:
        for _ in range(1000):
            if time.time() > deadline - 5:
                break
            e_idx = random.randint(0, num_edges - 1)
            current_c = best_colors[e_idx]
            min_valid = -1
            for c in range(0, current_c):
                if can_change_edge(e_idx, c, best_colors):
                    min_valid = c
                    break
            if min_valid != -1:
                best_colors[e_idx] = min_valid
        
        # Compact again
        compacted = compact_colors(best_colors)
        if max(compacted) + 1 < final_num:
            best_colors = compacted
    
    return best_colors

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print(f"Colors used: {max(res) + 1}")
    # Verify
    n = 16
    edge_index = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_index[(i,j)] = idx
            idx += 1
    valid = True
    for comb in itertools.combinations(range(n), 4):
        v0, v1, v2, v3 = comb
        e_indices = [
            edge_index[(v0, v1)],
            edge_index[(v0, v2)],
            edge_index[(v0, v3)],
            edge_index[(v1, v2)],
            edge_index[(v1, v3)],
            edge_index[(v2, v3)]
        ]
        if len({res[e] for e in e_indices}) < 5:
            valid = False
            break
    print(f"Valid: {valid}")
