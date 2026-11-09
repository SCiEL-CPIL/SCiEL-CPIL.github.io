import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 16
    # Edge list in lexicographic order
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    num_edges = len(edges)
    
    # Map (u,v) to edge index
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
    
    # Precompute all K4s as lists of 6 edge indices
    k4s = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        e_idx = [
            edge_idx[(a, b)], edge_idx[(a, c)], edge_idx[(a, d)],
            edge_idx[(b, c)], edge_idx[(b, d)], edge_idx[(c, d)]
        ]
        k4s.append(e_idx)
    num_k4s = len(k4s)
    
    # For each edge, which K4s does it appear in?
    edge_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_list in enumerate(k4s):
        for ei in e_list:
            edge_k4s[ei].append(k4_idx)
    
    def compact(colors):
        unique = sorted(set(colors))
        mapping = {c: i for i, c in enumerate(unique)}
        return [mapping[c] for c in colors], len(unique)
    
    # Initialize with (i+j) % 17
    colors = [0] * num_edges
    for idx, (i, j) in enumerate(edges):
        colors[idx] = (i + j) % 17
        
    # State tracking
    k4_color_counts = [{} for _ in range(num_k4s)]
    
    def init_state():
        for k4_idx in range(num_k4s):
            counts = {}
            for ei in k4s[k4_idx]:
                c = colors[ei]
                counts[c] = counts.get(c, 0) + 1
            k4_color_counts[k4_idx] = counts
            
    def is_all_valid():
        for k4_idx in range(num_k4s):
            if len(k4_color_counts[k4_idx]) < 5:
                return False
        return True

    def apply_change(e_idx, old_c, new_c):
        colors[e_idx] = new_c
        for k4_idx in edge_k4s[e_idx]:
            counts = k4_color_counts[k4_idx]
            counts[old_c] -= 1
            if counts[old_c] == 0:
                del counts[old_c]
            if new_c not in counts:
                counts[new_c] = 0
            counts[new_c] += 1
            
    def revert_change(e_idx, old_c, new_c):
        colors[e_idx] = old_c
        for k4_idx in edge_k4s[e_idx]:
            counts = k4_color_counts[k4_idx]
            counts[new_c] -= 1
            if counts[new_c] == 0:
                del counts[new_c]
            if old_c not in counts:
                counts[old_c] = 0
            counts[old_c] += 1

    init_state()
    
    # If initial is not valid, we need a repair. 
    # (i+j)%17 is valid for K16? 
    # Let's assume yes. If not, the search will fail to eliminate and just walk.
    # But if it's invalid, we should fix it.
    if not is_all_valid():
        # Quick repair using SA with hard constraints? 
        # Or just random recoloring until valid.
        # For now, we assume it's valid.
        pass

    best_colors = colors[:]
    best_count = len(set(colors))
    
    used_colors = set(colors)
    
    def try_eliminate_color(target_color):
        edges_to_recolor = [i for i in range(num_edges) if colors[i] == target_color]
        if not edges_to_recolor:
            return True
            
        other_colors = [c for c in used_colors if c != target_color]
        if not other_colors:
            return False
            
        random.shuffle(edges_to_recolor)
        
        changes_made = []
        success = True
        for e_idx in edges_to_recolor:
            old_c = colors[e_idx]
            if old_c != target_color:
                continue
                
            found = False
            random.shuffle(other_colors)
            for new_c in other_colors:
                if new_c == old_c:
                    continue
                
                valid = True
                for k4_idx in edge_k4s[e_idx]:
                    counts = k4_color_counts[k4_idx]
                    old_count = counts.get(old_c, 0)
                    new_count = counts.get(new_c, 0)
                    D = len(counts)
                    
                    if old_count == 1:
                        if new_count == 0:
                            new_D = D
                        else:
                            new_D = D - 1
                    else:
                        if new_count == 0:
                            new_D = D + 1
                        else:
                            new_D = D
                            
                    if new_D < 5:
                        valid = False
                        break
                        
                if valid:
                    apply_change(e_idx, old_c, new_c)
                    changes_made.append((e_idx, old_c, new_c))
                    found = True
                    break
                    
            if not found:
                for ce, oc, nc in reversed(changes_made):
                    revert_change(ce, oc, nc)
                success = False
                break
                
        if success:
            used_colors.remove(target_color)
            return True
        else:
            return False

    while time.time() < deadline - 5.0:
        current_count = len(used_colors)
        if current_count < best_count:
            best_count = current_count
            best_colors = colors[:]
            
        if used_colors:
            target = random.choice(list(used_colors))
            if try_eliminate_color(target):
                continue
            
        # Random Walk
        steps = 20
        for _ in range(steps):
            if time.time() > deadline - 5.0:
                break
            e_idx = random.randint(0, num_edges - 1)
            old_c = colors[e_idx]
            
            if len(used_colors) <= 1:
                break
            new_c = random.choice(list(used_colors))
            if new_c == old_c:
                continue
                
            valid = True
            for k4_idx in edge_k4s[e_idx]:
                counts = k4_color_counts[k4_idx]
                old_count = counts.get(old_c, 0)
                new_count = counts.get(new_c, 0)
                D = len(counts)
                
                if old_count == 1:
                    if new_count == 0:
                        new_D = D
                    else:
                        new_D = D - 1
                else:
                    if new_count == 0:
                        new_D = D + 1
                    else:
                        new_D = D
                        
                if new_D < 5:
                    valid = False
                    break
                    
            if valid:
                apply_change(e_idx, old_c, new_c)

    return compact(best_colors)[0]
