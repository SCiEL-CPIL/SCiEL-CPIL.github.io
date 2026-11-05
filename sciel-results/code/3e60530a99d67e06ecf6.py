import random
import time

def construct():
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    k4s = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    e_ab = (a, b)
                    e_ac = (a, c)
                    e_ad = (a, d)
                    e_bc = (b, c)
                    e_bd = (b, d)
                    e_cd = (c, d)
                    k4s.append([e_ab, e_ac, e_ad, e_bc, e_bd, e_cd])
    
    edge_to_idx = {e: idx for idx, e in enumerate(edges)}
    k4s_idx = []
    for k4 in k4s:
        k4s_idx.append([edge_to_idx[e] for e in k4])

    # Initialize with seed coloring
    colors = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    
    start_time = time.time()
    time_limit = 115

    k4s_containing = [[] for _ in range(num_edges)]
    for i, k4 in enumerate(k4s_idx):
        for idx in k4:
            k4s_containing[idx].append(k4)

    def is_valid(col):
        for k4 in k4s_idx:
            if len(set(col[idx] for idx in k4)) < 5:
                return False
        return True

    def try_eliminate_color(c, col):
        edges_to_change = [idx for idx, c_val in enumerate(col) if c_val == c]
        if not edges_to_change:
            return True
        
        available_colors = [c_val for c_val in set(col) if c_val != c]
        if not available_colors:
            return False
            
        random.shuffle(edges_to_change)
        assignment = {}
        
        def is_consistent(k4_indices, edge_idx, new_color):
            distinct = set()
            for idx in k4_indices:
                if idx in assignment:
                    distinct.add(assignment[idx])
                elif idx == edge_idx:
                    distinct.add(new_color)
                else:
                    distinct.add(col[idx])
            return len(distinct) >= 5

        def backtrack(k):
            if k == len(edges_to_change):
                return True
            
            idx = edges_to_change[k]
            cand_colors = available_colors[:]
            random.shuffle(cand_colors)
            
            for new_c in cand_colors:
                assignment[idx] = new_c
                
                valid = True
                for k4 in k4s_containing[idx]:
                    if not is_consistent(k4, idx, new_c):
                        valid = False
                        break
                
                if valid:
                    if backtrack(k + 1):
                        return True
                
                del assignment[idx]
            
            return False
        
        if backtrack(0):
            for idx, new_c in assignment.items():
                col[idx] = new_c
            return True
        else:
            return False

    def repair(col):
        # Fix violations
        max_iterations = 1000
        for _ in range(max_iterations):
            found_violation = False
            for k4 in k4s_idx:
                distinct = set(col[idx] for idx in k4)
                if len(distinct) < 5:
                    found_violation = True
                    # Try to fix by changing an edge in this K4
                    # Prefer colors not in this K4
                    existing = list(set(col))
                    random.shuffle(existing)
                    fixed = False
                    for idx in k4:
                        for new_c in existing:
                            if new_c not in distinct:
                                col[idx] = new_c
                                fixed = True
                                break
                        if fixed:
                            break
                    if not fixed:
                        # Use a new color
                        max_c = max(set(col))
                        col[k4[0]] = max_c + 1
                    break # Restart scan
            if not found_violation:
                break

    while time.time() - start_time < time_limit:
        current_colors = set(colors)
        if not current_colors:
            break
        
        color_freq = {}
        for c in colors:
            color_freq[c] = color_freq.get(c, 0) + 1
        
        # Try to eliminate rare colors first
        colors_list = sorted(current_colors, key=lambda x: color_freq[x])
        
        eliminated = False
        for c in colors_list:
            if time.time() - start_time > time_limit:
                break
            
            test_colors = colors[:]
            if try_eliminate_color(c, test_colors):
                colors = test_colors
                eliminated = True
                break
        
        if not eliminated:
            # Perturb
            for _ in range(random.randint(1, 3)):
                idx = random.randint(0, num_edges - 1)
                new_c = random.choice(list(set(colors)))
                colors[idx] = new_c
            repair(colors)

    return colors
