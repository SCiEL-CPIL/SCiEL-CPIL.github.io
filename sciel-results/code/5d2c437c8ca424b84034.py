import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Precompute edge index
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return (N - 1) * i - i * (i - 1) // 2 + (j - i - 1)
    
    # Precompute all 4-vertex subsets and their three 4-cycles
    four_sets = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    e1 = (edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a))
                    e2 = (edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a))
                    e3 = (edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a))
                    four_sets.append((e1, e2, e3))
    
    # Precompute for each edge, which 4-cycles it participates in
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, (e1, e2, e3) in enumerate(four_sets):
        for e in e1:
            edge_to_cycles[e].append(ci)
        for e in e2:
            edge_to_cycles[e].append(ci)
        for e in e3:
            edge_to_cycles[e].append(ci)
    
    def is_valid(colors):
        """Check if coloring is valid (no 4-cycle has 4 distinct colors)."""
        for (e1, e2, e3) in four_sets:
            c1, c2, c3, c4 = colors[e1[0]], colors[e1[1]], colors[e1[2]], colors[e1[3]]
            if len({c1, c2, c3, c4}) == 4:
                return False
            c1, c2, c3, c4 = colors[e2[0]], colors[e2[1]], colors[e2[2]], colors[e2[3]]
            if len({c1, c2, c3, c4}) == 4:
                return False
            c1, c2, c3, c4 = colors[e3[0]], colors[e3[1]], colors[e3[2]], colors[e3[3]]
            if len({c1, c2, c3, c4}) == 4:
                return False
        return True
    
    def check_cycle_valid(colors, e_tuple):
        """Check if a specific 4-cycle is valid."""
        c1, c2, c3, c4 = colors[e_tuple[0]], colors[e_tuple[1]], colors[e_tuple[2]], colors[e_tuple[3]]
        return len({c1, c2, c3, c4}) != 4
    
    # Base coloring: color edge (i,j) by i
    colors = [0] * NUM_EDGES
    for i in range(N - 1):
        for j in range(i + 1, N):
            colors[edge_idx(i, j)] = i
    
    assert is_valid(colors), "Base coloring should be valid"
    
    best_colors = colors[:]
    best_num_colors = len(set(best_colors))
    
    random.seed(42)
    
    # Strategy: Greedy introduction of new colors with smart edge selection
    # For each new color, find an edge that can take it without violating constraints
    # Prioritize edges with fewer constraints (fewer cycles they participate in)
    
    # Sort edges by number of cycles they're in (ascending) - easier to change
    edge_order = sorted(range(NUM_EDGES), key=lambda e: len(edge_to_cycles[e]))
    
    # Keep track of which edges have been "upgraded" to new colors
    # We'll try to introduce colors 15, 16, 17, ... one by one
    
    new_color = 15
    while time.time() < deadline - 5:
        if new_color >= 120:
            break
        
        # Try to find an edge that can take this new color
        # Shuffle the edge order for diversity
        shuffled_order = edge_order[:]
        random.shuffle(shuffled_order)
        
        found = False
        for e in shuffled_order:
            if time.time() > deadline - 5:
                break
            
            old_color = colors[e]
            if old_color == new_color:
                continue
            
            # Check if changing this edge to new_color is valid
            colors[e] = new_color
            valid = True
            for ci in edge_to_cycles[e]:
                e1, e2, e3 = four_sets[ci]
                if not check_cycle_valid(colors, e1):
                    valid = False
                    break
                if not check_cycle_valid(colors, e2):
                    valid = False
                    break
                if not check_cycle_valid(colors, e3):
                    valid = False
                    break
            
            if valid:
                found = True
                num_colors = len(set(colors))
                if num_colors > best_num_colors:
                    best_colors = colors[:]
                    best_num_colors = num_colors
                break
            else:
                colors[e] = old_color
        
        if not found:
            # Try a different approach: local search to free up space
            # Pick a random edge and try swapping colors with neighbors
            for _ in range(100):
                if time.time() > deadline - 5:
                    break
                
                e = random.randint(0, NUM_EDGES - 1)
                old_color = colors[e]
                
                # Try all 120 colors
                for new_c in range(120):
                    if new_c == old_color:
                        continue
                    colors[e] = new_c
                    valid = True
                    for ci in edge_to_cycles[e]:
                        e1, e2, e3 = four_sets[ci]
                        if not check_cycle_valid(colors, e1):
                            valid = False
                            break
                        if not check_cycle_valid(colors, e2):
                            valid = False
                            break
                        if not check_cycle_valid(colors, e3):
                            valid = False
                            break
                    
                    if valid:
                        num_colors = len(set(colors))
                        if num_colors > best_num_colors:
                            best_colors = colors[:]
                            best_num_colors = num_colors
                        break
                    else:
                        colors[e] = old_color
            
            # If we still can't add new_color, try next color
            new_color += 1
        else:
            # Successfully added new_color, try next
            new_color += 1
    
    # Final local search to squeeze out more colors
    while time.time() < deadline - 2:
        # Try to change random edges to random colors
        e = random.randint(0, NUM_EDGES - 1)
        old_color = colors[e]
        
        # Bias toward unused colors
        used = set(colors)
        unused = [c for c in range(120) if c not in used]
        if unused:
            new_color = random.choice(unused)
        else:
            new_color = random.randint(0, 119)
        
        if new_color == old_color:
            continue
        
        colors[e] = new_color
        valid = True
        for ci in edge_to_cycles[e]:
            e1, e2, e3 = four_sets[ci]
            if not check_cycle_valid(colors, e1):
                valid = False
                break
            if not check_cycle_valid(colors, e2):
                valid = False
                break
            if not check_cycle_valid(colors, e3):
                valid = False
                break
        
        if valid:
            num_colors = len(set(colors))
            if num_colors > best_num_colors:
                best_colors = colors[:]
                best_num_colors = num_colors
        else:
            colors[e] = old_color
    
    # Ensure best_colors is valid
    if not is_valid(best_colors):
        for i in range(N - 1):
            for j in range(i + 1, N):
                best_colors[edge_idx(i, j)] = i
    
    return best_colors
