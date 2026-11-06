import time
import random
import math
import sys

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLOR = 120

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Precompute all 4-cycles
CYCLES = []
for a in range(N):
    for b in range(a+1, N):
        for c in range(b+1, N):
            for d in range(c+1, N):
                # Cycle 1: a-b, b-c, c-d, d-a
                CYCLES.append((edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a)))
                # Cycle 2: a-b, b-d, d-c, c-a
                CYCLES.append((edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a)))
                # Cycle 3: a-c, c-b, b-d, d-a
                CYCLES.append((edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a)))

# Precompute for each edge, which cycles it belongs to
EDGE_CYCLES = [[] for _ in range(NUM_EDGES)]
for ci, cyc in enumerate(CYCLES):
    for e in cyc:
        EDGE_CYCLES[e].append(ci)

def is_valid_fast(colors, changed_edge, old_color):
    """Check validity only for cycles containing the changed edge."""
    for ci in EDGE_CYCLES[changed_edge]:
        cyc = CYCLES[ci]
        c0 = colors[cyc[0]]
        c1 = colors[cyc[1]]
        c2 = colors[cyc[2]]
        c3 = colors[cyc[3]]
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
    return True

def count_distinct_colors(colors):
    return len(set(colors))

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # 2s safety margin before 60s limit
    
    best_colors = None
    best_score = 0
    
    # Multiple restarts with different initial conditions
    max_restarts = 20
    
    for restart in range(max_restarts):
        if time.time() >= deadline - 5.0:
            break
            
        # Different initial colorings
        if restart == 0:
            # Original: color by min vertex
            current_colors = [0] * NUM_EDGES
            idx = 0
            for i in range(N):
                for j in range(i + 1, N):
                    current_colors[idx] = i
                    idx += 1
        elif restart == 1:
            # Random initial coloring
            current_colors = [random.randint(0, 30) for _ in range(NUM_EDGES)]
        elif restart == 2:
            # Color by edge weight (i+j) mod 40
            current_colors = [0] * NUM_EDGES
            idx = 0
            for i in range(N):
                for j in range(i + 1, N):
                    current_colors[idx] = (i + j) % 40
                    idx += 1
        elif restart == 3:
            # Color by (i*j) mod 50
            current_colors = [0] * NUM_EDGES
            idx = 0
            for i in range(N):
                for j in range(i + 1, N):
                    current_colors[idx] = (i * j) % 50
                    idx += 1
        elif restart == 4:
            # Color by edge index mod 60
            current_colors = [(idx % 60) for idx in range(NUM_EDGES)]
        else:
            # Mix of previous best and random
            if best_colors is not None:
                current_colors = best_colors[:]
                # Randomly perturb 20% of edges
                for _ in range(NUM_EDGES // 5):
                    e = random.randint(0, NUM_EDGES - 1)
                    current_colors[e] = random.randint(0, 119)
            else:
                current_colors = [random.randint(0, 50) for _ in range(NUM_EDGES)]
        
        # Ensure validity of initial coloring by fixing violations
        for e in range(NUM_EDGES):
            old_color = current_colors[e]
            # Try to find a valid color for this edge
            attempts = 0
            while attempts < 50:
                current_colors[e] = random.randint(0, 119)
                if is_valid_fast(current_colors, e, old_color):
                    break
                attempts += 1
            else:
                # If we can't find a valid color, revert
                current_colors[e] = old_color
        
        current_score = count_distinct_colors(current_colors)
        
        if current_score > best_score:
            best_score = current_score
            best_colors = current_colors[:]
        
        # Simulated Annealing
        T = 10.0 + random.uniform(0, 10)
        T_min = 0.01
        alpha = 0.9993 + random.uniform(0, 0.0004)
        
        used_colors = set(current_colors)
        color_counts = [0] * MAX_COLOR
        for c in current_colors:
            color_counts[c] += 1
        
        iter_count = 0
        no_improve_count = 0
        stagnation_limit = 5000 + random.randint(0, 3000)
        
        while time.time() < deadline - 2.0:
            # Pick random edge
            e = random.randint(0, NUM_EDGES - 1)
            old_color = current_colors[e]
            
            # Strategy: try to introduce new colors when beneficial
            if random.random() < 0.6:
                # Try unused colors first
                unused = [c for c in range(MAX_COLOR) if color_counts[c] == 0]
                if unused:
                    new_color = random.choice(unused)
                else:
                    new_color = random.randint(0, MAX_COLOR - 1)
            else:
                new_color = random.randint(0, MAX_COLOR - 1)
            
            if new_color == old_color:
                continue
            
            # Apply change
            current_colors[e] = new_color
            color_counts[old_color] -= 1
            color_counts[new_color] += 1
            
            # Check validity using fast incremental check
            if is_valid_fast(current_colors, e, old_color):
                # Calculate score change
                if color_counts[old_color] == 0:
                    delta = -1
                elif color_counts[new_color] == 1:
                    delta = 1
                else:
                    delta = 0
                
                new_score = current_score + delta
                
                # Accept/Reject based on Simulated Annealing
                if delta >= 0 or random.random() < math.exp(delta / T):
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_colors = current_colors[:]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    # Reject: revert
                    current_colors[e] = old_color
                    color_counts[old_color] += 1
                    color_counts[new_color] -= 1
            else:
                # Invalid: revert
                current_colors[e] = old_color
                color_counts[old_color] += 1
                color_counts[new_color] -= 1
            
            iter_count += 1
            no_improve_count += 1
            
            # Cool down
            T *= alpha
            if T < T_min:
                T = T_min
                # Occasionally reheat to escape local optima
                if random.random() < 0.03:
                    T = 5.0 + random.uniform(0, 5)
            
            # Restart if stagnant
            if no_improve_count > stagnation_limit:
                break
        
        no_improve_count = 0
    
    # Final verification and return
    if best_colors is None:
        # Fallback: simple valid coloring
        best_colors = [0] * NUM_EDGES
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                best_colors[idx] = i
                idx += 1
    
    return best_colors
