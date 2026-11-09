import time
import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2
DEADLINE = 85.0

def construct():
    start_time = time.time()
    
    # 1. Precomputation
    edge_idx = [[-1] * N for _ in range(N)]
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            idx = len(edges)
            edges.append((i, j))
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            
    cycles = []
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    
    vertices = list(range(N))
    for combo in itertools.combinations(vertices, 4):
        a, b, c, d = combo
        # Cycle 1: a-b-c-d-a
        e1, e2, e3, e4 = edge_idx[a][b], edge_idx[b][c], edge_idx[c][d], edge_idx[d][a]
        cyc_id = len(cycles)
        cycles.append((e1, e2, e3, e4))
        for e in (e1, e2, e3, e4): edge_to_cycles[e].append(cyc_id)
        
        # Cycle 2: a-b-d-c-a
        e1, e2, e3, e4 = edge_idx[a][b], edge_idx[b][d], edge_idx[d][c], edge_idx[c][a]
        cyc_id = len(cycles)
        cycles.append((e1, e2, e3, e4))
        for e in (e1, e2, e3, e4): edge_to_cycles[e].append(cyc_id)
        
        # Cycle 3: a-c-b-d-a
        e1, e2, e3, e4 = edge_idx[a][c], edge_idx[c][b], edge_idx[b][d], edge_idx[d][a]
        cyc_id = len(cycles)
        cycles.append((e1, e2, e3, e4))
        for e in (e1, e2, e3, e4): edge_to_cycles[e].append(cyc_id)

    def is_valid_change(colors, e, new_c):
        for cyc_id in edge_to_cycles[e]:
            c1, c2, c3, c4 = cycles[cyc_id]
            cols = [0, 0, 0, 0]
            for i, ce in enumerate((c1, c2, c3, c4)):
                cols[i] = new_c if ce == e else colors[ce]
            if len(set(cols)) == 4:
                return False
        return True

    # Initial coloring: color(i,j) = i
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            colors[edge_idx[i][j]] = i
            
    # Count array for colors 0..119
    color_counts = [0] * 120
    for c in colors:
        color_counts[c] += 1
        
    current_score = sum(1 for c in color_counts if c > 0)
    best_colors = colors[:]
    best_score = current_score
    
    # Local Search
    # We will try to increase the score.
    # Move: change color of a random edge to a random new color.
    # Accept if valid and (score increases OR (score same and random < 0.5) OR (score decreases and random < 0.01))
    
    while time.time() < start_time + DEADLINE:
        e = random.randint(0, NUM_EDGES - 1)
        old_c = colors[e]
        
        # Bias towards new colors
        if random.random() < 0.6:
            # Try a color that is currently unused or underused
            # Just pick random 0..119
            new_c = random.randint(0, 119)
        else:
            # Pick a random color from the current palette
            # To speed up, pick from indices where count > 0
            # This is O(120) which is fine.
            # Or just pick random and if not used, it's a "new" color attempt anyway.
            new_c = random.randint(0, 119)
            
        if new_c == old_c:
            continue
            
        if is_valid_change(colors, e, new_c):
            # Calculate score change
            delta = 0
            if color_counts[old_c] == 1:
                delta -= 1
            if color_counts[new_c] == 0:
                delta += 1
                
            new_score = current_score + delta
            
            # Acceptance criteria
            accept = False
            if delta > 0:
                accept = True
            elif delta == 0:
                accept = random.random() < 0.5
            else:
                accept = random.random() < 0.01
                
            if accept:
                # Update state
                colors[e] = new_c
                color_counts[old_c] -= 1
                color_counts[new_c] += 1
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_colors = colors[:]
                    
    return best_colors
