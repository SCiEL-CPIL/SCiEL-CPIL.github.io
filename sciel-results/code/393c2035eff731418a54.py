import time
import sys
import random
import itertools
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLORS = 120

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    # Precompute edge indices
    edge_idx = [[-1]*N for _ in range(N)]
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1
            
    # Precompute all 4-cycles and their edge indices
    cycles = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        cycles.append((edge_idx[a][b], edge_idx[b][c], edge_idx[c][d], edge_idx[d][a]))
        # Cycle 2: a-b-d-c-a
        cycles.append((edge_idx[a][b], edge_idx[b][d], edge_idx[d][c], edge_idx[c][a]))
        # Cycle 3: a-c-b-d-a
        cycles.append((edge_idx[a][c], edge_idx[c][b], edge_idx[b][d], edge_idx[d][a]))

    # Map edge index to list of cycle indices
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for i, cyc in enumerate(cycles):
        for e in cyc:
            edge_to_cycles[e].append(i)

    def get_seed_colors():
        colors = [0] * NUM_EDGES
        color_id = 0
        def process_partition(vertices):
            nonlocal color_id
            if len(vertices) <= 1:
                return
            mid = len(vertices) // 2
            left = vertices[:mid]
            right = vertices[mid:]
            for u in left:
                for v in right:
                    idx = edge_idx[u][v]
                    colors[idx] = color_id
            color_id += 1
            process_partition(left)
            process_partition(right)
        process_partition(list(range(N)))
        return colors

    current_colors = np.array(get_seed_colors(), dtype=np.int32)
    best_score = len(set(current_colors))
    best_colors = current_colors.copy()
    
    # Track frequency
    freq = np.zeros(MAX_COLORS, dtype=np.int32)
    for c in current_colors:
        freq[c] += 1
        
    used_colors_mask = freq > 0
    current_score = int(np.sum(used_colors_mask))
    
    random.seed(42)
    np.random.seed(42)
    
    # --- Randomized Backtracking Search ---
    # Strategy: greedily assign new colors to edges, checking constraints
    # For each edge, try to assign an unused color that doesn't create a rainbow 4-cycle
    
    def try_backtracking_search(colors, freq_arr, max_time=30.0):
        """
        Randomized backtracking: for each edge in random order, try to assign
        a new color that doesn't violate constraints.
        """
        colors = colors.copy()
        freq_arr = freq_arr.copy()
        score = int(np.sum(freq_arr > 0))
        best_local = score
        best_local_colors = colors.copy()
        
        # Get list of edges in random order
        edge_order = list(range(NUM_EDGES))
        random.shuffle(edge_order)
        
        # For each edge, try to introduce a new color
        for e in edge_order:
            if time.time() - start_time > max_time:
                break
                
            old_color = colors[e]
            
            # Get unused colors
            unused = [c for c in range(MAX_COLORS) if freq_arr[c] == 0]
            if not unused:
                break
            
            # Try each unused color (randomized)
            random.shuffle(unused)
            found = False
            for new_c in unused:
                # Check if assigning new_c to edge e creates any rainbow 4-cycle
                valid = True
                for c_idx in edge_to_cycles[e]:
                    cyc = cycles[c_idx]
                    # Get the 4 colors in this cycle with e assigned new_c
                    cols = []
                    for ce in cyc:
                        if ce == e:
                            cols.append(new_c)
                        else:
                            cols.append(colors[ce])
                    # Check if all 4 are distinct
                    if len(set(cols)) == 4:
                        valid = False
                        break
                
                if valid:
                    # Accept this assignment
                    freq_arr[old_color] -= 1
                    freq_arr[new_c] += 1
                    colors[e] = new_c
                    score += 1  # New color introduced
                    
                    if score > best_local:
                        best_local = score
                        best_local_colors = colors.copy()
                    found = True
                    break
        
            # If no new color worked, optionally try to reassign to a different used color
            # to free up a color for later (but this is complex; skip for now)
        
        return best_local_colors, best_local, freq_arr

    # Run backtracking search multiple times with different random seeds
    bt_start = time.time()
    bt_budget = 25.0  # Spend up to 25 seconds on backtracking
    while time.time() - bt_start < bt_budget:
        bt_colors, bt_score, bt_freq = try_backtracking_search(
            current_colors, freq, max_time=bt_budget - (time.time() - bt_start)
        )
        
        if bt_score > best_score:
            best_score = bt_score
            best_colors = bt_colors
            current_colors = bt_colors
            freq = bt_freq
            current_score = bt_score
        
        if best_score >= MAX_COLORS:
            break
    
    # --- Local Search Polish ---
    # Use remaining time for local search to refine
    temperature = 1.5
    cooling_rate = 0.99995
    min_temperature = 0.03
    
    steps = 0
    no_improve_steps = 0
    max_no_improve = 3000
    
    def is_valid_change(e, new_color):
        for c_idx in edge_to_cycles[e]:
            c = cycles[c_idx]
            c0 = new_color if c[0] == e else current_colors[c[0]]
            c1 = new_color if c[1] == e else current_colors[c[1]]
            c2 = new_color if c[2] == e else current_colors[c[2]]
            c3 = new_color if c[3] == e else current_colors[c[3]]
            
            if c0 != c1 and c0 != c2 and c0 != c3 and \
               c1 != c2 and c1 != c3 and \
               c2 != c3:
                return False
        return True

    # Main Loop
    while time.time() < deadline:
        steps += 1
        
        # Strategy: 60% try to introduce new color, 40% random SA move
        if random.random() < 0.60 and current_score < MAX_COLORS:
            unused_indices = np.where(freq == 0)[0]
            if len(unused_indices) == 0:
                new_color = random.randint(0, MAX_COLORS - 1)
                e = random.randint(0, NUM_EDGES - 1)
            else:
                new_color = int(random.choice(unused_indices))
                # Bias towards edges that are in fewer cycles (easier to assign)
                # Or just random
                e = random.randint(0, NUM_EDGES - 1)
        else:
            e = random.randint(0, NUM_EDGES - 1)
            new_color = random.randint(0, MAX_COLORS - 1)
            
        old_color = current_colors[e]
        
        if new_color == old_color:
            continue
            
        if is_valid_change(e, new_color):
            old_freq = freq[old_color]
            new_freq = freq[new_color]
            
            score_change = 0
            if old_freq == 1:
                score_change -= 1
            if new_freq == 0:
                score_change += 1
                
            new_score = current_score + score_change
            
            if new_score >= current_score:
                accept = True
            else:
                delta = current_score - new_score
                accept = random.random() < np.exp(-delta / temperature)
                
            if accept:
                freq[old_color] -= 1
                freq[new_color] += 1
                current_colors[e] = new_color
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_colors = current_colors.copy()
                    no_improve_steps = 0
                    temperature = 1.5
                else:
                    no_improve_steps += 1
                    
                temperature *= cooling_rate
                if temperature < min_temperature:
                    temperature = min_temperature
                    
                if no_improve_steps > max_no_improve:
                    current_colors = best_colors.copy()
                    freq = np.zeros(MAX_COLORS, dtype=np.int32)
                    for c in current_colors:
                        freq[c] += 1
                    current_score = best_score
                    temperature = 1.5
                    no_improve_steps = 0
                    for _ in range(15):
                        pe = random.randint(0, NUM_EDGES - 1)
                        pc = random.randint(0, MAX_COLORS - 1)
                        if pc != current_colors[pe] and is_valid_change(pe, pc):
                            old_pc = current_colors[pe]
                            freq[old_pc] -= 1
                            freq[pc] += 1
                            current_colors[pe] = pc
                            if freq[old_pc] == 0: current_score -= 1
                            if freq[pc] == 1: current_score += 1
                            break

    return best_colors.tolist()
