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
    deadline = start_time + 55.0  # Safety margin before 60s limit

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

    # Map edge index to list of cycle indices it belongs to
    # This allows fast validity check for a single edge change
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for i, cyc in enumerate(cycles):
        for e in cyc:
            edge_to_cycles[e].append(i)

    def is_valid_change(colors, edge, new_color):
        # Check all cycles containing this edge
        for c_idx in edge_to_cycles[edge]:
            c = cycles[c_idx]
            # Get colors of the 4 edges in this cycle
            # Note: one of them is the edge we are changing, so use new_color
            col0 = new_color if c[0] == edge else colors[c[0]]
            col1 = new_color if c[1] == edge else colors[c[1]]
            col2 = new_color if c[2] == edge else colors[c[2]]
            col3 = new_color if c[3] == edge else colors[c[3]]
            
            # Check if 4 distinct
            if col0 != col1 and col0 != col2 and col0 != col3 and \
               col1 != col2 and col1 != col3 and \
               col2 != col3:
                return False
        return True

    # Initialize with recursive partition seed
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

    current_colors = get_seed_colors()
    best_score = len(set(current_colors))
    best_colors = current_colors.copy()
    
    # Track frequency of each color for incremental score updates
    freq = [0] * MAX_COLORS
    for c in current_colors:
        freq[c] += 1
    
    random.seed(42)
    
    # Simulated Annealing with better color selection
    # Track which colors are currently in use
    used_colors = set(current_colors)
    current_score = len(used_colors)
    
    # Temperature for simulated annealing
    temperature = 2.0
    cooling_rate = 0.99995
    min_temperature = 0.05
    
    steps = 0
    no_improve_steps = 0
    max_no_improve = 8000
    
    # Track edges that are "constrained" (in many cycles) for smarter moves
    edge_constraint_count = [len(edge_to_cycles[e]) for e in range(NUM_EDGES)]
    # Edges with fewer constraints are easier to change
    sorted_edges_by_constraint = sorted(range(NUM_EDGES), key=lambda e: edge_constraint_count[e])
    
    # Weighted edge selection: prefer less constrained edges
    edge_weights = [1.0 / (1 + edge_constraint_count[e]) for e in range(NUM_EDGES)]
    total_weight = sum(edge_weights)
    cumulative_weights = []
    cum = 0
    for w in edge_weights:
        cum += w
        cumulative_weights.append(cum)
    
    def select_edge():
        r = random.random() * total_weight
        for i, cw in enumerate(cumulative_weights):
            if r <= cw:
                return i
        return NUM_EDGES - 1
    
    # Use a time-based loop
    while time.time() < deadline:
        steps += 1
        e = select_edge()
        old_color = current_colors[e]
        
        # Strategy: Bias toward unused colors to maximize distinct count
        # With some probability, pick an unused color; otherwise pick randomly
        if current_score < MAX_COLORS:
            # Find unused colors
            unused = [c for c in range(MAX_COLORS) if freq[c] == 0]
            if unused:
                # 70% chance to try an unused color, 30% random
                if random.random() < 0.75:
                    new_color = random.choice(unused)
                else:
                    new_color = random.randint(0, MAX_COLORS - 1)
            else:
                new_color = random.randint(0, MAX_COLORS - 1)
        else:
            new_color = random.randint(0, MAX_COLORS - 1)
        
        if new_color == old_color:
            continue
            
        if is_valid_change(current_colors, e, new_color):
            # Calculate what the new score would be
            old_score = current_score
            # If we're removing the last use of old_color, score decreases by 1
            # If we're adding the first use of new_color, score increases by 1
            score_change = 0
            if freq[old_color] == 1:
                score_change -= 1
            if freq[new_color] == 0:
                score_change += 1
            new_score = old_score + score_change
            
            # Acceptance criterion: simulated annealing
            if new_score >= current_score:
                accept = True
            else:
                # Accept with probability exp(-delta / temperature)
                delta = current_score - new_score
                accept = random.random() < np.exp(-delta / temperature)
            
            if accept:
                # Apply the move
                freq[old_color] -= 1
                freq[new_color] += 1
                current_colors[e] = new_color
                
                # Update used_colors and current_score
                if freq[old_color] == 0:
                    used_colors.discard(old_color)
                if freq[new_color] == 1:
                    used_colors.add(new_color)
                current_score = len(used_colors)
                
                if current_score > best_score:
                    best_score = current_score
                    best_colors = current_colors.copy()
                    no_improve_steps = 0
                    temperature = 2.0
                else:
                    no_improve_steps += 1
                
                temperature *= cooling_rate
                if temperature < min_temperature:
                    temperature = min_temperature
                    
                # Diversification: if stuck for too long, restart from best
                if no_improve_steps > max_no_improve:
                    current_colors = best_colors.copy()
                    freq = [0] * MAX_COLORS
                    for c in current_colors:
                        freq[c] += 1
                    used_colors = set(current_colors)
                    current_score = len(used_colors)
                    temperature = 2.0
                    no_improve_steps = 0
                    # Perturb with multiple valid random changes
                    perturb_count = 0
                    for _ in range(15):
                        pe = random.randint(0, NUM_EDGES - 1)
                        pc = random.randint(0, MAX_COLORS - 1)
                        if pc != current_colors[pe] and is_valid_change(current_colors, pe, pc):
                            old_pc = current_colors[pe]
                            freq[old_pc] -= 1
                            freq[pc] += 1
                            current_colors[pe] = pc
                            if freq[old_pc] == 0:
                                used_colors.discard(old_pc)
                            if freq[pc] == 1:
                                used_colors.add(pc)
                            current_score = len(used_colors)
                            perturb_count += 1
                    if perturb_count == 0:
                        # Try harder perturbation
                        for _ in range(30):
                            pe = random.randint(0, NUM_EDGES - 1)
                            pc = random.randint(0, MAX_COLORS - 1)
                            if pc != current_colors[pe] and is_valid_change(current_colors, pe, pc):
                                old_pc = current_colors[pe]
                                freq[old_pc] -= 1
                                freq[pc] += 1
                                current_colors[pe] = pc
                                if freq[old_pc] == 0:
                                    used_colors.discard(old_pc)
                                if freq[pc] == 1:
                                    used_colors.add(pc)
                                current_score = len(used_colors)
                                break
            else:
                # Rejected move
                pass
        else:
            # Invalid, do nothing
            pass

    return best_colors
