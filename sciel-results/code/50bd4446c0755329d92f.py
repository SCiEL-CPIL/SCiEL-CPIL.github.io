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
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for i, cyc in enumerate(cycles):
        for e in cyc:
            edge_to_cycles[e].append(i)

    # Precompute for each edge: the other 3 edges in each cycle it belongs to
    # This avoids repeated lookups in is_valid_change
    edge_cycle_other_edges = [[] for _ in range(NUM_EDGES)]
    for e in range(NUM_EDGES):
        for c_idx in edge_to_cycles[e]:
            c = cycles[c_idx]
            others = [c[k] for k in range(4) if c[k] != e]
            edge_cycle_other_edges[e].append(others)

    def is_valid_change(colors, edge, new_color):
        for others in edge_cycle_other_edges[edge]:
            col0 = new_color
            col1 = colors[others[0]]
            col2 = colors[others[1]]
            col3 = colors[others[2]]
            
            if col0 != col1 and col0 != col2 and col0 != col3 and \
               col1 != col2 and col1 != col3 and \
               col2 != col3:
                return False
        return True

    # Improved seed: use a more balanced recursive partition with varied splits
    def get_seed_colors(seed_val=42):
        random.seed(seed_val)
        colors = [0] * NUM_EDGES
        color_id = 0
        def process_partition(vertices, depth=0):
            nonlocal color_id
            if len(vertices) <= 2:
                return
            # Vary the split ratio based on depth and randomization for diversity
            if depth == 0:
                mid = len(vertices) // 2
            elif depth == 1:
                mid = max(2, min(len(vertices)-2, len(vertices) // 2 + random.randint(-1, 1)))
            else:
                # For deeper levels, try different splits
                mid = max(2, min(len(vertices)-2, len(vertices) // 2 + random.randint(-1, 1)))
            
            left = vertices[:mid]
            right = vertices[mid:]
            for u in left:
                for v in right:
                    idx = edge_idx[u][v]
                    colors[idx] = color_id
            color_id += 1
            process_partition(left, depth + 1)
            process_partition(right, depth + 1)
        process_partition(list(range(N)))
        return colors

    # Try multiple seed colorings and keep the best
    best_seed = None
    best_seed_score = 0
    for seed_val in range(10):
        seed = get_seed_colors(seed_val)
        score = len(set(seed))
        if score > best_seed_score:
            best_seed_score = score
            best_seed = seed
    
    current_colors = best_seed.copy()
    best_score = best_seed_score
    best_colors = best_seed.copy()
    
    freq = [0] * MAX_COLORS
    for c in current_colors:
        freq[c] += 1
    
    random.seed(42)
    
    used_colors = set(current_colors)
    current_score = len(used_colors)
    
    temperature = 2.0
    cooling_rate = 0.99998
    min_temp = 0.01
    
    steps = 0
    no_improve_steps = 0
    max_no_improve = 2000
    
    # Track edge constraint density
    edge_cycle_count = [len(edge_to_cycles[e]) for e in range(NUM_EDGES)]
    
    while time.time() < deadline:
        steps += 1
        
        # Smarter edge selection: balance between constraint density and color diversity
        # Edges with fewer cycles are easier to recolor
        # Edges with rare colors are more valuable to change
        r1 = random.random()
        if r1 < 0.4:
            # Bias toward less constrained edges
            weights = [1.0 / (1 + edge_cycle_count[e]) for e in range(NUM_EDGES)]
            total_w = sum(weights)
            r = random.random() * total_w
            cumulative = 0
            e = 0
            for i, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    e = i
                    break
        elif r1 < 0.7:
            # Bias toward edges with rare colors (higher potential for new colors)
            weights = [1.0 / (1 + freq[current_colors[e]]) for e in range(NUM_EDGES)]
            total_w = sum(weights)
            r = random.random() * total_w
            cumulative = 0
            e = 0
            for i, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    e = i
                    break
        else:
            e = random.randint(0, NUM_EDGES - 1)
        
        old_color = current_colors[e]
        
        # Strategy: prioritize introducing new colors when possible
        if current_score < MAX_COLORS:
            unused = [c for c in range(MAX_COLORS) if freq[c] == 0]
            if unused:
                # 85% chance to try an unused color
                if random.random() < 0.85:
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
            old_score = current_score
            score_change = 0
            if freq[old_color] == 1:
                score_change -= 1
            if freq[new_color] == 0:
                score_change += 1
            new_score = old_score + score_change
            
            # Acceptance criteria
            if new_score > current_score:
                accept = True
            elif new_score == current_score:
                # Accept neutral moves with moderate probability
                accept = random.random() < 0.4
            else:
                # Accept declines with temperature-dependent probability
                delta = current_score - new_score
                accept = random.random() < np.exp(-delta / temperature)
            
            if accept:
                freq[old_color] -= 1
                freq[new_color] += 1
                current_colors[e] = new_color
                
                if freq[old_color] == 0:
                    used_colors.discard(old_color)
                if freq[new_color] == 1:
                    used_colors.add(new_color)
                current_score = len(used_colors)
                
                if current_score > best_score:
                    best_score = current_score
                    best_colors = current_colors.copy()
                    no_improve_steps = 0
                    # Boost temperature slightly on improvement to explore more
                    temperature = min(2.5, temperature * 1.15)
                else:
                    no_improve_steps += 1
                
                temperature *= cooling_rate
                if temperature < min_temp:
                    temperature = min_temp
                    
                # Diversification: structured re-optimization
                if no_improve_steps > max_no_improve:
                    # Restore best solution
                    current_colors = best_colors.copy()
                    freq = [0] * MAX_COLORS
                    for c in current_colors:
                        freq[c] += 1
                    used_colors = set(current_colors)
                    current_score = len(used_colors)
                    temperature = 2.0
                    no_improve_steps = 0
                    
                    # Apply systematic perturbations: pick a subset of edges and re-optimize
                    num_perturb = random.randint(5, 12)
                    perturb_edges = random.sample(range(NUM_EDGES), num_perturb)
                    for pe in perturb_edges:
                        # Try to introduce a new color if possible
                        if current_score < MAX_COLORS:
                            unused = [c for c in range(MAX_COLORS) if freq[c] == 0]
                            if unused and random.random() < 0.7:
                                pc = random.choice(unused)
                            else:
                                pc = random.randint(0, MAX_COLORS - 1)
                        else:
                            pc = random.randint(0, MAX_COLORS - 1)
                        
                        if pc != current_colors[pe] and is_valid_change(current_colors, pe, pc):
                            old_c = current_colors[pe]
                            freq[old_c] -= 1
                            freq[pc] += 1
                            current_colors[pe] = pc
                            if freq[old_c] == 0:
                                used_colors.discard(old_c)
                            if freq[pc] == 1:
                                used_colors.add(pc)
                            current_score = len(used_colors)

    # Final polish: aggressive greedy color insertion
    if time.time() < deadline - 1.0:
        # Multiple passes of greedy color insertion
        for polish_pass in range(5):
            if time.time() > deadline - 0.5:
                break
            unused = [c for c in range(MAX_COLORS) if freq[c] == 0]
            if not unused:
                break
            # Sort edges by constraint density (fewer cycles first)
            edge_order = sorted(range(NUM_EDGES), key=lambda e: edge_cycle_count[e])
            for new_c in unused:
                if time.time() > deadline - 0.5:
                    break
                # Find an edge that can take this color, preferring less constrained edges
                for e in edge_order:
                    if current_colors[e] != new_c and is_valid_change(current_colors, e, new_c):
                        old_c = current_colors[e]
                        freq[old_c] -= 1
                        freq[new_c] += 1
                        current_colors[e] = new_c
                        if freq[old_c] == 0:
                            used_colors.discard(old_c)
                        if freq[new_c] == 1:
                            used_colors.add(new_c)
                        current_score = len(used_colors)
                        if current_score > best_score:
                            best_score = current_score
                            best_colors = current_colors.copy()
                        break  # Only one edge per new color in this pass

    return best_colors
