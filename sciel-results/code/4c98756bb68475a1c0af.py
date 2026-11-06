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
    
    # Precompute edge indices for all pairs (i, j) with 0 <= i < j < N
    edge_idx = [[-1]*N for _ in range(N)]
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1
            
    # Precompute all 4-cycles (sets of 4 vertices)
    cycles = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        cycles.append(((a,b), (b,c), (c,d), (d,a)))
        # Cycle 2: a-b-d-c-a
        cycles.append(((a,b), (b,d), (d,c), (c,a)))
        # Cycle 3: a-c-b-d-a
        cycles.append(((a,c), (c,b), (b,d), (d,a)))
        
    # Convert cycles to edge indices for faster lookup
    cycle_indices = []
    for c in cycles:
        cycle_indices.append(tuple(edge_idx[u][v] for u, v in c))
    
    # Precompute for each edge, the list of cycles it belongs to
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for i, c_idx in enumerate(cycle_indices):
        for e_idx in c_idx:
            edge_to_cycles[e_idx].append(i)
    
    # Convert to numpy arrays for vectorized operations
    cycle_indices_np = np.array(cycle_indices, dtype=np.int32)
    num_cycles = len(cycle_indices_np)
    
    # Precompute: for each cycle, which positions are which edges
    # cycle_edges_np[c, p] = edge index at position p in cycle c
    # We already have this in cycle_indices_np
    
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
    
    def count_colors(colors):
        return len(set(colors))
    
    def check_validity_np(colors_np):
        """Check if coloring is valid using numpy. Returns True if valid."""
        # For each cycle, get the 4 edge colors
        # cycle_indices_np has shape (num_cycles, 4)
        # colors_np has shape (NUM_EDGES,)
        cycle_colors = colors_np[cycle_indices_np]  # shape (num_cycles, 4)
        
        # Check if any cycle has 4 distinct colors
        # A cycle is invalid if all 4 colors are different
        # c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3
        c0 = cycle_colors[:, 0]
        c1 = cycle_colors[:, 1]
        c2 = cycle_colors[:, 2]
        c3 = cycle_colors[:, 3]
        
        # Count distinct colors in each cycle
        # A cycle has 4 distinct colors iff:
        # (c0!=c1) & (c0!=c2) & (c0!=c3) & (c1!=c2) & (c1!=c3) & (c2!=c3)
        all_distinct = (c0 != c1) & (c0 != c2) & (c0 != c3) & (c1 != c2) & (c1 != c3) & (c2 != c3)
        
        return not np.any(all_distinct)
    
    def is_valid_incremental(colors_list, changed_edge, new_color, old_color):
        """Check validity only for cycles involving changed_edge."""
        colors_np = np.array(colors_list, dtype=np.int32)
        colors_np[changed_edge] = new_color
        
        # Only check cycles that include changed_edge
        relevant_cycles = edge_to_cycles[changed_edge]
        if not relevant_cycles:
            return True
            
        for c_idx in relevant_cycles:
            ci = cycle_indices[c_idx]
            c0 = colors_np[ci[0]]
            c1 = colors_np[ci[1]]
            c2 = colors_np[ci[2]]
            c3 = colors_np[ci[3]]
            if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                return False
        return True
    
    # Multi-start local search
    best_score = 0
    best_colors = None
    
    random.seed(42)
    np.random.seed(42)
    
    # Number of restarts based on time budget
    # Each restart should take roughly equal time
    max_restarts = 8
    time_per_restart = (deadline - start_time) / max_restarts
    
    for restart in range(max_restarts):
        if time.time() > deadline - 2.0:
            break
            
        restart_start = time.time()
        restart_deadline = restart_start + time_per_restart
        
        # Initialize: use seed coloring with some random perturbations
        current_colors = get_seed_colors()
        
        # Apply random perturbations to escape local optima
        num_perturb = random.randint(5, 15)
        for _ in range(num_perturb):
            e = random.randint(0, NUM_EDGES - 1)
            new_c = random.randint(0, MAX_COLORS - 1)
            current_colors[e] = new_c
        
        # Check if still valid; if not, revert perturbations one by one until valid
        colors_np = np.array(current_colors, dtype=np.int32)
        if not check_validity_np(colors_np):
            # Revert to seed
            current_colors = get_seed_colors()
        
        current_colors = current_colors.copy()
        current_distinct = count_colors(current_colors)
        
        if current_distinct > best_score:
            best_score = current_distinct
            best_colors = current_colors.copy()
        
        # Track color frequencies
        freq = [0] * MAX_COLORS
        for c in current_colors:
            freq[c] += 1
        
        # Local search with simulated annealing
        T = 5.0
        T_min = 0.01
        cooling = 0.9995
        
        steps = 0
        no_improve_steps = 0
        max_no_improve = 2000
        
        while time.time() < restart_deadline:
            steps += 1
            
            # Pick a random edge
            e = random.randint(0, NUM_EDGES - 1)
            old_color = current_colors[e]
            
            # Smart color selection: bias toward unused or rare colors
            # 60% chance to pick from unused colors, 40% random
            if random.random() < 0.6:
                # Find unused colors
                unused = [c for c in range(MAX_COLORS) if freq[c] == 0]
                if unused:
                    new_color = random.choice(unused)
                else:
                    # All colors used, pick rare ones
                    # Weight by 1/(freq+1)
                    weights = [1.0 / (freq[c] + 1) for c in range(MAX_COLORS)]
                    total_w = sum(weights)
                    r = random.random() * total_w
                    cumulative = 0
                    new_color = MAX_COLORS - 1
                    for c in range(MAX_COLORS):
                        cumulative += weights[c]
                        if r <= cumulative:
                            new_color = c
                            break
            else:
                new_color = random.randint(0, MAX_COLORS - 1)
            
            if new_color == old_color:
                continue
            
            # Check validity efficiently - only cycles involving edge e
            valid = True
            for c_idx in edge_to_cycles[e]:
                ci = cycle_indices[c_idx]
                c0 = current_colors[ci[0]] if ci[0] != e else new_color
                c1 = current_colors[ci[1]] if ci[1] != e else new_color
                c2 = current_colors[ci[2]] if ci[2] != e else new_color
                c3 = current_colors[ci[3]] if ci[3] != e else new_color
                if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                    valid = False
                    break
            
            if valid:
                # Calculate new score incrementally
                old_freq = freq[old_color]
                new_freq = freq[new_color]
                
                # Update freq temporarily
                freq[old_color] -= 1
                freq[new_color] += 1
                
                # Update distinct count
                new_distinct = current_distinct
                if old_freq == 1:
                    new_distinct -= 1
                if new_freq == 0:
                    new_distinct += 1
                
                # Acceptance criterion
                if new_distinct >= current_distinct:
                    accept = True
                else:
                    delta = new_distinct - current_distinct  # negative
                    accept = random.random() < np.exp(delta / T)
                
                if accept:
                    current_colors[e] = new_color
                    current_distinct = new_distinct
                    no_improve_steps = 0
                    
                    if new_distinct > best_score:
                        best_score = new_distinct
                        best_colors = current_colors.copy()
                else:
                    # Revert
                    current_colors[e] = old_color
                    freq[old_color] += 1
                    freq[new_color] -= 1
                    no_improve_steps += 1
            else:
                no_improve_steps += 1
            
            # Cool down
            T *= cooling
            if T < T_min:
                T = T_min
            
            # Restart if stuck
            if no_improve_steps > max_no_improve:
                # Perturb randomly
                for _ in range(3):
                    pe = random.randint(0, NUM_EDGES - 1)
                    pc = random.randint(0, MAX_COLORS - 1)
                    current_colors[pe] = pc
                    # Update freq
                    old_pc = current_colors[pe]
                    # This is getting complex, just reset freq
                    freq = [0] * MAX_COLORS
                    for c in current_colors:
                        freq[c] += 1
                    current_distinct = count_colors(current_colors)
                    no_improve_steps = 0
                break
    
    return best_colors
