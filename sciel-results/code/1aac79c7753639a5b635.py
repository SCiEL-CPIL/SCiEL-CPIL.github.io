import time
import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLORS = 120

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill

    # Precompute edge indices
    edge_idx = [[-1]*N for _ in range(N)]
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1

    # Precompute cycles and their edge indices
    cycles = []
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        e1 = edge_idx[a][b]
        e2 = edge_idx[b][c]
        e3 = edge_idx[c][d]
        e4 = edge_idx[d][a]
        cid = len(cycles)
        cycles.append((e1, e2, e3, e4))
        edge_to_cycles[e1].append(cid)
        edge_to_cycles[e2].append(cid)
        edge_to_cycles[e3].append(cid)
        edge_to_cycles[e4].append(cid)
        
        # Cycle 2: a-b-d-c-a
        e1 = edge_idx[a][b]
        e2 = edge_idx[b][d]
        e3 = edge_idx[d][c]
        e4 = edge_idx[c][a]
        cid = len(cycles)
        cycles.append((e1, e2, e3, e4))
        edge_to_cycles[e1].append(cid)
        edge_to_cycles[e2].append(cid)
        edge_to_cycles[e3].append(cid)
        edge_to_cycles[e4].append(cid)
        
        # Cycle 3: a-c-b-d-a
        e1 = edge_idx[a][c]
        e2 = edge_idx[c][b]
        e3 = edge_idx[b][d]
        e4 = edge_idx[d][a]
        cid = len(cycles)
        cycles.append((e1, e2, e3, e4))
        edge_to_cycles[e1].append(cid)
        edge_to_cycles[e2].append(cid)
        edge_to_cycles[e3].append(cid)
        edge_to_cycles[e4].append(cid)

    def is_bad_cycle(colors, c_idx):
        e1, e2, e3, e4 = c_idx
        c1 = colors[e1]
        c2 = colors[e2]
        c3 = colors[e3]
        c4 = colors[e4]
        return c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4

    def count_distinct(colors):
        return len(set(colors))

    # Initialize with recursive bisection
    def make_initial_coloring(seed_offset=0):
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

    def local_search(colors, deadline, seed_val):
        random.seed(seed_val)
        
        freq = [0] * MAX_COLORS
        for c in colors:
            freq[c] += 1
        
        bad_count = 0
        for c_idx in cycles:
            if is_bad_cycle(colors, c_idx):
                bad_count += 1
            
        best_score = count_distinct(colors)
        best_colors = colors.copy()
        best_bad = bad_count
        
        # Track unused colors count
        unused_count = MAX_COLORS - best_score
        
        no_improve_steps = 0
        max_no_improve = 5000
        
        while time.time() < deadline:
            # Pick a random edge
            e = random.randint(0, NUM_EDGES - 1)
            old_color = colors[e]
            
            # Pick a new color
            if random.random() < 0.7:
                # Bias towards unused colors
                if unused_count > 0:
                    # Find an unused color
                    for i in range(MAX_COLORS):
                        if freq[i] == 0:
                            new_color = i
                            break
                    else:
                        new_color = random.randint(0, MAX_COLORS - 1)
                else:
                    new_color = random.randint(0, MAX_COLORS - 1)
            else:
                new_color = random.randint(0, MAX_COLORS - 1)

            if new_color == old_color:
                continue

            # Calculate delta in bad_count
            delta = 0
            for cid in edge_to_cycles[e]:
                c_idx = cycles[cid]
                was_bad = is_bad_cycle(colors, c_idx)
                colors[e] = new_color
                is_bad_now = is_bad_cycle(colors, c_idx)
                colors[e] = old_color
                
                if was_bad and not is_bad_now:
                    delta -= 1
                elif not was_bad and is_bad_now:
                    delta += 1
            
            new_bad_count = bad_count + delta
            
            if new_bad_count <= best_bad:
                # Accept if it doesn't increase bad cycles
                colors[e] = new_color
                freq[old_color] -= 1
                freq[new_color] += 1
                bad_count = new_bad_count
                
                # Update score tracking
                old_score = best_score
                if freq[old_color] == 0 and old_color != new_color:
                    # Old color disappeared
                    if freq[new_color] == 1:
                        # New color was just introduced
                        pass  # Net change 0
                    else:
                        best_score -= 1
                        unused_count += 1
                elif freq[new_color] == 1 and new_color != old_color:
                    # New color was just introduced
                    if freq[old_color] == 0:
                        pass  # Net change 0
                    else:
                        best_score += 1
                        unused_count -= 1
                
                if best_score > best_bad:
                    # This shouldn't happen, but check
                    pass
                
                if best_score > best_score:
                    pass
                
                # Check if we improved
                if best_score > best_score:
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
                
                # Actually, let's track best score properly
                current_score = count_distinct(colors)
                if current_score > best_score:
                    best_score = current_score
                    best_colors = colors.copy()
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
                
                # Recalculate freq-based score if needed
                if no_improve_steps > max_no_improve:
                    # Restart with different seed
                    return best_colors, best_score
            else:
                # Simulated annealing: accept with small probability
                if random.random() < 0.01:
                    colors[e] = new_color
                    freq[old_color] -= 1
                    freq[new_color] += 1
                    bad_count = new_bad_count
                    no_improve_steps += 1

        return best_colors, best_score

    # Multiple restarts with different seeds
    best_overall_score = 0
    best_overall_colors = make_initial_coloring()
    
    num_restarts = 5
    time_per_restart = (deadline - start_time) / num_restarts
    
    for r in range(num_restarts):
        if time.time() >= deadline - 2.0:
            break
        
        seed_val = 42 + r * 1000
        init_colors = make_initial_coloring(r)
        restart_deadline = min(deadline - 1.0, start_time + (r + 1) * time_per_restart)
        
        colors, score = local_search(init_colors, restart_deadline, seed_val)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_colors = colors

    return best_overall_colors
