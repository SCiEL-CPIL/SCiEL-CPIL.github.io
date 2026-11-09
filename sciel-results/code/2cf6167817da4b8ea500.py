import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin for 120s limit
    
    n = 16
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
        
    num_edges = len(edges)
    
    # Precompute K4s
    k4s = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        e_idx = [
            edge_idx[(a, b)], edge_idx[(a, c)], edge_idx[(a, d)],
            edge_idx[(b, c)], edge_idx[(b, d)], edge_idx[(c, d)]
        ]
        k4s.append(e_idx)
        
    num_k4s = len(k4s)
    
    # Edge to K4 mapping
    edge_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_list in enumerate(k4s):
        for ei in e_list:
            edge_k4s[ei].append(k4_idx)

    max_colors = 16
    
    # State variables
    # colors: list of ints
    # k4_counts: list of lists of counts for each color in each K4
    # k4_mask: list of int masks
    # color_counts: global count of each color to quickly get num_colors
    
    def initialize_colors(p):
        cols = [0] * num_edges
        for idx, (i, j) in enumerate(edges):
            cols[idx] = (i + j) % p
        return cols

    def update_state(colors):
        k4_counts = [[0] * max_colors for _ in range(num_k4s)]
        k4_mask = [0] * num_k4s
        color_counts = [0] * max_colors
        
        for ei in range(num_edges):
            c = colors[ei]
            color_counts[c] += 1
            for k4_idx in edge_k4s[ei]:
                k4_counts[k4_idx][c] += 1
                k4_mask[k4_idx] |= (1 << c)
        
        viol = 0
        for mask in k4_mask:
            if mask.bit_count() < 5:
                viol += 1
                
        nc = sum(1 for c in color_counts if c > 0)
        
        return k4_counts, k4_mask, viol, nc, color_counts

    def get_violations_from_masks(k4_mask):
        v = 0
        for mask in k4_mask:
            if mask.bit_count() < 5:
                v += 1
        return v

    # Initial solution
    # Try a simple valid initialization. (i+j)%17 is known to be valid for small n, but let's use a robust random start if that fails or just stick to a known good pattern.
    # Actually, for K16, (i+j)%17 might not be valid. Let's use a random start with repair.
    
    # Strategy: Start with a random coloring using ~15 colors, then repair violations.
    # Then minimize colors.
    
    best_valid_colors = None
    best_valid_nc = float('inf')
    
    # Try a few simple deterministic starts first, they are fast
    for p in [15, 16, 17]:
        init_cols = initialize_colors(p)
        kc, km, viol, nc, cc = update_state(init_cols)
        if viol == 0:
            if nc < best_valid_nc:
                best_valid_nc = nc
                best_valid_colors = init_cols[:]
            if nc <= 13:
                return init_cols

    if best_valid_colors is None:
        # Start with random colors in 0..15
        init_cols = [random.randint(0, 15) for _ in range(num_edges)]
        kc, km, viol, nc, cc = update_state(init_cols)
    else:
        init_cols = best_valid_colors[:]
        kc, km, viol, nc, cc = update_state(init_cols)

    current_colors = init_cols[:]
    current_kc = kc
    current_km = km
    current_viol = viol
    current_nc = nc
    current_cc = cc
    
    best_colors = current_colors[:]
    best_nc = current_nc
    best_viol = current_viol

    T = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        if current_viol > 0:
            # Repair phase
            # Find a violated K4
            # Instead of picking random edge, pick an edge from a violated K4
            # This is much more effective
            
            # Find indices of violated K4s
            # To save time, we can just pick a random K4 and check if it's violated
            # Or maintain a list? No, just pick random K4 until we find a violated one.
            
            violated_k4_idx = -1
            for _ in range(10): # Try a few times to find a violated K4
                k4_idx = random.randint(0, num_k4s - 1)
                if current_km[k4_idx].bit_count() < 5:
                    violated_k4_idx = k4_idx
                    break
            
            if violated_k4_idx == -1:
                # Should not happen if current_viol > 0, but safety
                # Force update
                current_viol = get_violations_from_masks(current_km)
                if current_viol == 0:
                    # Sync best
                    if current_nc < best_nc:
                        best_nc = current_nc
                        best_colors = current_colors[:]
                    # Switch to minimize mode
                    # But we need to update state? No, state is consistent.
                    # Just continue to next iteration which will go to else branch
                    pass
                else:
                    # Try again next loop
                    continue

            # Pick a random edge from this K4
            e_list = k4s[violated_k4_idx]
            e_idx = random.choice(e_list)
            old_c = current_colors[e_idx]
            
            # Try to change to a color that increases diversity in this K4
            # Ideally, a color not present in this K4
            present_in_k4 = current_km[violated_k4_idx]
            # Find a color not in present_in_k4
            # If none, pick a random one
            
            # Simple heuristic: pick a random color 0..15
            new_c = random.randint(0, max_colors - 1)
            if new_c == old_c:
                continue
            
            # Calculate delta violations for all K4s containing e_idx
            delta_v = 0
            affected_k4s = edge_k4s[e_idx]
            
            # We need to check if the move is valid or improves violation count
            # For repair, we accept if delta_v < 0
            # For moves that keep delta_v == 0, we might accept with prob T to escape local minima
            
            is_improvement = False
            is_equal = False
            
            for k4_idx in affected_k4s:
                old_count_old_c = current_kc[k4_idx][old_c]
                old_count_new_c = current_kc[k4_idx][new_c]
                
                new_count_old_c = old_count_old_c - 1
                new_count_new_c = old_count_new_c + 1
                
                old_mask = current_km[k4_idx]
                new_mask = old_mask
                if new_count_old_c == 0:
                    new_mask &= ~(1 << old_c)
                if new_count_new_c == 1:
                    new_mask |= (1 << new_c)
                
                old_valid = (old_mask.bit_count() >= 5)
                new_valid = (new_mask.bit_count() >= 5)
                
                if old_valid and not new_valid:
                    delta_v += 1
                elif not old_valid and new_valid:
                    delta_v -= 1
                elif old_valid and new_valid:
                    pass
                else: # both invalid
                    pass
            
            if delta_v < 0:
                is_improvement = True
            elif delta_v == 0:
                is_equal = True
            else:
                # Reject worsening moves in repair phase usually, 
                # but if we are stuck, we might need to allow some? 
                # Let's stick to strict improvement or equal for now to ensure progress.
                # Actually, if we are stuck in a local minimum of violations, we might need to accept equal moves.
                if random.random() < T:
                    is_equal = True
                else:
                    continue
            
            # Apply move
            current_colors[e_idx] = new_c
            for k4_idx in affected_k4s:
                current_kc[k4_idx][old_c] -= 1
                current_kc[k4_idx][new_c] += 1
                old_mask = current_km[k4_idx]
                new_mask = old_mask
                if current_kc[k4_idx][old_c] == 0:
                    new_mask &= ~(1 << old_c)
                if current_kc[k4_idx][new_c] == 1:
                    new_mask |= (1 << new_c)
                current_km[k4_idx] = new_mask
            
            # Update global color counts
            current_cc[old_c] -= 1
            current_cc[new_c] += 1
            
            current_viol += delta_v
            
            if current_viol == 0:
                # Recalculate nc
                current_nc = sum(1 for c in current_cc if c > 0)
                if current_nc < best_nc:
                    best_nc = current_nc
                    best_colors = current_colors[:]
        
        else:
            # Valid phase: Minimize colors
            # Pick a random edge
            e_idx = random.randint(0, num_edges - 1)
            old_c = current_colors[e_idx]
            
            # Try to change to a color that reduces the total number of colors
            # This happens if old_c is unique (count 1) and new_c is already used (count > 0)
            # OR if old_c is used multiple times, but we change to another existing color? No, that doesn't reduce count.
            # So, prioritize edges with unique colors.
            
            if current_cc[old_c] == 1:
                # This edge is the only one with color old_c.
                # If we change it to an existing color (count > 0), we reduce nc by 1.
                # We must ensure validity is preserved.
                
                # Pick a random existing color (count > 0)
                existing_colors = [c for c in range(max_colors) if current_cc[c] > 0]
                if not existing_colors:
                    break
                
                new_c = random.choice(existing_colors)
                if new_c == old_c:
                    continue
                
                # Check validity
                valid_move = True
                for k4_idx in edge_k4s[e_idx]:
                    old_count_old_c = current_kc[k4_idx][old_c]
                    old_count_new_c = current_kc[k4_idx][new_c]
                    
                    new_count_old_c = old_count_old_c - 1
                    new_count_new_c = old_count_new_c + 1
                    
                    old_mask = current_km[k4_idx]
                    new_mask = old_mask
                    if new_count_old_c == 0:
                        new_mask &= ~(1 << old_c)
                    if new_count_new_c == 1:
                        new_mask |= (1 << new_c)
                    
                    if new_mask.bit_count() < 5:
                        valid_move = False
                        break
                
                if valid_move:
                    # Apply
                    current_colors[e_idx] = new_c
                    for k4_idx in edge_k4s[e_idx]:
                        current_kc[k4_idx][old_c] -= 1
                        current_kc[k4_idx][new_c] += 1
                        old_mask = current_km[k4_idx]
                        new_mask = old_mask
                        if current_kc[k4_idx][old_c] == 0:
                            new_mask &= ~(1 << old_c)
                        if current_kc[k4_idx][new_c] == 1:
                            new_mask |= (1 << new_c)
                        current_km[k4_idx] = new_mask
                    
                    current_cc[old_c] -= 1
                    current_cc[new_c] += 1
                    
                    current_nc -= 1 # Since old_c became 0 and new_c was >0
                    if current_nc < best_nc:
                        best_nc = current_nc
                        best_colors = current_colors[:]
            else:
                # old_c is not unique. Changing it to another existing color doesn't reduce nc.
                # Changing it to a new color increases nc.
                # So we only do moves that keep nc same or reduce violations (if any creep in? No, we are in valid phase).
                # In valid phase, we should only accept moves that keep it valid.
                # To escape local minima in nc, we might need to allow moves that keep nc same but change structure?
                # Or allow moves that increase nc slightly to escape? 
                # Let's try: Pick a random edge, pick a random existing color.
                # If valid, accept. This shuffles the coloring, might allow a future unique-color move.
                
                existing_colors = [c for c in range(max_colors) if current_cc[c] > 0]
                if not existing_colors:
                    break
                new_c = random.choice(existing_colors)
                if new_c == old_c:
                    continue
                
                valid_move = True
                for k4_idx in edge_k4s[e_idx]:
                    old_count_old_c = current_kc[k4_idx][old_c]
                    old_count_new_c = current_kc[k4_idx][new_c]
                    
                    new_count_old_c = old_count_old_c - 1
                    new_count_new_c = old_count_new_c + 1
                    
                    old_mask = current_km[k4_idx]
                    new_mask = old_mask
                    if new_count_old_c == 0:
                        new_mask &= ~(1 << old_c)
                    if new_count_new_c == 1:
                        new_mask |= (1 << new_c)
                    
                    if new_mask.bit_count() < 5:
                        valid_move = False
                        break
                
                if valid_move:
                    # Apply
                    current_colors[e_idx] = new_c
                    for k4_idx in edge_k4s[e_idx]:
                        current_kc[k4_idx][old_c] -= 1
                        current_kc[k4_idx][new_c] += 1
                        old_mask = current_km[k4_idx]
                        new_mask = old_mask
                        if current_kc[k4_idx][old_c] == 0:
                            new_mask &= ~(1 << old_c)
                        if current_kc[k4_idx][new_c] == 1:
                            new_mask |= (1 << new_c)
                        current_km[k4_idx] = new_mask
                    
                    current_cc[old_c] -= 1
                    current_cc[new_c] += 1
                    # nc stays same
                    
        T *= cooling
        if T < 0.01:
            T = 0.01

    # Final check and compaction
    # Ensure best_colors is valid
    kc, km, viol, nc, cc = update_state(best_colors)
    if viol > 0:
        # Fallback to a known valid solution if best is invalid
        # Try (i+j)%17
        cols = initialize_colors(17)
        kc, km, viol, nc, cc = update_state(cols)
        if viol == 0:
            best_colors = cols
        else:
            # Absolute fallback: random valid? 
            # Just return the first valid we found or a trivial valid one.
            # (i+j)%17 is very likely valid.
            pass

    # Compact labels
    unique = sorted(set(best_colors))
    mapping = {c: i for i, c in enumerate(unique)}
    return [mapping[c] for c in best_colors]
