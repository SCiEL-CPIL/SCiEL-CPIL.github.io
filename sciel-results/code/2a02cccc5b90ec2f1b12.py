import random
import time

N = 16
EDGES = []
for i in range(N):
    for j in range(i+1, N):
        EDGES.append((i, j))

edge_index = {}
for idx, (u, v) in enumerate(EDGES):
    edge_index[(u, v)] = idx
    edge_index[(v, u)] = idx

# Precompute constraints
constraints = [[] for _ in range(len(EDGES))]

for idx, (u, v) in enumerate(EDGES):
    others = [k for k in range(N) if k != u and k != v]
    for x_idx in range(len(others)):
        for y_idx in range(x_idx + 1, len(others)):
            x = others[x_idx]
            y = others[y_idx]
            # Cycle 1: u-x-y-v-u
            e1 = edge_index[(u, x)]
            e2 = edge_index[(x, y)]
            e3 = edge_index[(y, v)]
            constraints[idx].append((e1, e2, e3))
            
            # Cycle 2: u-y-x-v-u
            e4 = edge_index[(u, y)]
            e5 = edge_index[(y, x)]
            e6 = edge_index[(x, v)]
            constraints[idx].append((e4, e5, e6))

def is_valid_color(colors, edge_idx, new_color):
    for a, b, c in constraints[edge_idx]:
        ca = colors[a]
        cb = colors[b]
        cc = colors[c]
        if ca != cb and cb != cc and ca != cc:
            if new_color != ca and new_color != cb and new_color != cc:
                return False
    return True

def construct():
    start_time = time.time()
    timeout = 55.0  # Leave 5s buffer
    
    best_count = 0
    best_colors = [0] * len(EDGES)
    
    # Multiple restarts
    while time.time() - start_time < timeout:
        colors = [0] * len(EDGES)
        next_color = 1
        used_colors = set([0])
        
        # Greedy phase: try to assign new colors
        # Randomize edge order
        edge_order = list(range(len(EDGES)))
        random.shuffle(edge_order)
        
        for e_idx in edge_order:
            if is_valid_color(colors, e_idx, next_color):
                colors[e_idx] = next_color
                used_colors.add(next_color)
                next_color += 1
        
        current_count = len(set(colors))
        
        # Local Search Phase
        # We try to improve the count by recoloring edges.
        # Move 1: Change an edge to a new color.
        # Move 2: Change an edge to an existing color to enable Move 1 elsewhere.
        
        improved = True
        no_improve_count = 0
        max_local_steps = 5000
        local_step = 0
        
        while improved and local_step < max_local_steps and time.time() - start_time < timeout:
            improved = False
            local_step += 1
            
            # Try to add a new color
            if random.random() < 0.5:
                # Try to introduce a new color
                new_c = max(used_colors) + 1
                # Pick a random edge
                e_idx = random.randint(0, len(EDGES)-1)
                if is_valid_color(colors, e_idx, new_c):
                    colors[e_idx] = new_c
                    used_colors.add(new_c)
                    improved = True
                    no_improve_count = 0
                    continue
            
            # Try to recolor an existing edge to a different existing color
            # This helps "shuffle" the board
            e_idx = random.randint(0, len(EDGES)-1)
            current_c = colors[e_idx]
            
            # Try all existing colors? Or just a few?
            # Let's try a random existing color different from current
            other_colors = [c for c in used_colors if c != current_c]
            if other_colors:
                new_c = random.choice(other_colors)
                if is_valid_color(colors, e_idx, new_c):
                    # Check if this reduces the count of used colors?
                    # We need to be careful not to lose a color.
                    # If current_c is used only on this edge, and we change it, we lose that color.
                    # Count usage of current_c
                    if colors.count(current_c) == 1:
                        # We would lose a color. Only do this if we think it helps?
                        # For now, let's avoid losing colors unless we gain one immediately.
                        # But gaining one requires a subsequent move.
                        # Let's only allow recoloring if the color is used > 1 time, 
                        # OR if we are willing to take a risk.
                        # Actually, if we lose a color, count decreases. 
                        # We want to maximize count. So we should avoid decreasing count.
                        # So only recolor if count(current_c) > 1.
                        if colors.count(current_c) > 1:
                            colors[e_idx] = new_c
                            improved = True
                            no_improve_count = 0
                            continue
                    else:
                        colors[e_idx] = new_c
                        improved = True
                        no_improve_count = 0
                        continue
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > 100:
                    break

        final_count = len(set(colors))
        if final_count > best_count:
            best_count = final_count
            best_colors = colors[:]
            
    return best_colors
