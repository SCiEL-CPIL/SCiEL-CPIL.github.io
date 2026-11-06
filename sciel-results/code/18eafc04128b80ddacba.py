import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s limit to be safe
    
    # 1. Generate the 19 points of the hexagon of radius 2
    hex_pts = []
    for x in range(-2, 3):
        for y in range(-2, 3):
            if abs(x) <= 2 and abs(y) <= 2 and abs(x + y) <= 2:
                hex_pts.append((x, y))
    
    # 2. Generate the 18 points of the shell of radius 3
    shell_pts = []
    for x in range(-3, 4):
        for y in range(-3, 4):
            if abs(x) <= 3 and abs(y) <= 3 and abs(x + y) <= 3:
                # Check if it's in the hexagon
                in_hex = False
                for px, py in hex_pts:
                    if px == x and py == y:
                        in_hex = True
                        break
                if not in_hex:
                    shell_pts.append((x, y))
    
    # 3. Greedily add points from the shell to maximize edges
    current_pts = list(hex_pts)
    
    def count_edges(points):
        edges = 0
        n = len(points)
        pts_set = set(points)
        for i in range(n):
            x1, y1 = points[i]
            # Check 6 neighbors
            if (x1 + 1, y1) in pts_set: edges += 1
            if (x1 - 1, y1) in pts_set: edges += 1
            if (x1, y1 + 1) in pts_set: edges += 1
            if (x1, y1 - 1) in pts_set: edges += 1
            if (x1 + 1, y1 - 1) in pts_set: edges += 1
            if (x1 - 1, y1 + 1) in pts_set: edges += 1
        return edges // 2

    # Initial edge count
    current_edges = count_edges(current_pts)
    
    # Greedy addition
    remaining_shell = list(shell_pts)
    while len(current_pts) < 32 and remaining_shell:
        best_gain = -1
        best_idx = -1
        for i, p in enumerate(remaining_shell):
            # Calculate gain
            new_pts = current_pts + [p]
            # Optimization: only count edges involving p
            gain = 0
            for q in current_pts:
                dx = p[0] - q[0]
                dy = p[1] - q[1]
                if abs(dx) + abs(dy) + abs(dx + dy) == 2: # Triangular lattice neighbor check
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx >= 0:
            current_pts.append(remaining_shell[best_idx])
            remaining_shell.pop(best_idx)
            current_edges += best_gain
        else:
            break
            
    # 4. Local Search (Hill Climbing) to ensure optimality
    # We have 32 points. We can try swapping points or moving them.
    # Since we are restricted to the lattice and bounds, we can try moving any point to a nearby empty spot.
    
    best_edges = current_edges
    best_config = [list(p) for p in current_pts]
    
    # Simple local search: try moving each point to a neighbor cell if it improves edge count
    # To keep it fast, we limit iterations.
    improved = True
    iterations = 0
    max_iterations = 5000
    
    while improved and iterations < max_iterations and time.time() < deadline:
        improved = False
        iterations += 1
        for i in range(32):
            x, y = best_config[i]
            # Try 6 neighbors
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]:
                nx, ny = x + dx, y + dy
                if abs(nx) > 64 or abs(ny) > 64:
                    continue
                # Check if target is occupied
                occupied = False
                for j in range(32):
                    if i != j and best_config[j] == [nx, ny]:
                        occupied = True
                        break
                if occupied:
                    continue
                
                # Calculate new edge count
                # Remove i, add (nx, ny)
                # This is O(N) per move. 32 * 6 * 32 ~ 6000 ops per iteration.
                # 5000 iterations * 6000 = 30M ops. Should be fine in Python if optimized.
                
                # Quick check: 
                # Edges lost: neighbors of (x,y) in current set (excluding self)
                # Edges gained: neighbors of (nx,ny) in current set (excluding self)
                
                old_neighbors = 0
                new_neighbors = 0
                for j in range(32):
                    if i == j:
                        continue
                    jx, jy = best_config[j]
                    # Check if (x,y) is neighbor of (jx,jy)
                    if abs(x - jx) + abs(y - jy) + abs((x - jx) + (y - jy)) == 2:
                        old_neighbors += 1
                    # Check if (nx,ny) is neighbor of (jx,jy)
                    if abs(nx - jx) + abs(ny - jy) + abs((nx - jx) + (ny - jy)) == 2:
                        new_neighbors += 1
                
                delta = new_neighbors - old_neighbors
                if delta > 0:
                    best_config[i] = [nx, ny]
                    best_edges += delta
                    improved = True
                    break # Restart scan after move
    
    # Convert to list of lists
    return best_config
