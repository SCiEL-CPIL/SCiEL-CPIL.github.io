<<<<<<< SEARCH
def simulated_annealing(initial_state, time_limit=40.0):
    """
    Perform Simulated Annealing.
    """
    current_state = list(initial_state)
    current_obj = objective(current_state)
    best_state = current_state[:]
    best_obj = current_obj
    
    start_time = time.time()
    t_end = start_time + time_limit
    
    T = 1.0
    T_min = 1e-6
    cooling_rate = 0.999 # Slower cooling to explore more
    
    n_moves = 0
    while time.time() < t_end:
        # Generate neighbor
        neighbor = current_state[:]
        idx = random.randint(0, len(neighbor) - 1)
        if idx % 3 == 2: # Angle
            step = random.uniform(-0.1, 0.1)
        else: # Position
            step = random.uniform(-0.5, 0.5)
        neighbor[idx] += step
        
        neighbor_obj = objective(neighbor)
        
        delta = neighbor_obj - current_obj
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_state = neighbor
            current_obj = neighbor_obj
            
            if current_obj < best_obj:
                best_state = current_state[:]
                best_obj = current_obj
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        n_moves += 1
        # Check time every iteration for strict compliance
        if time.time() > t_end:
            break
                
    return best_state, best_obj
=======
def repair_state(state):
    """
    Apply collision resolution to a state.
    Returns a new state with overlaps reduced.
    """
    positions = []
    angles = []
    for i in range(N):
        x = state[3*i]
        y = state[3*i+1]
        th = state[3*i+2]
        positions.append([x, y])
        angles.append(th)
        
    verts_list = []
    for i in range(N):
        px = positions[i][0]
        py = positions[i][1]
        ang = angles[i]
        v = hex_vertices(ang)
        v[:, 0] += px
        v[:, 1] += py
        verts_list.append(v)
        
    # Iterate a few times to resolve overlaps
    for _ in range(5):
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dist_sq = dx*dx + dy*dy
                if dist_sq > 4.0:
                    continue
                
                overlap, mtv_normal = hexagon_collision_mtv(verts_list[i], verts_list[j])
                if overlap > 1e-6:
                    # Move both apart by half the overlap
                    push = overlap * 0.5 * 1.01 # Slightly more to ensure separation
                    # Move i in direction of mtv_normal
                    positions[i][0] += push * mtv_normal[0]
                    positions[i][1] += push * mtv_normal[1]
                    # Move j in opposite direction
                    positions[j][0] -= push * mtv_normal[0]
                    positions[j][1] -= push * mtv_normal[1]
                    
                    # Update vertices for next check
                    verts_list[i] = hex_vertices(angles[i])
                    verts_list[i][:, 0] += positions[i][0]
                    verts_list[i][:, 1] += positions[i][1]
                    
                    verts_list[j] = hex_vertices(angles[j])
                    verts_list[j][:, 0] += positions[j][0]
                    verts_list[j][:, 1] += positions[j][1]
                    
                    moved = True
        if not moved:
            break
            
    new_state = []
    for i in range(N):
        new_state.extend([positions[i][0], positions[i][1], angles[i]])
    return new_state

def simulated_annealing(initial_state, time_limit=40.0):
    """
    Perform Simulated Annealing with collision repair.
    """
    current_state = list(initial_state)
    current_obj = objective(current_state)
    best_state = current_state[:]
    best_obj = current_obj
    
    start_time = time.time()
    t_end = start_time + time_limit
    
    T = 1.0
    T_min = 1e-6
    cooling_rate = 0.999
    
    n_moves = 0
    while time.time() < t_end:
        # Generate neighbor
        neighbor = current_state[:]
        idx = random.randint(0, len(neighbor) - 1)
        if idx % 3 == 2: # Angle
            step = random.uniform(-0.1, 0.1)
        else: # Position
            step = random.uniform(-0.5, 0.5)
        neighbor[idx] += step
        
        # Repair overlaps
        neighbor = repair_state(neighbor)
        
        neighbor_obj = objective(neighbor)
        
        delta = neighbor_obj - current_obj
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_state = neighbor
            current_obj = neighbor_obj
            
            if current_obj < best_obj:
                best_state = current_state[:]
                best_obj = current_obj
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        n_moves += 1
        if time.time() > t_end:
            break
                
    return best_state, best_obj
>>>>>>> REPLACE

Finally, we need to adjust the time limits in `construct` to ensure we finish well within 120s. The repair step adds overhead, so we might need to reduce the number of iterations or time per run. Let's set the deadline to 110s and run 2 passes of 40s each, or 1 pass of 50s. Given the repair step is faster than pure SA wandering in invalid states, 40s should yield better results.

