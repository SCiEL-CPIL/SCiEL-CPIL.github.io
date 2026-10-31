def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Grid initialization
    pos = np.zeros((N, 2))
    k = 0
    for i in range(5):
        for j in range(5):
            if k < N:
                pos[k, 0] = (i + 0.5) / 5.0
                pos[k, 1] = (j + 0.5) / 5.0
                k += 1
    if k < N:
        pos[k, 0] = 0.5
        pos[k, 1] = 0.5
    
    radii = np.full(N, 0.03)
    
    # Phase 1: Force-directed relaxation with growing radii
    dt = 0.005
    for step in range(50000):
        if time.time() > deadline - 10:
            break
        
        forces = np.zeros((N, 2))
        
        # Inter-circle repulsion
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-6
                min_dist = radii[i] + radii[j]
                overlap = min_dist - dist
                if overlap > 0:
                    # Spring force proportional to overlap
                    f = 100.0 * overlap
                    fx = f * dx / dist
                    fy = f * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Wall constraints
        for i in range(N):
            if pos[i, 0] < radii[i]:
                forces[i, 0] += 100.0 * (radii[i] - pos[i, 0])
            if pos[i, 0] > 1 - radii[i]:
                forces[i, 0] -= 100.0 * (pos[i, 0] + radii[i] - 1)
            if pos[i, 1] < radii[i]:
                forces[i, 1] += 100.0 * (radii[i] - pos[i, 1])
            if pos[i, 1] > 1 - radii[i]:
                forces[i, 1] -= 100.0 * (pos[i, 1] + radii[i] - 1)
        
        # Update
        pos += forces * dt
        pos = np.clip(pos, 0, 1)
        
        # Grow radii slowly
        radii += 0.00005
        
        # Cap radii by wall distance
        for i in range(N):
            max_r = min(pos[i,0], 1-pos[i,0], pos[i,1], 1-pos[i,1])
            if radii[i] > max_r:
                radii[i] = max_r
    
    # Phase 2: Greedy radius maximization
    for _ in range(200):
        if time.time() > deadline - 5:
            break
        for i in range(N):
            max_r = min(pos[i,0], 1-pos[i,0], pos[i,1], 1-pos[i,1])
            for j in range(N):
                if i != j:
                    dx = pos[i,0] - pos[j,0]
                    dy = pos[i,1] - pos[j,1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    max_r = min(max_r, dist - radii[j])
            if max_r > radii[i]:
                radii[i] = max_r
    
    # Phase 3: Local search - try moving each circle to allow larger radius
    rng = np.random.RandomState(42)
    for _ in range(5000):
        if time.time() > deadline - 2:
            break
        i = rng.randint(N)
        best_r = radii[i]
        best_x, best_y = pos[i, 0], pos[i, 1]
        
        # Try several position perturbations
        for _ in range(20):
            nx = pos[i, 0] + rng.uniform(-0.02, 0.02)
            ny = pos[i, 1] + rng.uniform(-0.02, 0.02)
            nx = np.clip(nx, 0, 1)
            ny = np.clip(ny, 0, 1)
            
            max_r = min(nx, 1-nx, ny, 1-ny)
            for j in range(N):
                if i != j:
                    dx = nx - pos[j, 0]
                    dy = ny - pos[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    max_r = min(max_r, dist - radii[j])
            
            if max_r > best_r:
                best_r = max_r
                best_x, best_y = nx, ny
        
        if best_r > radii[i] + 1e-6:
            pos[i, 0], pos[i, 1] = best_x, best_y
            radii[i] = best_r
    
    # Final enforcement
    for _ in range(100):
        changed = False
        for i in range(N):
            max_r = min(pos[i,0], 1-pos[i,0], pos[i,1], 1-pos[i,1])
            if radii[i] > max_r:
                radii[i] = max(0, max_r)
                changed = True
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r - 1e-9:
                    if sum_r > 1e-10:
                        scale = dist / sum_r
                        radii[i] *= scale
                        radii[j] *= scale
                    else:
                        radii[i] = radii[j] = 0
                    changed = True
        if not changed:
            break
    
    result = np.column_stack([pos, radii])
    return result
