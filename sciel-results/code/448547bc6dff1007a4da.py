"""Design a D4-symmetric photonic crystal unit cell using Gray-Scott Turing patterns."""
import numpy as np
import time

def _enforce_d4(cell):
    """Enforce full D4 symmetry: 90° rotation invariance, horizontal/vertical/diagonal reflections."""
    n = len(cell)
    # Work with float array for averaging
    arr = np.array(cell, dtype=np.float64)
    # 90° rotations
    rot90 = np.rot90(arr, 1)
    rot180 = np.rot90(arr, 2)
    rot270 = np.rot90(arr, 3)
    # Reflections
    refl_h = arr[::-1, :]
    refl_v = arr[:, ::-1]
    # Diagonal reflections
    diag_main = arr.T
    diag_anti = np.rot90(arr.T, 1)  # anti-diagonal reflection
    # Average all 8 symmetry operations
    sym_avg = (arr + rot90 + rot180 + rot270 + refl_h + refl_v + diag_main + diag_anti) / 8.0
    # Threshold to binary
    result = (sym_avg >= 0.5).astype(int)
    return result.tolist()

def _gray_scott_step(u, v, F, k, dt, dx, dy):
    """One step of Gray-Scott model."""
    # Laplacian using 5-point stencil
    lap_u = (np.roll(u, 1, 0) + np.roll(u, -1, 0) + np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4*u) / (dx*dx)
    lap_v = (np.roll(v, 1, 0) + np.roll(v, -1, 0) + np.roll(v, 1, 1) + np.roll(v, -1, 1) - 4*v) / (dy*dy)
    
    du = dt * (lap_u - u*v*v + F*(1-u))
    dv = dt * (lap_v + u*v*v - (F+k)*v)
    
    u_new = u + du
    v_new = v + dv
    return u_new, v_new

def _generate_turing_pattern(size, n_steps, F, k, seed_val=0.5):
    """Generate a D4-symmetric Turing pattern using Gray-Scott model."""
    # Initialize with D4-symmetric pattern
    u = np.ones((size, size))
    v = np.zeros((size, size))
    
    # Create a symmetric seed: place spots at symmetric positions
    center = (size - 1) / 2.0
    # Place initial v spots at D4-symmetric positions
    offset = size // 4
    for di in [-offset, offset]:
        for dj in [-offset, offset]:
            i = int(center + di)
            j = int(center + dj)
            if 0 <= i < size and 0 <= j < size:
                v[i, j] = 1.0
                u[i, j] = 0.0
            # Also place at reflected positions for full D4
            for ri in [i, size-1-i]:
                for rj in [j, size-1-j]:
                    if 0 <= ri < size and 0 <= rj < size:
                        v[ri, rj] = 1.0
                        u[ri, rj] = 0.0
    
    dx = 1.0
    dy = 1.0
    dt = 1.0
    
    for step in range(n_steps):
        u, v = _gray_scott_step(u, v, F, k, dt, dx, dy)
        # Enforce D4 symmetry periodically
        if step % 5 == 0:
            v_arr = v.copy()
            u_arr = u.copy()
            # Enforce symmetry on v (the pattern-defining variable)
            v_sym = (v_arr + np.rot90(v_arr,1) + np.rot90(v_arr,2) + np.rot90(v_arr,3) + 
                     v_arr[::-1,:] + v_arr[:,::-1] + v_arr.T + np.rot90(v_arr.T,1)) / 8.0
            v = v_sym
            # Keep u consistent
            u_sym = (u_arr + np.rot90(u_arr,1) + np.rot90(u_arr,2) + np.rot90(u_arr,3) + 
                     u_arr[::-1,:] + u_arr[:,::-1] + u_arr.T + np.rot90(u_arr.T,1)) / 8.0
            u = u_sym
    
    # Threshold v to get binary pattern
    threshold = np.mean(v)
    pattern = (v > threshold).astype(int)
    return pattern.tolist()

def design(problem):
    """Design a D4-symmetric photonic crystal unit cell for maximum TM band gap."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 230s limit
    
    size = problem["pixels"]
    
    # Try multiple Gray-Scott parameter sets to find good patterns
    # Parameters (F, k) that produce different Turing patterns
    param_sets = [
        (0.035, 0.065, 50),   # Spots
        (0.032, 0.062, 50),   # Different spots
        (0.028, 0.058, 50),   # Stripes
        (0.037, 0.068, 40),   # Mixed
        (0.030, 0.060, 60),   # Fine structure
        (0.034, 0.064, 45),   # Another spot regime
        (0.026, 0.055, 55),   # Coarser stripes
        (0.036, 0.066, 35),   # Larger spots
    ]
    
    best_cell = None
    best_score = -1.0
    
    # Also prepare some analytical fallback designs
    fallback_designs = []
    
    # Fallback 1: Concentric rings (D4 symmetric)
    cell_rings = []
    for i in range(size):
        row = []
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = x*x + y*y
            # Alternating rings
            ring = int(r * size * 0.5)
            row.append(1 if ring % 2 == 0 else 0)
        cell_rings.append(row)
    fallback_designs.append(_enforce_d4(cell_rings))
    
    # Fallback 2: Checkerboard with larger blocks
    cell_blocks = []
    block_size = 4
    for i in range(size):
        row = []
        for j in range(size):
            bi = i // block_size
            bj = j // block_size
            row.append(1 if (bi + bj) % 2 == 0 else 0)
        cell_blocks.append(row)
    fallback_designs.append(_enforce_d4(cell_blocks))
    
    # Fallback 3: Cross pattern
    cell_cross = []
    for i in range(size):
        row = []
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            # Dielectric where |x| < 0.15 or |y| < 0.15
            if abs(x) < 0.15 or abs(y) < 0.15:
                row.append(1)
            else:
                row.append(0)
        cell_cross.append(row)
    fallback_designs.append(_enforce_d4(cell_cross))
    
    # Fallback 4: Four-quadrant pattern
    cell_quad = []
    for i in range(size):
        row = []
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            # Dielectric in alternating quadrants with circular boundaries
            in_quad = (x > 0 and y > 0) or (x < 0 and y < 0)
            r = x*x + y*y
            if in_quad and r < 0.3**2:
                row.append(1)
            elif not in_quad and r < 0.2**2:
                row.append(1)
            else:
                row.append(0)
        cell_quad.append(row)
    fallback_designs.append(_enforce_d4(cell_quad))
    
    # Fallback 5: Original rod (baseline)
    cell_rod = []
    for i in range(size):
        row = []
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            row.append(1 if x*x + y*y < 0.2**2 else 0)
        cell_rod.append(row)
    fallback_designs.append(_enforce_d4(cell_rod))
    
    # Try Gray-Scott patterns
    for F, k, n_steps in param_sets:
        if time.time() > deadline - 5:
            break
        try:
            cell = _generate_turing_pattern(size, n_steps, F, k)
            cell = _enforce_d4(cell)
            fallback_designs.append(cell)
        except Exception:
            continue
    
    # We can't easily score locally without the full band solver,
    # so we'll use heuristics to pick the best design
    # Heuristic: prefer designs with:
    # 1. Roughly 40-60% dielectric filling
    # 2. Complex microstructure (not too regular)
    # 3. Good D4 symmetry (already enforced)
    
    def _heuristic_score(cell):
        """Heuristic score for design quality."""
        arr = np.array(cell)
        filling = arr.mean()
        # Penalize extreme fillings
        fill_score = 1.0 - abs(filling - 0.5) * 2
        
        # Measure complexity via edge density
        edges = 0
        for i in range(size):
            for j in range(size):
                if i+1 < size and arr[i,j] != arr[i+1,j]:
                    edges += 1
                if j+1 < size and arr[i,j] != arr[i,j+1]:
                    edges += 1
        edge_density = edges / (size * size)
        # Moderate edge density is good
        edge_score = 1.0 - abs(edge_density - 0.15)
        
        # Measure local variance (microstructure complexity)
        # Compute local means in 4x4 blocks
        local_vars = []
        for i in range(0, size, 4):
            for j in range(0, size, 4):
                block = arr[i:i+4, j:j+4]
                local_vars.append(block.var())
        avg_local_var = np.mean(local_vars) if local_vars else 0
        var_score = min(avg_local_var * 4, 1.0)
        
        return fill_score * 0.4 + edge_score * 0.3 + var_score * 0.3
    
    # Score all candidate designs
    scored = []
    for idx, cell in enumerate(fallback_designs):
        score = _heuristic_score(cell)
        scored.append((score, idx, cell))
    
    # Sort by heuristic score
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Return the best heuristic design
    if scored:
        return scored[0][2]
    else:
        # Ultimate fallback
        return fallback_designs[0]
