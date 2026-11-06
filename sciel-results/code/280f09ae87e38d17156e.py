"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _apply_symmetry(cell):
    """Enforce 4-fold symmetry about center: flip rows, flip cols, transpose."""
    size = cell.shape[0]
    # Flip vertically
    cell = np.minimum(cell, cell[::-1, :].copy())
    # Flip horizontally  
    cell = np.minimum(cell, cell[:, ::-1].copy())
    # Transpose
    cell = np.minimum(cell, cell.T.copy())
    return cell

def _ca_step_2d(grid, rule_bits):
    """One step of 2D CA on a binary grid.
    
    rule_bits: dict mapping (center, neighbors_tuple) -> output bit
    Neighbors: 8 surrounding cells in order [NW,N,NE,E,SE,S,SW,W]
    """
    size = grid.shape[0]
    new_grid = np.zeros_like(grid)
    for i in range(size):
        for j in range(size):
            center = grid[i, j]
            # Get 8 neighbors (with wrap-around for periodic BC)
            neighbors = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni = (i + di) % size
                    nj = (j + dj) % size
                    neighbors.append(grid[ni, nj])
            # neighbors in order: NW, N, NE, W, E, SW, S, SE
            # Reorder to standard: [NW, N, NE, E, SE, S, SW, W]
            # Original order from loops: di=-1,dj=-1(NW), di=-1,dj=0(N), di=-1,dj=1(NE),
            #                          di=0,dj=-1(W), di=0,dj=1(E), 
            #                          di=1,dj=-1(SW), di=1,dj=0(S), di=1,dj=1(SE)
            # So: [NW, N, NE, W, E, SW, S, SE]
            # We want: [NW, N, NE, E, SE, S, SW, W]
            nw, n, ne, w, e, sw, s, se = neighbors
            key = (center, (nw, n, ne, e, se, s, sw, w))
            if key in rule_bits:
                new_grid[i, j] = rule_bits[key]
            else:
                # Default: keep center value
                new_grid[i, j] = center
    return new_grid

def _make_ca_cell(size, seed_pattern, n_steps, rule_type="conservative", threshold=0.5):
    """Generate a 32x32 symmetric binary cell using 2D cellular automaton.
    
    seed_pattern: initial 8x8 or 16x16 pattern (will be upscaled and symmetrized)
    n_steps: number of CA evolution steps
    rule_type: 'conservative' (mass-conserving) or 'xor' or 'majority'
    threshold: for some rules
    """
    # Create initial grid from seed pattern
    if isinstance(seed_pattern, list):
        seed = np.array(seed_pattern, dtype=int)
    else:
        seed = seed_pattern
    
    # Upscale seed to full size by nearest neighbor
    seed_size = seed.shape[0]
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            si = int(i * seed_size / size)
            sj = int(j * seed_size / size)
            cell[i, j] = seed[si, sj]
    
    # Apply initial symmetry
    cell = _apply_symmetry(cell)
    
    # Define CA rule based on rule_type
    if rule_type == "conservative":
        # Mass-conserving rule: output = 1 if center+neighbors sum >= threshold
        # Use a simple rule: output = center XOR (majority of neighbors)
        # This creates complex patterns while conserving mass approximately
        rule_bits = {}
        for center in [0, 1]:
            for nw in [0, 1]:
                for n in [0, 1]:
                    for ne in [0, 1]:
                        for e in [0, 1]:
                            for se in [0, 1]:
                                for s in [0, 1]:
                                    for sw in [0, 1]:
                                        for w in [0, 1]:
                                            key = (center, (nw, n, ne, e, se, s, sw, w))
                                            # Rule: flip if exactly 4 of 8 neighbors are 1
                                            # This creates checkerboard-like patterns
                                            n_ones = nw + n + ne + e + se + s + sw + w
                                            if n_ones == 4:
                                                rule_bits[key] = 1 - center
                                            else:
                                                rule_bits[key] = center
    elif rule_type == "xor":
        # XOR rule: output = center XOR (NW XOR SE) XOR (NE XOR SW)
        rule_bits = {}
        for center in [0, 1]:
            for nw in [0, 1]:
                for n in [0, 1]:
                    for ne in [0, 1]:
                        for e in [0, 1]:
                            for se in [0, 1]:
                                for s in [0, 1]:
                                    for sw in [0, 1]:
                                        for w in [0, 1]:
                                            key = (center, (nw, n, ne, e, se, s, sw, w))
                                            out = center ^ (nw ^ se) ^ (ne ^ sw)
                                            rule_bits[key] = out
    elif rule_type == "majority":
        # Majority rule: output = majority of center + 4 orthogonal neighbors
        rule_bits = {}
        for center in [0, 1]:
            for nw in [0, 1]:
                for n in [0, 1]:
                    for ne in [0, 1]:
                        for e in [0, 1]:
                            for se in [0, 1]:
                                for s in [0, 1]:
                                    for sw in [0, 1]:
                                        for w in [0, 1]:
                                            key = (center, (nw, n, ne, e, se, s, sw, w))
                                            # Majority of center, N, S, E, W
                                            votes = center + n + s + e + w
                                            rule_bits[key] = 1 if votes >= 3 else 0
    elif rule_type == "rule110_2d":
        # 2D extension of Rule 110
        # Rule 110 in 1D: 01101010 -> binary 110 -> 
        # For 2D, use a simplified version
        rule_bits = {}
        for center in [0, 1]:
            for nw in [0, 1]:
                for n in [0, 1]:
                    for ne in [0, 1]:
                        for e in [0, 1]:
                            for se in [0, 1]:
                                for s in [0, 1]:
                                    for sw in [0, 1]:
                                        for w in [0, 1]:
                                            key = (center, (nw, n, ne, e, se, s, sw, w))
                                            # Simplified rule: output based on local configuration
                                            # Use a hash-like function for complexity
                                            val = (center * 128 + nw * 64 + n * 32 + ne * 16 + 
                                                   e * 8 + se * 4 + s * 2 + sw * 1 + w * 16)
                                            # Use a pattern that creates complex structures
                                            if (val % 3 == 0) or (center == 0 and n_ones_count(nw,n,ne,e,se,s,sw,w) >= 5):
                                                rule_bits[key] = 1
                                            else:
                                                rule_bits[key] = 0
    else:
        # Default: conservative
        rule_type = "conservative"
        rule_bits = {}
        for center in [0, 1]:
            for nw in [0, 1]:
                for n in [0, 1]:
                    for ne in [0, 1]:
                        for e in [0, 1]:
                            for se in [0, 1]:
                                for s in [0, 1]:
                                    for sw in [0, 1]:
                                        for w in [0, 1]:
                                            key = (center, (nw, n, ne, e, se, s, sw, w))
                                            n_ones = nw + n + ne + e + se + s + sw + w
                                            if n_ones == 4:
                                                rule_bits[key] = 1 - center
                                            else:
                                                rule_bits[key] = center
    
    # Evolve CA
    for step in range(n_steps):
        cell = _ca_step_2d(cell, rule_bits)
        # Re-apply symmetry after each step to maintain invariance
        cell = _apply_symmetry(cell)
    
    return cell.tolist()

def _n_ones_count(*args):
    """Count ones in a tuple of bits."""
    return sum(args)

def _surrogate_gap(cell, size=32):
    """Fast surrogate evaluation using fill fraction and boundary complexity.
    
    Returns a score proxy for the 4-5 TM band gap.
    """
    eps = np.array(cell, dtype=float)
    fill_fraction = np.sum(eps) / (size * size)
    
    # Compute boundary length
    boundary = 0
    for i in range(size):
        for j in range(size):
            if eps[i, j] == 1:
                if i > 0 and eps[i-1, j] == 0: boundary += 1
                if i < size-1 and eps[i+1, j] == 0: boundary += 1
                if j > 0 and eps[i, j-1] == 0: boundary += 1
                if j < size-1 and eps[i, j+1] == 0: boundary += 1
    
    complexity = boundary / (size * size)
    
    # Optimal fill fraction for 4-5 gap is around 0.3-0.5
    ff_optimal = 0.4
    ff_penalty = (fill_fraction - ff_optimal) ** 2
    
    surrogate_score = complexity * np.exp(-ff_penalty / 0.05)
    
    return surrogate_score

def design(problem):
    """Design a 32x32 photonic crystal unit cell to maximize the 4-5 TM band gap."""
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 210.0  # 20s safety margin before 230s kill
    
    best_cell = None
    best_score = -np.inf
    
    # Generate baseline: simple centered disk
    base = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.04:  # radius ~0.2
                base[i, j] = 1
    best_cell = base.tolist()
    best_score = _surrogate_gap(base.tolist(), size)
    
    # Try various CA configurations
    # Seed patterns: 8x8 patterns that will be upscaled
    seed_patterns = [
        # Single pixel center
        [[0]*8 for _ in range(8)],
        # Cross pattern
        [[0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0],
         [0,0,0,1,1,0,0,0],
         [0,0,0,1,1,0,0,0],
         [0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0]],
        # Four corner dots
        [[1,0,0,0,0,0,0,1],
         [0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0],
         [1,0,0,0,0,0,0,1]],
        # Checkerboard
        [[1,0,1,0,1,0,1,0],
         [0,1,0,1,0,1,0,1],
         [1,0,1,0,1,0,1,0],
         [0,1,0,1,0,1,0,1],
         [1,0,1,0,1,0,1,0],
         [0,1,0,1,0,1,0,1],
         [1,0,1,0,1,0,1,0],
         [0,1,0,1,0,1,0,1]],
        # Ring pattern
        [[0,0,0,0,0,0,0,0],
         [0,1,1,1,1,1,1,0],
         [0,1,0,0,0,0,1,0],
         [0,1,0,0,0,0,1,0],
         [0,1,0,0,0,0,1,0],
         [0,1,0,0,0,0,1,0],
         [0,1,1,1,1,1,1,0],
         [0,0,0,0,0,0,0,0]],
    ]
    
    rule_types = ["conservative", "xor", "majority"]
    n_steps_values = [1, 2, 3, 4, 5, 6, 8, 10]
    
    for seed in seed_patterns:
        if time.time() > deadline - 30.0:
            break
        for rule_type in rule_types:
            if time.time() > deadline - 20.0:
                break
            for n_steps in n_steps_values:
                if time.time() > deadline - 10.0:
                    break
                try:
                    cell = _make_ca_cell(size, seed, n_steps, rule_type)
                    score = _surrogate_gap(cell, size)
                    if score > best_score:
                        best_score = score
                        best_cell = cell
                except Exception:
                    continue
    
    # Also try some direct geometric patterns as fallback
    if time.time() < deadline - 15.0:
        # Four-rod pattern
        for offset in [0.15, 0.2, 0.25, 0.3]:
            if time.time() > deadline - 10.0:
                break
            cell = np.zeros((size, size), dtype=int)
            for i in range(size):
                for j in range(size):
                    x = (i + 0.5) / size - 0.5
                    y = (j + 0.5) / size - 0.5
                    best_dist = 1.0
                    for cx in [-offset, offset]:
                        for cy in [-offset, offset]:
                            dist2 = (x-cx)**2 + (y-cy)**2
                            if dist2 < best_dist:
                                best_dist = dist2
                    if best_dist < 0.02:  # radius ~0.14
                        cell[i, j] = 1
            cell = _apply_symmetry(cell)
            cell_list = cell.tolist()
            score = _surrogate_gap(cell_list, size)
            if score > best_score:
                best_score = score
                best_cell = cell_list
        
        # Single disk with varying radius
        for radius in [0.15, 0.2, 0.25, 0.3, 0.35]:
            if time.time() > deadline - 5.0:
                break
            cell = np.zeros((size, size), dtype=int)
            for i in range(size):
                for j in range(size):
                    x = (i + 0.5) / size - 0.5
                    y = (j + 0.5) / size - 0.5
                    if x*x + y*y < radius*radius:
                        cell[i, j] = 1
            cell = _apply_symmetry(cell)
            cell_list = cell.tolist()
            score = _surrogate_gap(cell_list, size)
            if score > best_score:
                best_score = score
                best_cell = cell_list
    
    return best_cell
# EVOLVE-BLOCK-END
