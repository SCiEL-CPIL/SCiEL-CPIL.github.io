"""3D Container Loading with spatial grid acceleration and diverse restarts."""
import time
import math
import random
from itertools import permutations

def _can_place(occ_grid, x, y, z, w, h, d, W, H, D):
    """Check if a box at (x,y,z) with dims (w,h,d) fits without collision using grid."""
    # Grid cells are unit cubes; we mark occupied cells.
    # For efficiency, we only check cells within the box volume.
    x0, y0, z0 = int(x), int(y), int(z)
    x1, y1, z1 = int(x + w), int(y + h), int(z + d)
    # Clamp to container bounds
    x0 = max(0, min(x0, W))
    y0 = max(0, min(y0, H))
    z0 = max(0, min(z0, D))
    x1 = max(0, min(x1, W))
    y1 = max(0, min(y1, H))
    z1 = max(0, min(z1, D))
    if x1 <= x0 or y1 <= y0 or z1 <= z0:
        return False
    # Check if any cell in the range is occupied
    for cx in range(x0, x1):
        for cy in range(y0, y1):
            for cz in range(z0, z1):
                if occ_grid[cx][cy][cz]:
                    return False
    return True

def _place_box(occ_grid, x, y, z, w, h, d):
    """Mark cells as occupied."""
    x0, y0, z0 = int(x), int(y), int(z)
    x1, y1, z1 = int(x + w), int(y + h), int(z + d)
    for cx in range(x0, x1):
        for cy in range(y0, y1):
            for cz in range(z0, z1):
                occ_grid[cx][cy][cz] = True

def _remove_box(occ_grid, x, y, z, w, h, d):
    """Mark cells as unoccupied."""
    x0, y0, z0 = int(x), int(y), int(z)
    x1, y1, z1 = int(x + w), int(y + h), int(z + d)
    for cx in range(x0, x1):
        for cy in range(y0, y1):
            for cz in range(z0, z1):
                occ_grid[cx][cy][cz] = False

def _generate_positions(W, H, D, w, h, d, rng):
    """Generate candidate positions for a box with dimensions (w,h,d)."""
    positions = []
    # Add corner and edge positions
    candidates = [
        (0, 0, 0),
        (W - w, 0, 0),
        (0, H - h, 0),
        (0, 0, D - d),
        (W - w, H - h, 0),
        (W - w, 0, D - d),
        (0, H - h, D - d),
        (W - w, H - h, D - d),
    ]
    for cx, cy, cz in candidates:
        if 0 <= cx <= W - w and 0 <= cy <= H - h and 0 <= cz <= D - d:
            positions.append((cx, cy, cz))
    
    # Add some random positions
    for _ in range(10):
        cx = rng.uniform(0, max(0, W - w))
        cy = rng.uniform(0, max(0, H - h))
        cz = rng.uniform(0, max(0, D - d))
        cx, cy, cz = int(cx), int(cy), int(cz)
        if 0 <= cx <= W - w and 0 <= cy <= H - h and 0 <= cz <= D - d:
            positions.append((cx, cy, cz))
    
    # Add a grid of positions
    step_x = max(1, w // 4)
    step_y = max(1, h // 4)
    step_z = max(1, d // 4)
    for cx in range(0, W - w + 1, step_x):
        for cy in range(0, H - h + 1, step_y):
            for cz in range(0, D - d + 1, step_z):
                positions.append((cx, cy, cz))
    
    return positions

def _solve_single(boxes, container, rng, deadline):
    """Solve a single instance with a given random seed strategy."""
    W, H, D = container
    n = len(boxes)
    
    # Create occupancy grid
    occ_grid = [[[False for _ in range(D)] for _ in range(H)] for _ in range(W)]
    
    # Try all permutations of box dimensions (6 rotations)
    def get_rotations(w, h, d):
        return list(set(permutations([w, h, d])))
    
    # Sort boxes by volume descending, with some randomization
    box_indices = list(range(n))
    # Use a mix of sorting strategies
    strategy = rng.choice(['volume_desc', 'volume_asc', 'max_dim_desc', 'random'])
    if strategy == 'volume_desc':
        box_indices.sort(key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)
    elif strategy == 'volume_asc':
        box_indices.sort(key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2])
    elif strategy == 'max_dim_desc':
        box_indices.sort(key=lambda i: max(boxes[i]), reverse=True)
    else:
        rng.shuffle(box_indices)
    
    placements = [(-1, 0, 0, 0, 0, 0, 0)] * n
    placed_volume = 0
    
    for idx in box_indices:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        rotations = get_rotations(w, h, d)
        
        best_pos = None
        best_rot = None
        best_score = -1
        
        for rw, rh, rd in rotations:
            positions = _generate_positions(W, H, D, rw, rh, rd, rng)
            for px, py, pz in positions:
                if _can_place(occ_grid, px, py, pz, rw, rh, rd, W, H, D):
                    # Score: prefer positions that use space efficiently
                    # Use a simple heuristic: prefer lower z, then y, then x
                    score = -(pz * W * H + py * W + px)
                    if score > best_score:
                        best_score = score
                        best_pos = (px, py, pz)
                        best_rot = (rw, rh, rd)
        
        if best_pos is not None:
            px, py, pz = best_pos
            rw, rh, rd = best_rot
            _place_box(occ_grid, px, py, pz, rw, rh, rd)
            placements[idx] = (0, px, py, pz, rw, rh, rd)
            placed_volume += rw * rh * rd
        else:
            # Try with original orientation
            for px in range(0, W - w + 1):
                for py in range(0, H - h + 1):
                    for pz in range(0, D - d + 1):
                        if _can_place(occ_grid, px, py, pz, w, h, d, W, H, D):
                            _place_box(occ_grid, px, py, pz, w, h, d)
                            placements[idx] = (0, px, py, pz, w, h, d)
                            placed_volume += w * h * d
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
    
    return placements, placed_volume

def _local_search(boxes, container, placements, rng, deadline):
    """Perform local search by swapping and moving boxes."""
    W, H, D = container
    n = len(boxes)
    container_vol = W * H * D
    
    # Build occupancy grid from current placements
    occ_grid = [[[False for _ in range(D)] for _ in range(H)] for _ in range(W)]
    for i, (bin, x, y, z, w, h, d) in enumerate(placements):
        if bin == 0:
            _place_box(occ_grid, x, y, z, w, h, d)
    
    current_volume = sum(w * h * d for (bin, x, y, z, w, h, d) in placements if bin == 0)
    
    for _ in range(50):  # Limited iterations
        if time.time() > deadline:
            break
        
        # Try to move a random placed box to a better position
        placed_indices = [i for i in range(n) if placements[i][0] == 0]
        if not placed_indices:
            break
        
        idx = rng.choice(placed_indices)
        bin, x, y, z, w, h, d = placements[idx]
        
        # Remove box
        _remove_box(occ_grid, x, y, z, w, h, d)
        
        # Try to place it in a new position
        rotations = list(set(permutations([w, h, d])))
        best_pos = None
        best_rot = None
        best_score = -1
        
        for rw, rh, rd in rotations:
            positions = _generate_positions(W, H, D, rw, rh, rd, rng)
            for px, py, pz in positions:
                if _can_place(occ_grid, px, py, pz, rw, rh, rd, W, H, D):
                    score = -(pz * W * H + py * W + px)
                    if score > best_score:
                        best_score = score
                        best_pos = (px, py, pz)
                        best_rot = (rw, rh, rd)
        
        if best_pos is not None:
            px, py, pz = best_pos
            rw, rh, rd = best_rot
            _place_box(occ_grid, px, py, pz, rw, rh, rd)
            placements[idx] = (0, px, py, pz, rw, rh, rd)
        else:
            # Restore original position
            _place_box(occ_grid, x, y, z, w, h, d)
    
    return placements

def load(boxes, container):
    """Main entry point for 3D container loading."""
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    W, H, D = container
    container_vol = W * H * D
    
    if container_vol == 0:
        return [(-1, 0, 0, 0, b[0], b[1], b[2]) for b in boxes]
    
    best_placements = None
    best_volume = -1
    
    # Multiple restarts with different random seeds
    rng = random.Random(42)
    
    max_restarts = 200
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        placements, volume = _solve_single(boxes, container, rng, deadline)
        
        if volume > best_volume:
            best_volume = volume
            best_placements = [list(p) for p in placements]
        
        # If we have good coverage, do local search
        if best_volume / container_vol > 0.5 and time.time() < deadline - 10:
            best_placements = _local_search(boxes, container, best_placements, rng, deadline)
            new_volume = sum(w * h * d for (bin, x, y, z, w, h, d) in best_placements if bin == 0)
            if new_volume > best_volume:
                best_volume = new_volume
    
    if best_placements is None:
        # Fallback: simple greedy
        best_placements = [(-1, 0, 0, 0, b[0], b[1], b[2]) for b in boxes]
    
    return [tuple(p) for p in best_placements]
