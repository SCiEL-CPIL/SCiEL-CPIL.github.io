"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import permutations

def _rotations(w, h, d):
    """Return unique rotations of a box as (w, h, d) tuples."""
    seen = set()
    result = []
    for perm in permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # Precompute all rotations for each box
    box_rots = [_rotations(w, h, d) for w, h, d in boxes]
    
    # Sort boxes by volume descending, then by max dimension descending
    order = sorted(range(n), key=lambda i: (-(boxes[i][0]*boxes[i][1]*boxes[i][2]), 
                                             -max(boxes[i])), reverse=False)
    # Actually we want largest first:
    order = sorted(range(n), key=lambda i: (-(boxes[i][0]*boxes[i][1]*boxes[i][2]), 
                                             -max(boxes[i])))
    
    def solve(order):
        # placements[i] = (x, y, z, w, h, d) or None
        placements = [None] * n
        # Track occupied space using a list of placed boxes for overlap check
        placed_boxes = []
        
        # For efficiency, we'll track the "floor" at each (x, y) position
        # But for simplicity with arbitrary rotations, we'll use a grid if container is small enough
        # Otherwise, we'll use a list-based approach
        
        # Let's use a 3D grid if container dimensions are manageable
        # For large containers, use list-based
        
        max_dim = max(W, H, D)
        use_grid = (W <= 200 and H <= 200 and D <= 200)
        
        if use_grid:
            # 3D grid: grid[x][y][z] = 1 if occupied
            grid = [[[0]*D for _ in range(H)] for _ in range(W)]
            
            def can_place(x, y, z, bw, bh, bd):
                if x < 0 or y < 0 or z < 0:
                    return False
                if x + bw > W or y + bh > H or z + bd > D:
                    return False
                for ix in range(x, x + bw):
                    for iy in range(y, y + bh):
                        for iz in range(z, z + bd):
                            if grid[ix][iy][iz]:
                                return False
                return True
            
            def place(x, y, z, bw, bh, bd):
                for ix in range(x, x + bw):
                    for iy in range(y, y + bh):
                        for iz in range(z, z + bd):
                            grid[ix][iy][iz] = 1
            
            def find_positions(bw, bh, bd):
                """Find all valid positions, sorted by z, y, x."""
                positions = []
                for z in range(D - bd + 1):
                    for y in range(H - bh + 1):
                        for x in range(W - bw + 1):
                            if can_place(x, y, z, bw, bh, bd):
                                positions.append((x, y, z))
                return positions
        else:
            # List-based approach
            def can_place(x, y, z, bw, bh, bd):
                if x < 0 or y < 0 or z < 0:
                    return False
                if x + bw > W or y + bh > H or z + bd > D:
                    return False
                for (px, py, pz, pw, ph, pd) in placed_boxes:
                    if not (x + bw <= px or px + pw <= x):
                        if not (y + bh <= py or py + ph <= y):
                            if not (z + bd <= pz or pz + pd <= z):
                                return False
                return True
            
            def place(x, y, z, bw, bh, bd):
                placed_boxes.append((x, y, z, bw, bh, bd))
            
            def find_positions(bw, bh, bd):
                """Find candidate positions by checking floor positions."""
                # Generate candidate x, y positions from existing placements and boundaries
                xs = set([0])
                ys = set([0])
                for (px, py, pz, pw, ph, pd) in placed_boxes:
                    xs.add(px)
                    xs.add(px + pw)
                    ys.add(py)
                    ys.add(py + ph)
                
                positions = []
                for x in xs:
                    for y in ys:
                        # Find minimum z where box fits
                        min_z = 0
                        for (px, py, pz, pw, ph, pd) in placed_boxes:
                            if not (x + bw <= px or px + pw <= x):
                                if not (y + bh <= py or py + ph <= y):
                                    min_z = max(min_z, pz + pd)
                        if min_z + bd <= D:
                            positions.append((x, y, min_z))
                # Also check if box can go at z=0 for some positions
                # Sort by z, then y, then x
                positions.sort(key=lambda p: (p[2], p[1], p[0]))
                return positions
        
        total_volume = 0
        for idx in order:
            placed = False
            for (bw, bh, bd) in box_rots[idx]:
                positions = find_positions(bw, bh, bd)
                if positions:
                    # Pick the first valid position (lowest z, then y, then x)
                    x, y, z = positions[0]
                    place(x, y, z, bw, bh, bd)
                    placements[idx] = (x, y, z, bw, bh, bd)
                    total_volume += bw * bh * bd
                    placed = True
                    break
        
        return placements, total_volume
    
    # Try the greedy approach first
    t0 = time.time()
    deadline = 110.0  # Leave 10s safety margin
    
    placements, total_volume = solve(order)
    
    # Check if we have time for a second pass with different ordering
    elapsed = time.time() - t0
    if elapsed < deadline - 20:
        # Try sorting by different criteria
        order2 = sorted(range(n), key=lambda i: (-max(boxes[i]), -(boxes[i][0]*boxes[i][1]*boxes[i][2])))
        placements2, total_volume2 = solve(order2)
        if total_volume2 > total_volume:
            placements, total_volume = placements2, total_volume2
    
    # Build result
    result = []
    for i in range(n):
        if placements[i] is not None:
            x, y, z, w, h, d = placements[i]
            result.append((0, x, y, z, w, h, d))
        else:
            w, h, d = boxes[i]
            result.append((-1, 0, 0, 0, w, h, d))
    
    return result
# EVOLVE-BLOCK-END
