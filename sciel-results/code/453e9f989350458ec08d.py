"""Rewritten 3D container loading with spatial grid acceleration and more restarts."""
import time
import itertools
import random
import math

# EVOLVE-BLOCK-START

def _orientations(w, h, d):
    """Return unique orientations (w,h,d) as tuples."""
    dims = (w, h, d)
    seen = set()
    result = []
    for perm in itertools.permutations(dims):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result


class SpatialGrid:
    """3D spatial grid for fast collision detection."""
    
    def __init__(self, W, H, D, cell_size=None):
        self.W = W
        self.H = H
        self.D = D
        # Choose cell size to balance memory and query efficiency
        if cell_size is None:
            # Aim for roughly 8-16 cells per dimension
            self.cell_size = max(1, min(W, H, D) // 8)
        else:
            self.cell_size = cell_size
        
        self.nx = max(1, (W + self.cell_size - 1) // self.cell_size)
        self.ny = max(1, (H + self.cell_size - 1) // self.cell_size)
        self.nz = max(1, (D + self.cell_size - 1) // self.cell_size)
        
        # Grid stores sets of box indices per cell
        self.grid = {}
        self.boxes = []  # list of (x, y, z, w, h, d)
    
    def _cell_range(self, x, y, z, w, h, d):
        """Get the range of cells that a box at (x,y,z) with dims (w,h,d) occupies."""
        x0 = x // self.cell_size
        y0 = y // self.cell_size
        z0 = z // self.cell_size
        x1 = (x + w - 1) // self.cell_size if w > 0 else x0
        y1 = (y + h - 1) // self.cell_size if h > 0 else y0
        z1 = (z + d - 1) // self.cell_size if d > 0 else z0
        
        x0 = max(0, min(x0, self.nx - 1))
        y0 = max(0, min(y0, self.ny - 1))
        z0 = max(0, min(z0, self.nz - 1))
        x1 = max(0, min(x1, self.nx - 1))
        y1 = max(0, min(y1, self.ny - 1))
        z1 = max(0, min(z1, self.nz - 1))
        
        return (x0, x1, y0, y1, z0, z1)
    
    def add_box(self, x, y, z, w, h, d):
        """Add a box to the grid. Returns the box index."""
        idx = len(self.boxes)
        self.boxes.append((x, y, z, w, h, d))
        
        x0, x1, y0, y1, z0, z1 = self._cell_range(x, y, z, w, h, d)
        for cx in range(x0, x1 + 1):
            for cy in range(y0, y1 + 1):
                for cz in range(z0, z1 + 1):
                    key = (cx, cy, cz)
                    if key not in self.grid:
                        self.grid[key] = set()
                    self.grid[key].add(idx)
        
        return idx
    
    def can_place(self, x, y, z, w, h, d):
        """Check if a box at (x,y,z) with dims (w,h,d) can be placed without overlap."""
        # Bounds check
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > self.W or y + h > self.H or z + d > self.D:
            return False
        
        if w <= 0 or h <= 0 or d <= 0:
            return True
        
        x0, x1, y0, y1, z0, z1 = self._cell_range(x, y, z, w, h, d)
        
        # Collect all potentially overlapping box indices
        candidate_indices = set()
        for cx in range(x0, x1 + 1):
            for cy in range(y0, y1 + 1):
                for cz in range(z0, z1 + 1):
                    key = (cx, cy, cz)
                    if key in self.grid:
                        candidate_indices.update(self.grid[key])
        
        # Check actual overlap with each candidate
        for idx in candidate_indices:
            qx, qy, qz, qw, qh, qd = self.boxes[idx]
            if (x < qx + qw and x + w > qx and
                y < qy + qh and y + h > qy and
                z < qz + qd and z + d > qz):
                return False
        
        return True


def _find_position(box_dims, grid, W, H, D, placed_list):
    """Find the best position for a box with given dimensions.
    Returns (x, y, z, bw, bh, bd) or None."""
    bw, bh, bd = box_dims
    
    # Generate candidate positions from all box-face corners plus axis-aligned boundaries
    candidates = set()
    
    # Origin
    candidates.add((0, 0, 0))
    
    # For each placed box, generate all 8 corners of its bounding box
    for (qx, qy, qz, qw, qh, qd) in placed_list:
        for dx in (0, qw):
            for dy in (0, qh):
                for dz in (0, qd):
                    candidates.add((qx + dx, qy + dy, qz + dz))
    
    # Also add positions aligned with container boundaries
    for (qx, qy, qz, qw, qh, qd) in placed_list:
        candidates.add((0, qy, qz))
        candidates.add((qx, 0, qz))
        candidates.add((qx, qy, 0))
        if W - bw >= 0:
            candidates.add((W - bw, qy, qz))
        if H - bh >= 0:
            candidates.add((qx, H - bh, qz))
        if D - bd >= 0:
            candidates.add((qx, qy, D - bd))
    
    # Filter valid candidates and find best
    best_pos = None
    best_score = None
    
    for (px, py, pz) in candidates:
        if px < 0 or py < 0 or pz < 0:
            continue
        if px + bw > W or py + bh > H or pz + bd > D:
            continue
        if not grid.can_place(px, py, pz, bw, bh, bd):
            continue
        # Score: prefer lower z (bottom-up), then lower y, then lower x
        score = (pz, py, px)
        if best_score is None or score < best_score:
            best_score = score
            best_pos = (px, py, pz)
    
    if best_pos is None:
        return None
    
    return (best_pos[0], best_pos[1], best_pos[2], bw, bh, bd)


def _greedy_pack(boxes, container, order, deadline):
    """Greedy pack boxes in given order. Returns placements list."""
    W, H, D = container
    grid = SpatialGrid(W, H, D)
    placed_list = []  # list of (x, y, z, w, h, d)
    placements = [None] * len(boxes)
    
    for idx in order:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        orients = _orientations(w, h, d)
        
        # Try orientations, pick the one that fits in a valid position
        best_result = None
        best_orient_score = None
        
        for orient in orients:
            bw, bh, bd = orient
            result = _find_position(orient, grid, W, H, D, placed_list)
            if result is not None:
                px, py, pz, rbw, rbh, rbd = result
                # Score: prefer positions closer to origin, lower z
                score = (pz, py, px)
                if best_orient_score is None or score < best_orient_score:
                    best_orient_score = score
                    best_result = (px, py, pz, rbw, rbh, rbd)
        
        if best_result is not None:
            px, py, pz, bw, bh, bd = best_result
            grid.add_box(px, py, pz, bw, bh, bd)
            placed_list.append((px, py, pz, bw, bh, bd))
            placements[idx] = (0, px, py, pz, bw, bh, bd)
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
    
    # Fill any remaining None entries
    for i in range(len(placements)):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    return placements


def _score_placements(placements, boxes, container):
    """Compute the fraction of container volume filled."""
    W, H, D = container
    container_vol = W * H * D
    if container_vol == 0:
        return 0.0
    filled = 0
    for i, p in enumerate(placements):
        if p[0] == 0:
            bw, bh, bd = p[4], p[5], p[6]
            filled += bw * bh * bd
    return filled / container_vol


def _local_search(placements, boxes, container, deadline):
    """Try to re-insert left-out boxes by removing some placed boxes."""
    W, H, D = container
    n = len(boxes)
    
    # Identify left-out boxes
    left_out = [i for i in range(n) if placements[i][0] == -1]
    if not left_out:
        return placements
    
    # Try to insert left-out boxes one at a time
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in left_out[:]:  # copy to allow modification
            w, h, d = boxes[idx]
            orients = _orientations(w, h, d)
            
            # Try to find a position for this box in current placement
            # Build grid from current placements
            grid = SpatialGrid(W, H, D)
            placed_list = []
            for i in range(n):
                if placements[i][0] == 0:
                    px, py, pz, bw, bh, bd = placements[i][1], placements[i][2], placements[i][3], placements[i][4], placements[i][5], placements[i][6]
                    grid.add_box(px, py, pz, bw, bh, bd)
                    placed_list.append((px, py, pz, bw, bh, bd))
            
            found = False
            for orient in orients:
                bw, bh, bd = orient
                result = _find_position(orient, grid, W, H, D, placed_list)
                if result is not None:
                    px, py, pz, rbw, rbh, rbd = result
                    grid.add_box(px, py, pz, rbw, rbh, rbd)
                    placed_list.append((px, py, pz, rbw, rbh, rbd))
                    placements[idx] = (0, px, py, pz, rbw, rbh, rbd)
                    left_out.remove(idx)
                    found = True
                    improved = True
                    break
            
            if found:
                continue
    
    return placements


def load(boxes, container):
    start_time = time.time()
    DEADLINE = start_time + 130.0  # Safety margin for 150s limit
    
    W, H, D = container
    container_vol = W * H * D
    n = len(boxes)
    
    if container_vol == 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Pre-compute orientations for each box
    all_orients = [_orientations(w, h, d) for w, h, d in boxes]
    
    # Strategy: try multiple orderings, keep best
    strategies = []
    
    # Volume descending
    vol_order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    strategies.append(vol_order)
    
    # Max dimension descending
    maxdim_order = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    strategies.append(maxdim_order)
    
    # Surface area descending
    sa_order = sorted(range(n), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
    strategies.append(sa_order)
    
    # Min dimension descending (helps with tight fits)
    mindim_order = sorted(range(n), key=lambda i: min(boxes[i]), reverse=True)
    strategies.append(mindim_order)
    
    # Volume-to-max-dim ratio (prefer boxes that are "chunky")
    ratio_order = sorted(range(n), key=lambda i: (boxes[i][0]*boxes[i][1]*boxes[i][2]) / max(boxes[i]), reverse=True)
    strategies.append(ratio_order)
    
    # Sort by max dimension / min dimension (prefer elongated boxes first)
    elong_order = sorted(range(n), key=lambda i: max(boxes[i]) / max(1, min(boxes[i])), reverse=True)
    strategies.append(elong_order)
    
    # Try each strategy
    best_placements = None
    best_score = -1
    
    for order in strategies:
        if time.time() > DEADLINE - 10.0:
            break
        placements = _greedy_pack(boxes, container, order, DEADLINE)
        score = _score_placements(placements, boxes, container)
        if score > best_score:
            best_score = score
            best_placements = placements
    
    # If time remains, try many random orderings (spatial grid makes this feasible)
    remaining_time = DEADLINE - time.time()
    if remaining_time > 10.0:
        rng = random.Random(42)
        # Estimate how many random restarts we can afford
        # Each greedy pack takes roughly O(n * candidates * grid_query)
        # With spatial grid, grid_query is O(1) average, so each pack is fast
        num_random = min(200, int(remaining_time / 0.5))  # up to 200 restarts
        for _ in range(num_random):
            if time.time() > DEADLINE - 5.0:
                break
            rand_order = list(range(n))
            rng.shuffle(rand_order)
            placements = _greedy_pack(boxes, container, rand_order, DEADLINE)
            score = _score_placements(placements, boxes, container)
            if score > best_score:
                best_score = score
                best_placements = placements
    
    # Try local search on the best placement if time remains
    if best_placements is not None and time.time() < DEADLINE - 3.0:
        best_placements = _local_search(best_placements, boxes, container, DEADLINE)
        best_score = _score_placements(best_placements, boxes, container)
    
    if best_placements is None:
        # Fallback: all left out
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_placements

# EVOLVE-BLOCK-END
