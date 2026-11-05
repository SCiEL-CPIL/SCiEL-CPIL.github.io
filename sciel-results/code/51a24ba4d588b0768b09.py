"""3D Container Loading: volume-aware largest-empty-space search with occupancy grid."""
import time
import itertools
import heapq

# EVOLVE-BLOCK-START
import time as _time


def _orientations(w, h, d):
    """Yield the 6 distinct axis-permutations of (w, h, d)."""
    seen = set()
    out = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            out.append(perm)
    return out


def _fits_inside(x, y, z, w, h, d, W, H, D):
    return (x >= 0 and y >= 0 and z >= 0 and
            x + w <= W and y + h <= H and z + d <= D)


class OccupancyGrid:
    """
    3D occupancy grid that tracks distance to nearest obstacle/boundary.
    Used to find the largest empty rectangular space for box placement.
    """
    def __init__(self, W, H, D, resolution=1):
        self.W = W
        self.H = H
        self.D = D
        self.res = resolution
        self.nx = max(1, int(W // resolution) + 1)
        self.ny = max(1, int(H // resolution) + 1)
        self.nz = max(1, int(D // resolution) + 1)
        # dist[i,j,k] = minimum distance from cell (i,j,k) to any obstacle or boundary
        self.dist = [[[-1]*self.nz for _ in range(self.ny)] for _ in range(self.nx)]
        self._init_dist()
    
    def _init_dist(self):
        """Initialize distance field: distance to nearest boundary."""
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    x = i * self.res
                    y = j * self.res
                    z = k * self.res
                    # Distance to each boundary
                    dx = min(x, self.W - x)
                    dy = min(y, self.H - y)
                    dz = min(z, self.D - z)
                    self.dist[i][j][k] = min(dx, dy, dz)
    
    def place_box(self, x, y, z, w, h, d):
        """Mark a box as occupied by setting dist to 0 in its footprint."""
        i0 = int(x // self.res)
        i1 = min(self.nx - 1, int((x + w) // self.res))
        j0 = int(y // self.res)
        j1 = min(self.ny - 1, int((y + h) // self.res))
        k0 = int(z // self.res)
        k1 = min(self.nz - 1, int((z + d) // self.res))
        
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                for k in range(k0, k1 + 1):
                    self.dist[i][j][k] = 0
    
    def find_best_position(self, w, h, d, placed_boxes):
        """
        Find the best position for a box of size (w,h,d) by maximizing
        the minimum distance value in its footprint (creates largest void).
        Returns (x, y, z) or None.
        """
        best_pos = None
        best_score = -1
        
        # Candidate positions: from placed boxes' corners + origin
        cands = set()
        cands.add((0, 0, 0))
        for (px, py, pz, pw, ph, pd) in placed_boxes:
            for cx in (px, px + pw):
                for cy in (py, py + ph):
                    for cz in (pz, pz + pd):
                        if _fits_inside(cx, cy, cz, w, h, d, self.W, self.H, self.D):
                            cands.add((cx, cy, cz))
        
        for (x, y, z) in cands:
            # Check overlap with placed boxes
            overlap = False
            for (px, py, pz, pw, ph, pd) in placed_boxes:
                if (x < px + pw and px < x + w and
                    y < py + ph and py < y + h and
                    z < pz + pd and pz < z + d):
                    overlap = True
                    break
            if overlap:
                continue
            
            # Compute minimum distance in footprint
            i0 = int(x // self.res)
            i1 = min(self.nx - 1, int((x + w) // self.res))
            j0 = int(y // self.res)
            j1 = min(self.ny - 1, int((y + h) // self.res))
            k0 = int(z // self.res)
            k1 = min(self.nz - 1, int((z + d) // self.res))
            
            min_dist = float('inf')
            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    for k in range(k0, k1 + 1):
                        if self.dist[i][j][k] < min_dist:
                            min_dist = self.dist[i][j][k]
            
            # Score: maximize min_dist (larger void), break ties by lower z, x, y
            if min_dist > best_score or (min_dist == best_score and best_pos is not None and
                    (z, x, y) < best_pos):
                best_score = min_dist
                best_pos = (x, y, z)
        
        return best_pos


def _pack_volume_aware(boxes, W, H, D, deadline):
    """
    Pack boxes using volume-aware largest-empty-space search.
    Returns per-box placements.
    """
    n = len(boxes)
    # Sort by volume descending, then max dimension
    order = sorted(range(n), key=lambda i: (-(boxes[i][0]*boxes[i][1]*boxes[i][2]),
                                            -max(boxes[i])))
    
    # Create occupancy grid with resolution 1 for precision
    grid = OccupancyGrid(W, H, D, resolution=1)
    placed = []  # list of (x,y,z,w,h,d)
    result = [None] * n
    
    for idx in order:
        if _time.time() >= deadline:
            break
        w, h, d = boxes[idx]
        found = False
        for (ow, oh, od) in _orientations(w, h, d):
            pos = grid.find_best_position(ow, oh, od, placed)
            if pos is not None:
                x, y, z = pos
                grid.place_box(x, y, z, ow, oh, od)
                placed.append((x, y, z, ow, oh, od))
                result[idx] = (0, x, y, z, ow, oh, od)
                found = True
                break
        if not found:
            result[idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(n):
        if result[i] is None:
            w, h, d = boxes[i]
            result[i] = (-1, 0, 0, 0, w, h, d)
    return result


def _pack_shelf(boxes, W, H, D):
    """
    A fast shelf-based fallback.
    Returns per-box placements.
    """
    n = len(boxes)
    order = sorted(range(n), key=lambda i: (-(boxes[i][0]*boxes[i][1]*boxes[i][2]),
                                            -max(boxes[i])))
    placed = [None] * n
    x = y = z = 0
    row_h = layer_d = 0
    for idx in order:
        w, h, d = boxes[idx]
        best_orient = None
        best_score = None
        for (ow, oh, od) in _orientations(w, h, d):
            score = (x + ow > W, y + oh > H, z + od > D)
            if not any(score):
                s = (ow, oh, od)
                if best_score is None or s < best_score:
                    best_score = s
                    best_orient = (ow, oh, od)
        if best_orient is None:
            placed[idx] = (-1, 0, 0, 0, w, h, d)
            continue
        ow, oh, od = best_orient
        if x + ow > W:
            x, y, row_h = 0, y + row_h, 0
        if y + oh > H:
            y, z, layer_d, x, row_h = 0, z + layer_d, 0, 0, 0
        if z + od > D:
            placed[idx] = (-1, 0, 0, 0, w, h, d)
            continue
        placed[idx] = (0, x, y, z, ow, oh, od)
        x, row_h, layer_d = x + ow, max(row_h, oh), max(layer_d, od)
    return placed


def _volume_filled(placements, W, H, D):
    total = 0
    for (b, x, y, z, w, h, d) in placements:
        if b == 0:
            total += w * h * d
    return total


def _validate(placements, boxes, W, H, D):
    """Check no overlaps and all inside container. Returns True if valid."""
    loaded = []
    for (b, x, y, z, w, h, d) in placements:
        if b != 0:
            continue
        if not _fits_inside(x, y, z, w, h, d, W, H, D):
            return False
        loaded.append((x, y, z, w, h, d))
    for i in range(len(loaded)):
        for j in range(i + 1, len(loaded)):
            a, b = loaded[i], loaded[j]
            if (a[0] < b[0]+b[3] and b[0] < a[0]+a[3] and
                a[1] < b[1]+b[4] and b[1] < a[1]+a[4] and
                a[2] < b[2]+b[5] and b[2] < a[2]+a[5]):
                return False
    return True


def load(boxes, container):
    start = _time.time()
    deadline = start + 115.0  # 5s safety margin under 120s limit

    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for (w, h, d) in boxes]

    # Fast shelf fallback first (cheap, always valid).
    shelf = _pack_shelf(boxes, W, H, D)
    best = shelf
    best_vol = _volume_filled(shelf, W, H, D)

    # Try volume-aware packing with occupancy grid.
    try:
        va = _pack_volume_aware(boxes, W, H, D, deadline)
        if _validate(va, boxes, W, H, D):
            vol = _volume_filled(va, W, H, D)
            if vol > best_vol:
                best = va
                best_vol = vol
    except Exception:
        pass

    # If we still have time, try a second pass with different ordering.
    if _time.time() < deadline - 10.0:
        try:
            n = len(boxes)
            order2 = sorted(range(n), key=lambda i: (-max(boxes[i]),
                                                     -(boxes[i][0]*boxes[i][1]*boxes[i][2])))
            grid = OccupancyGrid(W, H, D, resolution=1)
            placed = []
            result = [None] * n
            for idx in order2:
                if _time.time() >= deadline - 5.0:
                    break
                w, h, d = boxes[idx]
                found = False
                for (ow, oh, od) in _orientations(w, h, d):
                    pos = grid.find_best_position(ow, oh, od, placed)
                    if pos is not None:
                        x, y, z = pos
                        grid.place_box(x, y, z, ow, oh, od)
                        placed.append((x, y, z, ow, oh, od))
                        result[idx] = (0, x, y, z, ow, oh, od)
                        found = True
                        break
                if not found:
                    result[idx] = (-1, 0, 0, 0, w, h, d)
            for i in range(n):
                if result[i] is None:
                    w, h, d = boxes[i]
                    result[i] = (-1, 0, 0, 0, w, h, d)
            if _validate(result, boxes, W, H, D):
                vol = _volume_filled(result, W, H, D)
                if vol > best_vol:
                    best = result
                    best_vol = vol
        except Exception:
            pass

    # Final safety: ensure validity; if invalid, fall back to shelf.
    if not _validate(best, boxes, W, H, D):
        best = shelf

    return best
# EVOLVE-BLOCK-END
