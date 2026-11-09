"""3D Container Loading: interval-tree accelerated greedy + shake local search."""
import time
import random
import itertools
from bisect import bisect_left, bisect_right

def _rotations(w, h, d):
    """Return all 6 unique orientations of a box (w,h,d)."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

class _IntervalTree3D:
    """
    3D interval structure for fast overlap detection.
    Uses nested interval trees: outer on x, middle on y, inner on z.
    Each node stores a box as (x, y, z, w, h, d).
    """
    def __init__(self):
        self._boxes = []  # list of (x, y, z, w, h, d)
        self._x_sorted = []  # sorted by x_start
        self._x_max = []  # for each x_sorted entry, the max x_end among suffix

    def add(self, box):
        x, y, z, w, h, d = box
        self._boxes.append(box)
        self._x_sorted.append((x, x + w))
        self._x_sorted.sort(key=lambda t: t[0])
        # Rebuild x_max
        n = len(self._x_sorted)
        self._x_max = [0.0] * n
        if n > 0:
            self._x_max[n - 1] = self._x_sorted[n - 1][1]
            for i in range(n - 2, -1, -1):
                self._x_max[i] = max(self._x_sorted[i][1], self._x_max[i + 1])

    def remove(self, box):
        x, y, z, w, h, d = box
        # Find and remove
        for i, b in enumerate(self._boxes):
            if b == box:
                self._boxes.pop(i)
                break
        # Rebuild x_sorted
        self._x_sorted = [(b[0], b[0] + b[3]) for b in self._boxes]
        self._x_sorted.sort(key=lambda t: t[0])
        n = len(self._x_sorted)
        self._x_max = [0.0] * n
        if n > 0:
            self._x_max[n - 1] = self._x_sorted[n - 1][1]
            for i in range(n - 2, -1, -1):
                self._x_max[i] = max(self._x_sorted[i][1], self._x_max[i + 1])

    def query_overlap(self, x, y, z, w, h, d):
        """Check if any placed box overlaps with the query box at (x,y,z) size (w,h,d)."""
        x2 = x + w
        y2 = y + h
        z2 = z + d
        # Binary search for candidate boxes that could overlap in x
        # We need boxes with x_start < x2 and x_end > x
        # Find first box with x_start < x2
        lo = bisect_left(self._x_sorted, (x2, float('-inf')))
        # Actually we want boxes where x_start < x2 AND x_end > x
        # Let's iterate from the start of those with x_start < x2
        # More efficiently: find range where x_start < x2
        # bisect_right on x_start for x2 gives count of boxes with x_start <= x2
        # We want x_start < x2, so use bisect_left
        end_idx = bisect_left(self._x_sorted, (x2, float('-inf')))
        # Now check from 0 to end_idx, but we can prune using x_max
        # Actually, for a box i to overlap in x: x_start_i < x2 AND x_end_i > x
        # Since x_sorted is sorted by x_start, boxes after end_idx have x_start >= x2, so no overlap.
        # For boxes 0..end_idx-1, we need x_end_i > x.
        # We can use x_max to skip suffixes where max x_end <= x.
        # But x_max[i] is max x_end from i to end. If x_max[i] <= x, then no box from i onward overlaps in x.
        # So we can find the first index where x_max > x.
        if not self._x_sorted:
            return False
        # Find leftmost index where x_max[i] > x
        # x_max is non-increasing? No, x_max[i] = max(x_end_i, x_max[i+1]), so x_max is non-increasing.
        # So if x_max[end_idx-1] <= x, no overlap.
        if end_idx == 0:
            return False
        if self._x_max[end_idx - 1] <= x:
            return False
        # Binary search for first index where x_max[i] > x
        # x_max is non-increasing, so we want the first i where x_max[i] > x
        # This is equivalent to finding the rightmost i where x_max[i] <= x, then i+1
        lo_idx = 0
        hi_idx = end_idx
        while lo_idx < hi_idx:
            mid = (lo_idx + hi_idx) // 2
            if self._x_max[mid] > x:
                hi_idx = mid
            else:
                lo_idx = mid + 1
        # Now check boxes from lo_idx to end_idx-1
        for i in range(lo_idx, end_idx):
            bx, by, bz, bw, bh, bd = self._boxes[i]
            # Check y overlap
            if by + bh <= y or by >= y2:
                continue
            # Check z overlap
            if bz + bd <= z or bz >= z2:
                continue
            # Check x overlap (should be guaranteed but verify)
            if bx + bw <= x or bx >= x2:
                continue
            return True
        return False

def _extreme_points(placed, container):
    """Generate candidate (x,y,z) positions from placed boxes."""
    W, H, D = container
    xs = {0}
    ys = {0}
    zs = {0}
    for (px, py, pz, pw, ph, pd) in placed:
        xs.add(px + pw)
        ys.add(py + ph)
        zs.add(pz + pd)
    xs = [x for x in xs if x < W]
    ys = [y for y in ys if y < H]
    zs = [z for z in zs if z < D]
    points = []
    for x in xs:
        for y in ys:
            for z in zs:
                points.append((x, y, z))
    return points

def _face_points(placed, container):
    """Generate candidate positions from face-adjacent corners of placed boxes."""
    W, H, D = container
    candidates = set()
    candidates.add((0, 0, 0))
    for (px, py, pz, pw, ph, pd) in placed:
        candidates.add((px + pw, py, pz))
        candidates.add((px, py + ph, pz))
        candidates.add((px, py, pz + pd))
        candidates.add((px + pw, py + ph, pz))
        candidates.add((px + pw, py, pz + pd))
        candidates.add((px, py + ph, pz + pd))
        candidates.add((px + pw, py + ph, pz + pd))
    return [(x, y, z) for (x, y, z) in candidates if x < W and y < H and z < D]

def _fits(box_dims, pos, container, placed_tree):
    """Check if box fits at pos without overlap and within container using interval tree."""
    W, H, D = container
    x, y, z = pos
    w, h, d = box_dims
    if x + w > W or y + h > H or z + d > D:
        return False
    return not placed_tree.query_overlap(x, y, z, w, h, d)

def _score_position(pos, box_dims, placed, container):
    """Score a candidate position: prefer compact (low) placements."""
    W, H, D = container
    x, y, z = pos
    w, h, d = box_dims
    return (z + d, y + h, x + w)

def _greedy(boxes, container, order, deadline, mode='extreme'):
    """Greedy placement with interval tree for fast overlap checks."""
    W, H, D = container
    placed = []
    tree = _IntervalTree3D()
    placements = [None] * len(boxes)
    total_vol = 0.0
    n = len(boxes)
    
    for idx in order:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        rots = _rotations(w, h, d)
        
        if mode == 'extreme':
            candidates = _extreme_points(placed, container)
            if (0, 0, 0) not in candidates:
                candidates.append((0, 0, 0))
        elif mode == 'faces':
            candidates = _face_points(placed, container)
        else:  # compact
            candidates = _extreme_points(placed, container)
            if (0, 0, 0) not in candidates:
                candidates.append((0, 0, 0))
        
        best_pos = None
        best_rot = None
        best_key = None
        
        for pos in candidates:
            for rw, rh, rd in rots:
                if _fits((rw, rh, rd), pos, container, tree):
                    if mode == 'compact':
                        key = _score_position(pos, (rw, rh, rd), placed, container)
                    else:
                        key = (pos[2], pos[1], pos[0])
                    if best_key is None or key < best_key:
                        best_key = key
                        best_pos = pos
                        best_rot = (rw, rh, rd)
        
        if best_pos is not None:
            x, y, z = best_pos
            rw, rh, rd = best_rot
            box_tuple = (x, y, z, rw, rh, rd)
            placed.append(box_tuple)
            tree.add(box_tuple)
            placements[idx] = (0, x, y, z, rw, rh, rd)
            total_vol += rw * rh * rd
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    return placements, total_vol

def _local_search(placements, boxes, container, deadline, rng, max_iters=50):
    """Shake local search: remove and re-insert boxes to find better positions."""
    W, H, D = container
    n = len(boxes)
    placed = []
    for i in range(n):
        p = placements[i]
        if p[0] == 0:
            placed.append((p[1], p[2], p[3], p[4], p[5], p[6]))
    
    tree = _IntervalTree3D()
    for b in placed:
        tree.add(b)
    
    loaded_indices = [i for i in range(n) if placements[i][0] == 0]
    if not loaded_indices:
        return placements, 0.0
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        idx = rng.choice(loaded_indices)
        p = placements[idx]
        box_pos = (p[1], p[2], p[3], p[4], p[5], p[6])
        # Remove from placed and tree
        placed.remove(box_pos)
        tree.remove(box_pos)
        w, h, d = boxes[idx]
        rots = _rotations(w, h, d)
        
        candidates = _extreme_points(placed, container)
        if (0, 0, 0) not in candidates:
            candidates.append((0, 0, 0))
        
        best_pos = None
        best_rot = None
        best_key = None
        for pos in candidates:
            for rw, rh, rd in rots:
                if _fits((rw, rh, rd), pos, container, tree):
                    key = (pos[2], pos[1], pos[0])
                    if best_key is None or key < best_key:
                        best_key = key
                        best_pos = pos
                        best_rot = (rw, rh, rd)
        
        if best_pos is not None:
            x, y, z = best_pos
            rw, rh, rd = best_rot
            box_tuple = (x, y, z, rw, rh, rd)
            placed.append(box_tuple)
            tree.add(box_tuple)
            placements[idx] = (0, x, y, z, rw, rh, rd)
        else:
            placed.append(box_pos)
            tree.add(box_pos)
    
    total_vol = 0.0
    for i in range(n):
        if placements[i][0] == 0:
            total_vol += placements[i][4] * placements[i][5] * placements[i][6]
    
    return placements, total_vol

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0
    W, H, D = container
    n = len(boxes)
    
    vols = [w * h * d for w, h, d in boxes]
    
    sort_keys = [
        ('vol_desc', lambda i: vols[i], True),
        ('min_dim_desc', lambda i: min(boxes[i]), True),
        ('max_dim_desc', lambda i: max(boxes[i]), True),
        ('compactness', lambda i: max(boxes[i]) / max(min(boxes[i]), 1e-9), False),
    ]
    
    best_placements = None
    best_vol = 0.0
    
    modes = ['extreme', 'faces', 'compact']
    
    for name, key_fn, reverse in sort_keys:
        if time.time() > deadline:
            break
        order = list(range(n))
        order.sort(key=key_fn, reverse=reverse)
        
        for mode in modes:
            if time.time() > deadline:
                break
            placements, vol = _greedy(boxes, container, order, deadline, mode)
            if vol > best_vol:
                best_vol = vol
                best_placements = placements
    
    if best_placements is not None:
        rng = random.Random(42)
        best_placements, best_vol = _local_search(best_placements, boxes, container, deadline, rng, max_iters=40)
    
    rng = random.Random(123)
    base_order = list(range(n))
    base_order.sort(key=lambda i: vols[i], reverse=True)
    
    num_restarts = 200
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        order = base_order[:]
        num_swaps = rng.randint(2, min(8, n))
        for _ in range(num_swaps):
            a, b = rng.randrange(n), rng.randrange(n)
            if a != b:
                order[a], order[b] = order[b], order[a]
        
        mode = rng.choice(['extreme', 'faces', 'compact'])
        placements, vol = _greedy(boxes, container, order, deadline, mode)
        if vol > best_vol:
            best_vol = vol
            best_placements = placements
    
    if best_placements is not None:
        rng2 = random.Random(999)
        best_placements, best_vol = _local_search(best_placements, boxes, container, deadline, rng2, max_iters=40)
    
    if best_placements is None:
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_placements
