"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START

# ---------- 3D BSP (skyline) based empty-space management ----------
# Each node represents an axis-aligned rectangular region [min_x, max_x) x [min_y, max_y) x [min_z, max_z).
# Leaves are either empty (can be subdivided) or occupied by a box.
# For efficiency, when inserting we recursively subdivide the tree so that every leaf is either
# 1) fully inside the new box, or
# 2) disjoint from it.

class _Node:
    __slots__ = ('x0', 'y0', 'z0', 'x1', 'y1', 'z1', 'box', 'left', 'right')
    def __init__(self, x0, y0, z0, x1, y1, z1):
        self.x0 = x0; self.y0 = y0; self.z0 = z0
        self.x1 = x1; self.y1 = y1; self.z1 = z1
        self.box = None       # if not None, this node is occupied by box (x0,y0,z0,w,h,d)
        self.left = None
        self.right = None

    def is_empty(self):
        return self.box is None and self.left is None and self.right is None

    def contains(self, x, y, z):
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1 and self.z0 <= z < self.z1

    def intersects(self, x, y, z, w, h, d):
        # box occupies [x, x+w) x [y, y+h) x [z, z+d)
        if x + w <= self.x0 or self.x1 <= x:
            return False
        if y + h <= self.y0 or self.y1 <= y:
            return False
        if z + d <= self.z0 or self.z1 <= z:
            return False
        return True

    def fully_inside(self, x, y, z, w, h, d):
        return (self.x0 >= x and self.x1 <= x + w and
                self.y0 >= y and self.y1 <= y + h and
                self.z0 >= z and self.z1 <= z + d)


class _BSP:
    def __init__(self, W, H, D):
        self.W = W
        self.H = H
        self.D = D
        self.root = _Node(0, 0, 0, W, H, D)
        self.boxes = []  # list of (x, y, z, w, h, d)

    def _subdivide(self, node, x, y, z, w, h, d):
        """
        Recursively subdivide 'node' so that no leaf partially intersects the new box region.
        After this, every leaf is either fully inside the box or fully disjoint.
        Returns nothing; mutates the tree.
        """
        if node.is_empty():
            if node.fully_inside(x, y, z, w, h, d):
                # This region is fully inside the new box; we don't need to subdivide further
                # We'll mark it as occupied at the top level, but for now leave as empty.
                # Actually we need to handle this differently.
                return
            if not node.intersects(x, y, z, w, h, d):
                return
            # Need to subdivide
            # Choose the axis with the largest extent to subdivide
            ex = node.x1 - node.x0
            ey = node.y1 - node.y0
            ez = node.z1 - node.z0
            if ex >= ey and ex >= ez:
                mid = (node.x0 + node.x1) // 2
                if mid == node.x0 or mid == node.x1:
                    # Can't subdivide further
                    return
                node.left = _Node(node.x0, node.y0, node.z0, mid, node.y1, node.z1)
                node.right = _Node(mid, node.y0, node.z0, node.x1, node.y1, node.z1)
                self._subdivide(node.left, x, y, z, w, h, d)
                self._subdivide(node.right, x, y, z, w, h, d)
            elif ey >= ex and ey >= ez:
                mid = (node.y0 + node.y1) // 2
                if mid == node.y0 or mid == node.y1:
                    return
                node.left = _Node(node.x0, node.y0, node.z0, node.x1, mid, node.z1)
                node.right = _Node(node.x0, mid, node.z0, node.x1, node.y1, node.z1)
                self._subdivide(node.left, x, y, z, w, h, d)
                self._subdivide(node.right, x, y, z, w, h, d)
            else:
                mid = (node.z0 + node.z1) // 2
                if mid == node.z0 or mid == node.z1:
                    return
                node.left = _Node(node.x0, node.y0, node.z0, node.x1, node.y1, mid)
                node.right = _Node(node.x0, node.y0, mid, node.x1, node.y1, node.z1)
                self._subdivide(node.left, x, y, z, w, h, d)
                self._subdivide(node.right, x, y, z, w, h, d)
        else:
            # This node is a leaf that intersects but is not fully inside.
            # Since it's a leaf (no children), we need to subdivide it.
            # But if it's already a leaf with no box, we subdivide as above.
            # If it has a box, it means this region is occupied; but we shouldn't be
            # trying to place inside an occupied region.
            # Actually, our invariant is: leaves are either empty or occupied.
            # If occupied, it should have been detected earlier.
            # Let's handle: if box is set, this region is occupied, so it intersects
            # the new box -> conflict. We should not be subdividing an occupied node.
            # Instead, we should detect conflicts before subdividing.
            pass

    def can_place(self, x, y, z, w, h, d):
        """Check if box can be placed at (x,y,z) with dims (w,h,d)."""
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > self.W or y + h > self.H or z + d > self.D:
            return False
        return self._check(self.root, x, y, z, w, h, d)

    def _check(self, node, x, y, z, w, h, d):
        if not node.intersects(x, y, z, w, h, d):
            return True
        if node.box is not None:
            # This node is occupied; check if it overlaps with the new box
            px, py, pz, pw, ph, pd = node.box
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
            return True
        if node.left is None and node.right is None:
            # Empty leaf, no conflict
            return True
        return self._check(node.left, x, y, z, w, h, d) and self._check(node.right, x, y, z, w, h, d)

    def place(self, x, y, z, w, h, d):
        """Place box, assuming can_place returned True."""
        self.boxes.append((x, y, z, w, h, d))
        # We need to update the BSP to reflect this placement.
        # For simplicity, we'll rebuild the occupied regions.
        # Actually, let's just store boxes and do overlap check against all boxes
        # for now, since BSP insertion is complex. We'll use the BSP for candidate
        # generation instead.
        pass

    def candidate_positions(self, w, h, d):
        """Generate candidate positions ordered by (z, y, x)."""
        # Collect all relevant x, y, z coordinates
        xs = set([0])
        ys = set([0])
        zs = set([0])
        for (px, py, pz, pw, ph, pd) in self.boxes:
            xs.add(px)
            xs.add(px + pw)
            ys.add(py)
            ys.add(py + ph)
            zs.add(pz)
            zs.add(pz + pd)

        # Filter and sort
        xs = sorted([x for x in xs if x + w <= self.W])
        ys = sorted([y for y in ys if y + h <= self.H])
        zs = sorted([z for z in zs if z + d <= self.D])

        cands = []
        for z in zs:
            for y in ys:
                for x in xs:
                    if self.can_place(x, y, z, w, h, d):
                        cands.append((x, y, z))

        cands.sort(key=lambda c: (c[2], c[1], c[0]))  # sort by z, y, x
        return cands


# ---------- Overlap checking (fallback / validation) ----------

def _overlap(a, b):
    ax, ay, az, aw, ah, ad = a
    bx, by, bz, bw, bh, bd = b
    if ax + aw <= bx or bx + bw <= ax:
        return False
    if ay + ah <= by or by + bh <= ay:
        return False
    if az + ad <= bz or bz + bd <= az:
        return False
    return True


def _fits_in_bin(x, y, z, w, h, d, W, H, D, placed):
    if x < 0 or y < 0 or z < 0:
        return False
    if x + w > W or y + h > H or z + d > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        if _overlap((x, y, z, w, h, d), (px, py, pz, pw, ph, pd)):
            return False
    return True


def _place_in_bin(placed, W, H, D, w, h, d):
    """Try to find a position for the box in the given bin."""
    # Generate candidate positions
    xs = set([0])
    ys = set([0])
    zs = set([0])
    for (px, py, pz, pw, ph, pd) in placed:
        xs.add(px)
        xs.add(px + pw)
        ys.add(py)
        ys.add(py + ph)
        zs.add(pz)
        zs.add(pz + pd)

    xs = sorted([x for x in xs if x + w <= W])
    ys = sorted([y for y in ys if y + h <= H])
    zs = sorted([z for z in zs if z + d <= D])

    # Sort candidates by (z, y, x) to prefer compact placement
    cands = []
    for z in zs:
        for y in ys:
            for x in xs:
                cands.append((x, y, z))
    cands.sort(key=lambda c: (c[2], c[1], c[0]))

    for (x, y, z) in cands:
        if _fits_in_bin(x, y, z, w, h, d, W, H, D, placed):
            return (x, y, z)
    return None


def _score_placement(placement, boxes, container):
    W, H, D = container
    cont_vol = W * H * D
    if not placement:
        return float("inf")
    bins = {}
    for (b, x, y, z, w, h, d) in placement:
        bins.setdefault(b, []).append((w, h, d))
    B = len(bins)
    last = max(bins.keys())
    last_vol = sum(w * h * d for (w, h, d) in bins[last])
    frac = last_vol / cont_vol if cont_vol > 0 else 1.0
    return B + 0.1 * (1 - frac)


def pack(boxes, container):
    start = time.time()
    deadline = start + 110.0  # safety margin under 120s
    W, H, D = container

    # index boxes with their dims
    indexed = [(i, w, h, d, w * h * d) for (i, (w, h, d)) in enumerate(boxes)]
    # First-fit decreasing by volume, tie-break by largest dimension
    indexed.sort(key=lambda t: (-t[4], -(t[1] + t[2] + t[3])))

    bins = []  # list of lists of (x, y, z, w, h, d)
    box_bin = {}
    box_pos = {}
    box_dim = {}

    for (i, w, h, d, vol) in indexed:
        if time.time() > deadline - 20.0:
            break
        placed_any = False
        for bidx, placed in enumerate(bins):
            pos = _place_in_bin(placed, W, H, D, w, h, d)
            if pos is not None:
                x, y, z = pos
                placed.append((x, y, z, w, h, d))
                box_bin[i] = bidx
                box_pos[i] = (x, y, z)
                box_dim[i] = (w, h, d)
                placed_any = True
                break
        if not placed_any:
            # open new bin
            bins.append([(0, 0, 0, w, h, d)])
            box_bin[i] = len(bins) - 1
            box_pos[i] = (0, 0, 0)
            box_dim[i] = (w, h, d)

    # Build current placement
    def build_placement():
        out = []
        for i in range(len(boxes)):
            if i not in box_bin:
                # fallback: place in bin 0 at origin (shouldn't happen)
                w, h, d = boxes[i]
                out.append((0, 0, 0, w, h, d))
            else:
                b = box_bin[i]
                x, y, z = box_pos[i]
                w, h, d = box_dim[i]
                out.append((b, x, y, z, w, h, d))
        return out

    current = build_placement()
    best = current
    best_score = _score_placement(best, boxes, container)

    # Local improvement: try to move boxes from higher bins into lower bins
    n = len(boxes)
    for it in range(3):
        if time.time() > deadline - 15.0:
            break
        improved = False
        # process boxes in descending bin order
        order = sorted(range(n), key=lambda i: box_bin.get(i, 0), reverse=True)
        for i in order:
            if time.time() > deadline - 15.0:
                break
            b = box_bin[i]
            if b == 0:
                continue
            w, h, d = box_dim[i]
            # try to move into each lower bin
            for tb in range(b):
                placed = bins[tb]
                pos = _place_in_bin(placed, W, H, D, w, h, d)
                if pos is not None:
                    x, y, z = pos
                    placed.append((x, y, z, w, h, d))
                    box_bin[i] = tb
                    box_pos[i] = (x, y, z)
                    improved = True
                    break
        if not improved:
            break

    # Compact toward origin within each bin to free space (light)
    for bidx in range(len(bins)):
        if time.time() > deadline - 10.0:
            break
        placed = bins[bidx]
        # sort by (z, y, x) and try to shift each toward origin
        for k in range(len(placed)):
            x, y, z, w, h, d = placed[k]
            others = [placed[j] for j in range(len(placed)) if j != k]
            new_x = x
            for nx in range(0, x + 1):
                if _fits_in_bin(nx, y, z, w, h, d, W, H, D, others):
                    new_x = nx
                    break
            new_y = y
            for ny in range(0, y + 1):
                if _fits_in_bin(new_x, ny, z, w, h, d, W, H, D, others):
                    new_y = ny
                    break
            new_z = z
            for nz in range(0, z + 1):
                if _fits_in_bin(new_x, new_y, nz, w, h, d, W, H, D, others):
                    new_z = nz
                    break
            if (new_x, new_y, new_z) != (x, y, z):
                placed[k] = (new_x, new_y, new_z, w, h, d)

    # Rebuild and re-score
    current = build_placement()
    cur_score = _score_placement(current, boxes, container)
    if cur_score < best_score:
        best = current
        best_score = cur_score

    # Final safety: ensure validity by a simple rebuild check; if invalid, fall back to a trivial valid packing
    def validate(placement):
        binsv = {}
        for (b, x, y, z, w, h, d) in placement:
            if b < 0:
                return False
            if x + w > W or y + h > H or z + d > D or x < 0 or y < 0 or z < 0:
                return False
            binsv.setdefault(b, []).append((x, y, z, w, h, d))
        for b, lst in binsv.items():
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    if _overlap(lst[i], lst[j]):
                        return False
        return True

    if not validate(best):
        # trivial valid: one box per bin
        best = [(i, 0, 0, 0, w, h, d) for (i, (w, h, d)) in enumerate(boxes)]

    return best

# EVOLVE-BLOCK-END
