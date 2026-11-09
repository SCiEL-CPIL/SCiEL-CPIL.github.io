"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START

def _compute_score(placements, boxes, container):
    """Compute the score: B + 0.1*(1 - last-bin occupied volume fraction)."""
    if not placements:
        return 0.0
    
    W, H, D = container
    container_vol = W * H * D
    
    bin_volumes = {}
    for (b, x, y, z, w, h, d) in placements:
        bin_volumes[b] = bin_volumes.get(b, 0) + w * h * d
    
    B = len(bin_volumes)
    if B == 0:
        return 0.0
    
    last_bin = max(bin_volumes.keys())
    last_vol = bin_volumes[last_bin]
    last_frac = last_vol / container_vol if container_vol > 0 else 0.0
    
    return B + 0.1 * (1.0 - last_frac)


class _BSPNode:
    """A node in the BSP tree representing an empty rectangular subspace."""
    __slots__ = ['x', 'y', 'z', 'w', 'h', 'd', 'left', 'right', 'box']
    
    def __init__(self, x, y, z, w, h, d):
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        self.h = h
        self.d = d
        self.left = None
        self.right = None
        self.box = None  # (bx, by, bz, bw, bh, bd) if this node is a leaf with a box


def _bsp_find_position(root, w, h, d):
    """Find a valid position in the BSP tree for a box with dims (w,h,d).
    Returns (x, y, z) or None. Prefers positions minimizing z, then y, then x."""
    if root is None:
        return None
    
    # If this node is a leaf with a box, no space here
    if root.box is not None:
        return None
    
    # If this node is small enough to fit the box, check if it's fully empty
    if root.w >= w and root.h >= h and root.d >= d:
        # This subspace can fit the box. Place at the corner (x, y, z)
        # But we need to verify it's truly empty (no child boxes overlap)
        # Since BSP only stores boxes in leaves, and internal nodes are empty space,
        # if this node has no box and its children don't cover the placement, we're good.
        # Actually, in a proper BSP, if we can fit here, we should check children.
        # Let's use a recursive approach: try to find a spot in this subtree.
        
        best_pos = None
        best_score = float('inf')
        
        def _search(node):
            nonlocal best_pos, best_score
            if node is None or node.box is not None:
                return
            
            # If this node can fit the box
            if node.w >= w and node.h >= h and node.d >= d:
                # Try placing at (node.x, node.y, node.z)
                # But we need to make sure no existing box overlaps
                # In BSP, if we reach here without hitting a box, the space is free
                # Actually, we need to be more careful. Let's check children.
                
                # For now, let's try the corner position
                px, py, pz = node.x, node.y, node.z
                pos_score = pz * 1000000 + py * 1000 + px
                if pos_score < best_score:
                    best_score = pos_score
                    best_pos = (px, py, pz)
                
                # Also check if we can fit in children (they might have better positions)
                if node.left:
                    _search(node.left)
                if node.right:
                    _search(node.right)
        
        _search(root)
        return best_pos
    
    # If this node can't fit the box, search children
    if root.left:
        pos = _bsp_find_position(root.left, w, h, d)
        if pos is not None:
            return pos
    if root.right:
        pos = _bsp_find_position(root.right, w, h, d)
        if pos is not None:
            return pos
    
    return None


def _bsp_insert(root, bx, by, bz, bw, bh, bd):
    """Insert a box into the BSP tree, splitting nodes as needed.
    Returns the (possibly new) root."""
    if root is None:
        node = _BSPNode(bx, by, bz, bw, bh, bd)
        node.box = (bx, by, bz, bw, bh, bd)
        return node
    
    # If this node is a leaf with a box, we need to create a new internal node
    if root.box is not None:
        # Create a new internal node that contains both the old box and the new box
        new_root = _BSPNode(root.x, root.y, root.z, root.w, root.h, root.d)
        new_root.left = root
        # Now insert the new box into new_root
        root = new_root
    
    # Check if the new box fits entirely within this node
    if (bx >= root.x and by >= root.y and bz >= root.z and
        bx + bw <= root.x + root.w and by + bh <= root.y + root.h and bz + bd <= root.z + root.d):
        
        # The box fits in this node. Now we need to find a child or create one.
        # For simplicity, let's just store it here if there's no conflict
        # Actually, we need to split the space. Let's use a simpler approach:
        # Just check if the box overlaps with existing boxes in this subtree.
        
        # For now, let's just return the root as-is and handle placement differently.
        # This BSP approach is getting complex. Let's simplify.
        pass
    
    return root


def _find_position(bin_boxes, w, h, d, W, H, D):
    """Find a valid position in the bin for a box with dims (w,h,d).
    Returns (x, y, z) or None. Uses corner-candidate generation.
    Prefers positions that minimize z, then y, then x."""
    if not bin_boxes:
        return (0, 0, 0)
    
    # Generate candidate coordinates from existing box edges
    x_cands = {0}
    y_cands = {0}
    z_cands = {0}
    for (ox, oy, oz, ow, oh, od) in bin_boxes:
        x_cands.add(ox + ow)
        y_cands.add(oy + oh)
        z_cands.add(oz + od)
    
    x_cands = sorted([xc for xc in x_cands if xc + w <= W])
    y_cands = sorted([yc for yc in y_cands if yc + h <= H])
    z_cands = sorted([zc for zc in z_cands if zc + d <= D])
    
    if not x_cands or not y_cands or not z_cands:
        return None
    
    best_pos = None
    best_score = float('inf')
    
    for xc in x_cands:
        for yc in y_cands:
            for zc in z_cands:
                # Fast overlap check
                ok = True
                for (ox, oy, oz, ow, oh, od) in bin_boxes:
                    if (xc < ox + ow and ox < xc + w and
                        yc < oy + oh and oy < yc + h and
                        zc < oz + od and oz < zc + d):
                        ok = False
                        break
                if ok:
                    # Prefer positions that are lower in z, then y, then x
                    pos_score = zc * 1000000 + yc * 1000 + xc
                    if pos_score < best_score:
                        best_score = pos_score
                        best_pos = (xc, yc, zc)
                    # Early exit if we found the absolute best possible position
                    if xc == 0 and yc == 0 and zc == 0:
                        return (0, 0, 0)
    
    return best_pos


def _pack_bfd(boxes, container, order_key, time_limit=None, start_time=None, best_fit=False):
    """Best Fit Decreasing with a given ordering key.
    order_key: callable that takes (w,h,d) and returns sort key (ascending).
    We sort descending by this key.
    If best_fit=True, select the bin that gives the lowest (z,y,x) position."""
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 115 - (start_time - start_time)  # full budget
    
    W, H, D = container
    n = len(boxes)
    
    # Sort boxes by the key descending
    indexed = list(range(n))
    indexed.sort(key=lambda i: order_key(boxes[i]), reverse=True)
    
    bins = []  # list of lists of (x, y, z, w, h, d)
    results = [None] * n
    
    for idx in indexed:
        if time.time() > start_time + time_limit:
            # Time's up, place remaining boxes at origin of new bins (safe fallback)
            for j in range(idx, n):
                if results[j] is None:
                    b_idx = len(bins)
                    bins.append([(0, 0, 0, boxes[j][0], boxes[j][1], boxes[j][2])])
                    results[j] = (b_idx, 0, 0, 0)
            break
        
        w, h, d = boxes[idx]
        placed = False
        
        if best_fit and bins:
            # Best fit: find the bin that gives the best (lowest z, then y, then x) position
            best_b_idx = -1
            best_pos = None
            best_score = float('inf')
            
            for b_idx, bin_boxes in enumerate(bins):
                pos = _find_position(bin_boxes, w, h, d, W, H, D)
                if pos is not None:
                    x, y, z = pos
                    pos_score = z * 1000000 + y * 1000 + x
                    if pos_score < best_score:
                        best_score = pos_score
                        best_b_idx = b_idx
                        best_pos = pos
            
            if best_b_idx >= 0:
                x, y, z = best_pos
                bins[best_b_idx].append((x, y, z, w, h, d))
                results[idx] = (best_b_idx, x, y, z)
                placed = True
        
        if not placed:
            # First fit or no best-fit found
            for b_idx, bin_boxes in enumerate(bins):
                pos = _find_position(bin_boxes, w, h, d, W, H, D)
                if pos is not None:
                    x, y, z = pos
                    bin_boxes.append((x, y, z, w, h, d))
                    results[idx] = (b_idx, x, y, z)
                    placed = True
                    break
        
        if not placed:
            b_idx = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            results[idx] = (b_idx, 0, 0, 0)
    
    # Fill in any remaining unplaced boxes
    for j in range(n):
        if results[j] is None:
            b_idx = len(bins)
            bins.append([(0, 0, 0, boxes[j][0], boxes[j][1], boxes[j][2])])
            results[j] = (b_idx, 0, 0, 0)
    
    placements = []
    for i in range(n):
        b_idx, x, y, z = results[i]
        w, h, d = boxes[i]
        placements.append((b_idx, x, y, z, w, h, d))
    
    return placements


def _pack_shelf(boxes, container):
    """Shelf heuristic: rows along x, shelves along y, layers along z."""
    W, H, D = container
    placements = []
    b = 0
    x = y = z = 0
    row_h = 0
    layer_d = 0
    for w, h, d in boxes:
        if x + w > W:
            x = 0
            y += row_h
            row_h = 0
        if y + h > H:
            y = 0
            z += layer_d
            layer_d = 0
            x = 0
            row_h = 0
        if z + d > D:
            b += 1
            x = y = z = 0
            row_h = 0
            layer_d = 0
        placements.append((b, x, y, z, w, h, d))
        x += w
        row_h = max(row_h, h)
        layer_d = max(layer_d, d)
    return placements


def _pack_with_time_budget(boxes, container, deadline):
    """Try multiple heuristics within the time budget, return best placement."""
    start_time = time.time()
    
    best_placements = None
    best_score = float('inf')
    
    def update(placements):
        nonlocal best_placements, best_score
        if placements is None:
            return
        score = _compute_score(placements, boxes, container)
        if score < best_score:
            best_score = score
            best_placements = placements
    
    # Strategy 1: Shelf (very fast)
    update(_pack_shelf(boxes, container))
    
    if time.time() > deadline - 5:
        return best_placements
    
    # Strategy 2: BFD sorted by volume descending (first-fit)
    update(_pack_bfd(boxes, container, lambda b: b[0]*b[1]*b[2], best_fit=False))
    
    if time.time() > deadline - 3:
        return best_placements
    
    # Strategy 3: BFD sorted by max dimension descending (first-fit)
    update(_pack_bfd(boxes, container, lambda b: max(b), best_fit=False))
    
    if time.time() > deadline - 2:
        return best_placements
    
    # Strategy 4: BFD sorted by volume, then max dim (tie-break) (first-fit)
    update(_pack_bfd(boxes, container, lambda b: b[0]*b[1]*b[2] + 0.001*max(b), best_fit=False))
    
    if time.time() > deadline - 2:
        return best_placements
    
    # Strategy 5: BFD sorted by surface area descending (first-fit)
    update(_pack_bfd(boxes, container, lambda b: 2*(b[0]*b[1] + b[1]*b[2] + b[0]*b[2]), best_fit=False))
    
    if time.time() > deadline - 2:
        return best_placements
    
    # Strategy 6: BFD sorted by volume descending (best-fit)
    update(_pack_bfd(boxes, container, lambda b: b[0]*b[1]*b[2], best_fit=True))
    
    if time.time() > deadline - 2:
        return best_placements
    
    # Strategy 7: BFD sorted by max dimension descending (best-fit)
    update(_pack_bfd(boxes, container, lambda b: max(b), best_fit=True))
    
    if time.time() > deadline - 2:
        return best_placements
    
    # Strategy 8+: Randomized restarts with mixed ordering
    n = len(boxes)
    vol = [w * h * d for w, h, d in boxes]
    maxdim = [max(w, h, d) for w, h, d in boxes]
    surf = [2*(w*h + h*d + w*d) for w, h, d in boxes]
    
    attempts = 0
    while time.time() < deadline - 1:
        attempts += 1
        # Build a random-weighted ordering
        # Mix of volume, max-dim, and surface area with random perturbation
        w1 = random.random()
        w2 = random.random()
        w3 = random.random()
        total_w = w1 + w2 + w3
        if total_w == 0:
            total_w = 1
        
        weights = []
        for i in range(n):
            weight = (w1 * vol[i] + w2 * maxdim[i] * max(vol) / max(maxdim) + w3 * surf[i] * max(vol) / max(surf))
            weight += random.random() * 0.05 * max(vol)
            weights.append(weight)
        
        indexed = list(range(n))
        indexed.sort(key=lambda i: weights[i], reverse=True)
        
        # Randomly choose between first-fit and best-fit
        use_best_fit = random.random() < 0.5
        
        W, H, D = container
        bins = []
        results = [None] * n
        
        for idx in indexed:
            if time.time() > deadline - 1:
                break
            w, h, d = boxes[idx]
            placed = False
            
            if use_best_fit and bins:
                # Best fit: find the bin that gives the best (lowest z, then y, then x) position
                best_b_idx = -1
                best_pos = None
                best_score = float('inf')
                
                for b_idx, bin_boxes in enumerate(bins):
                    pos = _find_position(bin_boxes, w, h, d, W, H, D)
                    if pos is not None:
                        x, y, z = pos
                        pos_score = z * 1000000 + y * 1000 + x
                        if pos_score < best_score:
                            best_score = pos_score
                            best_b_idx = b_idx
                            best_pos = pos
                
                if best_b_idx >= 0:
                    x, y, z = best_pos
                    bins[best_b_idx].append((x, y, z, w, h, d))
                    results[idx] = (best_b_idx, x, y, z)
                    placed = True
            
            if not placed:
                # First fit or no best-fit found
                for b_idx, bin_boxes in enumerate(bins):
                    pos = _find_position(bin_boxes, w, h, d, W, H, D)
                    if pos is not None:
                        x, y, z = pos
                        bin_boxes.append((x, y, z, w, h, d))
                        results[idx] = (b_idx, x, y, z)
                        placed = True
                        break
            
            if not placed:
                b_idx = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                results[idx] = (b_idx, 0, 0, 0)
        
        # Fill in any remaining unplaced boxes
        for j in range(n):
            if results[j] is None:
                b_idx = len(bins)
                bins.append([(0, 0, 0, boxes[j][0], boxes[j][1], boxes[j][2])])
                results[j] = (b_idx, 0, 0, 0)
        
        placements = []
        for i in range(n):
            b_idx, x, y, z = results[i]
            w, h, d = boxes[i]
            placements.append((b_idx, x, y, z, w, h, d))
        
        update(placements)
        
        # Safety: stop if we've done many attempts
        if attempts > 100:
            break
    
    return best_placements


def pack(boxes, container):
    """
    Place every box into numbered bins.
    Returns one placement (bin, x, y, z, w, h, d) per box.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    if not boxes:
        return []
    
    placements = _pack_with_time_budget(boxes, container, deadline)
    
    return placements

# EVOLVE-BLOCK-END
