# EVOLVE-BLOCK-START
import time
import itertools

def pack(boxes, container):
    """
    3D bin packing: place every box unrotated without overlap into numbered bins.
    Strategy: Best-Fit Decreasing with exhaustive corner-position search per bin.
    """
    deadline = time.time() + 110.0  # safety margin before 120s limit

    W, H, D = container
    n = len(boxes)

    # Sort indices by volume descending (Best-Fit Decreasing)
    vol = [w * h * d for w, h, d in boxes]
    order = sorted(range(n), key=lambda i: -vol[i])

    # Each bin stores a list of placed boxes as (x, y, z, w, h, d)
    bins = []  # list of lists

    def get_corners(bin_data):
        """Generate candidate corner positions for a new box in this bin."""
        if not bin_data:
            return [(0, 0, 0)]

        corners = set()
        # Origin is always a candidate
        corners.add((0, 0, 0))

        for (bx, by, bz, bw, bh, bd) in bin_data:
            # Right face
            corners.add((bx + bw, by, bz))
            # Front face
            corners.add((bx, by + bh, bz))
            # Top face
            corners.add((bx, by, bz + bd))
            # Right-front edge
            corners.add((bx + bw, by + bh, bz))
            # Right-top edge
            corners.add((bx + bw, by, bz + bd))
            # Front-top edge
            corners.add((bx, by + bh, bz + bd))
            # Right-front-top corner
            corners.add((bx + bw, by + bh, bz + bd))

        # Filter corners that are within container bounds
        valid_corners = []
        for (cx, cy, cz) in corners:
            if cx < W and cy < H and cz < D:
                valid_corners.append((cx, cy, cz))

        return valid_corners

    def can_place(bin_data, x, y, z, w, h, d):
        """Check if a box at (x,y,z) with dims (w,h,d) fits without overlap."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (bx, by, bz, bw, bh, bd) in bin_data:
            # Check overlap
            if not (x + w <= bx or bx + bw <= x or
                    y + h <= by or by + bh <= y or
                    z + d <= bz or bz + bd <= z):
                return False
        return True

    def find_best_position(bin_data, w, h, d):
        """
        Find the best position for a box in this bin.
        'Best' means the position that leaves the most room (minimize wasted space).
        We use a simple heuristic: prefer positions closer to origin (pack tight).
        Returns (x, y, z) or None if no fit.
        """
        corners = get_corners(bin_data)
        best_pos = None
        best_score = float('inf')

        for (cx, cy, cz) in corners:
            if can_place(bin_data, cx, cy, cz, w, h, d):
                # Score: prefer positions closer to origin (tight packing)
                score = cx + cy + cz
                if score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)

        return best_pos

    # Place each box in sorted order
    placement_map = [None] * n  # index -> (bin_id, x, y, z, w, h, d)

    for idx in order:
        if time.time() > deadline:
            # Fallback: just place remaining boxes in a new bin each (shouldn't happen often)
            for j in range(idx, n):
                if placement_map[j] is None:
                    b = len(bins)
                    bins.append([(0, 0, 0, boxes[j][0], boxes[j][1], boxes[j][2])])
                    placement_map[j] = (b, 0, 0, 0, boxes[j][0], boxes[j][1], boxes[j][2])
            break

        w, h, d = boxes[idx]

        # Try to fit in existing bins first
        placed = False
        for b_idx in range(len(bins)):
            pos = find_best_position(bins[b_idx], w, h, d)
            if pos is not None:
                x, y, z = pos
                bins[b_idx].append((x, y, z, w, h, d))
                placement_map[idx] = (b_idx, x, y, z, w, h, d)
                placed = True
                break

        if not placed:
            # Open a new bin
            b_idx = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            placement_map[idx] = (b_idx, 0, 0, 0, w, h, d)

    # Build result in original order
    result = []
    for i in range(n):
        result.append(placement_map[i])

    return result
# EVOLVE-BLOCK-END
