"""Rewritten 3D bin packing: best-fit decreasing with corner-candidate placement search."""
import time
import math

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    # Sort boxes by volume descending (stable tie-break by dimensions)
    indexed = list(enumerate(boxes))
    indexed.sort(key=lambda t: (-t[1][0]*t[1][1]*t[1][2], -t[1][0], -t[1][1], -t[1][2]))
    order = [idx for idx, _ in indexed]

    # Deadline guard
    start = time.time()
    DEADLINE = 110.0  # safety margin under 120s

    # Each bin: list of (x, y, z, w, h, d)
    bins = []  # list of lists of placements

    def overlaps(p, existing):
        """Check if new placement p overlaps any existing in the same bin."""
        px, py, pz, pw, ph, pd = p
        for ex, ey, ez, ew, eh, ed in existing:
            if (px < ex + ew and ex < px + pw and
                py < ey + eh and ey < py + ph and
                pz < ez + ed and ez < pz + pd):
                return True
        return False

    def fits(p):
        px, py, pz, pw, ph, pd = p
        return px >= 0 and py >= 0 and pz >= 0 and px + pw <= W and py + ph <= H and pz + pd <= D

    def candidate_positions(existing, w, h, d):
        """Generate candidate (x, y, z) positions for a box of size (w,h,d) given existing placements."""
        cands = set()
        cands.add((0, 0, 0))
        for ex, ey, ez, ew, eh, ed in existing:
            # Place to the right, above, or in front of existing box
            if ex + ew <= W:
                cands.add((ex + ew, ey, ez))
            if ey + eh <= H:
                cands.add((ex, ey + eh, ez))
            if ez + ed <= D:
                cands.add((ex, ey, ez + ed))
            # Also try corners that are "supported"
            if ex + ew <= W and ey + eh <= H:
                cands.add((ex + ew, ey + eh, ez))
            if ex + ew <= W and ez + ed <= D:
                cands.add((ex + ew, ey, ez + ed))
            if ey + eh <= H and ez + ed <= D:
                cands.add((ex, ey + eh, ez + ed))
            if ex + ew <= W and ey + eh <= H and ez + ed <= D:
                cands.add((ex + ew, ey + eh, ez + ed))
        # Filter to valid positions
        valid = []
        for (cx, cy, cz) in cands:
            if cx + w <= W and cy + h <= H and cz + d <= D:
                valid.append((cx, cy, cz))
        return valid

    def placement_score(pos, w, h, d):
        """Lower is better: prefer positions closer to origin (minimize x+y+z)."""
        x, y, z = pos
        return x + y + z

    def best_fit_bin(box_idx, w, h, d):
        """Find the best (bin_index, position) for a box, or None."""
        best_bin = -1
        best_pos = None
        best_score = float('inf')
        for bi, existing in enumerate(bins):
            cands = candidate_positions(existing, w, h, d)
            for pos in cands:
                p = (pos[0], pos[1], pos[2], w, h, d)
                if not overlaps(p, existing):
                    s = placement_score(pos, w, h, d)
                    if s < best_score:
                        best_score = s
                        best_bin = bi
                        best_pos = pos
        if best_bin >= 0:
            return best_bin, best_pos
        return None, None

    def new_bin_position(w, h, d):
        """Position for a new bin (always 0,0,0)."""
        return (0, 0, 0)

    # Main loop: place each box
    placements_by_box = [None] * n  # store (bin, x, y, z, w, h, d) per original index

    for step in order:
        if time.time() - start > DEADLINE:
            break
        w, h, d = boxes[step]
        bi, pos = best_fit_bin(step, w, h, d)
        if bi is None:
            # Start a new bin
            bi = len(bins)
            bins.append([])
            pos = new_bin_position(w, h, d)
        bins[bi].append((pos[0], pos[1], pos[2], w, h, d))
        placements_by_box[step] = (bi, pos[0], pos[1], pos[2], w, h, d)

    # If we timed out, try to place remaining boxes greedily into existing or new bins
    if time.time() - start > DEADLINE:
        for step in order:
            if placements_by_box[step] is not None:
                continue
            w, h, d = boxes[step]
            placed = False
            for bi in range(len(bins)):
                cands = candidate_positions(bins[bi], w, h, d)
                for pos in cands:
                    p = (pos[0], pos[1], pos[2], w, h, d)
                    if not overlaps(p, bins[bi]):
                        bins[bi].append(p)
                        placements_by_box[step] = (bi, pos[0], pos[1], pos[2], w, h, d)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                bi = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                placements_by_box[step] = (bi, 0, 0, 0, w, h, d)

    # Build result
    result = []
    for i in range(n):
        if placements_by_box[i] is None:
            # Fallback: put in a new bin at origin (shouldn't happen)
            w, h, d = boxes[i]
            bi = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            result.append((bi, 0, 0, 0, w, h, d))
        else:
            result.append(placements_by_box[i])
    return result
# EVOLVE-BLOCK-END
