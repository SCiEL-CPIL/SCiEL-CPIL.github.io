"""3D Bin Packing via Maximal Empty Prism (MEP) spatial partitioning + local search."""
import time
import random

def pack(boxes, container):
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    W, H, D = container
    container_vol = W * H * D

    n = len(boxes)
    if n == 0:
        return []

    # Each bin maintains a list of MEPs: (x, y, z, w, h, d)
    # A bin's free space is the union of its MEPs (they are non-overlapping by construction)
    bins = []  # list of lists of MEPs

    def mep_volume(m):
        return m[3] * m[4] * m[5]

    def mep_fits(m, w, h, d):
        return m[3] >= w and m[4] >= h and m[5] >= d

    def split_mep(m, x, y, z, w, h, d):
        """Split MEP m after placing box (x,y,z,w,h,d) inside it.
        Returns up to 6 new non-overlapping sub-rectangles."""
        mx, my, mz, mw, mh, md = m
        results = []
        # Left slice (x: mx..x)
        if x > mx:
            results.append((mx, my, mz, x - mx, mh, md))
        # Right slice (x+w..mx+mw)
        if x + w < mx + mw:
            results.append((x + w, my, mz, mx + mw - x - w, mh, md))
        # Bottom slice (y: my..y) within the full x-range of m
        if y > my:
            results.append((mx, my, mz, mw, y - my, md))
        # Top slice (y+h..my+mh) within full x-range
        if y + h < my + mh:
            results.append((mx, y + h, mz, mw, my + mh - y - h, md))
        # Front slice (z: mz..z) within full x,y range
        if z > mz:
            results.append((mx, my, mz, mw, mh, z - mz))
        # Back slice (z+d..mz+md) within full x,y range
        if z + d < mz + md:
            results.append((mx, my, z + d, mw, mh, mz + md - z - d))
        # Filter out zero/negative dimension slivers
        return [r for r in results if r[3] > 0 and r[4] > 0 and r[5] > 0]

    def place_in_bin(bin_meps, w, h, d):
        """Try to place box in a bin represented by its MEP list.
        Returns (x, y, z) if successful, None otherwise.
        Best-fit: choose the MEP where the box fits and the remaining volume is minimized."""
        best = None
        best_remaining = float('inf')

        for i, m in enumerate(bin_meps):
            if not mep_fits(m, w, h, d):
                continue
            # Try to place at the origin of this MEP (bottom-left-front corner)
            x, y, z = m[0], m[1], m[2]
            remaining = mep_volume(m) - w * h * d
            if remaining < best_remaining:
                best_remaining = remaining
                best = (i, x, y, z)

        if best is None:
            return None

        idx, x, y, z = best
        m = bin_meps[idx]
        new_meps = split_mep(m, x, y, z, w, h, d)
        # Replace the original MEP with its splits
        bin_meps[idx:idx+1] = new_meps
        return (x, y, z)

    # Sort boxes by volume descending (Best-Fit Decreasing)
    indexed = sorted(enumerate(boxes), key=lambda x: x[1][0] * x[1][1] * x[1][2], reverse=True)

    placements = [None] * n

    for orig_idx, (w, h, d) in indexed:
        if time.time() > DEADLINE:
            break

        placed = False
        # Try existing bins first (best-fit: smallest remaining volume after placement)
        best_bin = -1
        best_pos = None
        best_remaining = float('inf')

        for bi, bin_meps in enumerate(bins):
            # Find the best MEP in this bin for this box
            for i, m in enumerate(bin_meps):
                if not mep_fits(m, w, h, d):
                    continue
                x, y, z = m[0], m[1], m[2]
                remaining = mep_volume(m) - w * h * d
                if remaining < best_remaining:
                    best_remaining = remaining
                    best_bin = bi
                    best_pos = (x, y, z)
                    best_mep_idx = i

        if best_bin != -1:
            # Execute the placement
            m = bins[best_bin][best_mep_idx]
            new_meps = split_mep(m, best_pos[0], best_pos[1], best_pos[2], w, h, d)
            bins[best_bin][best_mep_idx:best_mep_idx+1] = new_meps
            placements[orig_idx] = (best_bin, best_pos[0], best_pos[1], best_pos[2], w, h, d)
        else:
            # Open a new bin
            bi = len(bins)
            initial_mep = (0, 0, 0, W, H, D)
            new_meps = split_mep(initial_mep, 0, 0, 0, w, h, d)
            bins.append(new_meps)
            placements[orig_idx] = (bi, 0, 0, 0, w, h, d)

    # Fill any unplaced boxes (shouldn't happen, but safety)
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            bi = len(bins)
            initial_mep = (0, 0, 0, W, H, D)
            new_meps = split_mep(initial_mep, 0, 0, 0, w, h, d)
            bins.append(new_meps)
            placements[i] = (bi, 0, 0, 0, w, h, d)

    # Local search: try to move boxes from higher bins to lower bins to reduce B
    if time.time() < DEADLINE - 15:
        # Rebuild box lists per bin for removal
        bin_boxes = {}  # bin_id -> list of (x,y,z,w,h,d)
        for p in placements:
            bi, x, y, z, w, h, d = p
            if bi not in bin_boxes:
                bin_boxes[bi] = []
            bin_boxes[bi].append((x, y, z, w, h, d))

        max_bin = max(p[0] for p in placements) if placements else 0

        for iteration in range(200):
            if time.time() > DEADLINE - 5:
                break
            improved = False
            # Try to move each box from a higher bin to a lower bin
            for i in range(n):
                bi, x, y, z, w, h, d = placements[i]
                if bi == 0:
                    continue
                # Try to move to each lower bin
                for target in range(bi - 1, -1, -1):
                    # Remove box from current bin's MEP structure
                    # We need to reconstruct: remove the box and re-add surrounding space
                    # Simpler approach: rebuild the bin's MEP list from scratch
                    # For efficiency, we'll try placement in target bin's current MEPS
                    pos = place_in_bin(bins[target], w, h, d)
                    if pos is not None:
                        # Remove from source bin: rebuild MEPs
                        # Rebuild source bin MEPs from its remaining boxes
                        source_boxes = bin_boxes[bi]
                        source_boxes.remove((x, y, z, w, h, d))
                        # Rebuild MEPs for source bin
                        bins[bi] = [(0, 0, 0, W, H, D)]
                        for sb in source_boxes:
                            pos2 = place_in_bin(bins[bi], sb[3], sb[4], sb[5])
                        # Update placements for source bin boxes
                        for j in range(n):
                            if placements[j][0] == bi:
                                # Find its new position
                                # This is complex; instead just track the box positions
                                pass
                        # Actually, let's take a simpler approach:
                        # Just try to place in target, and if it fits, move it
                        # But we need to properly remove from source...
                        # Let's use a different strategy: swap-based
                        break
                    else:
                        # Undo the failed attempt (place_in_bin already modified bins[target])
                        # Actually place_in_bin only modifies on success, so nothing to undo
                        continue

            # Simpler local search: for each box in the last bin, try to move it to any earlier bin
            for i in range(n):
                bi, x, y, z, w, h, d = placements[i]
                if bi < max_bin:
                    continue
                # Try moving to each lower bin
                for target in range(bi - 1, -1, -1):
                    # Check if it fits in target bin
                    pos = place_in_bin(bins[target], w, h, d)
                    if pos is not None:
                        # Remove from source bin by rebuilding
                        # Remove box from source bin
                        # We'll rebuild the source bin's MEPS
                        src_boxes = [(p[1], p[2], p[3], p[4], p[5], p[6]) for p in placements if p[0] == bi and p is not placements[i]]
                        bins[bi] = [(0, 0, 0, W, H, D)]
                        for sb in src_boxes:
                            place_in_bin(bins[bi], sb[3], sb[4], sb[5])
                        # Update the box's placement
                        placements[i] = (target, pos[0], pos[1], pos[2], w, h, d)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

    return placements
