<<<<<<< SEARCH
def _find_position(box, placements, W, H, D, grid=None):
    """
    Find the best position for a box in a bin.
    Strategy: Bottom-Left-Far (Lowest Z, then Lowest Y, then Lowest X).
    We generate candidate coordinates from existing box boundaries.
    """
    w, h, d = box
    if w > W or h > H or d > D:
        return None

    # Candidate coordinates are 0 and right/top/front faces of existing boxes
    xs = set([0])
    ys = set([0])
    zs = set([0])

    for (_, px, py, pz, pw, ph, pd) in placements:
        xs.add(px + pw)
        ys.add(py + ph)
        zs.add(pz + pd)

    # Filter candidates that fit within container
    valid_xs = sorted([x for x in xs if x + w <= W])
    valid_ys = sorted([y for y in ys if y + h <= H])
    valid_zs = sorted([z for z in zs if z + d <= D])

    if not valid_xs or not valid_ys or not valid_zs:
        return None

    # Fast path using occupancy grid:
    # Build a boolean array of size (W, H, D) marking occupied cells.
    # For each candidate z, y, x, check if the sub-array [x:x+w, y:y+h, z:z+d] is all False.
    # We want the lexicographically smallest (z, y, x).
    # Since we iterate z, y, x in sorted order, the first valid one is the answer.
    if grid is not None:
        # grid is a 3D boolean array of shape (W, H, D)
        # Note: NumPy indexing is [x, y, z]
        for z in valid_zs:
            for y in valid_ys:
                for x in valid_xs:
                    if not grid[x:x+w, y:y+h, z:z+d].any():
                        return (x, y, z)
        return None

    # Fallback: manual check
    for z in valid_zs:
        for y in valid_ys:
            for x in valid_xs:
                if _can_place(box, (x, y, z), placements, W, H, D, grid):
                    return (x, y, z)
    return None
=======
def _find_position(box, placements, W, H, D, grid=None):
    """
    Find the best position for a box in a bin.
    Strategy: Bottom-Left-Far (Lowest Z, then Lowest Y, then Lowest X).
    We generate candidate coordinates from existing box boundaries.
    Uses NumPy for fast overlap checking.
    """
    w, h, d = box
    if w > W or h > H or d > D:
        return None

    # Candidate coordinates are 0 and right/top/front faces of existing boxes
    xs = set([0])
    ys = set([0])
    zs = set([0])

    for (_, px, py, pz, pw, ph, pd) in placements:
        xs.add(px + pw)
        ys.add(py + ph)
        zs.add(pz + pd)

    # Filter candidates that fit within container
    valid_xs = sorted([x for x in xs if x + w <= W])
    valid_ys = sorted([y for y in ys if y + h <= H])
    valid_zs = sorted([z for z in zs if z + d <= D])

    if not valid_xs or not valid_ys or not valid_zs:
        return None

    if grid is not None:
        # Use vectorized NumPy approach: for each z, compute which (x,y) positions are valid
        # by checking if the sub-array is all False.
        for z in valid_zs:
            # Get the 2D slice at this z: shape (W, H)
            z_slice = grid[:, :, z]  # shape (W, H)
            
            # For each y in valid_ys, check if there's an x in valid_xs where
            # grid[x:x+w, y:y+h, z:z+d] is all False
            # Equivalent: for each y, check if min over x of (z_slice[x:x+w, y:y+h].any()) is False
            
            # More efficient: precompute for each y which x ranges are free
            # For a given z and y, the box occupies x in [x, x+w) and y in [y, y+h)
            # We need z_slice[x:x+w, y:y+h].any() == False for all such (x,y)
            
            # Vectorized approach: for each candidate y, compute the OR over the h rows
            # Then for each candidate x, check if the column range is all False
            found = False
            for y in valid_ys:
                if y + h > H:
                    continue
                # Check if any part of the box at this y overlaps with occupied cells
                # z_slice[:, y:y+h] has shape (W, h)
                # We need to find x such that z_slice[x:x+w, y:y+h].any() == False
                # i.e., all values in that sub-array are False
                # This is equivalent to: for all i in [0,w), z_slice[x+i, y:y+h].all(axis=1) == True
                # Better: compute col_valid = ~z_slice[:, y:y+h].any(axis=1)  # shape (W,)
                # Then we need a run of w consecutive True values starting at x
                # But we only care about x in valid_xs
                
                col_valid = ~z_slice[:, y:y+h].any(axis=1)  # shape (W,)
                # Now for each x in valid_xs, check if col_valid[x:x+w].all()
                # We can vectorize this: for each x, check if min(col_valid[x:x+w]) == 1
                # But valid_xs is small, so just loop
                for x in valid_xs:
                    if x + w <= W and col_valid[x:x+w].all():
                        return (x, y, z)
            # Continue to next z
        return None

    # Fallback: manual check
    for z in valid_zs:
        for y in valid_ys:
            for x in valid_xs:
                if _can_place(box, (x, y, z), placements, W, H, D, grid):
                    return (x, y, z)
    return None
>>>>>>> REPLACE

