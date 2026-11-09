<<<<<<< SEARCH
def _score_position(pos, box_dims, placed, container):
    """Score a position: prefer compact (close to origin) and tight fits."""
    W, H, D = container
    x, y, z = pos
    w, h, d = box_dims
    # Prefer positions closer to origin (compact)
    dist = x + y + z
    # Prefer tighter fits (less wasted space along each axis)
    slack = (W - x - w) + (H - y - h) + (D - z - d)
    return dist * 0.5 + slack * 0.1
=======
def _score_position(pos, box_dims, placed, container):
    """Score a position: prefer compact (close to origin) and tight fits."""
    W, H, D = container
    x, y, z = pos
    w, h, d = box_dims
    # Prefer positions closer to origin (compact)
    dist = x + y + z
    # Prefer tighter fits (less wasted space along each axis)
    slack = (W - x - w) + (H - y - h) + (D - z - d)
    # Prefer positions that leave more room in the largest dimension
    max_slack = max(W - x - w, H - y - h, D - z - d)
    return dist * 0.6 + slack * 0.05 + max_slack * 0.1
>>>>>>> REPLACE

