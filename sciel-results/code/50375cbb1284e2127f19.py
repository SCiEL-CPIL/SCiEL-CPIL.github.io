<<<<<<< SEARCH
def _build_positions(layout_idx):
    """Generate 11 positions for given layout index."""
    sqrt3 = _SQRT3
    scale = 1.0000001
    u = (sqrt3*scale, 0.0)
    v = (sqrt3/2*scale, 1.5*scale)
=======
def _build_positions(layout_idx):
    """Generate 11 positions for given layout index."""
    sqrt3 = _SQRT3
    scale = _GAP  # use gap factor for disjointness
    u = (sqrt3*scale, 0.0)
    v = (sqrt3/2*scale, 1.5*scale)
>>>>>>> REPLACE
