<<<<<<< SEARCH
def _hex_to_cartesian(q, r, scale=1.0):
    """Convert axial coordinates (q, r) to Cartesian for pointy-top hexagons."""
    x = scale * math.sqrt(3) * (q + r / 2.0)
    y = scale * 1.5 * r
    return x, y
=======
def _hex_to_cartesian(q, r, spacing_factor=1.0):
    """Convert axial coordinates (q, r) to Cartesian for pointy-top hexagons.
    spacing_factor > 1.0 ensures disjoint hexagons."""
    x = spacing_factor * math.sqrt(3) * (q + r / 2.0)
    y = spacing_factor * 1.5 * r
    return x, y
>>>>>>> REPLACE

