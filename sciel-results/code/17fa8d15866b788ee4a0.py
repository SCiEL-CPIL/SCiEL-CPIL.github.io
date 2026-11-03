"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    """Place 19 unit hexagons on a hexagonal lattice (edge-sharing, centers at
    integer (i,j) combinations of basis vectors) inside a regular hexagon of
    side length 3.0, centered at origin, flat-top orientation.

    Lattice basis (center-to-center distance = sqrt(3), interiors disjoint):
        u = (1.5, sqrt(3)/2)
        v = (0.5, sqrt(3)/2)
    Positions: (i*u + j*v) for i,j in {-1,0,1} with i+j in {-1,0,1} (19 sites).
    """
    s3 = math.sqrt(3.0)
    u = (1.5, s3 / 2.0)
    v = (0.5, s3 / 2.0)

    inner = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            # Hexagonal neighborhood: |i|<=1, |j|<=1, |i+j|<=1
            if abs(i + j) > 1:
                continue
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            inner.append([x, y, 0.0])

    # Outer hexagon: center (0,0), side length 3.0, rotation 0 (flat-top)
    return inner, (0.0, 0.0), 3.0, 0.0
# EVOLVE-BLOCK-END
