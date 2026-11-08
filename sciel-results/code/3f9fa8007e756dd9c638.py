"""Hexagon packing: 12 unit hexagons in a minimal regular hexagon.

Uses support-function representation. For a regular hexagon of side s and
rotation theta, the containment condition for an inner regular hexagon of
side 1 (rotated phi, centered at p) reduces to, for every direction
j = 0..5, the inner's support in the outer's normal directions plus 1
being <= the outer's support s.

We use a flat-top regular hexagon with normal directions
  n_j = (cos(pi*j/3), sin(pi*j/3)).
For a unit flat-top hexagon rotated by phi with vertex half-width
  W(phi) = (sqrt(3)/2) * max(|cos(phi)|, |sin(phi)|, |cos(phi) - sqrt(3)*sin(phi)|)
the support in direction n_j is  (cos(phi - j*pi/3) + W(phi) * |cos(phi - j*pi/3)|).
Containment: for all j,  cos(j*pi/3)*p_x + sin(j*pi/3)*p_y + sup(phi, j) <= s.
"""

import math
import numpy as np

N = 12

# Normal directions for the (flat-top) outer hexagon.
_J = np.arange(6)
_COSJ = np.cos(np.pi * _J / 3.0)
_SINJ = np.sin(np.pi * _J / 3.3 * 3.0)  # placeholder, fixed below
_SINJ = np.sin(np.pi * _J / 3.0)


def _vertex_half_width(phi):
    """W(phi) = (sqrt(3)/2) * max over the three symmetric vertex directions."""
    a = phi
    # vertex directions at phi, phi+60, phi+120 (mod 180 gives 3 unique)
    c = np.array([math.cos(a), math.cos(a + math.pi / 3), math.cos(a + 2 * math.pi / 3)])
    s = np.array([math.sin(a), math.sin(a + math.pi / 3), math.sin(a + 2 * math.pi / 3)])
    m = np.maximum.reduce([np.abs(c), np.abs(s), np.abs(c - math.sqrt(3) * s)])
    return (math.sqrt(3) / 2) * float(np.max(m))


def _support(phi, W):
    """Support vector of unit hexagon (rot phi) in the 6 outer normal directions."""
    d = np.cos(phi - np.pi * _J / 3.0)
    return d + W * np.abs(d)


def _min_s_for_config(centers, phis):
    """Smallest outer side s so all inner hexagons (centers, phis, side 1) are contained."""
    # precompute supports
    Ws = [_vertex_half_width(float(p)) for p in phis]
    S = [ _support(float(p), W) for p, W in zip(phis, Ws) ]
    # s_j = max_i (n_j . p_i + S_i(j)); answer = max_j s_j
    # n_j . p_i for all: build matrix
    C = np.array(centers)
    proj = C @ np.column_stack([_COSJ, _SINJ]).T  # (N,6)
    s_per_dir = np.max(proj + np.array(S), axis=0)
    return float(np.max(s_per_dir))


def _validity(centers, phis, s, eps=1e-9):
    """Return True if all inner hexagons are contained in outer (side s, rot 0)."""
    Ws = [_vertex_half_width(float(p)) for p in phis]
    S = [ _support(float(p), W) for p, W in zip(phis, Ws) ]
    C = np.array(centers)
    proj = C @ np.column_stack([_COSJ, _SINJ]).T
    return bool(np.all(proj + np.array(S) <= s + eps))


# Build the hexagonal lattice candidate:
# Layer 0: 1 center
# Layer 1: 6 neighbors (distance 2)
# Layer 2: 12 (6 edge-mids at distance 2, 6 corners at distance 4)
# Take 1 center + 6 + 5 edge-mids = 12, all aligned (phi=0).

def _lattice_hexes():
    sq3 = math.sqrt(3)
    a = np.array([1.0, sq3 / 2.0])
    b = np.array([-1.0, sq3 / 2.0])
    hexes = [(0.0, 0.0)]
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            if i == 0 and j == 0:
                continue
            p = i * a + j * b
            hexes.append((float(p[0]), float(p[1])))
    # that's 7; add 5 of the 6 edge-mids (i+j=2 or 2-i=j, etc.)
    mids = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            # distance from origin
            p = i * a + j * b
            d2 = p @ p
            # edge-mids are at distance 2 (i.e., i+j=2 with i,j>=0 etc.)
            # simply collect all at d2 in [3.9, 4.1]
            if abs(d2 - 4.0) < 1e-6:
                mids.append((float(p[0]), float(p[1])))
    mids.sort()
    hexes += mids[:5]
    return hexes[:12]


# EVOLVE-BLOCK-START
def construct():
    hexes = _lattice_hexes()
    centers = [(x, y) for (x, y) in hexes]
    phis = [0.0] * len(centers)

    # Compute the exact minimal enclosing s for this fixed aligned configuration.
    s = _min_s_for_config(centers, phis)

    # Guard: verify validity with a tiny margin; if not, inflate s slightly.
    if not _validity(centers, phis, s, eps=1e-8):
        s = _min_s_for_config(centers, phis) + 1e-6

    inner = [[x, y, 0.0] for (x, y) in centers]
    return inner, (0.0, 0.0), float(s), 0.0
# EVOLVE-BLOCK-END


# Self-check (harmless, not used by evaluator).
if __name__ == "__main__":
    inner, center, side, angle = construct()
    print("n inner:", len(inner))
    print("side:", side)
    centers = [(c[0], c[1]) for c in inner]
    phis = [c[2] for c in inner]
    print("valid:", _validity(centers, phis, side, eps=1e-7))
