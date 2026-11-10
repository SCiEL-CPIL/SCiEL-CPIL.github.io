# EVOLVE-BLOCK-START
import numpy as np


def hexagon_packing_12():
    """
    Pack 12 unit regular hexagons in the smallest enclosing regular hexagon.

    Arrangement: 1 center + 6 first-ring (dist 2) + 5 second-ring (dist 4).
    Inner hexagons oriented with angle 30° (flat side facing center).
    Outer hexagon centered at origin, angle 0°, side length s ≈ 2*5/sqrt(3).
    """
    # First ring: 6 hexagons at distance 2 from origin
    ring1_angles = np.deg2rad(np.arange(0, 360, 60))
    ring1 = np.column_stack([2.0 * np.cos(ring1_angles), 2.0 * np.sin(ring1_angles)])

    # Second ring: 5 hexagons at distance 4 (along directions 0, 60, 120, 180, 240)
    ring2_angles = np.deg2rad(np.array([0, 60, 120, 180, 240]))
    ring2 = np.column_stack([4.0 * np.cos(ring2_angles), 4.0 * np.sin(ring2_angles)])

    # Center hexagon
    center = np.array([[0.0, 0.0]])

    # Combine all 12 positions
    positions = np.vstack([center, ring1, ring2])  # shape (12, 2)

    # All inner hexagons have angle 30 degrees
    inner_hex_data = np.column_stack([positions, np.full(12, 30.0)])

    # Outer hexagon: centered at origin, angle 0 degrees
    # Outermost vertices of inner hexagons are at radius 5 (ring2 at dist 4 + 1).
    # Outer hexagon with angle 0 has apothem = s * cos(30°) = s * sqrt(3)/2.
    # Need apothem >= 5 + margin, so s >= (5 + margin) / (sqrt(3)/2)
    margin = 1e-5
    outer_hex_side_length = 2.0 * (5.0 + margin) / np.sqrt(3.0)

    outer_hex_data = np.array([0.0, 0.0, 0.0])

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
