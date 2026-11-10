# EVOLVE-BLOCK-START
import numpy as np


def moving_sofa_2d() -> np.ndarray:
    """
    2D moving-sofa problem: return a path of corridor placements (x, y, angle_degrees)
    where (x, y) is the inner corner of the width-1 L-shaped corridor and the angle is
    its rotation in degrees. The evaluator prepends (0,0,0) and appends (-2,0,90), then
    densifies. The sofa is the intersection of all corridor placements; we maximize its
    area.

    New approach (analytic / Hammersley-style):
    A placement (c, a) blocks a point p iff R(-a)(p-c) lies in the base L. Its two
    arms give two half-plane constraints on the sofa:
        n_a . p >= n_a . c   and   t_a . p >= t_a . c
    where n_a = (-sin a, cos a) is the inner-wall normal and t_a = (cos a, sin a) is
    the other wall's normal. Hence the sofa is the intersection of half-planes
    n_a . p >= f_a, t_a . p >= g_a with f_a = max_{path} n_a.c(a), g_a = max t_a.c(a).
    To enlarge the sofa we minimise f_a and g_a for every a. The minimum is attained
    when the inner corner rides the two OUTER walls of the corridor, i.e.
        c(a) = (1 + cos a, 1 + sin a)
    (the corner's arms lie along the outer walls). This is the classic Hammersley
    "asymmetric" sofa construction, known to beat the symmetric path. The base L's
    arms are 3 units long, so the corner stays inside the arms for all a in [0, 90].
    """
    # Sample rotations strictly between the evaluator's endpoints (0 and 90 deg).
    # Using the analytic corner c(a) = (1+cos a, 1+sin a) at each sampled angle.
    angles_deg = np.arange(1.0, 90.0, 1.0)  # 1, 2, ..., 89  (89 points)
    a = np.deg2rad(angles_deg)
    cos_a = np.cos(a)
    sin_a = np.sin(a)
    x = 1.0 + cos_a
    y = 1.0 + sin_a

    path = np.column_stack([x, y, angles_deg]).astype(np.float64)
    return path


# EVOLVE-BLOCK-END
