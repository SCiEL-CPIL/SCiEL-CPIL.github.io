<<<<<<< SEARCH
def _min_side_for_hexagons(positions, angles):
    """Minimal outer side length (centered at origin, any rotation) that contains
    all hexagons. We search over container rotation in steps and compute the
    required S for that rotation, take the minimum."""
    best = float('inf')
    # container rotation search
    for Srot_deg in np.linspace(0.0, 60.0, 13):
        Srot = math.radians(Srot_deg)
        req = 0.0
        for cx, cy, ang in zip(positions, angles):
            corners = _hex_corners(cx, cy, ang)
            for k in range(6):
                th = Srot + k * math.pi / 3
                nx = math.cos(th)
                ny = math.sin(th)
                support = np.max(corners[:, 0] * nx + corners[:, 1] * ny)
                s_req = 2.0 * support / math.sqrt(3.0)
                if s_req > req:
                    req = s_req
        if req < best:
            best = req
    return best
=======
def _min_side_for_hexagons(positions, angles):
    """Minimal outer side length (centered at origin, any rotation) that contains
    all hexagons. We search over container rotation in steps and compute the
    required S for that rotation, take the minimum."""
    best = float('inf')
    # container rotation search
    for Srot_deg in np.linspace(0.0, 60.0, 13):
        Srot = math.radians(Srot_deg)
        req = 0.0
        for i in range(len(positions)):
            cx, cy = positions[i]
            ang = angles[i]
            corners = _hex_corners(cx, cy, ang)
            for k in range(6):
                th = Srot + k * math.pi / 3
                nx = math.cos(th)
                ny = math.sin(th)
                support = np.max(corners[:, 0] * nx + corners[:, 1] * ny)
                s_req = 2.0 * support / math.sqrt(3.0)
                if s_req > req:
                    req = s_req
        if req < best:
            best = req
    return best
>>>>>>> REPLACE
