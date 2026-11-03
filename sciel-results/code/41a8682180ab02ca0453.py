<<<<<<< SEARCH
    cx, cy = center
    angle_deg = math.degrees(beta)

    # Final scaling to ensure all hexagons are inside
    SCALE = 1.0 + 1e-14
    pts_scaled = opt_pts.copy()
    for i in range(len(pts_scaled)):
        pts_scaled[i, 0] = cx + (pts_scaled[i, 0] - cx) * SCALE
        pts_scaled[i, 1] = cy + (pts_scaled[i, 1] - cy) * SCALE

    max_proj = 0
    for k in range(6):
        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))

    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM

    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)

    return inner, center, side, rotation
=======
    cx, cy = center
    angle_deg = math.degrees(beta)

    # Use the optimized points directly (no artificial scaling that worsens S)
    pts_final = opt_pts

    max_proj = 0
    for k in range(6):
        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_final[:, 0] - cx) * nx + (pts_final[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))

    D_final = max_proj
    S_final = (D_final + APOTHEM) / APOTHEM

    inner = []
    for x, y in pts_final:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(angle_deg)

    return inner, center, side, rotation
>>>>>>> REPLACE
