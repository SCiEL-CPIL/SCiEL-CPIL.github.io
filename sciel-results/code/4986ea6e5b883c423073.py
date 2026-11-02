<<<<<<< SEARCH
    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    SCALE = 1.0 + 1e-12
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))

    pts_scaled = np.array(pts_scaled)

    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
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
    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # No scaling - use the exact positions from optimization
    pts_final = best_global_pts

    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
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

Now let me add a few more diverse candidate configurations and improve the evaluation loop:

