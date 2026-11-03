<<<<<<< SEARCH
def _build_layouts():
    """Generate multiple candidate layouts for the 11 inner hexagons.
    Returns list of (positions_list, inner_angle)."""
    layouts = []
    scale = 1.000001
    u = (_SQRT3 * scale, 0.0)
    v = (_SQRT3 / 2 * scale, 1.5 * scale)

    # Layout 0: hexagonal flower + 4 from second ring (compact)
    target = [(0, 0)]
    ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    target.extend(ring1)
    ring2 = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0),
             (-1, -1), (0, -2), (1, -2), (2, -1)]
    ring2_sorted = sorted(ring2, key=lambda p: p[0] ** 2 + p[1] ** 2)
    target.extend(ring2_sorted[:4])

    for inner_angle in [0.0, 30.0]:
        positions = []
        for ti, tj in target:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions.append((x, y))
        layouts.append((positions[:N], inner_angle))

    # Layout 1: row-based 3-4-4
    positions = []
    y1 = -1.5 * scale
    for k in range(-1, 2):
        x = k * _SQRT3 * scale + _SQRT3 / 2 * scale
        positions.append((x, y1))
    y2 = 0.0
    for k in range(-1, 3):
        x = k * _SQRT3 * scale
        positions.append((x, y2))
    y3 = 1.5 * scale
    for k in range(-1, 3):
        x = k * _SQRT3 * scale + _SQRT3 / 2 * scale
        positions.append((x, y3))
    for inner_angle in [0.0, 30.0]:
        layouts.append((positions[:N], inner_angle))

    # Layout 2: 4-3-4 shifted
    positions = []
    y1 = -1.5 * scale
    for k in range(-2, 2):
        x = k * _SQRT3 * scale
        positions.append((x, y1))
    y2 = 0.0
    for k in range(-1, 2):
        x = k * _SQRT3 * scale + _SQRT3 / 2 * scale
        positions.append((x, y2))
    y3 = 1.5 * scale
    for k in range(-2, 2):
        x = k * _SQRT3 * scale
        positions.append((x, y3))
    for inner_angle in [0.0, 30.0]:
        layouts.append((positions[:N], inner_angle))

    # Layout 3: greedy nearest-neighbor from origin on triangular lattice
    coords = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            coords.append((x, y))
    coords_sorted = sorted(coords, key=lambda t: t[0] ** 2 + t[1] ** 2)
    positions = coords_sorted[:N]
    for inner_angle in [0.0, 30.0]:
        layouts.append((positions, inner_angle))

    # Layout 4: slightly different second-ring selection (pick 4 that minimize bounding)
    # Try different 4-of-10 combinations from ring2
    best_combo_side = float('inf')
    best_combo = None
    for combo in itertools.combinations(ring2, 4):
        target_test = [(0, 0)] + ring1 + list(combo)
        positions_test = []
        for ti, tj in target_test:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions_test.append((x, y))
        # Quick bounding check: use axis-aligned bounding box as proxy
        xs = [p[0] for p in positions_test]
        ys = [p[1] for p in positions_test]
        bbox = (max(xs) - min(xs)) + (max(ys) - min(ys))
        if bbox < best_combo_side:
            best_combo_side = bbox
            best_combo = combo

    if best_combo:
        target = [(0, 0)] + ring1 + list(best_combo)
        positions = []
        for ti, tj in target:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions.append((x, y))
        for inner_angle in [0.0, 30.0]:
            layouts.append((positions[:N], inner_angle))

    return layouts
=======
def _build_layouts():
    """Generate multiple candidate layouts for the 11 inner hexagons.
    Returns list of (positions_list, inner_angle)."""
    layouts = []
    scale = 1.000001
    u = (_SQRT3 * scale, 0.0)
    v = (_SQRT3 / 2 * scale, 1.5 * scale)

    # Layout 0: hexagonal flower + 4 from second ring (compact)
    target = [(0, 0)]
    ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    target.extend(ring1)
    ring2 = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0),
             (-1, -1), (0, -2), (1, -2), (2, -1)]
    ring2_sorted = sorted(ring2, key=lambda p: p[0] ** 2 + p[1] ** 2)
    target.extend(ring2_sorted[:4])

    for inner_angle in [0.0, 30.0]:
        positions = []
        for ti, tj in target:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions.append((x, y))
        layouts.append((positions[:N], inner_angle))

    # Layout 1: row-based 3-4-4
    positions = []
    y1 = -1.5 * scale
    for k in range(-1, 2):
        x = k * _SQRT3 * scale + _SQRT3 / 2 * scale
        positions.append((x, y1))
    y2 = 0.0
    for k in range(-1, 3):
        x = k * _SQRT3 * scale
        positions.append((x, y2))
    y3 = 1.5 * scale
    for k in range(-1, 3):
        x = k * _SQRT3 * scale + _SQRT3 / 2 * scale
        positions.append((x, y3))
    for inner_angle in [0.0, 30.0]:
        layouts.append((positions[:N], inner_angle))

    # Layout 2: 4-3-4 shifted
    positions = []
    y1 = -1.5 * scale
    for k in range(-2, 2):
        x = k * _SQRT3 * scale
        positions.append((x, y1))
    y2 = 0.0
    for k in range(-1, 2):
        x = k * _SQRT3 * scale + _SQRT3 / 2 * scale
        positions.append((x, y2))
    y3 = 1.5 * scale
    for k in range(-2, 2):
        x = k * _SQRT3 * scale
        positions.append((x, y3))
    for inner_angle in [0.0, 30.0]:
        layouts.append((positions[:N], inner_angle))

    # Layout 3: greedy nearest-neighbor from origin on triangular lattice
    coords = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            coords.append((x, y))
    coords_sorted = sorted(coords, key=lambda t: t[0] ** 2 + t[1] ** 2)
    positions = coords_sorted[:N]
    for inner_angle in [0.0, 30.0]:
        layouts.append((positions, inner_angle))

    # Layout 4: try all 4-of-10 combinations from ring2, keep top few by bbox
    combos = list(itertools.combinations(ring2, 4))
    combo_scores = []
    for combo in combos:
        target_test = [(0, 0)] + ring1 + list(combo)
        positions_test = []
        for ti, tj in target_test:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions_test.append((x, y))
        xs = [p[0] for p in positions_test]
        ys = [p[1] for p in positions_test]
        bbox = (max(xs) - min(xs)) + (max(ys) - min(ys))
        combo_scores.append((bbox, combo))
    combo_scores.sort()
    for _, combo in combo_scores[:5]:
        target = [(0, 0)] + ring1 + list(combo)
        positions = []
        for ti, tj in target:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions.append((x, y))
        for inner_angle in [0.0, 30.0]:
            layouts.append((positions[:N], inner_angle))

    # Layout 5: compact cluster - use 7 central + 4 adjacent but with slight compression
    # Try a few different inner angles for the central flower
    for inner_angle in [0.0, 15.0, 30.0, 45.0]:
        target = [(0, 0)] + ring1 + ring2_sorted[:4]
        positions = []
        for ti, tj in target:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions.append((x, y))
        layouts.append((positions[:N], inner_angle))

    return layouts
>>>>>>> REPLACE

