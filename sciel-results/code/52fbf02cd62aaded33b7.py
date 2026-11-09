def _honeycomb_seed():
    d = np.sqrt(3.0)
    pts = [(0, 0)]
    for i in range(6):
        a = np.deg2rad(60.0 * i)
        pts.append((d * np.cos(a), d * np.sin(a)))
    for i in range(6):
        a = np.deg2rad(60.0 * i + 30.0)
        pts.append((2.0 * d * np.cos(a), 2.0 * d * np.sin(a)))
    # Take only the first 11 points
    pts = pts[:11]
    x = np.zeros((11, 3))
    for k, (px, py) in enumerate(pts):
        x[k, 0] = px
        x[k, 1] = py
    return x
