def _init_grid(m_x, m_y, m_z):
    tx = (np.arange(m_x) + 0.5) / m_x - 0.5
    ty = (np.arange(m_y) + 0.5) / m_y - 0.5
    tz = (np.arange(m_z) + 0.5) / m_z - 0.5
    grid = np.array([(x, y, z) for z in tz for y in ty for x in tx])
    return grid[:N]

# In construct:
inits = []
inits.append(_init_grid(4, 4, 4))
inits.append(_init_grid(3, 4, 5))
inits.append(np.random.uniform(-0.4, 0.4, (N, 3)))

best_init = None
best_init_radius = 0
for init in inits:
    r, _, _, _, _ = _compute_radius_and_bottleneck(init)
    if r > best_init_radius:
        best_init_radius = r
        best_init = init

centers = best_init.copy()
best_centers = centers.copy()
best_radius = best_init_radius
