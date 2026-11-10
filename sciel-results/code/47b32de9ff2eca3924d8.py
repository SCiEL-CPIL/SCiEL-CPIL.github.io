    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    p13 = np.array([1.0, 0.0, 0.0])
    p14 = np.array([-1.0, 0.0, 0.0])
    
    init_points = np.vstack([ico, p13, p14])
    
    quats = np.array([point_to_quat(p) for p in init_points])
