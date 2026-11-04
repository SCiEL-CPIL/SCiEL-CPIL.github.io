    centers = np.array([
        [0.0, 0.0],
    ])
    
    # Ring 1
    for k in range(6):
        a = k * np.pi / 3.0
        centers = np.vstack([centers, [S3 * np.cos(a), S3 * np.sin(a)]])
