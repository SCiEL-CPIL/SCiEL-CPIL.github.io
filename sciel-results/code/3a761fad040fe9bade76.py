def equation(b, s, temp, pH, params):
    bn = (b - 1.0) / 4.0
    sn = (s - 1.0) / 4.0
    tn = (temp - 1.0) / 39.0
    pn = (pH - 2.0) / 10.0
    
    centers = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.5, 0.5, 0.5, 0.5],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.5, 0.0, 0.5, 1.0],
        [0.5, 1.0, 0.5, 0.0],
        [0.0, 0.5, 1.0, 0.5],
    ])
    
    diff = np.stack([bn, sn, tn, pn], axis=-1)[:, :, None] - centers[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    width_sq = 0.49  # 0.7^2
    rbf = np.exp(-dist_sq / (2.0 * width_sq))
    
    return rbf @ params
