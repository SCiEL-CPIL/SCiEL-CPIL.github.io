import numpy as np


def hexagon_packing_12():
    a1 = np.array([np.sqrt(3), 0.0])
    a2 = np.array([np.sqrt(3) / 2.0, 1.5])

    def gen_config(coords):
        return np.array([i * a1 + j * a2 for i, j in coords], dtype=float)

    coord_sets = [
        [(i, j) for i in range(3) for j in range(4)],
        [(i, j) for i in range(4) for j in range(3)],
        [(i, j) for i in range(2) for j in range(6)],
        [(i, j) for i in range(6) for j in range(2)],
        [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
         (2, 0), (-2, 0), (1, 1), (-1, -1), (2, -1)],
        [(i, j) for i in range(4) for j in range(3) if not (i == 3 and j == 2)] + [(0, 3)],
        [(i, j) for i in range(3) for j in range(4) if not (i == 2 and j == 3)] + [(3, 0)],
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (1, 3),
         (2, 0), (2, 1), (2, 2), (2, 3), (3, 0)],
        [(i, j) for i in range(-1, 3) for j in range(0, 4)
         if not (i == -1 and j < 1) and not (i == 2 and j > 2)],
    ]

    configs = []
    for cs in coord_sets:
        if len(cs) == 12:
            configs.append(gen_config(cs))

    best_s = float('inf')
    best_centered = None
    best_angle = 0.0

    for config in configs:
        center = np.mean(config, axis=0)
        centered = config - center

        for angle_deg in np.arange(0, 60, 0.1):
            theta = np.radians(angle_deg)
            R = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
            rotated = centered @ R.T

            max_proj = 0.0
            for k in range(6):
                alpha = np.radians(angle_deg + 30 + k * 60)
                n = np.array([np.cos(alpha), np.sin(alpha)])
                projs = rotated @ n
                max_proj = max(max_proj, np.max(projs))

            s = 2.0 * max_proj / np.sqrt(3)

            if s < best_s:
                best_s = s
                best_centered = centered.copy()
                best_angle = angle_deg

    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = best_centered[i, 0]
        inner_hex_data[i, 1] = best_centered[i, 1]
        inner_hex_data[i, 2] = best_angle

    outer_hex_data = np.array([0.0, 0.0, best_angle])

    return inner_hex_data, outer_hex_data, best_s
