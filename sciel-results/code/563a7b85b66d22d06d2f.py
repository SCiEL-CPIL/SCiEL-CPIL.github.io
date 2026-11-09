<<<<<<< SEARCH
    x0 = np.array(init_positions + init_angles).flatten()
    x0 = np.concatenate([x0, [0.0, 0.0, 5.0, 0.0]])
=======
    # Construct x0 correctly: [x1, y1, a1, ..., x12, y12, a12, cx, cy, S, A]
    x0_flat = []
    for i in range(12):
        x0_flat.extend([init_positions[i][0], init_positions[i][1], init_angles[i]])
    x0_flat.extend([0.0, 0.0, 5.0, 0.0])
    x0 = np.array(x0_flat)
>>>>>>> REPLACE

