<<<<<<< SEARCH
        normals = np.stack([
            np.cos(theta + k * np.pi / 3),
            np.sin(theta + k * np.pi / 3)
        ] for k in range(6)).T  # shape (6, 2)
=======
        normals = np.array([
            [np.cos(theta + k * np.pi / 3), np.sin(theta + k * np.pi / 3)]
            for k in range(6)
        ])  # shape (6, 2)
>>>>>>> REPLACE
