<<<<<<< SEARCH
        # Combine: (11, 1, 2) + (1, 6, 2) -> (11, 6, 2)
        all_verts = rotated_coords[:, np.newaxis, :] + rotated_hex_verts[np.newaxis, :, :]
        return all_verts.flatten()
=======
        # Combine: (11, 1, 2) + (1, 6, 2) -> (11, 6, 2)
        all_verts = rotated_coords[:, np.newaxis, :] + rotated_hex_verts[np.newaxis, :, :]
        return all_verts.reshape(-1, 2)
>>>>>>> REPLACE
