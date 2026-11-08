<<<<<<< SEARCH
        # The required apothem is the max absolute projection over all points and all normals
        apothem = np.max(np.abs(projs))
        
        # Side length S = 2 * apothem / sqrt(3)
        side = 2 * apothem / np.sqrt(3)
        return side
=======
        # The required apothem is the max absolute projection of centers + apothem of inner hexagon
        # Inner hexagon side 1 has apothem sqrt(3)/2.
        # Since inner hexagons are aligned with outer (angle 0), the max projection of inner hexagon on normal n is its apothem.
        inner_apothem = np.sqrt(3) / 2.0
        apothem = np.max(np.abs(projs)) + inner_apothem
        
        # Side length S = 2 * apothem / sqrt(3)
        side = 2 * apothem / np.sqrt(3)
        return side
>>>>>>> REPLACE
