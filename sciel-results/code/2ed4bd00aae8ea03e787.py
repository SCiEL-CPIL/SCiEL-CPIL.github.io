<<<<<<< SEARCH
    # Create the final inner hexagons
    if best_positions is None:
        # Fallback to the original configuration
        positions = []
        for _, x, y in candidates:
            positions.append((x, y))
            if len(positions) == N:
                break
        best_positions = positions
        inner_hexes = [[pos[0], pos[1], 30.0] for pos in positions]
        cx, cy, best_side, best_angle = _min_enclosing_hex_side(inner_hexes)
        best_center = (cx, cy)
    else:
        inner_hexes = [[pos[0], pos[1], 30.0] for pos in best_positions]
    
    # Final verification and return
    return inner_hexes, (best_center[0], best_center[1]), best_side, best_angle
=======
    # Create the final inner hexagons
    if best_positions is None:
        # Fallback to the original configuration
        positions = []
        for _, x, y in candidates:
            positions.append((x, y))
            if len(positions) == N:
                break
        best_positions = positions
        inner_hexes = [[pos[0], pos[1], 30.0] for pos in positions]
        cx, cy, best_side, best_angle = _min_enclosing_hex_side(inner_hexes)
        best_center = (cx, cy)
    else:
        # Use the best inner angles found
        if 'best_inner_angles' in locals() and best_inner_angles is not None:
            inner_hexes = [[pos[0], pos[1], best_inner_angles[i]] for i, pos in enumerate(best_positions)]
        else:
            inner_hexes = [[pos[0], pos[1], 30.0] for pos in best_positions]
    
    # Final verification and return
    return inner_hexes, (best_center[0], best_center[1]), best_side, best_angle
>>>>>>> REPLACE
