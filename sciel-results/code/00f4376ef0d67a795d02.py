        if verify_no_collision(new_positions, best_angles):
            new_side = min_containing_hexagon_side(new_positions, best_angles)
            if new_side < best_side - 1e-9:
                best_positions = new_positions
   