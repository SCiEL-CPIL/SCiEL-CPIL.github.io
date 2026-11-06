<<<<<<< SEARCH
    # If no valid cell was found, use a default
    if best_cell is None:
        best_cell = make_cell_from_pattern(four_rod_pattern(0.10, 0.25))
    
    return best_cell
=======
    # If no valid cell was found, use a default
    if best_cell is None:
        best_cell = make_cell_from_pattern(four_rod_pattern(0.10, 0.25))
    
    # Final safety check: enforce symmetry on the best cell found
    # This guarantees the output satisfies the C4v symmetry requirement
    best_cell = enforce_symmetry(best_cell)
    
    return best_cell
>>>>>>> REPLACE
