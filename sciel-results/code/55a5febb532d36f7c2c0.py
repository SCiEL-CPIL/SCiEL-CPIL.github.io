else: # Blue -> Red
    # Quads
    # Becomes Red K4 if others are Red
    # all_red = ~np.any(others_q == 1, axis=1)
    # delta += np.sum(all_red)
    # Correct.
    
    # Quints
    # Was Blue K5 if others were Blue
    # all_blue = ~np.any(others_k == 0, axis=1)
    # delta -= np.sum(all_blue)
    # Correct.
