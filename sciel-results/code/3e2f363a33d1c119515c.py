    for r in range(2):
        j = r
        target_r = j * 3 + 2
        V223[target_r, 7:11] = V2[r, 7:11] # Wait, V2 is 2x4, cols 0-3.
        # Correction: V2[r, 0:4]
        V223[target_r, 7:11] = V2[r, 0:4]
