        if best_bin >= 0:
            x, y, z = best_pos
            bins[best_bin] = (bins[best_bin][0] + [(x, y, z, w, h, d)],
                              bins[best_bin][1] + vol)
        else:
            # Open new bin
            bins.append([(0, 0, 0, w, h, d)], vol)  # <--- ERROR HERE
