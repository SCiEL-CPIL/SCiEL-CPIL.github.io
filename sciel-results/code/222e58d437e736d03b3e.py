            for (rw, rh, rd) in rots:
                if rh > H or rw > W or rd > D:
                    continue
                
                found_z0 = False
                for x in range(W - rw + 1):
                    if found_z0:
                        break
                    for z in range(D - rd + 1):
                        z_bottom = np.max(height_map[x:x+rw, z:z+rd])
                        if z_bottom + rh > H:
                            continue
                        if z_bottom < best_z_bottom:
                            best_z_bottom = z_bottom
                            best_pos = (x, z)
                            best_rot = (rw, rh, rd)
                            if z_bottom == 0:
                                found_z0 = True
                                break
                if found_z0:
                    break # Found optimal height 0, no need to check other rotations
