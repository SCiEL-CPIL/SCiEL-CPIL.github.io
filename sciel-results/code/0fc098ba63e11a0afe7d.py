<<<<<<< SEARCH
            nv = _vol([best[i] if i != idx else (0, x, y, z, bw, bh, bd) for i in range(n)])
            if nv > best_v:
                best[idx] = (0, x, y, z, bw, bh, bd)
                best_v = nv
            else:
                grid.remove(x, y, z, bw, bh, bd)
                grid.place(old_pos[0], old_pos[1], old_pos[2], old_rot[0], old_rot[1], old_rot[2])
=======
            old_vol = old_rot[0] * old_rot[1] * old_rot[2]
            new_vol = bw * bh * bd
            nv = best_v - old_vol + new_vol
            if nv > best_v:
                best[idx] = (0, x, y, z, bw, bh, bd)
                best_v = nv
            else:
                grid.remove(x, y, z, bw, bh, bd)
                grid.place(old_pos[0], old_pos[1], old_pos[2], old_rot[0], old_rot[1], old_rot[2])
>>>>>>> REPLACE

