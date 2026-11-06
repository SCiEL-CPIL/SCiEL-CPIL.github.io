    def delta_move(pts, i, nx, ny, pair):
        xi, yi = pts[i]
        new_pt = (nx, ny)
        others = [j for j in range(16) if j != i]
        old_keys = []
        new_keys = []
        for j in others:
            old_keys.append(pair[(i, j)] if i < j else pair[(j, i)])
            new_keys.append(_line_key_coords(new_pt, pts[j]))
        
        oldcnt = defaultdict(int)
        for k in old_keys:
            oldcnt[k] += 1
        newcnt = defaultdict(int)
        for k in new_keys:
            newcnt[k] += 1
        gain = 0
        for k in set(oldcnt) | set(newcnt):
            o = oldcnt.get(k, 0)
            nn = newcnt.get(k, 0)
            gain += (1 if nn == 4 else 0) - (1 if o == 4 else 0)
        return gain
