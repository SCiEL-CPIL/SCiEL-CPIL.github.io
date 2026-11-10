import numpy as np
import time

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # 580s safety margin
    
    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3
    
    qr = set((u * u) % p for u in range(p))
    
    def point_idx(xa, xb, ya, yb):
        return xa * p**3 + xb * p**2 + ya * p + yb
    
    def idx_to_point(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // p**2) % p
        xa = (idx // p**3) % p
        return xa, xb, ya, yb
    
    all_directions = []
    for ma in range(p):
        for mb in range(p):
            all_directions.append((1, ma, mb))
    all_directions.append((0, 0, 1))
    
    lines = []
    for d_idx, (dxa, dya, dyb) in enumerate(all_directions):
        dir_lines = []
        visited = set()
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        pt = point_idx(xa, xb, ya, yb)
                        if pt in visited:
                            continue
                        line_points = []
                        for t_a in range(p):
                            for t_b in range(p):
                                if dxa == 1:
                                    nxa = (xa + t_a) % p
                                    nxb = (xb + t_b) % p
                                    nya = (ya + t_a * dya + t_b * dyb * k) % p
                                    nyb = (yb + t_a * dyb + t_b * dya) % p
                                else:
                                    nxa = xa
                                    nxb = xb
                                    nya = (ya + t_a) % p
                                    nyb = (yb + t_b) % p
                                npt = point_idx(nxa, nxb, nya, nyb)
                                line_points.append(npt)
                                visited.add(npt)
                        dir_lines.append(line_points)
        lines.append(dir_lines)
    
    point_lines = [[] for _ in range(q * q)]
    for d_idx in range(len(all_directions)):
        for l_idx in range(len(lines[d_idx])):
            for pt in lines[d_idx][l_idx]:
                point_lines[pt].append((d_idx, l_idx))
    
    in_N = [True] * (q * q)
    
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    pt = point_idx(xa, xb, ya, yb)
                    in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                    ya_is_qr = ya in qr
                    if in_unital and ya_is_qr:
                        in_N[pt] = False
    
    valid_witness = [set() for _ in range(q * q)]
    for d_idx in range(len(all_directions)):
        for l_idx in range(len(lines[d_idx])):
            line_pts = lines[d_idx][l_idx]
            all_in_N = all(in_N[pt] for pt in line_pts)
            if all_in_N:
                for pt in line_pts:
                    valid_witness[pt].add(d_idx)
    
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        for x in range(q * q):
            if not in_N[x]:
                continue
            if not valid_witness[x]:
                continue
            affected = []
            for (d_idx, l_idx) in point_lines[x]:
                line_pts = lines[d_idx][l_idx]
                for y in line_pts:
                    if y != x:
                        affected.append((y, d_idx))
            can_remove = True
            for (y, d_idx) in affected:
                if d_idx in valid_witness[y]:
                    if len(valid_witness[y]) == 1:
                        can_remove = False
                        break
            if can_remove:
                in_N[x] = False
                for (y, d_idx) in affected:
                    if d_idx in valid_witness[y]:
                        valid_witness[y].discard(d_idx)
                changed = True
    
    points = []
    for pt in range(q * q):
        if in_N[pt]:
            xa, xb, ya, yb = idx_to_point(pt)
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)
