<<<<<<< SEARCH
        # inside test: cross(edge, p - edge_start) >= 0
        def inside(p):
            return np.cross(edge, p - edge_start) >= -1e-12
        input_pts = output
        output = []
        n_in = len(input_pts)
        for j in range(n_in):
            cur = input_pts[j]
            nxt = input_pts[(j + 1) % n_in]
            cur_in = inside(cur)
            nxt_in = inside(nxt)
            if cur_in:
                output.append(cur)
                if not nxt_in:
                    # exit: intersect
                    d = nxt - cur
                    denom = np.cross(edge, d)
                    if abs(denom) > 1e-15:
                        t = np.cross(edge_start - cur, d) / denom
                        output.append(cur + t * d)
            else:
                if nxt_in:
                    # enter: intersect
                    d = nxt - cur
                    denom = np.cross(edge, d)
                    if abs(denom) > 1e-15:
                        t = np.cross(edge_start - cur, d) / denom
                        output.append(cur + t * d)
=======
        # inside test: cross(edge, p - edge_start) >= 0
        # Manual 2D cross product: e0*d1 - e1*d0
        def cross2d(a, b):
            return a[0]*b[1] - a[1]*b[0]
            
        def inside(p):
            return cross2d(edge, p - edge_start) >= -1e-12
        input_pts = output
        output = []
        n_in = len(input_pts)
        for j in range(n_in):
            cur = input_pts[j]
            nxt = input_pts[(j + 1) % n_in]
            cur_in = inside(cur)
            nxt_in = inside(nxt)
            if cur_in:
                output.append(cur)
                if not nxt_in:
                    # exit: intersect
                    d = nxt - cur
                    denom = cross2d(edge, d)
                    if abs(denom) > 1e-15:
                        t = cross2d(edge_start - cur, d) / denom
                        output.append(cur + t * d)
            else:
                if nxt_in:
                    # enter: intersect
                    d = nxt - cur
                    denom = cross2d(edge, d)
                    if abs(denom) > 1e-15:
                        t = cross2d(edge_start - cur, d) / denom
                        output.append(cur + t * d)
>>>>>>> REPLACE

