    def overlap_penalty(params):
        verts = get_inner_verts(params)
        penalty = 0.0
        eps = 0.1
        for i in range(N):
            for j in range(i+1, N):
                ci = verts[i].mean(axis=0)
                cj = verts[j].mean(axis=0)
                d_c = np.linalg.norm(ci - cj)
                if d_c > 2.0:
                    continue
                d = poly_dist(verts[i], verts[j])
                if d < eps:
                    penalty += (eps - d)**2
        return penalty
