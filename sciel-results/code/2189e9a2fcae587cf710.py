    def _constraint_funcs(flat_pts):
        pts = flat_pts.reshape(N, 2)
        vals = []
        for i in range(N):
            x, y = pts[i]
            vals.append(y)
            vals.append(np.sqrt(3) * x - y)
            vals.append(np.sqrt(3) * (1 - x) - y)
        return vals

    constraints = [
        {'type': 'ineq', 'fun': lambda x: _constraint_funcs(x)[3*i]} for i in range(N)
    ] + [
        {'type': 'ineq', 'fun': lambda x: _constraint_funcs(x)[3*i+1]} for i in range(N)
    ] + [
        {'type': 'ineq', 'fun': lambda x: _constraint_funcs(x)[3*i+2]} for i in range(N)
    ]
