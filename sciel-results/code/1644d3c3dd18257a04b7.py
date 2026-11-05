<<<<<<< SEARCH
def _objective(params, outer_angle_deg, n):
    """
    Objective function: outer side length + overlap penalty.
    params: flat array of length 3*n: [x1, y1, a1, x2, y2, a2, ...]
    """
    inner_params = params.reshape(n, 3)
    side = _compute_side(inner_params, outer_angle_deg)
    overlap_pen = _overlap_penalty(inner_params)
    return side + overlap_pen
=======
def _objective(params, outer_angle_deg, n):
    """
    Objective function: outer side length + overlap penalty.
    params: flat array of length 3*n: [x1, y1, a1, x2, y2, a2, ...]
    """
    inner_params = params.reshape(n, 3)
    side = _compute_side(inner_params, outer_angle_deg)
    overlap_pen = _overlap_penalty(inner_params)
    return side + overlap_pen

def _objective_smooth(params, outer_angle_deg, n, beta=50.0):
    """
    Smooth surrogate objective using soft-max for the side length.
    Instead of hard max over projections, use log-sum-exp for smoothness.
    """
    inner_params = params.reshape(n, 3)
    normals = _outer_normals(outer_angle_deg)  # (6, 2)
    cos30 = math.cos(math.pi / 6)
    
    # Collect all projections
    all_projs = []
    for i in range(n):
        x = inner_params[i, 0]
        y = inner_params[i, 1]
        a = inner_params[i, 2]
        offsets = _hex_vertex_offsets(a)
        verts = offsets + np.array([x, y])
        projs = verts @ normals.T  # (6, 6)
        all_projs.append(projs.flatten())
    
    all_projs = np.concatenate(all_projs)  # (36*n,)
    # Soft-max: (1/beta) * log(sum(exp(beta * p)))
    scaled = beta * all_projs
    max_s = np.max(scaled)
    log_sum_exp = max_s + np.log(np.sum(np.exp(scaled - max_s)))
    soft_max_proj = log_sum_exp / beta
    
    side = soft_max_proj / cos30
    overlap_pen = _overlap_penalty(inner_params)
    return side + overlap_pen
>>>>>>> REPLACE

