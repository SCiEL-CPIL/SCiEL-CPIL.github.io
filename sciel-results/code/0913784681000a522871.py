<<<<<<< SEARCH
def _coordinate_descent(objective, x0, lo, hi, deadline, max_rounds=100):
    """
    Cyclic coordinate descent with line search.
    """
    x = np.array(x0, dtype=float)
    f_current = objective(x)
    n = len(x)
    
    for rnd in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            x_old = x[i]
            best_x = x_old
            best_f = f_current
            # Try several steps
            for step_frac in [0.5, -0.5, 0.25, -0.25, 0.1, -0.1, 0.75, -0.75]:
                x_new = x.copy()
                x_new[i] = np.clip(x_old + step_frac * (hi[i] - lo[i]) * 0.2, lo[i], hi[i])
                f_new = objective(x_new)
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new[i]
            if best_x != x_old:
                x[i] = best_x
                f_current = best_f
                improved = True
        if not improved:
            break
    
    return x, f_current
=======
def _coordinate_descent(objective, x0, lo, hi, deadline, max_rounds=100):
    """
    Cyclic coordinate descent with line search.
    """
    x = np.array(x0, dtype=float)
    f_current = objective(x)
    n = len(x)
    
    for rnd in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            x_old = x[i]
            best_x = x_old
            best_f = f_current
            # Try several steps
            for step_frac in [0.5, -0.5, 0.25, -0.25, 0.1, -0.1, 0.75, -0.75]:
                x_new = x.copy()
                x_new[i] = np.clip(x_old + step_frac * (hi[i] - lo[i]) * 0.2, lo[i], hi[i])
                f_new = objective(x_new)
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new[i]
            if best_x != x_old:
                x[i] = best_x
                f_current = best_f
                improved = True
        if not improved:
            break
    
    return x, f_current

def _sigmoid_profile(z, params, n_ambient, n_substrate):
    """
    Compute continuous refractive index profile from sigmoid parameters.
    params: array of shape (2*K,) where for each k: [center_k, width_k]
    Returns n(z) values at positions z.
    """
    K = len(params) // 2
    n = np.full_like(z, n_ambient, dtype=float)
    for k in range(K):
        center = params[2*k]
        width = params[2*k+1]
        if width < 1e-6:
            width = 1e-6
        # Each sigmoid contributes a step from n_ambient toward n_substrate
        # Weight is 1/K so that sum approaches n_substrate
        step = (n_substrate - n_ambient) / K
        sig = 1.0 / (1.0 + np.exp(-(z - center) / width))
        n += step * sig
    return n

def _discretize_profile(n_profile, z, n_L, n_H, max_layers, max_total, max_layer):
    """
    Discretize a continuous index profile into L/H layers.
    Returns (materials, thicknesses) or None if infeasible.
    """
    n_layers = len(z) - 1
    if n_layers == 0:
        return None
    
    # Compute average index in each layer
    layer_indices = []
    for i in range(n_layers):
        avg_n = (n_profile[i] + n_profile[i+1]) / 2.0
        layer_indices.append(avg_n)
    
    # Decide L or H based on which is closer
    materials = []
    for avg_n in layer_indices:
        if abs(avg_n - n_L) < abs(avg_n - n_H):
            materials.append('L')
        else:
            materials.append('H')
    
    # Remove consecutive duplicates (merge them)
    merged_mats = []
    merged_ths = []
    i = 0
    dz = z[1] - z[0]
    while i < len(materials):
        mat = materials[i]
        count = 1
        while i + count < len(materials) and materials[i+count] == mat:
            count += 1
        merged_mats.append(mat)
        merged_ths.append(count * dz)
        i += count
    
    # If too many layers, merge adjacent pairs
    while len(merged_mats) > max_layers:
        # Merge the two adjacent layers with most similar indices
        best_merge_idx = 0
        best_sim = -1
        for j in range(len(merged_mats) - 1):
            n1 = n_L if merged_mats[j] == 'L' else n_H
            n2 = n_L if merged_mats[j+1] == 'L' else n_H
            sim = 1.0 - abs(n1 - n2) / (n_H - n_L)
            if sim > best_sim:
                best_sim = sim
                best_merge_idx = j
        # Merge layers at best_merge_idx and best_merge_idx+1
        new_th = merged_ths[best_merge_idx] + merged_ths[best_merge_idx+1]
        # Determine material by weighted average
        n1 = n_L if merged_mats[best_merge_idx] == 'L' else n_H
        n2 = n_L if merged_mats[best_merge_idx+1] == 'L' else n_H
        w1 = merged_ths[best_merge_idx]
        w2 = merged_ths[best_merge_idx+1]
        avg_n = (n1*w1 + n2*w2) / (w1+w2)
        new_mat = 'L' if abs(avg_n - n_L) < abs(avg_n - n_H) else 'H'
        merged_mats[best_merge_idx] = new_mat
        merged_ths[best_merge_idx] = new_th
        del merged_mats[best_merge_idx+1]
        del merged_ths[best_merge_idx+1]
    
    # Scale to fit total thickness constraint
    total_th = sum(merged_ths)
    if total_th > max_total:
        scale = max_total / total_th
        merged_ths = [t * scale for t in merged_ths]
    
    # Check individual layer constraints
    for t in merged_ths:
        if t > max_layer + 1e-9:
            return None
    
    return merged_mats, merged_ths
>>>>>>> REPLACE

