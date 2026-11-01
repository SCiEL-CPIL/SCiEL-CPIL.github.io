<<<<<<< SEARCH
def _high_precision_polish(z, time_budget):
    """
    Final high-precision polish: iterative expand + reposition with very small steps.
    Uses binary search for maximum valid radius at each position.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    max_rounds = 50
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break

        improved = False

        for i in range(n):
            # Compute max possible radius for circle i
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)

            if r_max > r[i] + 1e-8:
                # Binary search for max valid radius
                lo = r[i]
                hi = r_max
                for _ in range(15):
                    mid = (lo + hi) / 2.0
                    r_trial = r.copy()
                    r_trial[i] = mid
                    if _is_valid(x, y, r_trial, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-9:
                    r[i] = lo
                    improved = True

            # Try small repositions to create more space
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1),
                          (0.707, 0.707), (-0.707, 0.707),
                          (0.707, -0.707), (-0.707, -0.707)]
            for dx_dir, dy_dir in directions:
                for step in [0.005, 0.002, 0.001]:
                    new_x = x[i] + dx_dir * step
                    new_y = y[i] + dy_dir * step
                    if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                        x[i] = new_x
                        y[i] = new_y
                        improved = True
                        break
                if improved:
                    break

        if not improved:
            break

    # Final uniform expansion
    for _ in range(50):
        if time.time() > time_budget:
            break
        r_trial = r * 1.0005
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None
=======
def _high_precision_polish(z, time_budget):
    """
    Contact-graph-based local optimizer: identifies blocking pairs and performs
    targeted repositioning to maximize radius sum.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    def compute_contact_graph(x, y, r, w, h, tol=1e-4):
        """Find pairs of circles that are nearly touching."""
        contacts = []
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if abs(d - (r[i] + r[j])) < tol:
                    contacts.append((i, j))
        # Also check wall contacts
        for i in range(n):
            if abs(x[i] - r[i]) < tol:
                contacts.append((i, 'left'))
            if abs(x[i] + r[i] - w) < tol:
                contacts.append((i, 'right'))
            if abs(y[i] - r[i]) < tol:
                contacts.append((i, 'bottom'))
            if abs(y[i] + r[i] - h) < tol:
                contacts.append((i, 'top'))
        return contacts

    max_rounds = 30
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break

        improved = False
        contacts = compute_contact_graph(x, y, r, w, h)

        # For each contact pair, try to separate them slightly
        for contact in contacts:
            if len(contact) == 2:
                i, j = contact
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < 1e-6:
                    continue
                ux = dx / d
                uy = dy / d
                
                # Try to move i away from j
                for step in [0.005, 0.003, 0.002, 0.001]:
                    new_xi = x[i] + ux * step
                    new_yi = y[i] + uy * step
                    if _is_valid_move(x, y, r, w, h, i, new_xi, new_yi):
                        x[i] = new_xi
                        y[i] = new_yi
                        improved = True
                        break
            else:
                i, wall = contact
                # Move away from wall
                if wall == 'left':
                    dx_dir, dy_dir = 1, 0
                elif wall == 'right':
                    dx_dir, dy_dir = -1, 0
                elif wall == 'bottom':
                    dx_dir, dy_dir = 0, 1
                else:
                    dx_dir, dy_dir = 0, -1
                
                for step in [0.005, 0.003, 0.002, 0.001]:
                    new_xi = x[i] + dx_dir * step
                    new_yi = y[i] + dy_dir * step
                    if _is_valid_move(x, y, r, w, h, i, new_xi, new_yi):
                        x[i] = new_xi
                        y[i] = new_yi
                        improved = True
                        break

        # Try to grow all radii uniformly
        if improved or rnd % 5 == 0:
            for _ in range(50):
                if time.time() > time_budget:
                    break
                r_trial = r * 1.0003
                if _is_valid(x, y, r_trial, w, h):
                    r = r_trial
                else:
                    break

        if not improved:
            # Try one more time with smaller steps
            contacts = compute_contact_graph(x, y, r, w, h, tol=5e-5)
            for contact in contacts:
                if len(contact) == 2:
                    i, j = contact
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    if d < 1e-6:
                        continue
                    ux = dx / d
                    uy = dy / d
                    for step in [0.001, 0.0005, 0.0002]:
                        new_xi = x[i] + ux * step
                        new_yi = y[i] + uy * step
                        if _is_valid_move(x, y, r, w, h, i, new_xi, new_yi):
                            x[i] = new_xi
                            y[i] = new_yi
                            improved = True
                            break
            if not improved:
                break

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None
>>>>>>> REPLACE
