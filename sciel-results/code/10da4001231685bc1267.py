import numpy as np
import time
import scipy.optimize
import scipy.spatial.distance


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) points.
    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = 600 - 10  # 590 seconds to be safe

    def get_score(points):
        """Calculate the actual score R = (d_min/d_max)^2"""
        pts = np.asarray(points).reshape(n, d)
        dists = scipy.spatial.distance.pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = float(np.min(dists))
        d_max = float(np.max(dists))
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def normalize(pts):
        dists = scipy.spatial.distance.pdist(pts)
        d_max = float(np.max(dists))
        if d_max > 0:
            return pts / d_max
        return pts

    def planar_objective(x, beta=50.0):
        """
        Smoothed planar objective: minimize -(soft_min / soft_max).
        """
        pts = x.reshape(n, d)
        dists = scipy.spatial.distance.pdist(pts)
        if len(dists) == 0:
            return 0.0
        max_d = float(np.max(dists))
        if max_d < 1e-12:
            return 0.0

        # soft min
        u = beta * (max_d - dists)
        # stabilize
        u = np.clip(u, -700.0, 700.0)
        term_min = np.exp(u)
        s_min = np.sum(term_min)
        if s_min <= 0:
            return 0.0
        d_min_soft = max_d - (1.0 / beta) * np.log(s_min)

        # soft max
        v = beta * (dists - max_d)
        v = np.clip(v, -700.0, 700.0)
        term_max = np.exp(v)
        s_max = np.sum(term_max)
        if s_max <= 0:
            return 0.0
        d_max_soft = max_d + (1.0 / beta) * np.log(s_max)

        if d_max_soft == 0:
            return 0.0
        return -(d_min_soft / d_max_soft)

    def spherical_objective(theta_phi, beta=50.0):
        """
        Spherical representation via stereographic projection.
        theta in [0, pi], phi in [0, 2*pi].
        Project to plane via z = cot(theta/2), x = csc(theta/2) cos phi, y = csc(theta/2) sin phi
        Then evaluate planar objective (smoothed) on projected points.
        """
        theta = theta_phi[0::2]
        phi = theta_phi[1::2]

        # Avoid numerical issues at poles
        th = np.clip(theta, 1e-6, np.pi - 1e-6)
        half = 0.5 * th
        c = np.cos(half)
        s = np.sin(half)
        # stereographic: from north pole (theta=0) projection onto plane z=0 (equator plane)
        # Actually standard: point on sphere (sin th cos ph, sin th sin ph, cos th)
        # Stereographic from north pole (0,0,1) to plane z=0:
        # x = sin th cos ph / (1 - cos th), y = sin th sin ph / (1 - cos th)
        # 1 - cos th = 2 sin^2(th/2)
        # sin th = 2 sin(th/2) cos(th/2)
        # So x = cos(th/2)/sin(th/2) * cos ph = cot(th/2) cos ph
        # y = cot(th/2) sin ph
        cot = c / s  # = cot(th/2)
        x = cot * np.cos(phi)
        y = cot * np.sin(phi)
        pts = np.column_stack((x, y))

        dists = scipy.spatial.distance.pdist(pts)
        if len(dists) == 0:
            return 0.0
        max_d = float(np.max(dists))
        if max_d < 1e-12:
            return 0.0

        # soft min
        u = beta * (max_d - dists)
        u = np.clip(u, -700.0, 700.0)
        term_min = np.exp(u)
        s_min = np.sum(term_min)
        if s_min <= 0:
            return 0.0
        d_min_soft = max_d - (1.0 / beta) * np.log(s_min)

        # soft max
        v = beta * (dists - max_d)
        v = np.clip(v, -700.0, 700.0)
        term_max = np.exp(v)
        s_max = np.sum(term_max)
        if s_max <= 0:
            return 0.0
        d_max_soft = max_d + (1.0 / beta) * np.log(s_max)

        if d_max_soft == 0:
            return 0.0
        return -(d_min_soft / d_max_soft)

    def project_sphere(theta, phi):
        th = np.clip(theta, 1e-6, np.pi - 1e-6)
        half = 0.5 * th
        c = np.cos(half)
        s = np.sin(half)
        cot = c / s
        x = cot * np.cos(phi)
        y = cot * np.sin(phi)
        return np.column_stack((x, y))

    def sphere_from_planar(pts):
        """
        Map planar points to spherical coords via inverse stereographic projection.
        For point (x,y) in plane z=0 (equator plane), inverse from north pole:
        On sphere: 
          sin th cos ph = 2x / (1+x^2+y^2)
          sin th sin ph = 2y / (1+x^2+y^2)
          cos th = (1 - x^2 - y^2) / (1 + x^2 + y^2)
        So th = arccos(cos th), ph = arctan2(y,x)
        """
        x = pts[:, 0]
        y = pts[:, 1]
        r2 = x**2 + y**2
        denom = 1.0 + r2
        cos_th = (1.0 - r2) / denom
        cos_th = np.clip(cos_th, -1.0, 1.0)
        th = np.arccos(cos_th)
        ph = np.arctan2(y, x)
        # map to [0, 2pi]
        ph = np.mod(ph, 2 * np.pi)
        return th, ph

    # ---------- Initial candidates ----------
    candidates = []

    # Candidate 1: 4x4 grid
    grid_x, grid_y = np.meshgrid(np.linspace(-1.5, 1.5, 4), np.linspace(-1.5, 1.5, 4))
    grid_pts = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    candidates.append(normalize(grid_pts))

    # Candidate 2: Regular 16-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    circle_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    candidates.append(normalize(circle_pts))

    # Candidate 3: Hexagonal-like
    hex_pts = grid_pts.copy()
    for i in range(1, 4, 2):
        hex_pts[i * 4:(i + 1) * 4, 0] += 0.5
    candidates.append(normalize(hex_pts))

    # Candidate 4: Multi-ring
    c1 = np.array([0.0, 0.0])
    a2 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 5
    r2 = 1.0
    inner = np.column_stack((r2 * np.cos(a2), r2 * np.sin(a2)))
    a3 = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    r3 = 2.0
    outer = np.column_stack((r3 * np.cos(a3), r3 * np.sin(a3)))
    ring_pts = np.vstack([c1, inner, outer])
    candidates.append(normalize(ring_pts))

    # Candidate 5: Fibonacci sphere -> stereographic projection
    # Fibonacci sphere: points on unit sphere
    i_arr = np.arange(n)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z_vals = 1.0 - 2.0 * (i_arr + 0.5) / n  # from 1 to -1
    z_vals = np.clip(z_vals, -1.0 + 1e-6, 1.0 - 1e-6)
    r_vals = np.sqrt(np.maximum(0.0, 1.0 - z_vals**2))
    ph_vals = golden_angle * i_arr
    x_sphere = r_vals * np.cos(ph_vals)
    y_sphere = r_vals * np.sin(ph_vals)
    # Convert to theta, phi: theta = arccos(z), phi = ph
    theta_fib = np.arccos(z_vals)
    phi_fib = ph_vals
    fib_pts = project_sphere(theta_fib, phi_fib)
    candidates.append(normalize(fib_pts))

    # Candidate 6: Spherical code initial (16 points on sphere, projected)
    # Try a structured spherical code: icosahedron-like + more
    # For 16 points, use a cube + octahedron? Cube has 8, octahedron 6 = 14, need 2 more.
    # Let's just use random spherical and project.
    np.random.seed(42)
    for _ in range(12):
        # random planar
        rand_pts = np.random.uniform(-2, 2, (n, d))
        candidates.append(normalize(rand_pts))

    # Add a few random spherical projections
    for _ in range(6):
        theta_r = np.random.uniform(0.05, np.pi - 0.05, n)
        phi_r = np.random.uniform(0, 2 * np.pi, n)
        sp_pts = project_sphere(theta_r, phi_r)
        candidates.append(normalize(sp_pts))

    # Evaluate initial candidates
    best_points = None
    best_score = -np.inf
    for c in candidates:
        s = get_score(c)
        if s > best_score:
            best_score = s
            best_points = c.copy()

    # ---------- Population-based optimization in two representations ----------
    # Population A: planar
    # Population B: spherical (theta, phi)
    pop_size = 12

    # Initialize planar population
    pop_planar = [c.copy() for c in candidates[:pop_size]]
    while len(pop_planar) < pop_size:
        rand_pts = np.random.uniform(-2, 2, (n, d))
        pop_planar.append(normalize(rand_pts))

    # Initialize spherical population by mapping some planar candidates to sphere
    pop_sphere = []
    for c in candidates[:pop_size]:
        th, ph = sphere_from_planar(c)
        pop_sphere.append(np.column_stack((th, ph)).flatten())
    while len(pop_sphere) < pop_size:
        theta_r = np.random.uniform(0.05, np.pi - 0.05, n)
        phi_r = np.random.uniform(0, 2 * np.pi, n)
        pop_sphere.append(np.column_stack((theta_r, phi_r)).flatten())

    bounds_planar = [(-5.0, 5.0)] * (n * d)
    # For spherical: theta in [eps, pi-eps], phi in [0, 2pi]
    eps_th = 1e-4
    bounds_sphere = []
    for _ in range(n):
        bounds_sphere.append((eps_th, np.pi - eps_th))
        bounds_sphere.append((0.0, 2 * np.pi))

    beta_planar = 50.0
    beta_sphere = 50.0

    stagnation_planar = 0
    stagnation_sphere = 0
    max_stagnation = 3

    # Main loop: alternate between planar and spherical optimization
    iteration = 0
    while time.time() < deadline - 15:
        iteration += 1

        # Decide which representation to optimize this round (alternate or based on performance)
        if iteration % 2 == 1:
            # Planar round
            improved = False
            for i in range(len(pop_planar)):
                if time.time() > deadline - 10:
                    break
                try:
                    res = scipy.optimize.minimize(
                        planar_objective,
                        pop_planar[i].flatten(),
                        args=(beta_planar,),
                        method='L-BFGS-B',
                        bounds=bounds_planar,
                        options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    if res.success:
                        polished = res.x.reshape(n, d)
                        s = get_score(polished)
                        pop_planar[i] = polished
                        if s > best_score:
                            best_score = s
                            best_points = polished.copy()
                            improved = True
                except Exception:
                    pass

            if improved:
                stagnation_planar = 0
            else:
                stagnation_planar += 1

            if stagnation_planar >= max_stagnation:
                # Inject diversity: perturb best points
                for i in range(len(pop_planar) // 2):
                    if best_points is not None:
                        noise = np.random.normal(0, 0.08, (n, d))
                        new_pt = normalize(best_points + noise)
                    else:
                        new_pt = normalize(np.random.uniform(-2, 2, (n, d)))
                    idx = np.random.randint(0, len(pop_planar))
                    pop_planar[idx] = new_pt
                stagnation_planar = 0

                # Also inject into sphere population by mapping
                th, ph = sphere_from_planar(best_points if best_points is not None else pop_planar[0])
                new_sphere = np.column_stack((th, ph)).flatten()
                idx = np.random.randint(0, len(pop_sphere))
                pop_sphere[idx] = new_sphere

        else:
            # Spherical round
            improved = False
            for i in range(len(pop_sphere)):
                if time.time() > deadline - 10:
                    break
                try:
                    res = scipy.optimize.minimize(
                        spherical_objective,
                        pop_sphere[i].flatten(),
                        args=(beta_sphere,),
                        method='L-BFGS-B',
                        bounds=bounds_sphere,
                        options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    if res.success:
                        x_opt = res.x
                        th_opt = x_opt[0::2]
                        ph_opt = x_opt[1::2]
                        pts = project_sphere(th_opt, ph_opt)
                        s = get_score(pts)
                        pop_sphere[i] = x_opt
                        if s > best_score:
                            best_score = s
                            best_points = pts.copy()
                            improved = True
                            # Also update planar population with this good point
                            idx_p = np.random.randint(0, len(pop_planar))
                            pop_planar[idx_p] = pts.copy()
                except Exception:
                    pass

            if improved:
                stagnation_sphere = 0
            else:
                stagnation_sphere += 1

            if stagnation_sphere >= max_stagnation:
                # Inject diversity in spherical space
                for i in range(len(pop_sphere) // 2):
                    if best_points is not None:
                        th_best, ph_best = sphere_from_planar(best_points)
                        th_noise = np.random.normal(0, 0.05, n)
                        ph_noise = np.random.normal(0, 0.05, n)
                        th_new = np.clip(th_best + th_noise, 0.02, np.pi - 0.02)
                        ph_new = np.mod(ph_best + ph_noise, 2 * np.pi)
                    else:
                        th_new = np.random.uniform(0.05, np.pi - 0.05, n)
                        ph_new = np.random.uniform(0, 2 * np.pi, n)
                    new_sphere = np.column_stack((th_new, ph_new)).flatten()
                    idx = np.random.randint(0, len(pop_sphere))
                    pop_sphere[idx] = new_sphere
                stagnation_sphere = 0

                # Also inject into planar
                if best_points is not None:
                    noise = np.random.normal(0, 0.08, (n, d))
                    new_pt = normalize(best_points + noise)
                    idx_p = np.random.randint(0, len(pop_planar))
                    pop_planar[idx_p] = new_pt

        # Cross-pollination every few iterations
        if iteration % 4 == 0 and best_points is not None:
            # Map best planar to sphere
            th, ph = sphere_from_planar(best_points)
            new_sphere = np.column_stack((th, ph)).flatten()
            pop_sphere[np.random.randint(0, len(pop_sphere))] = new_sphere

            # Map best sphere to planar: project a random sphere member
            idx_s = np.random.randint(0, len(pop_sphere))
            x_s = pop_sphere[idx_s]
            th_s = x_s[0::2]
            ph_s = x_s[1::2]
            pts_s = project_sphere(th_s, ph_s)
            pop_planar[np.random.randint(0, len(pop_planar))] = normalize(pts_s)

    # Final polish in both representations
    if best_points is not None and time.time() < deadline - 3:
        # Polish in planar with high beta
        try:
            res_final = scipy.optimize.minimize(
                planar_objective,
                best_points.flatten(),
                args=(200.0,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            if res_final.success:
                final_pts = res_final.x.reshape(n, d)
                final_score = get_score(final_pts)
                if final_score > best_score:
                    best_score = final_score
                    best_points = final_pts
        except Exception:
            pass

        # Polish in spherical with high beta
        if time.time() < deadline - 1:
            try:
                th, ph = sphere_from_planar(best_points)
                x0 = np.column_stack((th, ph)).flatten()
                res_final_s = scipy.optimize.minimize(
                    spherical_objective,
                    x0,
                    args=(200.0,),
                    method='L-BFGS-B',
                    bounds=bounds_sphere,
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10}
                )
                if res_final_s.success:
                    th_opt = res_final_s.x[0::2]
                    ph_opt = res_final_s.x[1::2]
                    pts_opt = project_sphere(th_opt, ph_opt)
                    s_opt = get_score(pts_opt)
                    if s_opt > best_score:
                        best_score = s_opt
                        best_points = pts_opt
            except Exception:
                pass

    if best_points is None:
        best_points = candidates[0]

    return best_points
