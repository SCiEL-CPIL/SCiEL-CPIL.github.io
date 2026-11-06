import numpy as np
import time
import concurrent.futures
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

def construct(instance):
    """
    Find low-energy structures for Dzugutov clusters using Basin Hopping 
    initialized from polytetrahedral/icosahedral motifs.
    """
    n = instance["n"]
    potential = instance["potential"]
    
    # Time budget: 300s total / 6 instances = 50s per instance, use 45s to be safe
    time_limit_per_instance = 45.0
    start_time = time.time()
    
    def generate_icosahedral_initial(n):
        """Generate initial cluster based on icosahedral shells (Dzugutov favors icosahedral order)."""
        # icosahedron vertices
        phi = (1 + np.sqrt(5)) / 2
        ico_vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        # Normalize to unit sphere
        ico_vertices = ico_vertices / np.linalg.norm(ico_vertices, axis=1, keepdims=True)
        
        points = []
        # Center
        points.append(np.zeros(3))
        
        # Shell 1: icosahedron (12 atoms) at radius ~1.9
        r1 = 1.9
        if n > 1:
            count1 = min(12, n - 1)
            for i in range(count1):
                points.append(ico_vertices[i % 12] * r1)
        
        # Shell 2: 30 atoms at radius ~3.2
        if n > 13:
            r2 = 3.2
            golden_angle = np.pi * (3 - np.sqrt(5))
            shell2_points = []
            for i in range(30):
                z = 1 - 2 * (i + 0.5) / 30
                radius = np.sqrt(max(0, 1 - z**2))
                theta = golden_angle * i
                x = radius * np.cos(theta)
                y = radius * np.sin(theta)
                shell2_points.append(np.array([x, y, z]))
            shell2_points = np.array(shell2_points)
            shell2_points = shell2_points / np.linalg.norm(shell2_points, axis=1, keepdims=True)
            
            count2 = min(30, n - len(points))
            for i in range(count2):
                points.append(shell2_points[i] * r2)
        
        # If we need more atoms, add more
        if len(points) < n:
            rng = np.random.default_rng(42)
            for _ in range(n - len(points)):
                theta = rng.uniform(0, 2 * np.pi)
                z = rng.uniform(-1, 1)
                radius = np.sqrt(max(0, 1 - z**2))
                r = 3.2 + rng.uniform(0.5, 2.0)
                p = np.array([radius * np.cos(theta), radius * np.sin(theta), z]) * r
                valid = True
                for q in points:
                    if np.linalg.norm(p - q) < 0.8:
                        valid = False
                        break
                if valid:
                    points.append(p)
        
        points = np.array(points[:n])
        points = points - np.mean(points, axis=0)
        return points

    def generate_bcc_initial(n):
        """Generate an initial cluster by cutting a BCC lattice."""
        a_lattice = 2.2
        size_box = int(np.ceil(n ** (1/3) * a_lattice * 2)) + 2
        points = []
        for i in range(-size_box, size_box+1):
            for j in range(-size_box, size_box+1):
                for k in range(-size_box, size_box+1):
                    points.append([i*a_lattice, j*a_lattice, k*a_lattice])
                    points.append([(i+0.5)*a_lattice, (j+0.5)*a_lattice, (k+0.5)*a_lattice])
        
        points = np.array(points)
        com = np.mean(points, axis=0)
        points = points - com
        
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)
        selected = points[indices[:n]]
        selected = selected - np.mean(selected, axis=0)
        return selected

    def generate_pentagonal_initial(n):
        """Generate initial cluster based on pentagonal bipyramid motifs."""
        # Pentagon vertices
        angles = np.linspace(0, 2*np.pi, 6)[:-1]
        pentagon = np.array([[np.cos(a), np.sin(a), 0] for a in angles])
        
        # Build a cluster from pentagonal bipyramids
        points = []
        
        # Central pentagonal bipyramid (7 atoms)
        r_pyra = 1.9
        points.append(np.array([0, 0, 0]))
        for v in pentagon:
            points.append(v * r_pyra)
        points.append(np.array([0, 0, r_pyra * 0.8]))
        points.append(np.array([0, 0, -r_pyra * 0.8]))
        
        # Add more pentagonal bipyramids around
        rng = np.random.default_rng(42)
        while len(points) < n:
            # Pick a random existing point and add a new pentagonal bipyramid nearby
            center = points[rng.integers(len(points))]
            offset = rng.normal(0, 1.5, 3)
            new_center = center + offset
            
            # Check if new center is far enough from all existing points
            if all(np.linalg.norm(new_center - p) > 1.0 for p in points):
                # Add pentagonal bipyramid centered at new_center
                for v in pentagon:
                    p = new_center + v * r_pyra * 0.8
                    if all(np.linalg.norm(p - q) > 0.8 for q in points):
                        points.append(p)
                p_top = new_center + np.array([0, 0, r_pyra * 0.6])
                if all(np.linalg.norm(p_top - q) > 0.8 for q in points):
                    points.append(p_top)
                p_bot = new_center + np.array([0, 0, -r_pyra * 0.6])
                if all(np.linalg.norm(p_bot - q) > 0.8 for q in points):
                    points.append(p_bot)
        
        points = np.array(points[:n])
        points = points - np.mean(points, axis=0)
        return points

    def local_minimize(positions):
        """Perform local minimization using L-BFGS-B."""
        def energy_and_grad(x):
            pos = x.reshape(-1, 3)
            e, grad = dzugutov_energy_gradient(pos, potential)
            return e, grad.flatten()
            
        x0 = positions.flatten()
        res = minimize(
            fun=energy_and_grad,
            x0=x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10, 'maxls': 30}
        )
        return res.fun, res.x.reshape(-1, 3)

    def basin_hopping_trajectory(seed, init_type='ico'):
        """Run a single basin hopping trajectory with adaptive step size."""
        rng = np.random.default_rng(seed)
        
        if init_type == 'ico':
            x = generate_icosahedral_initial(n)
        elif init_type == 'bcc':
            x = generate_bcc_initial(n)
        else:
            x = generate_pentagonal_initial(n)
        
        x += rng.normal(0, 0.03, x.shape)
        
        try:
            e_curr, x_curr = local_minimize(x)
        except Exception:
            e_curr, x_curr = local_minimize(x + rng.normal(0, 0.1, x.shape))
        
        best_e = e_curr
        best_x = x_curr.copy()
        
        step_size = 0.4
        n_iter = 80
        T = 0.15
        no_improve_count = 0
        
        for i in range(n_iter):
            if time.time() - start_time > time_limit_per_instance - 3:
                break
                
            x_trial = x_curr + rng.normal(0, step_size, x_curr.shape)
            
            try:
                e_trial, x_trial_min = local_minimize(x_trial)
            except Exception:
                continue
            
            delta_e = e_trial - e_curr
            if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
                e_curr, x_curr = e_trial, x_trial_min
                if e_curr < best_e:
                    best_e = e_curr
                    best_x = x_curr.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
            
            if no_improve_count > 10:
                step_size *= 1.3
                no_improve_count = 0
            elif best_e < e_curr:
                step_size = max(0.2, step_size * 0.9)
        
        return best_e, best_x

    n_threads = 4
    best_energy = float('inf')
    best_positions = None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = []
        init_types = ['ico', 'bcc', 'ico', 'pent']
        for i in range(n_threads):
            seed = 42 + i * 1000
            futures.append(executor.submit(basin_hopping_trajectory, seed, init_types[i]))
            
        for future in concurrent.futures.as_completed(futures):
            try:
                e, pos = future.result()
                if e < best_energy:
                    best_energy = e
                    best_positions = pos
            except Exception:
                pass
    
    if best_positions is None:
        radius = 1.1 * n ** (1.0 / 3.0) * 0.75
        rng = np.random.default_rng(n)
        points = []
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 for q in points):
                points.append(p)
        _, best_positions = relax(np.array(points), potential)
    
    return best_positions

# EVOLVE-BLOCK-END
