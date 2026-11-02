"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles.
Strategy: Force-based optimization with adaptive repulsion + simulated annealing.
Instead of NLP with all-pair constraints (O(N^2) per iteration), we use a physics-based
approach where overlapping circles repel each other with a force proportional to
overlap depth. This is O(N^2) per step but each step is very cheap (no sparse matrices,
no linear solves). We combine this with simulated annealing for global exploration
and a final Delaunay-based polish for local refinement.
"""
import time
import numpy as np
from scipy.spatial import Delaunay, Voronoi
from scipy.optimize import minimize

N = 26
EPS = 1e-9
DEADLINE = 145.0  # Safety margin for 150s limit


def _max_radius(pos, r):
    """Maximal radius for each circle given current positions and radii."""
    n = len(r)
    mr = np.empty(n)
    for i in range(n):
        dmin = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        for j in range(n):
            if i != j:
                dmin = min(dmin, np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1]) - r[j])
        mr[i] = max(0.0, dmin)
    return mr


def _score(cands):
    """Compute sum of radii for a valid packing, or -inf if invalid."""
    if len(cands) < N:
        return -1e10
    arr = np.array(cands)
    pos = arr[:, :2]
    rad = arr[:, 2]
    
    # Check boundary
    for i in range(N):
        if pos[i, 0] < rad[i] + 1e-8 or pos[i, 0] > 1 - rad[i] - 1e-8:
            return -1e10
        if pos[i, 1] < rad[i] + 1e-8 or pos[i, 1] > 1 - rad[i] - 1e-8:
            return -1e10
    
    # Check pairwise
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
            if d < rad[i] + rad[j] - 1e-8:
                return -1e10
    
    return float(np.sum(rad))


def _force_step(pos, rad, dt, repulsion_strength=1.0):
    """
    One step of force-based optimization.
    Forces:
    1. Repulsion between overlapping circles (proportional to overlap)
    2. Attraction toward maximizing radius (grow radii)
    3. Boundary repulsion
    """
    n = len(rad)
    forces = np.zeros((n, 2))
    dr = np.zeros(n)
    
    # Pairwise repulsion
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.hypot(dx, dy)
            if d < 1e-10:
                # Random small perturbation
                angle = np.random.uniform(0, 2 * np.pi)
                dx, dy = np.cos(angle) * 1e-10, np.sin(angle) * 1e-10
                d = 1e-10
            
            overlap = rad[i] + rad[j] - d
            if overlap > 0:
                # Force proportional to overlap, directed along separation
                fx = repulsion_strength * overlap * dx / d
                fy = repulsion_strength * overlap * dy / d
                forces[i, 0] += fx
                forces[i, 1] += fy
                forces[j, 0] -= fx
                forces[j, 1] -= fy
                # Slightly reduce radii when overlapping
                dr[i] -= 0.5 * overlap
                dr[j] -= 0.5 * overlap
    
    # Boundary forces
    for i in range(n):
        # Left boundary
        if pos[i, 0] < rad[i]:
            forces[i, 0] += repulsion_strength * (rad[i] - pos[i, 0])
        # Right boundary
        if pos[i, 0] > 1 - rad[i]:
            forces[i, 0] -= repulsion_strength * (pos[i, 0] - (1 - rad[i]))
        # Bottom boundary
        if pos[i, 1] < rad[i]:
            forces[i, 1] += repulsion_strength * (rad[i] - pos[i, 1])
        # Top boundary
        if pos[i, 1] > 1 - rad[i]:
            forces[i, 1] -= repulsion_strength * (pos[i, 1] - (1 - rad[i]))
    
    # Apply forces
    pos_new = pos + dt * forces
    rad_new = rad + dt * dr
    rad_new = np.maximum(rad_new, 0.01)
    
    # Keep in bounds
    pos_new[:, 0] = np.clip(pos_new[:, 0], 0.01, 0.99)
    pos_new[:, 1] = np.clip(pos_new[:, 1], 0.01, 0.99)
    
    return pos_new, rad_new


def _simulated_annealing(pos, rad, t_start, max_time=30.0):
    """Simulated annealing with force-based moves."""
    best_pos = pos.copy()
    best_rad = rad.copy()
    best_score = _score(list(zip(pos[:, 0], pos[:, 1], rad)))
    
    current_pos = pos.copy()
    current_rad = rad.copy()
    current_score = best_score
    
    # Adaptive parameters
    base_temp = 0.05
    cooling_rate = 0.995
    move_scale = 0.02
    
    step = 0
    while time.time() < t_start + max_time:
        temp = base_temp * (cooling_rate ** step)
        
        # Generate move
        move_type = np.random.random()
        if move_type < 0.6:
            # Move a random circle
            i = np.random.randint(N)
            dx = np.random.normal(0, move_scale)
            dy = np.random.normal(0, move_scale)
            new_pos = current_pos.copy()
            new_pos[i, 0] += dx
            new_pos[i, 1] += dy
            new_rad = current_rad.copy()
        elif move_type < 0.85:
            # Grow/shrink a random circle
            i = np.random.randint(N)
            new_pos = current_pos.copy()
            new_rad = current_rad.copy()
            new_rad[i] *= (1 + np.random.normal(0, 0.1))
            new_rad[i] = max(0.01, new_rad[i])
        else:
            # Force-based step
            new_pos, new_rad = _force_step(current_pos, current_rad, dt=0.1)
        
        new_score = _score(list(zip(new_pos[:, 0], new_pos[:, 1], new_rad)))
        
        if new_score > current_score:
            current_pos = new_pos
            current_rad = new_rad
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pos = new_pos.copy()
                best_rad = new_rad.copy()
        else:
            # Accept with probability exp((new - current) / temp)
            delta = new_score - current_score
            if temp > 1e-8 and np.random.random() < np.exp(delta / temp):
                current_pos = new_pos
                current_rad = new_rad
                current_score = new_score
        
        step += 1
        # Adaptive move scale
        if step % 100 == 0:
            move_scale *= 0.99
    
    return best_pos, best_rad, best_score


def _delaunay_polish(pos, rad, max_iter=50):
    """Polish using Delaunay triangulation to identify contact pairs."""
    for _ in range(max_iter):
        if time.time() > DEADLINE - 5:
            break
        
        # Get Delaunay triangulation
        try:
            tri = Delaunay(pos)
            # Get unique edges
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        a, b = simplex[i], simplex[j]
                        edges.add((min(a, b), max(a, b)))
        except Exception:
            break
        
        # For each edge, check if circles are in contact or nearly so
        for i, j in edges:
            d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
            r_sum = rad[i] + rad[j]
            
            if d < r_sum + 0.01:  # Near contact
                # Try to increase radii slightly if feasible
                # Check if we can grow both radii
                growth = 0.001
                new_ri = rad[i] + growth
                new_rj = rad[j] + growth
                
                # Check boundary constraints
                if (pos[i, 0] >= new_ri and pos[i, 0] <= 1 - new_ri and
                    pos[i, 1] >= new_ri and pos[i, 1] <= 1 - new_ri and
                    pos[j, 0] >= new_rj and pos[j, 0] <= 1 - new_rj and
                    pos[j, 1] >= new_rj and pos[j, 1] <= 1 - new_rj):
                    # Check other pairs
                    feasible = True
                    for k in range(N):
                        if k == i or k == j:
                            continue
                        if np.hypot(pos[i, 0] - pos[k, 0], pos[i, 1] - pos[k, 1]) < new_ri + rad[k] - 1e-9:
                            feasible = False
                            break
                        if np.hypot(pos[j, 0] - pos[k, 0], pos[j, 1] - pos[k, 1]) < new_rj + rad[k] - 1e-9:
                            feasible = False
                            break
                    
                    if feasible:
                        rad[i] = new_ri
                        rad[j] = new_rj
    
    # Final max radius computation
    for _ in range(10):
        rad = _max_radius(pos, rad)
    
    return pos, rad


def _grid_initial(side):
    """Generate grid initial configuration."""
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append(((i + 0.5) / side, (j + 0.5) / side))
    cells = cells[:N]
    pos = np.array(cells, dtype=float)
    r = np.full(N, 0.5 / side - 0.01)
    for _ in range(20):
        r = _max_radius(pos, r)
    return pos, r


def _random_initial(rng):
    """Generate random initial configuration."""
    pos = rng.random((N, 2)) * 0.8 + 0.1
    r = np.full(N, 0.02)
    for _ in range(20):
        r = _max_radius(pos, r)
    return pos, r


def _hex_initial():
    """Generate hexagonal initial configuration."""
    pos_list = []
    side = int(np.ceil(np.sqrt(N)))
    for i in range(side):
        for j in range(side):
            if len(pos_list) >= N:
                break
            x = (j + 0.5 * (i % 2)) / side
            y = (i + 0.5) / side
            pos_list.append((x, y))
    pos = np.array(pos_list[:N])
    pos = np.clip(pos, 0.1, 0.9)
    r = np.full(N, 0.05)
    for _ in range(20):
        r = _max_radius(pos, r)
    return pos, r


def _lbfgs_refine(pos, rad):
    """Use L-BFGS-B for local refinement with penalty method."""
    def penalty_objective(x):
        p = x[:2 * N].reshape(N, 2)
        r = x[2 * N:]
        # Penalty for overlaps
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(p[i, 0] - p[j, 0], p[i, 1] - p[j, 1])
                overlap = max(0, r[i] + r[j] - d)
                penalty += overlap ** 2
            # Boundary penalties
            for coord in [0, 1]:
                for axis in range(2):
                    val = p[i, axis]
                    r_i = r[i]
                    if val < r_i:
                        penalty += (r_i - val) ** 2
                    if val > 1 - r_i:
                        penalty += (val - (1 - r_i)) ** 2
        return -np.sum(r) + 1000 * penalty
    
    x0 = np.concatenate([pos.flatten(), rad])
    bounds = [(0, 1)] * (2 * N) + [(0.01, 0.5)] * N
    
    try:
        res = minimize(
            penalty_objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10}
        )
        if res.success or res.fun < penalty_objective(x0):
            new_pos = res.x[:2 * N].reshape(N, 2)
            new_rad = res.x[2 * N:]
            # Verify validity
            s = _score(list(zip(new_pos[:, 0], new_pos[:, 1], new_rad)))
            if s > -1e9:
                return new_pos, new_rad
    except Exception:
        pass
    
    return pos, rad


def construct():
    t_start = time.time()
    rng = np.random.default_rng(42)
    
    best_pos = None
    best_rad = None
    best_score = -1e10
    
    def consider(pos, rad):
        nonlocal best_pos, best_rad, best_score
        s = _score(list(zip(pos[:, 0], pos[:, 1], rad)))
        if s > best_score:
            best_score = s
            best_pos = pos.copy()
            best_rad = rad.copy()
    
    # Generate diverse initializations
    inits = []
    
    # Grid inits
    for side in [4, 5, 6]:
        if time.time() > DEADLINE:
            break
        pos, rad = _grid_initial(side)
        inits.append((pos, rad))
    
    # Hex initial
    if time.time() < DEADLINE:
        pos, rad = _hex_initial()
        inits.append((pos, rad))
    
    # Random inits
    for _ in range(5):
        if time.time() > DEADLINE:
            break
        pos, rad = _random_initial(rng)
        inits.append((pos, rad))
    
    # Perturbed grids
    for side in [5, 6]:
        if time.time() > DEADLINE:
            break
        pos, rad = _grid_initial(side)
        for _ in range(3):
            if time.time() > DEADLINE:
                break
            p2 = pos + 0.05 * (rng.random((N, 2)) * 2 - 1)
            p2 = np.clip(p2, 0.05, 0.95)
            r2 = rad.copy()
            r2 = _max_radius(p2, r2)
            inits.append((p2, r2))
    
    # Run simulated annealing on each init
    for pos, rad in inits:
        if time.time() > DEADLINE - 40:
            break
        
        # Force-based pre-optimize
        for _ in range(200):
            if time.time() > DEADLINE - 40:
                break
            pos, rad = _force_step(pos, rad, dt=0.05)
            rad = np.maximum(rad, 0.01)
        
        # Simulated annealing
        pos, rad, s = _simulated_annealing(pos, rad, t_start, max_time=15.0)
        consider(pos, rad)
        
        # L-BFGS refinement
        if time.time() < DEADLINE - 20:
            pos, rad = _lbfgs_refine(pos, rad)
            consider(pos, rad)
        
        # Delaunay polish
        if time.time() < DEADLINE - 10:
            pos, rad = _delaunay_polish(pos, rad)
            consider(pos, rad)
    
    # Final polish on best
    if best_pos is not None:
        if time.time() < DEADLINE - 5:
            best_pos, best_rad = _delaunay_polish(best_pos, best_rad)
            consider(best_pos, best_rad)
        
        # One more force-based refinement
        if time.time() < DEADLINE - 2:
            for _ in range(100):
                if time.time() > DEADLINE - 2:
                    break
                best_pos, best_rad = _force_step(best_pos, best_rad, dt=0.02)
                best_rad = np.maximum(best_rad, 0.01)
            best_rad = _max_radius(best_pos, best_rad)
            consider(best_pos, best_rad)
    
    if best_pos is None:
        # Fallback
        pos, rad = _grid_initial(5)
        return np.column_stack([pos, rad])
    
    return np.column_stack([best_pos, best_rad])


if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
    print(out)
