"""Program for 200 equal circles in a unit square."""
import numpy as np
import time
from scipy.spatial import Delaunay, KDTree
from scipy.spatial.distance import cdist

N = 200
TIME_LIMIT = 145.0  # seconds, leaving margin before 150s kill


def _compute_radius(centres):
    """Compute the packing radius for a set of centres."""
    if len(centres) != N:
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # Check bounds: centres must be inside [-0.5, 0.5]^2
    if np.any(centres < -0.5) or np.any(centres > 0.5):
        return 0.0
    # Distance to walls
    wall_dists = np.min(np.stack([
        0.5 - centres[:, 0],
        0.5 + centres[:, 0],
        0.5 - centres[:, 1],
        0.5 + centres[:, 1]
    ], axis=1), axis=1)
    min_wall = np.min(wall_dists)
    if min_wall <= 0:
        return 0.0
    # Minimum pairwise distance using cdist
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    return min(min_dist / 2.0, min_wall)


def _fibonacci_lattice():
    """Create a Fibonacci lattice (sunflower pattern) initialization."""
    # Generate N points using the golden angle
    n = N
    theta = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
    i = np.arange(n)
    r = np.sqrt(i / n) * 0.5  # scale to fit in radius 0.5
    angles = i * theta
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    centres = np.column_stack((x, y))
    # Clip to ensure inside square
    centres = np.clip(centres, -0.49, 0.49)
    return centres


def _hexagonal_packing():
    """Create a hexagonal packing pattern, optimized for N points."""
    best_centres = None
    best_r = -1
    
    # Try various numbers of rows
    for n_rows in range(12, 22):
        for start_offset in [0, 1]:
            for n_cols in range(12, 20):
                # Calculate how many points fit in this configuration
                total_points = 0
                for row in range(n_rows):
                    if (row + start_offset) % 2 == 0:
                        total_points += n_cols
                    else:
                        total_points += n_cols - 1
                
                if total_points < N:
                    continue
                
                # Calculate spacing
                s_horiz = 1.0 / n_cols
                if n_rows > 1:
                    s_vert = 2.0 / ((n_rows - 1) * np.sqrt(3))
                else:
                    s_vert = 1.0
                
                s = min(s_horiz, s_vert)
                
                # Create the grid
                centres = []
                y_extent = (n_rows - 1) * s * np.sqrt(3) / 2
                y_start = -0.5 + (1.0 - y_extent) / 2 + s * np.sqrt(3) / 4
                for row in range(n_rows):
                    y = y_start + row * s * np.sqrt(3) / 2
                    if (row + start_offset) % 2 == 0:
                        x_extent = (n_cols - 1) * s
                        x_start = -0.5 + (1.0 - x_extent) / 2
                        for col in range(n_cols):
                            x = x_start + col * s
                            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                                centres.append((x, y))
                    else:
                        x_extent = (n_cols - 2) * s
                        x_start = -0.5 + (1.0 - x_extent) / 2 + s / 2
                        for col in range(n_cols - 1):
                            x = x_start + col * s
                            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                                centres.append((x, y))
                
                centres = np.array(centres)
                if len(centres) < N:
                    continue
                
                centres = centres[:N]
                actual_r = _compute_radius(centres)
                if actual_r > best_r:
                    best_r = actual_r
                    best_centres = centres.copy()
    
    if best_centres is not None:
        return best_centres
    
    # Fallback to grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _delaunay_push_optimization(centres, deadline):
    """Delaunay-triangulation-based discrete graph optimization.
    
    Constructs the Delaunay triangulation to identify constraint edges,
    then iteratively performs exact geometric push operations along the
    gradient of the tightest constraint (minimum pair distance or wall proximity).
    """
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    
    no_improve_count = 0
    push_magnitude = max(0.002, best_r * 0.15)
    
    for it in range(8000):
        if time.time() > deadline:
            break
        
        # Compute current constraints
        wall_dists = np.min(np.stack([
            0.5 - best_centres[:, 0],
            0.5 + best_centres[:, 0],
            0.5 - best_centres[:, 1],
            0.5 + best_centres[:, 1]
        ], axis=1), axis=1)
        
        # Build Delaunay triangulation to find neighbor graph
        try:
            tri = Delaunay(best_centres)
            # Extract unique edges from triangles
            edges_set = set()
            for simplex in tri.simplices:
                for a in range(3):
                    for b in range(a + 1, 3):
                        i, j = simplex[a], simplex[b]
                        if i > j:
                            i, j = j, i
                        edges_set.add((i, j))
            edges = list(edges_set)
        except Exception:
            # Fallback to KDTree nearest neighbors
            tree = KDTree(best_centres)
            dists, indices = tree.query(best_centres, k=3)
            edges = []
            for i in range(N):
                for k in range(1, 3):
                    j = indices[i, k]
                    if i < j:
                        edges.append((i, j))
        
        # Compute constraint values for each edge
        if len(edges) > 0:
            edge_i = np.array([e[0] for e in edges])
            edge_j = np.array([e[1] for e in edges])
            edge_vecs = best_centres[edge_i] - best_centres[edge_j]
            edge_dists = np.linalg.norm(edge_vecs, axis=1)
            edge_constraints = edge_dists / 2.0
        else:
            edge_constraints = np.array([])
        
        # Combine pair constraints and wall constraints into a unified view
        # For each point, find the tightest constraint
        point_constraints = np.full(N, np.inf)
        point_constraint_type = np.zeros(N, dtype=int)  # 0=pair, 1=wall
        point_constraint_other = np.full(N, -1, dtype=int)
        
        for idx, (i, j) in enumerate(edges):
            c = edge_constraints[idx]
            if c < point_constraints[i]:
                point_constraints[i] = c
                point_constraint_type[i] = 0
                point_constraint_other[i] = j
            if c < point_constraints[j]:
                point_constraints[j] = c
                point_constraint_type[j] = 0
                point_constraint_other[j] = i
        
        # Wall constraints
        for i in range(N):
            if wall_dists[i] < point_constraints[i]:
                point_constraints[i] = wall_dists[i]
                point_constraint_type[i] = 1
                point_constraint_other[i] = -1
        
        # Find the globally tightest constraint
        min_idx = np.argmin(point_constraints)
        min_val = point_constraints[min_idx]
        
        # If the tightest constraint is already well above zero, we're good
        if min_val > best_r * 0.95:
            no_improve_count += 1
            if no_improve_count > 30:
                break
            # Try a smaller push
            push_magnitude *= 0.9
            continue
        
        # Determine push direction for the tightest point
        push_dir = np.zeros(2)
        if point_constraint_type[min_idx] == 0:
            # Pair constraint: push away from the other point
            other = point_constraint_other[min_idx]
            vec = best_centres[min_idx] - best_centres[other]
            norm = np.linalg.norm(vec)
            if norm > 1e-12:
                push_dir = vec / norm
        else:
            # Wall constraint: push away from the closest wall
            dx = np.array([
                0.5 - best_centres[min_idx, 0],
                0.5 + best_centres[min_idx, 0],
                0.5 - best_centres[min_idx, 1],
                0.5 + best_centres[min_idx, 1]
            ])
            wall_idx = np.argmin(dx)
            if wall_idx == 0:
                push_dir = np.array([-1.0, 0.0])
            elif wall_idx == 1:
                push_dir = np.array([1.0, 0.0])
            elif wall_idx == 2:
                push_dir = np.array([0.0, -1.0])
            else:
                push_dir = np.array([0.0, 1.0])
        
        # Also compute push for the "other" point if it's a pair constraint
        other_push_dir = np.zeros(2)
        has_other_push = False
        if point_constraint_type[min_idx] == 0:
            other = point_constraint_other[min_idx]
            other_vec = best_centres[other] - best_centres[min_idx]
            other_norm = np.linalg.norm(other_vec)
            if other_norm > 1e-12:
                other_push_dir = other_vec / other_norm
                has_other_push = True
        
        # Try multiple push magnitudes
        improved = False
        for scale in [1.0, 0.5, 0.25, 2.0, 0.1]:
            if time.time() > deadline:
                break
            
            delta = push_magnitude * scale
            new_centres = best_centres.copy()
            new_centres[min_idx] += delta * push_dir
            if has_other_push:
                new_centres[point_constraint_other[min_idx]] += delta * other_push_dir
            
            # Check bounds
            if np.any(new_centres < -0.5 + 1e-12) or np.any(new_centres > 0.5 - 1e-12):
                continue
            
            new_r = _compute_radius(new_centres)
            if new_r > best_r + 1e-13:
                best_r = new_r
                best_centres = new_centres
                improved = True
                no_improve_count = 0
                break
        
        if improved:
            push_magnitude *= 0.97
        else:
            no_improve_count += 1
            push_magnitude *= 0.85
            if push_magnitude < 1e-10 or no_improve_count > 50:
                break
    
    return best_centres, best_r


def _delaunay_multi_push(centres, deadline):
    """Multi-point Delaunay-based push: resolve top-K tightest constraints simultaneously.
    
    Uses the Delaunay graph to identify all active constraints, then pushes
    the most constrained points in a coordinated manner.
    """
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    
    push_magnitude = max(0.001, best_r * 0.1)
    no_improve_count = 0
    
    for it in range(6000):
        if time.time() > deadline:
            break
        
        # Compute wall distances
        wall_dists = np.min(np.stack([
            0.5 - best_centres[:, 0],
            0.5 + best_centres[:, 0],
            0.5 - best_centres[:, 1],
            0.5 + best_centres[:, 1]
        ], axis=1), axis=1)
        
        # Build Delaunay triangulation
        try:
            tri = Delaunay(best_centres)
            edges_set = set()
            for simplex in tri.simplices:
                for a in range(3):
                    for b in range(a + 1, 3):
                        i, j = simplex[a], simplex[b]
                        if i > j:
                            i, j = j, i
                        edges_set.add((i, j))
            edges = list(edges_set)
        except Exception:
            tree = KDTree(best_centres)
            dists, indices = tree.query(best_centres, k=3)
            edges = []
            for i in range(N):
                for k in range(1, 3):
                    j = indices[i, k]
                    if i < j:
                        edges.append((i, j))
        
        # Compute per-point tightest constraint
        point_constraints = np.full(N, np.inf)
        point_type = np.zeros(N, dtype=int)
        point_other = np.full(N, -1, dtype=int)
        
        if len(edges) > 0:
            edge_i = np.array([e[0] for e in edges])
            edge_j = np.array([e[1] for e in edges])
            edge_vecs = best_centres[edge_i] - best_centres[edge_j]
            edge_dists = np.linalg.norm(edge_vecs, axis=1)
            edge_c = edge_dists / 2.0
            
            for idx, (i, j) in enumerate(edges):
                c = edge_c[idx]
                if c < point_constraints[i]:
                    point_constraints[i] = c
                    point_type[i] = 0
                    point_other[i] = j
                if c < point_constraints[j]:
                    point_constraints[j] = c
                    point_type[j] = 0
                    point_other[j] = i
        
        for i in range(N):
            if wall_dists[i] < point_constraints[i]:
                point_constraints[i] = wall_dists[i]
                point_type[i] = 1
                point_other[i] = -1
        
        # Select top K most constrained points
        K = min(4, N)
        constrained_idx = np.argsort(point_constraints)[:K]
        
        # Compute push directions
        displacements = np.zeros_like(best_centres)
        for idx in constrained_idx:
            if point_type[idx] == 0:
                other = point_other[idx]
                vec = best_centres[idx] - best_centres[other]
                norm = np.linalg.norm(vec)
                if norm > 1e-12:
                    displacements[idx] = vec / norm
            else:
                dx = np.array([
                    0.5 - best_centres[idx, 0],
                    0.5 + best_centres[idx, 0],
                    0.5 - best_centres[idx, 1],
                    0.5 + best_centres[idx, 1]
                ])
                wall_idx = np.argmin(dx)
                if wall_idx == 0:
                    displacements[idx] = np.array([-1.0, 0.0])
                elif wall_idx == 1:
                    displacements[idx] = np.array([1.0, 0.0])
                elif wall_idx == 2:
                    displacements[idx] = np.array([0.0, -1.0])
                else:
                    displacements[idx] = np.array([0.0, 1.0])
        
        # Try multiple step sizes
        improved = False
        for scale in [1.0, 0.5, 0.25, 2.0, 0.1]:
            if time.time() > deadline:
                break
            
            new_centres = best_centres + push_magnitude * scale * displacements
            
            if np.any(new_centres < -0.5 + 1e-12) or np.any(new_centres > 0.5 - 1e-12):
                continue
            
            new_r = _compute_radius(new_centres)
            if new_r > best_r + 1e-13:
                best_r = new_r
                best_centres = new_centres
                improved = True
                no_improve_count = 0
                break
        
        if improved:
            push_magnitude *= 0.96
        else:
            no_improve_count += 1
            push_magnitude *= 0.85
            if push_magnitude < 1e-10 or no_improve_count > 40:
                break
    
    return best_centres, best_r


def _polish_pass(centres, deadline):
    """High-precision polish using Delaunay-based exact constraint resolution.
    
    Alternates between single-point and paired pushes with very fine steps,
    using the Delaunay graph for neighbor identification.
    """
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    
    step_size = max(0.0005, best_r * 0.05)
    no_improve_count = 0
    
    for it in range(4000):
        if time.time() > deadline:
            break
        
        # Compute wall distances
        wall_dists = np.min(np.stack([
            0.5 - best_centres[:, 0],
            0.5 + best_centres[:, 0],
            0.5 - best_centres[:, 1],
            0.5 + best_centres[:, 1]
        ], axis=1), axis=1)
        
        # Build Delaunay for neighbor info
        try:
            tri = Delaunay(best_centres)
            edges_set = set()
            for simplex in tri.simplices:
                for a in range(3):
                    for b in range(a + 1, 3):
                        i, j = simplex[a], simplex[b]
                        if i > j:
                            i, j = j, i
                        edges_set.add((i, j))
            edges = list(edges_set)
        except Exception:
            tree = KDTree(best_centres)
            dists, indices = tree.query(best_centres, k=2)
            edges = [(i, indices[i, 1]) for i in range(N) if i < indices[i, 1]]
        
        # Per-point tightest constraint
        point_constraints = np.full(N, np.inf)
        point_type = np.zeros(N, dtype=int)
        point_other = np.full(N, -1, dtype=int)
        
        if len(edges) > 0:
            for (i, j) in edges:
                d = np.linalg.norm(best_centres[i] - best_centres[j])
                c = d / 2.0
                if c < point_constraints[i]:
                    point_constraints[i] = c
                    point_type[i] = 0
                    point_other[i] = j
                if c < point_constraints[j]:
                    point_constraints[j] = c
                    point_type[j] = 0
                    point_other[j] = i
        
        for i in range(N):
            if wall_dists[i] < point_constraints[i]:
                point_constraints[i] = wall_dists[i]
                point_type[i] = 1
                point_other[i] = -1
        
        # Alternate: single point on even, pair on odd
        if it % 2 == 0:
            idx = np.argmin(point_constraints)
            targets = [idx]
        else:
            targets = np.argsort(point_constraints)[:2]
        
        displacements = np.zeros_like(best_centres)
        for idx in targets:
            if point_type[idx] == 0:
                other = point_other[idx]
                vec = best_centres[idx] - best_centres[other]
                norm = np.linalg.norm(vec)
                if norm > 1e-12:
                    displacements[idx] = vec / norm
            else:
                dx = np.array([
                    0.5 - best_centres[idx, 0],
                    0.5 + best_centres[idx, 0],
                    0.5 - best_centres[idx, 1],
                    0.5 + best_centres[idx, 1]
                ])
                wall_idx = np.argmin(dx)
                if wall_idx == 0:
                    displacements[idx] = np.array([-1.0, 0.0])
                elif wall_idx == 1:
                    displacements[idx] = np.array([1.0, 0.0])
                elif wall_idx == 2:
                    displacements[idx] = np.array([0.0, -1.0])
                else:
                    displacements[idx] = np.array([0.0, 1.0])
        
        improved = False
        for scale in [1.0, 0.5, 0.25, 0.1, 2.0]:
            if time.time() > deadline:
                break
            
            new_centres = best_centres + step_size * scale * displacements
            
            if np.any(new_centres < -0.5 + 1e-12) or np.any(new_centres > 0.5 - 1e-12):
                continue
            
            new_r = _compute_radius(new_centres)
            if new_r > best_r + 1e-14:
                best_r = new_r
                best_centres = new_centres
                improved = True
                no_improve_count = 0
                break
        
        if improved:
            step_size *= 0.95
        else:
            no_improve_count += 1
            step_size *= 0.85
            if step_size < 1e-11 or no_improve_count > 40:
                break
    
    return best_centres, best_r


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - 3  # Leave 3s safety margin
    
    best_centres = None
    best_r = -1
    
    # Try multiple initializations for diversity
    initializations = []
    
    # 1. Fibonacci lattice
    fib_centres = _fibonacci_lattice()
    initializations.append(("fibonacci", fib_centres))
    
    # 2. Hexagonal packing
    hex_centres = _hexagonal_packing()
    initializations.append(("hexagonal", hex_centres))
    
    # 3. Random perturbations of hexagonal
    for seed in [42, 123, 456]:
        np.random.seed(seed)
        perturbed = hex_centres.copy()
        perturbation = np.random.uniform(-0.005, 0.005, size=perturbed.shape)
        perturbed += perturbation
        perturbed = np.clip(perturbed, -0.499, 0.499)
        initializations.append((f"hex_perturb_{seed}", perturbed))
    
    for name, centres in initializations:
        if time.time() > deadline:
            break
        
        current_r = _compute_radius(centres)
        
        # Delaunay single-push optimization
        if time.time() < deadline - 25:
            centres, current_r = _delaunay_push_optimization(centres, deadline)
        
        # Delaunay multi-point push
        if time.time() < deadline - 12:
            centres, current_r = _delaunay_multi_push(centres, deadline)
        
        # High-precision polish pass
        if time.time() < deadline - 5:
            centres, current_r = _polish_pass(centres, deadline)
        
        if current_r > best_r:
            best_r = current_r
            best_centres = centres.copy()
    
    # Final check
    if best_centres is not None:
        final_r = _compute_radius(best_centres)
        if final_r > 0:
            return best_centres
    
    # Fallback to a valid grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]
