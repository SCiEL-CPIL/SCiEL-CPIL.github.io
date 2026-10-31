"""Circle packing in perimeter-4 rectangle using topological graph model with contact edges."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random
import time


def _build_contact_graph(centers, radii, tolerance=0.01):
    """
    Build a contact graph where edges represent circles that are in contact or nearly in contact.
    Returns adjacency list and edge list.
    """
    n = len(centers)
    adj = [[] for _ in range(n)]
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            
            # Consider circles in contact if they are within tolerance
            if abs(dist - min_dist) < tolerance:
                adj[i].append(j)
                adj[j].append(i)
                edges.append((i, j))
    
    return adj, edges


def _compute_spring_forces(centers, radii, w, h, adj, edges, k_repel=10.0, k_contact=5.0, k_wall=2.0):
    """
    Compute spring forces based on the contact graph.
    - Repulsive forces for non-contact circles that overlap
    - Spring forces along contact edges to maintain contact distance
    - Wall forces to push circles against boundaries
    """
    n = len(centers)
    forces = np.zeros_like(centers)
    
    # Repulsive forces for overlapping circles (not in contact graph)
    contact_set = set()
    for i, j in edges:
        contact_set.add((i, j))
        contact_set.add((j, i))
    
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in contact_set:
                continue
            
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            
            if dist < min_dist and dist > 1e-10:
                # Overlap: strong repulsive force
                overlap = min_dist - dist
                fx = (dx / dist) * overlap * k_repel
                fy = (dy / dist) * overlap * k_repel
                forces[i, 0] += fx
                forces[i, 1] += fy
                forces[j, 0] -= fx
                forces[j, 1] -= fy
    
    # Spring forces along contact edges
    for i, j in edges:
        dx = centers[i, 0] - centers[j, 0]
        dy = centers[i, 1] - centers[j, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        target_dist = radii[i] + radii[j]
        
        if dist > 1e-10:
            # Spring force to maintain contact distance
            stretch = dist - target_dist
            fx = (dx / dist) * stretch * k_contact
            fy = (dy / dist) * stretch * k_contact
            forces[i, 0] -= fx
            forces[i, 1] -= fy
            forces[j, 0] += fx
            forces[j, 1] += fy
    
    # Wall forces: push circles against walls to maximize radii
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        
        # Left wall
        if x - r < 0.05:
            forces[i, 0] += k_wall * (0.05 - (x - r))
        # Right wall
        if x + r > w - 0.05:
            forces[i, 0] -= k_wall * ((x + r) - (w - 0.05))
        # Bottom wall
        if y - r < 0.05:
            forces[i, 1] += k_wall * (0.05 - (y - r))
        # Top wall
        if y + r > h - 0.05:
            forces[i, 1] -= k_wall * ((y + r) - (h - 0.05))
    
    return forces


def _topological_relaxation(centers, w, h, radii, max_iterations=10000, initial_lr=0.02, lr_decay=0.999):
    """
    Topological relaxation using contact graph model.
    Iteratively builds contact graph, computes forces, and updates positions.
    """
    n = len(centers)
    centers = centers.copy()
    radii = radii.copy()
    lr = initial_lr
    
    for it in range(max_iterations):
        # Build contact graph
        adj, edges = _build_contact_graph(centers, radii, tolerance=0.01)
        
        # Compute forces
        forces = _compute_spring_forces(centers, radii, w, h, adj, edges)
        
        # Apply forces with learning rate decay
        centers += lr * forces
        
        # Clamp to rectangle
        for i in range(n):
            centers[i, 0] = np.clip(centers[i, 0], radii[i], w - radii[i])
            centers[i, 1] = np.clip(centers[i, 1], radii[i], h - radii[i])
        
        # Decay learning rate
        lr *= lr_decay
        
        # Gradually increase radii
        radii *= (1.0 + 1e-6)
        
        # Early stopping if forces are small
        if np.max(np.abs(forces)) < 1e-6:
            break
    
    return centers, radii


def _evaluate(centers, w, h, radii):
    """Evaluate validity and objective. Returns (is_valid, objective)."""
    n = len(centers)
    
    # Check bounding box perimeter
    if w + h > 2.0 + 1e-6:
        return False, 0.0
    
    # Check all circles are within the rectangle
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-6 or x + r > w + 1e-6 or y - r < -1e-6 or y + r > h + 1e-6:
            return False, 0.0
    
    # Check no overlap
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - 1e-6:
                return False, 0.0
    
    # Objective: sum of radii
    return True, np.sum(radii)


def _initial_layout(n, w, h):
    """Generate an initial layout of n circles in a rectangle with perimeter 4."""
    # Place circles in a grid pattern initially
    cols = int(np.ceil(np.sqrt(n * w / h)))
    rows = int(np.ceil(n / cols))
    
    centers = []
    for i in range(rows):
        for j in range(cols):
            if len(centers) >= n:
                break
            x = (j + 0.5) * w / cols
            y = (i + 0.5) * h / rows
            centers.append([x, y])
    
    centers = np.array(centers[:n])
    return centers


def construct():
    """
    Pack 21 circles with disjoint interiors in a variable rectangle of perimeter 4
    and maximize the sum of their radii.
    
    Returns: array of shape (n, 3) with [x, y, r] for each circle.
    """
    n = N
    start_time = time.time()
    time_limit = 115  # Leave some margin
    
    best_solution = None
    best_score = -np.inf
    
    # Try multiple initial configurations with different aspect ratios
    aspect_ratios = [
        (1.0, 1.0),      # Square
        (0.8, 1.2),      # Tall
        (1.2, 0.8),      # Wide
        (0.9, 1.1),      # Slightly tall
        (1.1, 0.9),      # Slightly wide
    ]
    
    for trial in range(30):
        if time.time() - start_time > time_limit:
            break
        
        random.seed(trial * 12345 + 678)
        np.random.seed(trial * 12345 + 678)
        
        # Cycle through aspect ratios
        idx = trial % len(aspect_ratios)
        w, h = aspect_ratios[idx]
        
        # Add some randomness to initial positions
        centers = _initial_layout(n, w, h)
        centers += np.random.uniform(-0.02, 0.02, size=centers.shape)
        
        # Initial radii: estimate from grid spacing
        cols = int(np.ceil(np.sqrt(n * w / h)))
        rows = int(np.ceil(n / cols))
        base_r = min(w / cols, h / rows) * 0.4
        radii = np.full(n, base_r)
        
        # Topological relaxation
        centers, radii = _topological_relaxation(centers, w, h, radii, max_iterations=5000, initial_lr=0.03)
        
        # Check validity and get score
        is_valid, score = _evaluate(centers, w, h, radii)
        
        if is_valid and score > best_score:
            best_score = score
            best_solution = np.column_stack([centers, radii])
    
    if best_solution is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_solution
# EVOLVE-BLOCK-END
