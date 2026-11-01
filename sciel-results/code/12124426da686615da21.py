import numpy as np
import time
import random

N = 12
SQRT3 = np.sqrt(3.0)

# Precompute vertex offsets for a unit hexagon at angle 0
BASE_ANGLES = np.radians(np.arange(0, 360, 60))
BASE_COS = np.cos(BASE_ANGLES)
BASE_SIN = np.sin(BASE_ANGLES)
BASE_VERTS = np.column_stack([BASE_COS, BASE_SIN])

def hexagon_vertices(cx, cy, angle_deg):
    """
    Returns (6, 2) array of vertices for a unit hexagon at (cx, cy) with rotation angle_deg.
    """
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    verts = (R @ BASE_VERTS.T).T
    return verts + np.array([cx, cy])

def check_validity(positions, angles, tol=1e-5):
    """
    Check if all hexagons are pairwise disjoint using SAT.
    positions: (N, 2)
    angles: (N,)
    """
    vertices_list = []
    for i in range(N):
        x, y = positions[i]
        verts = hexagon_vertices(x, y, angles[i])
        vertices_list.append(verts)
    
    for i in range(N):
        for j in range(i + 1, N):
            p1 = vertices_list[i]
            p2 = vertices_list[j]
            
            # SAT check
            disjoint = False
            # Check edges of p1
            for i_sat in range(6):
                j_sat = (i_sat + 1) % 6
                edge = p1[j_sat] - p1[i_sat]
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    continue
                normal /= norm
                proj1 = p1.dot(normal)
                proj2 = p2.dot(normal)
                if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                    disjoint = True
                    break
            
            if not disjoint:
                # Check edges of p2
                for i_sat in range(6):
                    j_sat = (i_sat + 1) % 6
                    edge = p2[j_sat] - p2[i_sat]
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm < 1e-10:
                        continue
                    normal /= norm
                    proj1 = p1.dot(normal)
                    proj2 = p2.dot(normal)
                    if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                        disjoint = True
                        break
                if not disjoint:
                    return False
    return True

def required_outer_side(positions, angles):
    """
    Compute the exact side length of the smallest regular hexagon (angle 0, centered at origin)
    that contains all unit hexagons at given positions and angles.
    """
    verts_flat = np.zeros((N * 6, 2))
    for i in range(N):
        verts_flat[i*6:(i+1)*6] = hexagon_vertices(positions[i, 0], positions[i, 1], angles[i])
    
    x = verts_flat[:, 0]
    y = verts_flat[:, 1]
    
    L1 = x + y / SQRT3
    L2 = y * 2.0 / SQRT3
    L3 = -x + y / SQRT3
    L4 = -x - y / SQRT3
    L5 = -y * 2.0 / SQRT3
    L6 = x - y / SQRT3
    
    all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
    return np.max(all_L)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Initial configuration: Hexagonal lattice
    # This is a known good starting point.
    # Centers of hexagons in a lattice with spacing 2 (touching)
    # But we need to fit them in a smaller hexagon.
    # Let's start with a compact cluster.
    
    def get_initial_config():
        # Init 1: Lattice
        coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
        ]
        pos = []
        ang = []
        for q, r in coords:
            x = SQRT3 * (q + r / 2.0)
            y = 1.5 * r
            pos.append([x, y])
            ang.append(0.0)
        return np.array(pos), np.array(ang)

    positions, angles = get_initial_config()
    
    # Ensure initial validity
    if not check_validity(positions, angles):
        # Scale up until valid
        scale = 1.0
        while scale < 5.0:
            pos_scaled = positions * scale
            if check_validity(pos_scaled, angles):
                positions = pos_scaled
                break
            scale += 0.1
    
    current_side = required_outer_side(positions, angles)
    best_positions = positions.copy()
    best_angles = angles.copy()
    best_side = current_side
    
    # Rigid-Body RRT* Strategy
    # State: (x_offset, y_offset, global_rotation_rad)
    # We treat the 12 hexagons as a rigid body.
    # We expand a tree of these states.
    
    class Node:
        def __init__(self, x, y, theta, parent=None, cost=None):
            self.x = x
            self.y = y
            self.theta = theta
            self.parent = parent
            self.cost = cost if cost is not None else 0.0
            self.children = []
            
    root = Node(0.0, 0.0, 0.0)
    root.cost = current_side
    best_node = root
    
    tree_nodes = [root]
    
    # RRT* parameters
    step_size = 1.0
    max_nodes = 20000
    gamma = 3.0 # Rewiring factor
    
    def apply_transform(pos, ang, dx, dy, dtheta):
        # Rotate positions around origin, then translate
        # Note: angles are also rotated
        c, s = np.cos(dtheta), np.sin(dtheta)
        R = np.array([[c, -s], [s, c]])
        
        new_pos = np.zeros_like(pos)
        new_ang = np.zeros_like(ang)
        
        for i in range(N):
            # Rotate position vector
            p = pos[i]
            p_rot = R @ p
            p_new = p_rot + np.array([dx, dy])
            new_pos[i] = p_new
            
            # Rotate angle
            new_ang[i] = (ang[i] + np.degrees(dtheta)) % 360
            
        return new_pos, new_ang

    def steer(node1, node2):
        # Move from node1 towards node2 by step_size
        dx = node2.x - node1.x
        dy = node2.y - node1.y
        dtheta = node2.theta - node1.theta
        
        # Normalize theta difference to [-pi, pi]
        while dtheta > np.pi: dtheta -= 2 * np.pi
        while dtheta < -np.pi: dtheta += 2 * np.pi
        
        dist_xy = np.sqrt(dx*dx + dy*dy)
        
        if dist_xy < 1e-6:
            step_x = 0
            step_y = 0
            step_theta = dtheta
        else:
            # We need to scale the step to be 'step_size' in some metric.
            # Let's use Euclidean distance for x,y and radians for theta.
            # Simple approach: step in x,y is step_size, step in theta is proportional.
            # Actually, let's just step in x,y by step_size and theta by a fixed small amount or proportional.
            # To keep it simple, let's step in x,y by step_size.
            # And step theta by (dtheta/dist_xy) * step_size? No, that couples them.
            # Let's just move x,y by step_size towards target, and theta by a fraction of dtheta.
            
            factor = step_size / dist_xy if dist_xy > step_size else 1.0
            step_x = dx * factor
            step_y = dy * factor
            step_theta = dtheta * factor
            
        return Node(node1.x + step_x, node1.y + step_y, node1.theta + step_theta, parent=node1)

    def get_cost(node):
        pos, ang = apply_transform(positions, angles, node.x, node.y, node.theta)
        if not check_validity(pos, ang):
            return float('inf')
        return required_outer_side(pos, ang)

    def find_nearest(node):
        # Find nearest node in tree to 'node'
        min_dist = float('inf')
        nearest = None
        for n in tree_nodes:
            dx = n.x - node.x
            dy = n.y - node.y
            dtheta = n.theta - node.theta
            # Normalize dtheta
            while dtheta > np.pi: dtheta -= 2 * np.pi
            while dtheta < -np.pi: dtheta += 2 * np.pi
            
            dist = np.sqrt(dx*dx + dy*dy) + 0.1 * abs(dtheta)
            if dist < min_dist:
                min_dist = dist
                nearest = n
        return nearest

    def rewire(new_node):
        # RRT* rewiring: check neighbors within radius and update parent if cost improves
        radius = gamma * step_size
        best_parent = new_node.parent
        best_cost = new_node.cost
        
        # We need to check if connecting to a neighbor reduces cost.
        # Cost is path length + final objective? 
        # In RRT*, cost is usually path length. Here, our "cost" is the objective function value.
        # This is not a standard shortest path problem.
        # However, we can treat the objective as the cost to minimize.
        # If a neighbor has a lower objective, and we can reach it, we might want to re-parent.
        # But since we are doing RRT, we are exploring.
        # Let's just use standard RRT (no star) for simplicity and speed, 
        # but with a bias towards the goal (minimizing side).
        
        # Actually, let's stick to a simple RRT with random sampling biased towards center.
        # The "star" part is complex for non-metric costs.
        pass

    # Main RRT Loop
    while time.time() < deadline and len(tree_nodes) < max_nodes:
        # Sample random state
        # Bias towards origin (center of outer hexagon)
        if random.random() < 0.7:
            # Sample in a disk of radius 5
            r = random.uniform(0, 5)
            theta = random.uniform(0, 2 * np.pi)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
        else:
            # Sample in a larger area
            x = random.uniform(-10, 10)
            y = random.uniform(-10, 10)
            
        dtheta = random.uniform(-np.pi/6, np.pi/6)
        
        target_node = Node(x, y, dtheta)
        
        # Find nearest
        nearest = find_nearest(target_node)
        if nearest is None:
            continue
            
        # Steer
        new_node = steer(nearest, target_node)
        
        # Check validity and cost
        new_node.cost = get_cost(new_node)
        
        if new_node.cost < float('inf'):
            # Valid
            new_node.parent = nearest
            nearest.children.append(new_node)
            tree_nodes.append(new_node)
            
            if new_node.cost < best_side:
                best_side = new_node.cost
                best_node = new_node
                # Update best positions
                pos, ang = apply_transform(positions, angles, new_node.x, new_node.y, new_node.theta)
                best_positions = pos.copy()
                best_angles = ang.copy()
                
    # Final polish: Local search around the best node found
    # We can try to refine the best rigid body transform slightly
    if time.time() < deadline - 5.0:
        # Simple gradient-free local search on (x, y, theta)
        for _ in range(500):
            if time.time() > deadline - 5.0:
                break
            dx = random.uniform(-0.1, 0.1)
            dy = random.uniform(-0.1, 0.1)
            dtheta = random.uniform(-0.05, 0.05)
            
            new_x = best_node.x + dx
            new_y = best_node.y + dy
            new_theta = best_node.theta + dtheta
            
            pos, ang = apply_transform(positions, angles, new_x, new_y, new_theta)
            if not check_validity(pos, ang):
                continue
            
            side = required_outer_side(pos, ang)
            if side < best_side:
                best_side = side
                best_positions = pos.copy()
                best_angles = ang.copy()
                best_node.x = new_x
                best_node.y = new_y
                best_node.theta = new_theta

    # Return best found
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
    
    return inner, (0.0, 0.0), float(best_side), 0.0

# EVOLVE-BLOCK-END
