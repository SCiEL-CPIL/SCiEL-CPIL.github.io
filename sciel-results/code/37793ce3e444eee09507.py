"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import linprog

N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    best_sum = 0.0
    best_result = None
    
    # Initial grid placement
    side = int(np.ceil(np.sqrt(N)))
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                positions.append([(i + 0.5) / side, (j + 0.5) / side])
    positions = np.array(positions[:N])
    radii = np.full(N, 0.5 / side - 1e-4)
    
    def compute_contact_graph(pos, rad, eps=0.01):
        """Compute contact graph: circles that are close to touching."""
        n = len(pos)
        contact = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(pos[i] - pos[j])
                # Consider contact if distance is close to sum of radii
                if dist < rad[i] + rad[j] + eps:
                    contact[i, j] = contact[j, i] = True
        return contact
    
    def solve_lp_for_radii(pos, contact):
        """
        Solve LP to maximize sum of radii given positions and contact constraints.
        Variables: r_0, ..., r_{n-1}
        Constraints:
          - r_i + r_j <= ||pos_i - pos_j|| for all contacts
          - r_i <= pos[i][0] (left wall)
          - r_i <= 1 - pos[i][0] (right wall)
          - r_i <= pos[i][1] (bottom wall)
          - r_i <= 1 - pos[i][1] (top wall)
          - r_i >= 0
        """
        n = len(pos)
        
        # Objective: maximize sum(r_i) => minimize -sum(r_i)
        c = -np.ones(n)
        
        # Constraints in form A_ub @ x <= b_ub
        A_ub = []
        b_ub = []
        
        # Pairwise distance constraints for contacts
        for i in range(n):
            for j in range(i+1, n):
                if contact[i, j]:
                    dist = np.linalg.norm(pos[i] - pos[j])
                    row = np.zeros(n)
                    row[i] = 1
                    row[j] = 1
                    A_ub.append(row)
                    b_ub.append(dist)
        
        # Wall constraints
        for i in range(n):
            # Left wall: r_i <= pos[i][0]
            row = np.zeros(n)
            row[i] = 1
            A_ub.append(row)
            b_ub.append(pos[i][0])
            
            # Right wall: r_i <= 1 - pos[i][0]
            row = np.zeros(n)
            row[i] = 1
            A_ub.append(row)
            b_ub.append(1 - pos[i][0])
            
            # Bottom wall: r_i <= pos[i][1]
            row = np.zeros(n)
            row[i] = 1
            A_ub.append(row)
            b_ub.append(pos[i][1])
            
            # Top wall: r_i <= 1 - pos[i][1]
            row = np.zeros(n)
            row[i] = 1
            A_ub.append(1 - pos[i][1])
        
        A_ub = np.array(A_ub) if A_ub else None
        b_ub = np.array(b_ub) if b_ub else None
        
        # Bounds: 0 <= r_i <= 0.5
        bounds = [(0, 0.5)] * n
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                r = -res.x
                # Clip to non-negative
                r = np.maximum(r, 0)
                return r
        except Exception:
            pass
        
        return None
    
    def validate(pos, rad):
        """Check if circles are valid (inside unit square, no overlaps)."""
        for i in range(len(pos)):
            x, y = pos[i]
            r = rad[i]
            if r < 0 or x < -1e-9 or y < -1e-9 or x + r > 1 + 1e-9 or y + r > 1 + 1e-9:
                return False
        for i in range(len(pos)):
            for j in range(i+1, len(pos)):
                dist = np.linalg.norm(pos[i] - pos[j])
                if dist < rad[i] + rad[j] - 1e-9:
                    return False
        return True
    
    def get_score(pos, rad):
        """Compute the sum of radii."""
        return np.sum(rad)
    
    # Greedy optimization loop
    max_iterations = 5000
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Compute contact graph
        contact = compute_contact_graph(positions, radii, eps=0.02)
        
        # Solve LP for optimal radii given current positions
        new_radii = solve_lp_for_radii(positions, contact)
        if new_radii is not None:
            radii = new_radii
        
        # Check validity
        if not validate(positions, radii):
            # If invalid, shrink radii slightly
            radii = radii * 0.99
            continue
        
        current_sum = get_score(positions, radii)
        if current_sum > best_sum:
            best_sum = current_sum
            best_result = np.column_stack([positions, radii]).copy()
        
        # Try to improve positions using gradient-like moves
        # For each circle, try small perturbations
        improved = False
        n = len(positions)
        
        for i in range(n):
            if time.time() > deadline:
                break
            
            # Try small moves in 8 directions
            for dx in [-0.01, 0, 0.01]:
                for dy in [-0.01, 0, 0.01]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    new_pos = positions.copy()
                    new_pos[i] = [positions[i][0] + dx, positions[i][1] + dy]
                    
                    # Check if new position is valid (inside square)
                    if new_pos[i][0] < 0 or new_pos[i][0] > 1 or new_pos[i][1] < 0 or new_pos[i][1] > 1:
                        continue
                    
                    # Recompute contact graph and solve LP
                    new_contact = compute_contact_graph(new_pos, radii, eps=0.02)
                    new_rad = solve_lp_for_radii(new_pos, new_contact)
                    if new_rad is None:
                        continue
                    
                    # Validate
                    if not validate(new_pos, new_rad):
                        continue
                    
                    new_sum = get_score(new_pos, new_rad)
                    if new_sum > current_sum + 1e-8:
                        positions = new_pos
                        radii = new_rad
                        current_sum = new_sum
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        # If no improvement after several iterations, add random perturbation
        if not improved and iteration % 100 == 99:
            # Random restart with small perturbation
            for i in range(n):
                positions[i] = positions[i] + np.random.uniform(-0.05, 0.05, 2)
                positions[i] = np.clip(positions[i], 0, 1)
            # Recompute radii
            contact = compute_contact_graph(positions, radii, eps=0.02)
            new_radii = solve_lp_for_radii(positions, contact)
            if new_radii is not None:
                radii = new_radii
    
    # Final polish: more precise LP solve
    if best_result is not None:
        final_pos = best_result[:, :2]
        final_rad = best_result[:, 2]
        
        # Multiple LP refinements
        for _ in range(20):
            if time.time() > deadline:
                break
            contact = compute_contact_graph(final_pos, final_rad, eps=0.005)
            new_rad = solve_lp_for_radii(final_pos, contact)
            if new_rad is not None:
                final_rad = new_rad
                if not validate(final_pos, final_rad):
                    final_rad = final_rad * 0.999
                    continue
                if get_score(final_pos, final_rad) > get_score(final_pos, final_rad):
                    best_result = np.column_stack([final_pos, final_rad]).copy()
    
    if best_result is not None:
        return best_result
    else:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
# EVOLVE-BLOCK-END
