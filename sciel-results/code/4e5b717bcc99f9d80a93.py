"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Discrete state-space search (Simulated Annealing on a grid) for circle packing.
    """
    n = N
    # Grid dimensions
    # Perimeter 4 => w+h=2. Let's assume square w=1, h=1 for the grid.
    # We can refine aspect ratio later if needed, but square is a good start.
    w, h = 1.0, 1.0
    grid_size = 40  # 40x40 grid points
    # Grid points: x in [0, 1], y in [0, 1]
    # Step = 1 / (grid_size - 1)
    step = 1.0 / (grid_size - 1)
    
    # Generate all grid points
    # We want points slightly inside to allow radius > 0 easily?
    # If a point is at 0, max radius is limited by neighbors.
    # Let's use the grid points directly.
    xs = np.linspace(0.0, 1.0, grid_size)
    ys = np.linspace(0.0, 1.0, grid_size)
    grid_points = np.array([(x, y) for y in ys for x in xs])
    num_points = len(grid_points)
    
    # State: list of indices into grid_points for the n circles
    # Initialize randomly
    current_state = np.random.choice(num_points, n, replace=False)
    
    def calc_radii(state):
        """
        Calculate radii for circles at grid indices in state.
        Returns array of radii.
        """
        radii = np.zeros(n)
        coords = grid_points[state]
        
        for i in range(n):
            x, y = coords[i]
            # Distance to boundaries
            r = min(x, 1.0 - x, y, 1.0 - y)
            # Distance to other circles
            for j in range(n):
                if i == j:
                    continue
                xj, yj = coords[j]
                dist = np.sqrt((x - xj)**2 + (y - yj)**2)
                # Radius is half the distance to avoid overlap
                # But we also need to respect other circles' radii.
                # The "lattice gas" model usually implies equal radii or 
                # radii determined by local density.
                # The prompt says "radii dynamically calculated as the minimum 
                # distance to grid boundaries or neighbors".
                # This usually means r_i = min( dist_to_boundary, 0.5 * min_j dist(i,j) )
                # However, this is not symmetric. If circle A is small, it can be close to B,
                # but B's radius is limited by A?
                # Standard packing: dist(i,j) >= r_i + r_j.
                # If we define r_i = min( boundary, 0.5 * min_j dist(i,j) ),
                # then dist(i,j) >= 2*r_i and dist(i,j) >= 2*r_j is NOT guaranteed 
                # if r_i != r_j.
                # Example: dist = 1. r_i = 0.5, r_j = 0.5. Sum = 1. OK.
                # If dist = 1. r_i = 0.4, r_j = 0.4. Sum = 0.8. OK.
                # But if we calculate r_i based on min dist, and r_j based on min dist,
                # do they overlap?
                # Let r_i = min( b_i, 0.5 * d_i_min ).
                # Let r_j = min( b_j, 0.5 * d_j_min ).
                # We know d_ij is the distance.
                # If d_ij is the min distance for i, then r_i <= 0.5 d_ij.
                # If d_ij is the min distance for j, then r_j <= 0.5 d_ij.
                # Then r_i + r_j <= d_ij.
                # So this definition ensures no overlap!
                
                # So, r_i = min( x_i, 1-x_i, y_i, 1-y_i, 0.5 * min_{j!=i} dist(i,j) )
                
                # Optimization: precompute distances?
                # For N=21, O(N^2) is fine.
                pass
            
            # Recalculate properly
            min_dist = float('inf')
            for j in range(n):
                if i == j:
                    continue
                xj, yj = coords[j]
                dist = np.sqrt((x - xj)**2 + (y - yj)**2)
                if dist < min_dist:
                    min_dist = dist
            
            r = min(r, 0.5 * min_dist)
            radii[i] = r
            
        return radii

    def get_score(state):
        radii = calc_radii(state)
        return np.sum(radii)
    
    current_score = get_score(current_state)
    best_state = current_state.copy()
    best_score = current_score
    
    # Simulated Annealing
    T = 0.05
    T_min = 1e-5
    cooling = 0.995
    steps = 50000
    
    for step in range(steps):
        # Move: pick a random circle, move to a random empty spot
        idx = np.random.randint(n)
        old_pos = current_state[idx]
        
        # Find an empty spot
        # Create a mask of occupied spots
        # For efficiency, we can just pick a random index and check if it's occupied.
        # With 1600 spots and 21 occupied, most picks are empty.
        new_pos = np.random.randint(num_points)
        if new_pos == old_pos:
            continue
        
        # Check if new_pos is occupied by another circle
        # Use a set for fast lookup? Or just check if in state.
        # N=21, checking if new_pos in current_state is O(N).
        if new_pos in current_state:
            continue
            
        # Apply move
        current_state[idx] = new_pos
        
        new_score = get_score(current_state)
        delta = new_score - current_score
        
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_state = current_state.copy()
        else:
            # Revert
            current_state[idx] = old_pos
            
        T *= cooling
        if T < T_min:
            T = T_min
            
    # Finalize
    final_state = best_state
    radii = calc_radii(final_state)
    coords = grid_points[final_state]
    
    result = np.zeros((n, 3))
    for i in range(n):
        result[i, 0] = coords[i, 0]
        result[i, 1] = coords[i, 1]
        result[i, 2] = radii[i]
        
    return result
# EVOLVE-BLOCK-END
