import numpy as np
import time
from itertools import product

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = 600  # seconds
    
    # Grid dimensions
    # Try 8x7 and 7x8
    best_score = 0
    best_points = None
    best_lines = None
    
    for W, H in [(8, 7), (7, 8)]:
        # Generate grid points
        xs = np.arange(W)
        ys = np.arange(H)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        grid_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        n_grid = grid_points.shape[0]
        
        # Generate candidate lines
        # A*x + B*y + C = 0
        candidates = []
        
        # Vertical lines x = c => 1*x + 0*y - c = 0
        for c in range(W):
            candidates.append((1, 0, -c))
            
        # Horizontal lines y = c => 0*x + 1*y - c = 0
        for c in range(H):
            candidates.append((0, 1, -c))
            
        # Diagonals with slope 1: y - x = c => -1*x + 1*y - c = 0
        # c ranges from -(W-1) to H-1
        for c in range(-(W-1), H):
            candidates.append((-1, 1, -c))
            
        # Diagonals with slope -1: y + x = c => 1*x + 1*y - c = 0
        # c ranges from 0 to W-1+H-1
        for c in range(W+H-1):
            candidates.append((1, 1, -c))
            
        # Add some other low slope lines if we need more candidates
        # Slope 2: 2y - x = c => -1*x + 2*y - c = 0
        # Slope 1/2: y - 2x = c => -2*x + 1*y - c = 0
        # We have 8+7+14+14 = 43 candidates already. We need 27.
        # This is enough.
        
        lines_arr = np.array(candidates, dtype=np.int64)
        
        # Function to count incidences for a set of lines and points
        def count_inc(points, lines):
            # points: (N, 2), lines: (M, 3)
            # Check A*x + B*y + C == 0
            # Vectorized: 
            # A: (M, 1), B: (M, 1), C: (M, 1)
            # x: (1, N), y: (1, N)
            A = lines[:, 0:1]
            B = lines[:, 1:2]
            C = lines[:, 2:3]
            x = points[:, 0:1].T
            y = points[:, 1:2].T
            res = A * x + B * y + C
            return np.sum(res == 0)
        
        # Iterative refinement
        # 1. Initial line selection: top 27 by coverage of full grid
        # Calculate coverage of each line on full grid
        A = lines_arr[:, 0:1]
        B = lines_arr[:, 1:2]
        C = lines_arr[:, 2:3]
        x = grid_points[:, 0:1].T
        y = grid_points[:, 1:2].T
        res = A * x + B * y + C
        line_cov = np.sum(res == 0, axis=1)
        
        # Get indices of top 27 lines
        top_line_indices = np.argsort(line_cov)[-27:]
        current_lines = lines_arr[top_line_indices]
        
        # 2. Initial point selection: top 54 by coverage of current lines
        A = current_lines[:, 0:1]
        B = current_lines[:, 1:2]
        C = current_lines[:, 2:3]
        x = grid_points[:, 0:1].T
        y = grid_points[:, 1:2].T
        res = A * x + B * y + C
        point_cov = np.sum(res == 0, axis=0)
        
        top_point_indices = np.argsort(point_cov)[-54:]
        current_points = grid_points[top_point_indices]
        
        best_local_score = count_inc(current_points, current_lines)
        
        # Iterate
        for _ in range(50):
            if time.time() - start_time > deadline - 10:
                break
                
            # Update lines based on current points
            A = lines_arr[:, 0:1]
            B = lines_arr[:, 1:2]
            C = lines_arr[:, 2:3]
            x = current_points[:, 0:1].T
            y = current_points[:, 1:2].T
            res = A * x + B * y + C
            line_cov = np.sum(res == 0, axis=1)
            
            top_line_indices = np.argsort(line_cov)[-27:]
            new_lines = lines_arr[top_line_indices]
            
            # Update points based on new lines
            A = new_lines[:, 0:1]
            B = new_lines[:, 1:2]
            C = new_lines[:, 2:3]
            x = grid_points[:, 0:1].T
            y = grid_points[:, 1:2].T
            res = A * x + B * y + C
            point_cov = np.sum(res == 0, axis=0)
            
            top_point_indices = np.argsort(point_cov)[-54:]
            new_points = grid_points[top_point_indices]
            
            new_score = count_inc(new_points, new_lines)
            
            if new_score > best_local_score:
                best_local_score = new_score
                current_lines = new_lines
                current_points = new_points
            else:
                # Converged or stuck, try a slight perturbation or break
                # To avoid getting stuck in a local minimum, we can randomize initial lines
                # But for now, let's just break if no improvement
                # Or try a different subset of candidates?
                # Let's just break for this grid size
                break
        
        if best_local_score > best_score:
            best_score = best_local_score
            best_points = current_points
            best_lines = current_lines

    # Return the best found
    # Ensure types are correct
    return best_points, best_lines
