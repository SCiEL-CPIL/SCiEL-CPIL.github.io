# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    """
    Finds a large subset of the 8x8x8 grid with no 5 co-spherical points.
    Uses a greedy initialization followed by simulated annealing to maximize the number of points.
    """
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    
    def can_add(arr: np.ndarray, p: np.ndarray) -> bool:
        """
        Checks if point p can be added to set arr without creating 5 co-spherical points.
        This is equivalent to checking that p does not lie on any sphere defined by 4 points in arr.
        """
        m = len(arr)
        if m < 4:
            return True
        
        # Construct the 4x4 matrix rows for each point in arr relative to p
        # Row i: [q_i - p, |q_i|^2 - |p|^2]
        # If any 4 rows are linearly dependent, the points are co-spherical.
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = (arr ** 2).sum(axis=1) - (p ** 2).sum()
        
        if m == 4:
            return abs(np.linalg.det(w)) >= 0.5
            
        # Generate all combinations of 4 points from arr
        combos = np.array(list(itertools.combinations(range(m), 4)))
        sub = w[combos]  # Shape (C, 4, 4)
        
        # Compute determinants
        dets = np.linalg.det(sub)
        
        # If any determinant is close to 0, the points are co-spherical
        return not np.any(np.abs(dets) < 0.5)

    def greedy_initial(num_restarts=10):
        """
        Generates an initial valid set using a greedy approach with random restarts.
        """
        np.random.seed(42)
        best = None
        for _ in range(num_restarts):
            order = grid[np.random.permutation(len(grid))]
            arr = np.empty((0, 3), dtype=np.int64)
            for p in order:
                if can_add(arr, p):
                    if len(arr) == 0:
                        arr = p[None, :]
                    else:
                        arr = np.vstack([arr, p])
            if best is None or len(arr) > len(best):
                best = arr
        return best

    # Initialize with a good greedy solution
    best_set = greedy_initial()
    current_set = best_set.copy()
    
    # Simulated Annealing Parameters
    T = 1.0
    cooling = 0.9998
    min_T = 0.01
    
    # Deadline guard
    deadline = time.time() + 550
    
    while time.time() < deadline:
        if T < min_T:
            break
            
        # Generate a move
        # 70% chance to try adding a point, 30% chance to try removing one
        if np.random.random() < 0.7:
            # Pick a random point from the grid
            p = grid[np.random.randint(len(grid))]
            
            # Check if p is already in current_set
            # Using np.any with broadcasting
            if len(current_set) > 0 and np.any((current_set == p).all(axis=1)):
                continue # Point already exists, no change
            
            # Check if p can be added
            if can_add(current_set, p):
                # Accept the addition (increases size, always good)
                if len(current_set) == 0:
                    current_set = p[None, :]
                else:
                    current_set = np.vstack([current_set, p])
                
                # Update best if this set is larger
                if len(current_set) > len(best_set):
                    best_set = current_set.copy()
        else:
            # Try to remove a random point to escape local optima
            if len(current_set) == 0:
                continue
                
            idx = np.random.randint(len(current_set))
            
            # Create new set without the point at idx
            if len(current_set) == 1:
                new_set = np.empty((0, 3), dtype=np.int64)
            else:
                new_set = np.delete(current_set, idx, axis=0)
            
            # Accept removal with probability exp(-1/T)
            # This allows the set to shrink temporarily to explore other regions
            if np.random.random() < np.exp(-1.0 / T):
                current_set = new_set
        
        # Cool down
        T *= cooling
        
    return best_set.astype(np.int64)


# EVOLVE-BLOCK-END
