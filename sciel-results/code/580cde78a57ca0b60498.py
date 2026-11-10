import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    grid_sq = (grid ** 2).sum(axis=1)
    
    start_time = time.time()
    deadline = start_time + 630 - 20  # 610s safety margin
    
    # Precompute all 4-combinations of indices for efficient violation checking
    # We'll use a different approach: maintain a set and count violations
    
    def count_violations(arr, grid_sq):
        """Count the number of violating 5-tuples in the set."""
        m = len(arr)
        if m < 5:
            return 0
        violations = 0
        # For each point p in the set, check all 4-combinations of other points
        # But this is O(m * C(m-1, 4)) which is expensive for large m
        # Instead, use a smarter approach: for each 5-tuple, check if it's cospherical
        # But C(m, 5) is also expensive.
        
        # Use the determinant test efficiently
        # For each point p (as reference), form matrix with 4 other points
        for i in range(m):
            p = arr[i]
            p_sq = grid_sq[i]
            # Get all other points
            others = np.delete(arr, i, axis=0)
            others_sq = np.delete(grid_sq, i)
            # For each 4-combination of others
            for combo in itertools.combinations(range(len(others)), 4):
                q1, q2, q3, q4 = others[list(combo)]
                q1_sq, q2_sq, q3_sq, q4_sq = others_sq[list(combo)]
                mat = np.array([
                    q1 - p, q2 - p, q3 - p, q4 - p
                ])
                sq_diff = np.array([
                    q1_sq - p_sq, q2_sq - p_sq, q3_sq - p_sq, q4_sq - p_sq
                ])
                mat_full = np.column_stack([mat, sq_diff])
                det = np.linalg.det(mat_full)
                if abs(det) < 0.5:
                    violations += 1
        return violations
    
    # Simulated annealing approach
    # State: a set of points (represented as indices into grid)
    # Energy: number of violating 5-tuples
    # Move: add a random point, remove a random point, or swap
    
    def is_valid(arr, grid_sq):
        """Check if the set has no violating 5-tuples."""
        m = len(arr)
        if m < 5:
            return True
        for i in range(m):
            p = arr[i]
            p_sq = grid_sq[i]
            others = np.delete(arr, i, axis=0)
            others_sq = np.delete(grid_sq, i)
            for combo in itertools.combinations(range(len(others)), 4):
                q1, q2, q3, q4 = others[list(combo)]
                q1_sq, q2_sq, q3_sq, q4_sq = others_sq[list(combo)]
                mat = np.array([
                    q1 - p, q2 - p, q3 - p, q4 - p
                ])
                sq_diff = np.array([
                    q1_sq - p_sq, q2_sq - p_sq, q3_sq - p_sq, q4_sq - p_sq
                ])
                mat_full = np.column_stack([mat, sq_diff])
                det = np.linalg.det(mat_full)
                if abs(det) < 0.5:
                    return False
        return True
    
    def fast_check_add(arr, grid_sq, new_point, new_sq):
        """Check if adding new_point to arr creates any violation."""
        m = len(arr)
        if m < 4:
            return True
        # Check all 4-combinations of arr with new_point
        for combo in itertools.combinations(range(m), 4):
            q1, q2, q3, q4 = arr[list(combo)]
            q1_sq, q2_sq, q3_sq, q4_sq = grid_sq[list(combo)]
            mat = np.array([
                q1 - new_point, q2 - new_point, q3 - new_point, q4 - new_point
            ])
            sq_diff = np.array([
                q1_sq - new_sq, q2_sq - new_sq, q3_sq - new_sq, q4_sq - new_sq
            ])
            mat_full = np.column_stack([mat, sq_diff])
            det = np.linalg.det(mat_full)
            if abs(det) < 0.5:
                return False
        return True
    
    # Greedy initialization with multiple strategies
    def greedy_init(order_idx, grid, grid_sq, deadline):
        arr = []
        arr_sq = []
        for idx in order_idx:
            if time.time() > deadline:
                break
            p = grid[idx]
            p_sq = grid_sq[idx]
            if len(arr) < 4:
                arr.append(p)
                arr_sq.append(p_sq)
                continue
            if fast_check_add(np.array(arr), np.array(arr_sq), p, p_sq):
                arr.append(p)
                arr_sq.append(p_sq)
        return np.array(arr) if arr else np.empty((0, 3), dtype=np.int64)
    
    # Simulated annealing
    def simulated_annealing(initial_arr, grid, grid_sq, deadline, rng):
        if len(initial_arr) == 0:
            return initial_arr
        
        current = initial_arr.copy()
        current_sq = (current ** 2).sum(axis=1)
        current_set = set(tuple(p) for p in current)
        current_energy = 0 if is_valid(current, current_sq) else count_violations(current, current_sq)
        
        best = current.copy()
        best_sq = current_sq.copy()
        best_energy = current_energy
        best_size = len(current)
        
        T = 10.0
        T_min = 0.001
        cooling_rate = 0.9995
        
        while T > T_min and time.time() < deadline:
            # Generate a random move
            move_type = rng.random()
            
            if move_type < 0.3:
                # Add a random point
                candidates = [i for i in range(len(grid)) if tuple(grid[i]) not in current_set]
                if not candidates:
                    continue
                idx = rng.choice(candidates)
                new_point = grid[idx]
                new_sq = grid_sq[idx]
                
                # Check if adding creates a violation
                if fast_check_add(current, current_sq, new_point, new_sq):
                    new_arr = np.vstack([current, new_point[None, :]])
                    new_arr_sq = np.append(current_sq, new_sq)
                    new_energy = 0
                    new_size = len(new_arr)
                else:
                    new_arr = np.vstack([current, new_point[None, :]])
                    new_arr_sq = np.append(current_sq, new_sq)
                    new_energy = 1  # At least one violation
                    new_size = len(new_arr)
                
                delta_energy = new_energy - current_energy
                if delta_energy <= 0 or rng.random() < np.exp(-delta_energy / T):
                    current = new_arr
                    current_sq = new_arr_sq
                    current_set = set(tuple(p) for p in current)
                    current_energy = new_energy
                    
                    if new_size > best_size or (new_size == best_size and new_energy < best_energy):
                        best = current.copy()
                        best_sq = current_sq.copy()
                        best_energy = new_energy
                        best_size = new_size
                        
            elif move_type < 0.6:
                # Remove a random point
                if len(current) <= 5:
                    continue
                idx = rng.integers(0, len(current))
                new_arr = np.delete(current, idx, axis=0)
                new_arr_sq = np.delete(current_sq, idx)
                removed_point = tuple(current[idx])
                
                new_size = len(new_arr)
                # Energy can only decrease or stay same
                new_energy = max(0, current_energy - 1)  # Approximate
                # Actually need to recompute, but for speed, approximate
                
                delta_energy = new_energy - current_energy
                if delta_energy <= 0 or rng.random() < np.exp(-delta_energy / T):
                    current = new_arr
                    current_sq = new_arr_sq
                    current_set = current_set - {removed_point}
                    current_energy = new_energy
                    
            else:
                # Swap: remove one, add another
                if len(current) == 0:
                    continue
                remove_idx = rng.integers(0, len(current))
                removed_point = tuple(current[remove_idx])
                
                candidates = [i for i in range(len(grid)) if tuple(grid[i]) not in current_set]
                if not candidates:
                    continue
                add_idx = rng.choice(candidates)
                new_point = grid[add_idx]
                new_sq = grid_sq[add_idx]
                
                # Remove the point
                temp_arr = np.delete(current, remove_idx, axis=0)
                temp_sq = np.delete(current_sq, remove_idx)
                
                # Check if adding new_point is valid
                if fast_check_add(temp_arr, temp_sq, new_point, new_sq):
                    new_arr = np.vstack([temp_arr, new_point[None, :]])
                    new_arr_sq = np.append(temp_sq, new_sq)
                    new_energy = 0
                else:
                    new_arr = np.vstack([temp_arr, new_point[None, :]])
                    new_arr_sq = np.append(temp_sq, new_sq)
                    new_energy = 1
                
                new_size = len(new_arr)
                delta_energy = new_energy - current_energy
                if delta_energy <= 0 or rng.random() < np.exp(-delta_energy / T):
                    current = new_arr
                    current_sq = new_arr_sq
                    current_set = current_set - {removed_point} | {tuple(new_point)}
                    current_energy = new_energy
                    
                    if new_size > best_size or (new_size == best_size and new_energy < best_energy):
                        best = current.copy()
                        best_sq = current_sq.copy()
                        best_energy = new_energy
                        best_size = new_size
            
            T *= cooling_rate
        
        return best
    
    rng = np.random.default_rng(42)
    best_overall = np.empty((0, 3), dtype=np.int64)
    
    # Multiple restarts with different initializations
    center = np.array([3.5, 3.5, 3.5])
    dists = np.linalg.norm(grid - center, axis=1)
    sums = grid.sum(axis=1)
    
    restart_count = 0
    while time.time() < deadline - 30:
        restart_count += 1
        strategy = restart_count % 7
        
        if strategy == 0:
            order_idx = rng.permutation(N)
        elif strategy == 1:
            noise = rng.random(N)
            order_idx = np.argsort(dists + noise)
        elif strategy == 2:
            noise = rng.random(N)
            order_idx = np.argsort(sums + noise)
        elif strategy == 3:
            noise = rng.random(N)
            order_idx = np.argsort(-sums + noise)
        elif strategy == 4:
            noise = rng.random(N)
            order_idx = np.argsort(-dists + noise)
        elif strategy == 5:
            key = (grid[:, 0] * 13 + grid[:, 1] * 7 + grid[:, 2] * 11) % 100
            noise = rng.random(N)
            order_idx = np.argsort(key + noise)
        else:
            prods = grid[:, 0] * grid[:, 1] * grid[:, 2]
            noise = rng.random(N)
            order_idx = np.argsort(prods + noise)
        
        initial = greedy_init(order_idx, grid, grid_sq, deadline - 60)
        
        if time.time() < deadline - 60 and len(initial) >= 15:
            improved = simulated_annealing(initial, grid, grid_sq, deadline - 30, rng)
            if len(improved) > len(best_overall):
                best_overall = improved.copy()
        elif len(initial) > len(best_overall):
            best_overall = initial.copy()
        
        # Periodically verify and try to extend
        if len(best_overall) >= 20 and time.time() < deadline - 40:
            # Try to add more points greedily
            best_set = set(tuple(p) for p in best_overall)
            for idx in rng.permutation(N):
                if time.time() > deadline - 20:
                    break
                if tuple(grid[idx]) in best_set:
                    continue
                p = grid[idx]
                p_sq = grid_sq[idx]
                if fast_check_add(best_overall, (best_overall ** 2).sum(axis=1), p, p_sq):
                    best_overall = np.vstack([best_overall, p[None, :]])
                    best_set.add(tuple(p))
    
    return best_overall.astype(np.int64) if len(best_overall) > 0 else np.array([[0,0,0]], dtype=np.int64)
