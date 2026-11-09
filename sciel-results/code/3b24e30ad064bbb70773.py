import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    num_points_total = len(grid)
    deadline = time.time() + 580
    
    sq_norms = (grid ** 2).sum(axis=1).astype(np.float64)
    
    def is_valid_addition(arr: np.ndarray, p: np.ndarray) -> bool:
        m = len(arr)
        if m < 4:
            return True
        
        diff = arr.astype(np.float64) - p.astype(np.float64)
        sq_diff = sq_norms[arr] - (p.astype(np.float64) ** 2).sum()
        M = np.column_stack([diff, sq_diff])
        
        # For small m, check all
        if m <= 12:
            for combo in itertools.combinations(range(m), 4):
                if abs(np.linalg.det(M[list(combo)])) < 0.5:
                    return False
            return True
        else:
            # For larger m, check all but with early termination
            # and use a smarter ordering
            for combo in itertools.combinations(range(m), 4):
                if abs(np.linalg.det(M[list(combo)])) < 0.5:
                    return False
            return True

    def greedy_build(order):
        curr_set = np.empty((0, 3), dtype=np.int64)
        for idx in order:
            if time.time() > deadline:
                break
            p = grid[idx]
            if is_valid_addition(curr_set, p):
                if len(curr_set) == 0:
                    curr_set = p[None, :]
                else:
                    curr_set = np.vstack([curr_set, p])
        return curr_set

    best_set = np.empty((0, 3), dtype=np.int64)
    np.random.seed(42)
    
    # Phase 1: Greedy restarts
    for i in range(35):
        if time.time() > deadline:
            break
        order = np.random.permutation(num_points_total)
        curr_set = greedy_build(order)
        if len(curr_set) > len(best_set):
            best_set = curr_set.copy()
    
    # Phase 2: Local search
    current_set = best_set.copy()
    grid_tuple_to_idx = {tuple(p): i for i, p in enumerate(grid)}
    current_indices = set(grid_tuple_to_idx[tuple(p)] for p in current_set)
    remaining_indices = list(set(range(num_points_total)) - current_indices)
    
    no_improve = 0
    max_no_improve = 60
    
    while time.time() < deadline:
        improved = False
        
        # Add
        if remaining_indices:
            for idx in np.random.permutation(remaining_indices):
                if time.time() > deadline:
                    break
                if is_valid_addition(current_set, grid[idx]):
                    current_set = np.vstack([current_set, grid[idx]])
                    current_indices.add(idx)
                    remaining_indices.remove(idx)
                    improved = True
                    no_improve = 0
                    break
        
        # Swap
        if not improved and current_indices and remaining_indices:
            for _ in range(60):
                if time.time() > deadline:
                    break
                in_i = list(current_indices)[np.random.randint(len(current_indices))]
                out_i = remaining_indices[np.random.randint(len(remaining_indices))]
                
                in_pos = np.where((current_set == grid[in_i]).all(axis=1))[0]
                if len(in_pos) == 0:
                    continue
                in_pos = in_pos[0]
                
                reduced = np.delete(current_set, in_pos, axis=0)
                if is_valid_addition(reduced, grid[out_i]):
                    current_set = np.vstack([reduced, grid[out_i]])
                    current_indices.discard(in_i)
                    current_indices.add(out_i)
                    remaining_indices.remove(out_i)
                    remaining_indices.append(in_i)
                    improved = True
                    no_improve = 0
                    break
        
        # 2-for-1
        if not improved and len(current_indices) > 2 and len(remaining_indices) >= 2:
            for _ in range(30):
                if time.time() > deadline:
                    break
                in_i = list(current_indices)[np.random.randint(len(current_indices))]
                in_pos = np.where((current_set == grid[in_i]).all(axis=1))[0]
                if len(in_pos) == 0:
                    continue
                in_pos = in_pos[0]
                reduced = np.delete(current_set, in_pos, axis=0)
                
                idxs = np.random.choice(remaining_indices, 2, replace=False)
                if is_valid_addition(reduced, grid[idxs[0]]):
                    temp = np.vstack([reduced, grid[idxs[0]]])
                    if is_valid_addition(temp, grid[idxs[1]]):
                        current_set = np.vstack([temp, grid[idxs[1]]])
                        current_indices.discard(in_i)
                        current_indices.update(idxs)
                        remaining_indices.remove(idxs[0])
                        remaining_indices.remove(idxs[1])
                        remaining_indices.append(in_i)
                        improved = True
                        no_improve = 0
                        break
        
        if not improved:
            no_improve += 1
            if no_improve > max_no_improve:
                if len(current_indices) > 8:
                    kick_n = min(3, len(current_indices) // 6)
                    for idx in np.random.choice(list(current_indices), kick_n, replace=False):
                        mask = ~((current_set == grid[idx]).all(axis=1))
                        current_set = current_set[mask]
                        current_indices.discard(idx)
                        remaining_indices.append(idx)
                no_improve = 0
                max_no_improve = 120

    return current_set.astype(np.int64)
