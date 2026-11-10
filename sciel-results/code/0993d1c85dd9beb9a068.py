import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    sq_norms = (grid ** 2).sum(axis=1)
    
    # Precompute indices for faster access
    # We will represent the current set as a boolean mask of size N
    # Or a list of indices. A list is easier for removal/addition logic in Python if N is small (512).
    
    def can_add_to_set(points_indices, new_point_idx):
        """
        Check if new_point_idx can be added to the set of points given by indices in points_indices.
        Returns True if no 5 points (including new_point_idx) are cospherical.
        """
        if len(points_indices) < 4:
            return True
        
        p_new = grid[new_point_idx]
        sq_new = sq_norms[new_point_idx]
        
        # We need to check all combinations of 4 points from points_indices
        # with p_new.
        # The 5 points are cospherical iff the determinant of the 5x5 matrix is 0.
        # Matrix rows: [x, y, z, x^2+y^2+z^2, 1]
        # Alternatively, using the relative vector method from the seed:
        # For any 4 points q1..q4 in the set, form 4x4 matrix with rows (q - p_new, |q|^2 - |p_new|^2).
        # If |det| < 0.5, invalid.
        
        w = np.empty((len(points_indices), 4), dtype=np.float64)
        w[:, 0:3] = grid[points_indices] - p_new
        w[:, 3] = sq_norms[points_indices] - sq_new
        
        # Check all 4-combinations
        # To speed this up, we can vectorize the determinant calculation if we generate combinations.
        # C(40, 4) is ~91k. C(100, 4) is ~3.9M. 
        # If k is small, itertools is fine. If k grows, we need to be careful.
        # Given we expect k ~ 40-50, C(50,4) ~ 230k. This is fast in Python loops? 
        # No, 230k determinant calculations in a Python loop might be slow (~1-2s).
        # We should vectorize.
        
        combos = np.array(list(itertools.combinations(range(len(points_indices)), 4)))
        if len(combos) == 0:
            return True
            
        sub = w[combos] # (C, 4, 4)
        dets = np.linalg.det(sub)
        return not np.any(np.abs(dets) < 0.5)

    def is_valid_set(indices):
        """
        Check if the set of points given by indices is valid.
        """
        # We can just check all 5-combinations? No, that's O(k^5).
        # But we build it incrementally. 
        # For the initial check of a greedy set, we can just assume it's valid if built correctly.
        # For swaps, we use can_add_to_set logic on the reduced set.
        pass

    # Greedy Initialization
    def greedy_init(seed):
        np.random.seed(seed)
        order = np.random.permutation(N)
        selected_indices = []
        for idx in order:
            if can_add_to_set(selected_indices, idx):
                selected_indices.append(idx)
        return selected_indices

    # Local Search
    def local_search(initial_indices, max_time):
        current_indices = list(initial_indices)
        current_set = set(current_indices)
        best_indices = list(current_indices)
        best_size = len(current_indices)
        
        start_time = time.time()
        no_improve_count = 0
        
        while time.time() - start_time < max_time:
            # 1. Try to add any point
            added = False
            # Sort candidates by some heuristic? Or random?
            # Random is better for diversity.
            candidates = [i for i in range(N) if i not in current_set]
            np.random.shuffle(candidates)
            
            for idx in candidates:
                if can_add_to_set(current_indices, idx):
                    current_indices.append(idx)
                    current_set.add(idx)
                    added = True
                    if len(current_indices) > best_size:
                        best_indices = list(current_indices)
                        best_size = len(current_indices)
                        no_improve_count = 0
                    break # Add one, then restart scan? Or continue? 
                    # Restarting scan ensures we always try to add the "easiest" or next valid one.
                    # Actually, just break and loop again.
            
            if added:
                continue # Restart the loop to try adding more
            
            # 2. No point could be added. Try 1-swap.
            swapped = False
            # Pick a random point to remove
            out_idx = np.random.choice(current_indices)
            out_set = current_set - {out_idx}
            remaining_indices = [i for i in current_indices if i != out_idx]
            
            # Try to add a random point from candidates
            candidates = [i for i in range(N) if i not in out_set]
            np.random.shuffle(candidates)
            
            for in_idx in candidates:
                if can_add_to_set(remaining_indices, in_idx):
                    # Swap is valid
                    current_indices = remaining_indices + [in_idx]
                    current_set = out_set | {in_idx}
                    swapped = True
                    # Size is same, so best_size doesn't change, but we might unblock additions
                    # Reset no_improve_count partially?
                    # Let's keep a counter for swaps without size increase
                    no_improve_count += 1
                    break
            
            if not swapped:
                # Try a few more random out_idx if first one failed
                # Or just break and restart the whole local search phase?
                # For now, if no swap found, we are stuck.
                # We can try 2-swaps here if time permits, but let's keep it simple first.
                # If we are stuck, we might need to accept a smaller set or restart.
                # But we want to maximize.
                # Let's try a different out_idx
                # To avoid infinite loop on stuck state, we limit attempts.
                # Actually, if can_add_to_set fails for all candidates for a specific out_idx, 
                # it doesn't mean all out_idx fail.
                # Let's try 10 random swaps.
                found_swap = False
                for _ in range(10):
                    out_idx = np.random.choice(current_indices)
                    out_set = current_set - {out_idx}
                    remaining_indices = [i for i in current_indices if i != out_idx]
                    candidates = [i for i in range(N) if i not in out_set]
                    np.random.shuffle(candidates)
                    for in_idx in candidates[:20]: # Limit candidates to check for speed
                        if can_add_to_set(remaining_indices, in_idx):
                            current_indices = remaining_indices + [in_idx]
                            current_set = out_set | {in_idx}
                            found_swap = True
                            no_improve_count += 1
                            break
                    if found_swap:
                        break
                
                if not found_swap:
                    # Try more aggressive random swaps before breaking
                    # We might be in a deep local optimum
                    found_swap = False
                    for _ in range(50):
                        out_idx = np.random.choice(current_indices)
                        out_set = current_set - {out_idx}
                        remaining_indices = [i for i in current_indices if i != out_idx]
                        candidates = [i for i in range(N) if i not in out_set]
                        np.random.shuffle(candidates)
                        for in_idx in candidates[:50]: # Limit candidates to check for speed
                            if can_add_to_set(remaining_indices, in_idx):
                                current_indices = remaining_indices + [in_idx]
                                current_set = out_set | {in_idx}
                                found_swap = True
                                no_improve_count += 1
                                break
                        if found_swap:
                            break
                    if not found_swap:
                        break

        return best_indices

    # Run multiple restarts
    best_overall = []
    start_total = time.time()
    time_limit = 600.0
    safety_margin = 30.0 # seconds
    
    num_restarts = 0
    # Strategy: Use larger time slices for initial restarts to allow deeper search.
    # First restart: 300s. Second: 150s. Third: 75s. Etc.
    # This focuses effort on the most promising (earliest) good solutions.
    
    while time.time() - start_total < time_limit - safety_margin:
        num_restarts += 1
        seed = num_restarts
        init_indices = greedy_init(seed)
        
        # Allocate remaining time to local search
        elapsed = time.time() - start_total
        remaining = time_limit - safety_margin - elapsed
        if remaining < 10:
            break
        
        # Dynamic time allocation: 300s, 150s, 75s, 37s...
        # Cap at remaining time.
        base_time = 300.0 / (2 ** (num_restarts - 1))
        ls_time = min(base_time, remaining)
        
        if ls_time < 1:
            break
            
        search_result = local_search(init_indices, ls_time)
        
        if len(search_result) > len(best_overall):
            best_overall = search_result
            
            # Deterministic Polish Pass
            current = list(best_overall)
            current_set = set(current)
            for idx in range(N):
                if idx not in current_set:
                    if can_add_to_set(current, idx):
                        current.append(idx)
                        current_set.add(idx)
                        
            if len(current) > len(best_overall):
                best_overall = current

    # Final polish on the absolute best
    if best_overall:
        current = list(best_overall)
        current_set = set(current)
        # Try adding in sorted order
        for idx in range(N):
            if idx not in current_set:
                if can_add_to_set(current, idx):
                    current.append(idx)
                    current_set.add(idx)
        best_overall = current

    if not best_overall:
        return np.empty((0, 3), dtype=np.int64)
        
    return grid[best_overall].astype(np.int64)
