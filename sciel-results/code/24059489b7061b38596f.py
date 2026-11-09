# EVOLVE-BLOCK-START
import time
import numpy as np

def _precompute(n=128, max_colors=4):
    """Precompute for each position the list of (indices_tuple, required_colors) constraints."""
    n2 = n - 1
    constraints = [[] for _ in range(n2)]
    for a in range(n2):
        for d in range(1, (n2 - a) // 3 + 1):
            i0 = a
            i1 = a + d
            i2 = a + 2 * d
            i3 = a + 3 * d
            # For each position, record the other 3 positions and required colors
            constraints[i0].append(((i1, i2, i3), 2))
            constraints[i1].append(((i0, i2, i3), 2))
            constraints[i2].append(((i0, i1, i3), 2))
            constraints[i3].append(((i0, i1, i2), 2))
    return constraints

def _count_colors(arr):
    return len(set(arr.tolist()))

def construct():
    t_start = time.time()
    deadline = t_start + 85.0  # 5s safety margin
    n = 128
    n2 = n - 1
    
    constraints = _precompute(n2)
    
    # Precompute constraint arrays for vectorized checking
    # For each position, store as arrays
    cons_idx = [np.array([c[0] for c in cl], dtype=np.int32) if cl else np.empty((0,3), dtype=np.int32) for cl in constraints]
    cons_req = [np.array([c[1] for c in cl], dtype=np.int32) if cl else np.empty(0, dtype=np.int32) for cl in constraints]
    
    def is_valid(arr):
        """Check if all constraints are satisfied. Returns True if valid."""
        for i in range(n2):
            if not cons_idx[i].size:
                continue
            others = arr[cons_idx[i]]  # shape (k, 3)
            # For each constraint, count distinct colors among the 3 others
            # Must be >= 2 (i.e., not all same)
            # Check: for each row, are all 3 elements equal?
            all_same = np.all(others == others[:, 0:1], axis=1)
            if np.any(all_same):
                return False
        return True
    
    def is_valid_fast(arr):
        """Faster validity check using the precomputed structure."""
        for i in range(n2):
            idx = cons_idx[i]
            if not idx.size:
                continue
            # Get the colors at the other positions
            c0 = arr[idx[:, 0]]
            c1 = arr[idx[:, 1]]
            c2 = arr[idx[:, 2]]
            # If any triplet has all three equal, invalid
            if np.any((c0 == c1) & (c1 == c2)):
                return False
        return True
    
    # Start with a valid 4-coloring
    # Use a simple pattern that avoids 4-term monochromatic progressions
    # Pattern based on base-4 representation
    arr = np.zeros(n2, dtype=np.int32)
    for i in range(n2):
        # Use a pattern that's known to work
        # Simple: cycle through 0,1,2,3 but with some variation
        arr[i] = i % 4
    
    # Check if valid, if not, try other patterns
    if not is_valid_fast(arr):
        # Try pattern based on van der Waerden construction
        for i in range(n2):
            arr[i] = (i % 4) + ((i // 4) % 2) * 2
        arr = np.minimum(arr, 3)
    
    if not is_valid_fast(arr):
        # Try more patterns
        patterns = [
            lambda i: i % 4,
            lambda i: (i % 4) + ((i // 8) % 2),
            lambda i: ((i % 2) * 2) + ((i // 2) % 2),
            lambda i: (i * 7) % 4,
            lambda i: (i * 3) % 4,
            lambda i: (i % 2) + 2 * ((i // 2) % 2),
        ]
        for pat in patterns:
            arr = np.array([pat(i) for i in range(n2)], dtype=np.int32)
            if is_valid_fast(arr):
                break
    
    if not is_valid_fast(arr):
        # Use greedy construction as fallback
        arr = np.zeros(n2, dtype=np.int32)
        for i in range(n2):
            for c in range(4):
                arr[i] = c
                if is_valid_fast(arr):
                    break
                arr[i] = 0
    
    best_arr = arr.copy()
    best_count = _count_colors(arr)
    
    # Local search: try to reduce colors
    # First, try to eliminate each color
    for target_color in range(3, -1, -1):  # Try to eliminate color 3, then 2, etc.
        if target_color not in set(arr.tolist()):
            continue
        # Try to reassign all positions with target_color to other colors
        positions_with_target = np.where(arr == target_color)[0]
        if len(positions_with_target) == 0:
            continue
        
        # Try one by one
        improved = True
        while improved:
            improved = False
            for pos in positions_with_target:
                if arr[pos] != target_color:
                    continue
                for new_color in range(4):
                    if new_color == target_color:
                        continue
                    arr[pos] = new_color
                    if is_valid_fast(arr):
                        improved = True
                        break
                if not improved:
                    arr[pos] = target_color
        
        # Check if we can now eliminate the target color
        if target_color not in set(arr.tolist()):
            best_arr = arr.copy()
            best_count = _count_colors(arr)
            break
        else:
            # Restore and try next
            pass
    
    # Now try local search to reduce further
    # Simulated annealing style
    current_arr = best_arr.copy()
    current_count = best_count
    
    for step in range(20000):
        if time.time() > deadline:
            break
        
        # Pick a random position
        pos = np.random.randint(n2)
        old_color = current_arr[pos]
        
        # Try changing to a random other color
        new_color = np.random.randint(4)
        if new_color == old_color:
            continue
        
        current_arr[pos] = new_color
        
        if is_valid_fast(current_arr):
            new_count = _count_colors(current_arr)
            if new_count < current_count:
                current_count = new_count
                best_arr = current_arr.copy()
                best_count = new_count
                if best_count <= 3:
                    break
            # Accept with some probability even if same count (for exploration)
        else:
            # Reject
            current_arr[pos] = old_color
    
    # Final check
    if not is_valid_fast(best_arr):
        # Fallback to greedy
        arr = np.zeros(n2, dtype=np.int32)
        for i in range(n2):
            for c in range(4):
                arr[i] = c
                if is_valid_fast(arr):
                    break
        best_arr = arr
    
    return best_arr.tolist()
# EVOLVE-BLOCK-END
