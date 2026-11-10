# EVOLVE-BLOCK-START
import time
import numpy as np


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set N in F_49^2 with q = 49, p = 7.
    Maximize |N^c| = |S| (complement size).
    """
    p = 7
    q = p * p
    NPTS = q * q  # 2401

    start = time.time()
    deadline = start + 580.0  # safety margin before 630s kill

    # --- Field F_p[alpha], alpha^2 = 3 ---
    k_const = 3

    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)

    def mul(a, b):
        return ((a[0] * b[0] + a[1] * b[1] * k_const) % p,
                (a[0] * b[1] + a[1] * b[0]) % p)

    def pt_to_idx(pt):
        xa, xb = pt[0]
        ya, yb = pt[1]
        return xa * 343 + xb * 49 + ya * 7 + yb

    def idx_to_pt(idx):
        xa = idx // 343
        idx %= 343
        xb = idx // 49
        idx %= 49
        ya = idx // 7
        yb = idx % 7
        return [[xa, xb], [ya, yb]]

    # --- Directions: (1, m) for all m in F_q, plus (0, 1) ---
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))
    directions.append(((0, 0), (1, 0)))
    num_dirs = len(directions)  # 50

    # --- Precompute affected[p, d, t] ---
    # affected[i, d, j] is the index of the point on the line through i in direction d, at offset j
    affected = np.zeros((NPTS, num_dirs, q - 1), dtype=np.int32)
    for p_idx in range(NPTS):
        px, py = idx_to_pt(p_idx)[0], idx_to_pt(p_idx)[1]
        for d_idx in range(num_dirs):
            vx, vy = directions[d_idx]
            count = 0
            for ta in range(p):
                for tb in range(p):
                    if ta == 0 and tb == 0:
                        continue
                    tvx = mul((ta, tb), vx)
                    tvy = mul((ta, tb), vy)
                    nx = add(px, tvx)
                    ny = add(py, tvy)
                    affected[p_idx, d_idx, count] = pt_to_idx([[nx[0], nx[1]], [ny[0], ny[1]]])
                    count += 1

    # --- State arrays ---
    in_S = np.zeros(NPTS, dtype=np.int32)
    line_count = np.zeros((NPTS, num_dirs), dtype=np.int32)
    zero_counts = np.zeros(NPTS, dtype=np.int32)

    def reset_state():
        in_S[:] = 0
        line_count[:] = 0
        zero_counts[:] = num_dirs

    def is_valid():
        return np.all(zero_counts > 0)

    def try_add(p_idx):
        """
        Try to add p_idx to S.
        Returns list of (i, d) changes to apply, or None if invalid.
        """
        changes = []
        for d in range(num_dirs):
            for i in affected[p_idx, d, :]:
                if line_count[i, d] == 0:
                    if zero_counts[i] == 1:
                        return None
                    changes.append((i, d))
        return changes

    def apply_add(p_idx, changes):
        in_S[p_idx] = 1
        for i, d in changes:
            line_count[i, d] += 1
            zero_counts[i] -= 1

    def try_remove(p_idx):
        """
        Try to remove p_idx from S.
        Returns list of (i, d) changes to apply.
        Removal is always valid (only helps).
        """
        changes = []
        for d in range(num_dirs):
            for i in affected[p_idx, d, :]:
                if line_count[i, d] > 0:
                    if line_count[i, d] == 1:
                        changes.append((i, d))
        return changes

    def apply_remove(p_idx, changes):
        in_S[p_idx] = 0
        for i, d in changes:
            line_count[i, d] -= 1
            if line_count[i, d] == 0:
                zero_counts[i] += 1

    def get_add_cost(p_idx):
        """
        Returns the number of (i, d) pairs where line_count[i, d] == 0.
        This is the cost of adding p_idx (reduction in slack).
        """
        cost = 0
        for d in range(num_dirs):
            for i in affected[p_idx, d, :]:
                if line_count[i, d] == 0:
                    cost += 1
        return cost

    def greedy_add_pass(order):
        """
        Greedily add points in the given order if possible.
        Returns number of points added.
        """
        added = 0
        for p_idx in order:
            if time.time() > deadline - 10:
                break
            if in_S[p_idx]:
                continue
            changes = try_add(p_idx)
            if changes is not None:
                apply_add(p_idx, changes)
                added += 1
        return added

    def local_search_with_swaps(max_iterations=500):
        """
        Local search: try to improve by swapping (remove p, add q) or just adding.
        Returns new size of S.
        """
        for _ in range(max_iterations):
            if time.time() > deadline - 10:
                break
            improved = False
            
            # Try to add any point
            for p_idx in np.random.permutation(NPTS):
                if in_S[p_idx]:
                    continue
                changes = try_add(p_idx)
                if changes is not None:
                    apply_add(p_idx, changes)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swaps: remove p, add q
            # Only consider removing points that have high "cost" (many zero lines)
            # and adding points that have low cost
            candidates_remove = []
            for p_idx in range(NPTS):
                if in_S[p_idx]:
                    # Count how many zero lines this point is on (i.e., how many i, d have line_count[i,d]==1 and this point is on that line)
                    # Actually, removing p_idx decreases line_count[i,d] for i in affected[p_idx, d, :]
                    # The "cost" of keeping p_idx is the number of (i,d) where it's the only S point
                    cost = 0
                    for d in range(num_dirs):
                        for i in affected[p_idx, d, :]:
                            if line_count[i, d] == 1:
                                cost += 1
                    candidates_remove.append((cost, p_idx))
            
            candidates_remove.sort(reverse=True)
            
            for cost_r, p_idx in candidates_remove[:20]:
                if time.time() > deadline - 10:
                    break
                # Try removing p_idx
                rem_changes = try_remove(p_idx)
                # Temporarily apply removal
                in_S[p_idx] = 0
                for i, d in rem_changes:
                    line_count[i, d] -= 1
                    if line_count[i, d] == 0:
                        zero_counts[i] += 1
                
                # Try adding some q
                for q_idx in np.random.permutation(NPTS):
                    if in_S[q_idx]:
                        continue
                    add_changes = try_add(q_idx)
                    if add_changes is not None:
                        apply_add(q_idx, add_changes)
                        improved = True
                        break
                
                if not improved:
                    # Undo removal
                    in_S[p_idx] = 1
                    for i, d in rem_changes:
                        line_count[i, d] += 1
                        if line_count[i, d] == 1:
                            zero_counts[i] -= 1
                else:
                    break
            
            if not improved:
                break
        
        return np.sum(in_S)

    # --- Main optimization loop ---
    best_S_size = 0
    best_in_S = None
    
    np.random.seed(12345)
    
    # Phase 1: Multiple greedy restarts with different orderings
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        reset_state()
        
        # Different random orderings
        order = np.random.permutation(NPTS)
        
        added = greedy_add_pass(order)
        
        # Additional passes to catch more
        for _ in range(2):
            if time.time() > deadline - 30:
                break
            order2 = np.random.permutation(NPTS)
            greedy_add_pass(order2)
        
        current_size = np.sum(in_S)
        if current_size > best_S_size:
            best_S_size = current_size
            best_in_S = in_S.copy()
            best_line_count = line_count.copy()
            best_zero_counts = zero_counts.copy()
        
        # Local search with swaps
        if time.time() < deadline - 30:
            local_search_with_swaps(max_iterations=200)
            current_size = np.sum(in_S)
            if current_size > best_S_size:
                best_S_size = current_size
                best_in_S = in_S.copy()
                best_line_count = line_count.copy()
                best_zero_counts = zero_counts.copy()

    # Phase 2: Simulated annealing-like approach on the best solution
    if best_in_S is not None and time.time() < deadline - 30:
        in_S = best_in_S.copy()
        line_count = best_line_count.copy()
        zero_counts = best_zero_counts.copy()
        
        # Try to improve further with more aggressive local search
        for _ in range(100):
            if time.time() > deadline - 10:
                break
            
            # Randomly remove a point, then try to add others
            p_idx = np.random.randint(NPTS)
            if in_S[p_idx]:
                rem_changes = try_remove(p_idx)
                in_S[p_idx] = 0
                for i, d in rem_changes:
                    line_count[i, d] -= 1
                    if line_count[i, d] == 0:
                        zero_counts[i] += 1
                
                # Try to add back or add different points
                for q_idx in np.random.permutation(NPTS):
                    if in_S[q_idx]:
                        continue
                    add_changes = try_add(q_idx)
                    if add_changes is not None:
                        apply_add(q_idx, add_changes)
                        break
        
        current_size = np.sum(in_S)
        if current_size > best_S_size:
            best_S_size = current_size
            best_in_S = in_S.copy()

    # Use the best found S
    if best_in_S is None:
        in_S = np.zeros(NPTS, dtype=np.int32)
    else:
        in_S = best_in_S

    # Construct N = complement of S
    arr = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    idx = pt_to_idx([[xa, xb], [ya, yb]])
                    if in_S[idx] == 0:
                        arr.append([[xa % p, xb % p], [ya % p, yb % p]])
    
    return np.array(arr, dtype=np.int64)

# EVOLVE-BLOCK-END
