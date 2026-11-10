# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    
    Strategy:
    1. Precompute field operations for F_49 = F_7[alpha], alpha^2 = 3.
    2. Precompute line structures for all 50 directions.
    3. Start with a valid Nikodym set (complement of a known valid set).
    4. Use a greedy algorithm to minimize N by removing points while maintaining validity.
    5. Use local search / simulated annealing to further optimize.
    
    Key insight: We want to MAXIMIZE the complement S = F_49^2 \ N.
    A point can be removed from N (added to S) if for every point x, 
    there still exists a direction d such that all other points on line_d(x) are in N.
    """
    start_time = time.time()
    deadline = 620.0  # Safety margin before 630s kill

    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3, non-residue mod 7

    # Precompute field operations tables
    def idx(a, b):
        return a + b * p

    def a_of(i):
        return i % p

    def b_of(i):
        return i // p

    # Multiplication table
    mul_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ai, bi = a_of(i), b_of(i)
        for j in range(q):
            aj, bj = a_of(j), b_of(j)
            res_a = (ai * aj + bi * bj * k) % p
            res_b = (ai * bj + bi * aj) % p
            mul_table[i, j] = idx(res_a, res_b)

    # Addition table
    add_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ai, bi = a_of(i), b_of(i)
        for j in range(q):
            aj, bj = a_of(j), b_of(j)
            res_a = (ai + aj) % p
            res_b = (bi + bj) % p
            add_table[i, j] = idx(res_a, res_b)

    # Negation table
    neg_table = np.zeros(q, dtype=np.int32)
    for i in range(q):
        ai, bi = a_of(i), b_of(i)
        neg_a = (-ai) % p
        neg_b = (-bi) % p
        neg_table[i] = idx(neg_a, neg_b)

    total_points = q * q  # 2401
    num_dirs = 50  # 49 directions (1, m) + 1 direction (0, 1)

    # Precompute line_other[d, x, :] = the 48 other points on the line through x in direction d
    line_other = np.zeros((num_dirs, total_points, q - 1), dtype=np.int32)
    
    for d in range(num_dirs):
        for x_idx in range(total_points):
            x = x_idx // q
            y = x_idx % q
            count = 0
            if d < 49:
                m = d
                for t in range(1, q):
                    nx = add_table[x, t]
                    ny = add_table[y, mul_table[t, m]]
                    line_other[d, x_idx, count] = nx * q + ny
                    count += 1
            else:
                for t in range(1, q):
                    nx = x
                    ny = add_table[y, t]
                    line_other[d, x_idx, count] = nx * q + ny
                    count += 1

    # Precompute affected[s, d, :] = points x such that s is on line_d(x) and s != x
    affected = np.zeros((total_points, num_dirs, q - 1), dtype=np.int32)
    
    for s_idx in range(total_points):
        sx = s_idx // q
        sy = s_idx % q
        for d in range(num_dirs):
            count = 0
            if d < 49:
                m = d
                for t in range(1, q):
                    neg_t = neg_table[t]
                    nx = add_table[sx, neg_t]
                    ny = add_table[sy, mul_table[neg_t, m]]
                    affected[s_idx, d, count] = nx * q + ny
                    count += 1
            else:
                for t in range(1, q):
                    neg_t = neg_table[t]
                    nx = sx
                    ny = add_table[sy, neg_t]
                    affected[s_idx, d, count] = nx * q + ny
                    count += 1

    # Initialize S (complement of N) with a valid starting set.
    # Start with S = empty (N = all points), which is trivially valid.
    # Then greedily add points to S.
    
    S = set()
    N_mask = np.ones(total_points, dtype=bool)  # True if in N

    # valid_count[x] = number of directions d such that all 48 other points on line_d(x) are in N
    # valid_dir_mask[x, d] = True if direction d is valid for x
    valid_count = np.zeros(total_points, dtype=np.int32)
    valid_dir_mask = np.zeros((total_points, num_dirs), dtype=bool)

    # Initially, all points are in N, so for every x and d, all 48 other points are in N.
    # So valid_count[x] = 50 for all x, and valid_dir_mask is all True.
    valid_count[:] = num_dirs
    valid_dir_mask[:] = True

    # Greedy expansion: try to add each point to S (remove from N)
    # We'll do multiple passes with different orderings to escape local optima.
    
    best_complement_size = 0
    best_S = set()
    best_N_mask = N_mask.copy()
    best_valid_count = valid_count.copy()
    best_valid_dir_mask = valid_dir_mask.copy()

    def try_add_point(s, S_set, N_mask_arr, valid_count_arr, valid_dir_mask_arr):
        """Try to add point s to S. Returns True if successful."""
        if s in S_set:
            return False
            
        # Collect affected (x, d) pairs
        lost_counts = np.zeros(total_points, dtype=np.int32)
        
        for d in range(num_dirs):
            xs = affected[s, d, :]
            for x in xs:
                if valid_dir_mask_arr[x, d]:
                    lost_counts[x] += 1
        
        # Check if any point is broken
        affected_points = np.nonzero(lost_counts)[0]
        for x in affected_points:
            if valid_count_arr[x] - lost_counts[x] <= 0:
                return False
        
        # Commit the change
        S_set.add(s)
        N_mask_arr[s] = False
        for d in range(num_dirs):
            xs = affected[s, d, :]
            for x in xs:
                if valid_dir_mask_arr[x, d]:
                    valid_dir_mask_arr[x, d] = False
                    valid_count_arr[x] -= 1
        
        return True

    def remove_point(s, S_set, N_mask_arr, valid_count_arr, valid_dir_mask_arr):
        """Remove point s from S (add back to N)."""
        if s not in S_set:
            return
            
        S_set.remove(s)
        N_mask_arr[s] = True
        for d in range(num_dirs):
            xs = affected[s, d, :]
            for x in xs:
                # Check if this direction becomes valid again
                others = line_other[d, x, :]
                if np.all(N_mask_arr[others]):
                    if not valid_dir_mask_arr[x, d]:
                        valid_dir_mask_arr[x, d] = True
                        valid_count_arr[x] += 1

    # Multiple greedy passes with different random orderings
    np.random.seed(42)
    
    for pass_num in range(20):
        if time.time() - start_time > deadline - 30:
            break
            
        # Reset to best known state
        S = set(best_S)
        N_mask = best_N_mask.copy()
        valid_count = best_valid_count.copy()
        valid_dir_mask = best_valid_dir_mask.copy()
        
        # Get candidates: all points not in S
        candidates = [i for i in range(total_points) if i not in S]
        
        # Random shuffle
        random.shuffle(candidates)
        
        # Try to add each candidate
        for s in candidates:
            if time.time() - start_time > deadline - 30:
                break
            if s in S:
                continue
            if try_add_point(s, S, N_mask, valid_count, valid_dir_mask):
                # Update best if improved
                if len(S) > best_complement_size:
                    best_complement_size = len(S)
                    best_S = set(S)
                    best_N_mask = N_mask.copy()
                    best_valid_count = valid_count.copy()
                    best_valid_dir_mask = valid_dir_mask.copy()

    # Local search: try to remove points from S and re-add different ones
    S = set(best_S)
    N_mask = best_N_mask.copy()
    valid_count = best_valid_count.copy()
    valid_dir_mask = best_valid_dir_mask.copy()
    
    # Simulated annealing style: randomly remove a point, then greedily re-optimize
    np.random.seed(123)
    
    for it in range(100):
        if time.time() - start_time > deadline - 30:
            break
            
        # Randomly remove 1-3 points from S
        num_remove = random.randint(1, 3)
        to_remove = random.sample(list(S), min(num_remove, len(S))) if S else []
        
        for s in to_remove:
            remove_point(s, S, N_mask, valid_count, valid_dir_mask)
        
        # Greedily re-optimize
        candidates = [i for i in range(total_points) if i not in S]
        random.shuffle(candidates)
        
        for s in candidates:
            if time.time() - start_time > deadline - 30:
                break
            if s in S:
                continue
            if try_add_point(s, S, N_mask, valid_count, valid_dir_mask):
                if len(S) > best_complement_size:
                    best_complement_size = len(S)
                    best_S = set(S)
                    best_N_mask = N_mask.copy()
                    best_valid_count = valid_count.copy()
                    best_valid_dir_mask = valid_dir_mask.copy()

    # Build result from best_S
    points = []
    for i in range(total_points):
        if i not in best_S:
            x = i // q
            y = i % q
            xa, xb = a_of(x), b_of(x)
            ya, yb = a_of(y), b_of(y)
            points.append([[xa, xb], [ya, yb]])

    result = np.array(points, dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
