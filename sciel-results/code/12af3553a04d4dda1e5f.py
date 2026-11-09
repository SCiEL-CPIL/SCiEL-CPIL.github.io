import numpy as np
import itertools
import time
import random

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = 590  # safety margin before 600s limit
    
    # Type A points: ±2*e_i
    type_a = []
    for i in range(d):
        p1 = np.zeros(d, dtype=np.int64)
        p1[i] = 2
        type_a.append(p1)
        p2 = np.zeros(d, dtype=np.int64)
        p2[i] = -2
        type_a.append(p2)
    
    # All type B points: all 4-subsets × 16 sign patterns (norm²=4)
    all_combos = list(itertools.combinations(range(d), 4))
    b_points_list = []
    for c in all_combos:
        for mask in range(16):
            p = np.zeros(d, dtype=np.int64)
            for k, idx in enumerate(c):
                if (mask >> k) & 1:
                    p[idx] = -1
                else:
                    p[idx] = 1
            b_points_list.append(p)
    b_arr = np.array(b_points_list, dtype=np.int64)  # shape (7392, 11)
    n_b = b_arr.shape[0]
    
    # All type B points are compatible with type A (verified analytically)
    # Need pairwise dist_sq >= 4 among type B: equivalent to dot product <= 2
    
    def greedy_select(order):
        selected_arr = []
        for idx in order:
            p = b_arr[idx]
            if len(selected_arr) == 0:
                selected_arr.append(p)
            else:
                sel_stack = np.array(selected_arr)
                dots = sel_stack @ p
                if np.all(dots <= 2):
                    selected_arr.append(p)
        return selected_arr
    
    best_result = []
    
    # Multiple random restarts with greedy selection
    for seed in range(300):
        if time.time() - start_time > deadline:
            break
        rng = random.Random(seed)
        order = list(range(n_b))
        rng.shuffle(order)
        result = greedy_select(order)
        if len(result) > len(best_result):
            best_result = result
    
    points = type_a + best_result
    points_array = np.array(points, dtype=np.int64)
    return points_array
