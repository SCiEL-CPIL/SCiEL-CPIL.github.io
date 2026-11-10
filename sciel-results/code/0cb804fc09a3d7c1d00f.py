import time
import random
import numpy as np

N = 64
NUM_POINTS = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Precompute squared distances
    xs = np.arange(N)
    ys = np.arange(N)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    # Flatten
    x_flat = X.flatten()
    y_flat = Y.flatten()
    
    # dist2[i, j] = (x_i - x_j)^2 + (y_i - y_j)^2
    # Using broadcasting
    # x_flat shape (4096,)
    # diff_x = x_flat[:, None] - x_flat[None, :]
    # This creates a 4096x4096 matrix.
    diff_x = x_flat[:, None] - x_flat[None, :]
    diff_y = y_flat[:, None] - y_flat[None, :]
    dist2 = diff_x**2 + diff_y**2
    dist2 = dist2.astype(np.int32)
    
    best_set = set()
    rng = random.Random(42)
    
    def greedy(order):
        selected = []
        selected_set = set()
        # dist_sets[p] = set of squared distances from p to other selected points
        dist_sets = {}
        
        for p in order:
            if p in selected_set:
                continue
            
            # 1. Check distances from p to all selected points
            if not selected:
                can_add = True
                new_dists = set()
            else:
                # Get indices of selected points
                # This is O(|S|)
                sel_indices = selected
                # Vectorized distance from p to all selected
                # dist2[p, sel_indices]
                d_to_sel = dist2[p, sel_indices]
                
                # Check for duplicates (apex at p)
                if len(set(d_to_sel)) != len(d_to_sel):
                    continue
                
                # Check for conflicts with existing points (apex at q in selected)
                can_add = True
                for i, q in enumerate(sel_indices):
                    d = d_to_sel[i]
                    if d in dist_sets[q]:
                        can_add = False
                        break
                
                if not can_add:
                    continue
                new_dists = set(d_to_sel)
            
            # Add p
            selected.append(p)
            selected_set.add(p)
            dist_sets[p] = new_dists
            
            # Update dist_sets for existing points
            for q in selected:
                if q != p:
                    dist_sets[q].add(dist2[p, q])
        
        return selected_set

    # Multiple restarts
    # Different orders:
    # 1. Random
    # 2. Sorted by x, then y
    # 3. Spiral
    # 4. Random with bias to corners
    
    orders = []
    
    # Random
    for _ in range(5):
        o = list(range(NUM_POINTS))
        rng.shuffle(o)
        orders.append(o)
        
    # Sorted
    orders.append(list(range(NUM_POINTS)))
    
    # Corner bias
    o = list(range(NUM_POINTS))
    # Sort by distance to center? Or random?
    # Let's just do more random restarts as they are effective for such constraints.
    
    for _ in range(500):
        if time.time() > deadline:
            break
        o = list(range(NUM_POINTS))
        rng.shuffle(o)
        s = greedy(o)
        if len(s) > len(best_set):
            best_set = s
            # If we found a very good set, maybe break early? No, keep searching.
            
    return [(p // N, p % N) for p in best_set]
