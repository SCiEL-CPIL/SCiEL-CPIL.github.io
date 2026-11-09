"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def _precompute(n):
    """Precompute triple list and index maps for fast access."""
    triple_list = list(combinations(range(n), 3))
    num_triples = len(triple_list)
    
    # Build reverse index: for each triple index, store (a,b,c)
    triple_data = [(a, b, c) for a, b, c in triple_list]
    
    # For each triple, precompute which 4-sets it belongs to (for red K4)
    # and which 5-sets it belongs to (for blue K5)
    # This is expensive, so we'll compute on the fly but efficiently
    
    return triple_list, triple_data, num_triples

def _count_violations_fast(colors, n, triple_data):
    """Count red K4 and blue K5 violations efficiently."""
    colors_arr = np.array(colors)
    num_triples = len(triple_data)
    
    # Create a lookup: for each triple (a,b,c), get its color quickly
    # We need to map from (a,b,c) to index
    idx_map = {}
    for i, (a, b, c) in enumerate(triple_data):
        idx_map[(a, b, c)] = i
    
    total_violations = 0
    
    # Check red K4: for each 4-set, count red triples
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    count = 0
                    count += colors_arr[idx_map[(i,j,k)]] == 0
                    count += colors_arr[idx_map[(i,j,l)]] == 0
                    count += colors_arr[idx_map[(i,k,l)]] == 0
                    count += colors_arr[idx_map[(j,k,l)]] == 0
                    if count == 4:
                        total_violations += 1
    
    # Check blue K5: for each 5-set, count blue triples
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    for m in range(l+1, n):
                        count = 0
                        count += colors_arr[idx_map[(i,j,k)]] == 1
                        count += colors_arr[idx_map[(i,j,l)]] == 1
                        count += colors_arr[idx_map[(i,j,m)]] == 1
                        count += colors_arr[idx_map[(i,k,l)]] == 1
                        count += colors_arr[idx_map[(i,k,m)]] == 1
                        count += colors_arr[idx_map[(i,l,m)]] == 1
                        count += colors_arr[idx_map[(j,k,l)]] == 1
                        count += colors_arr[idx_map[(j,k,m)]] == 1
                        count += colors_arr[idx_map[(j,l,m)]] == 1
                        count += colors_arr[idx_map[(k,l,m)]] == 1
                        if count == 10:
                            total_violations += 1
    
    return total_violations

def _local_search(colors, n, triple_data, max_iter=10000, deadline=None):
    """Local search to reduce violations. Returns improved colors and violation count."""
    colors_arr = np.array(colors, dtype=np.int8)
    num_triples = len(triple_data)
    
    # Build idx_map
    idx_map = {}
    for i, (a, b, c) in enumerate(triple_data):
        idx_map[(a, b, c)] = i
    
    # Precompute for each triple, the list of 4-sets it's part of
    # and the list of 5-sets it's part of
    # For triple (a,b,c), 4-sets are (a,b,c,d) for d>c, and also cases where 
    # the triple isn't the first three... actually all 4-sets containing {a,b,c}
    # A 4-set {w,x,y,z} contains triple {a,b,c} iff {a,b,c} ⊆ {w,x,y,z}
    
    # For efficiency, let's precompute:
    # For each triple t, list of 4-sets (as tuples) that contain it
    # For each triple t, list of 5-sets (as tuples) that contain it
    
    # This is O(n^3 * n + n^3 * n^2) which is too much for large n
    # Instead, let's use a different approach: maintain violation counts incrementally
    
    # Actually, let's use a simpler but effective approach:
    # For each flip, compute the change in violations by checking affected sets
    
    best_colors = colors_arr.copy()
    best_viol = _count_violations_fast(best_colors, n, triple_data)
    
    if best_viol == 0:
        return best_colors, 0
    
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iter:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        iter_count += 1
        
        # Try flipping a random triple
        t_idx = random.randint(0, num_triples - 1)
        old_color = best_colors[t_idx]
        new_color = 1 - old_color
        
        a, b, c = triple_data[t_idx]
        
        # Compute change in violations
        # Red K4: flipping affects all 4-sets containing (a,b,c)
        # If old_color == 0 (red), flipping to blue reduces red count in those 4-sets
        # If old_color == 1 (blue), flipping to red increases red count in those 4-sets
        
        delta_viol = 0
        
        # Check all 4-sets containing (a,b,c)
        # 4-sets containing {a,b,c}: add one more vertex d
        # d can be any vertex not in {a,b,c}
        for d in range(n):
            if d == a or d == b or d == c:
                continue
            # The 4-set is {a,b,c,d}, sorted
            verts = sorted([a, b, c, d])
            i, j, k, l = verts
            
            # Count red triples in this 4-set
            red_count = 0
            red_count += best_colors[idx_map[(i,j,k)]] == 0
            red_count += best_colors[idx_map[(i,j,l)]] == 0
            red_count += best_colors[idx_map[(i,k,l)]] == 0
            red_count += best_colors[idx_map[(j,k,l)]] == 0
            
            # Before flip:
            # After flip:
            if old_color == 0:
                # Was red, now blue: red_count decreases by 1
                before_viol = 1 if red_count == 4 else 0
                after_viol = 1 if (red_count - 1) == 4 else 0
                delta_viol += after_viol - before_viol
            else:
                # Was blue, now red: red_count increases by 1
                before_viol = 1 if red_count == 4 else 0
                after_viol = 1 if (red_count + 1) == 4 else 0
                delta_viol += after_viol - before_viol
        
        # Check all 5-sets containing (a,b,c)
        # 5-sets containing {a,b,c}: add two more vertices d, e
        other_vertices = [v for v in range(n) if v != a and v != b and v != c]
        for d_idx in range(len(other_vertices)):
            for e_idx in range(d_idx + 1, len(other_vertices)):
                d = other_vertices[d_idx]
                e = other_vertices[e_idx]
                verts = sorted([a, b, c, d, e])
                i, j, k, l, m = verts
                
                # Count blue triples in this 5-set
                blue_count = 0
                blue_count += best_colors[idx_map[(i,j,k)]] == 1
                blue_count += best_colors[idx_map[(i,j,l)]] == 1
                blue_count += best_colors[idx_map[(i,j,m)]] == 1
                blue_count += best_colors[idx_map[(i,k,l)]] == 1
                blue_count += best_colors[idx_map[(i,k,m)]] == 1
                blue_count += best_colors[idx_map[(i,l,m)]] == 1
                blue_count += best_colors[idx_map[(j,k,l)]] == 1
                blue_count += best_colors[idx_map[(j,k,m)]] == 1
                blue_count += best_colors[idx_map[(j,l,m)]] == 1
                blue_count += best_colors[idx_map[(k,l,m)]] == 1
                
                if old_color == 1:
                    # Was blue, now red: blue_count decreases by 1
                    before_viol = 1 if blue_count == 10 else 0
                    after_viol = 1 if (blue_count - 1) == 10 else 0
                    delta_viol += after_viol - before_viol
                else:
                    # Was red, now blue: blue_count increases by 1
                    before_viol = 1 if blue_count == 10 else 0
                    after_viol = 1 if (blue_count + 1) == 10 else 0
                    delta_viol += after_viol - before_viol
        
        if delta_viol < 0:
            best_colors[t_idx] = new_color
            best_viol += delta_viol
            improved = True
            if best_viol == 0:
                break
    
    return best_colors, best_viol

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_n = 0
    best_colors = None
    
    # Try increasing n from small to large
    # Start with n=5 and go up
    for n in range(5, 35):
        if time.time() > deadline - 5:
            break
        
        triple_list, triple_data, num_triples = _precompute(n)
        
        # Multiple random restarts
        found = False
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random coloring
            colors = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
            
            # Check initial violations
            viol = _count_violations_fast(colors, n, triple_data)
            
            if viol == 0:
                best_n = n
                best_colors = colors.tolist()
                found = True
                break
            
            # Local search
            colors, viol = _local_search(colors, n, triple_data, max_iter=5000, 
                                          deadline=deadline - 2)
            
            if viol == 0:
                best_n = n
                best_colors = colors.tolist()
                found = True
                break
        
        if not found:
            # If we couldn't find a valid coloring for this n, try a larger n
            # might work due to random initialization luck, but usually if n is 
            # too hard, larger n will be harder too. Let's continue anyway.
            pass
    
    if best_colors is None:
        # Fallback: return the seed program's result
        n = 9
        return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 
                                   for a, b, c in combinations(range(n), 3)]}
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
