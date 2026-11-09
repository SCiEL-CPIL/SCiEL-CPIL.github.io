import random
import time
import math

N = 64

def construct():
    all_points = [(x, y) for y in range(N) for x in range(N)]
    point_indices = {p: i for i, p in enumerate(all_points)}
    
    # Precompute squared distances? No, just compute on the fly.
    
    def is_isosceles(p, q, r):
        d_pq = (p[0]-q[0])**2 + (p[1]-q[1])**2
        d_pr = (p[0]-r[0])**2 + (p[1]-r[1])**2
        d_qr = (q[0]-r[0])**2 + (q[1]-r[1])**2
        return d_pq == d_pr or d_pq == d_qr or d_pr == d_qr

    def can_add(p, selected):
        # Check if p can be added to selected without forming an isosceles triangle
        # Iterate over pairs in selected
        sel_list = list(selected)
        n = len(sel_list)
        for i in range(n):
            for j in range(i+1, n):
                if is_isosceles(p, sel_list[i], sel_list[j]):
                    return False
        return True

    def greedy_fill(selected, available):
        # Add points from available to selected if valid
        # available is a set of points not in selected
        # We shuffle available to randomize
        avail_list = list(available)
        random.shuffle(avail_list)
        for p in avail_list:
            if p in selected:
                continue
            if can_add(p, selected):
                selected.add(p)
                available.remove(p)

    start_time = time.time()
    time_limit = 290.0
    
    best_set = set()
    best_size = 0
    
    # Initial random set of points to start with? Or start empty?
    # Start empty is safer for validity.
    
    current_set = set()
    available = set(all_points)
    
    # Initial Greedy Fill
    greedy_fill(current_set, available)
    
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = set(current_set)
        
    while time.time() - start_time < time_limit:
        # Perturbation: Remove a random point from current_set
        if len(current_set) > 0:
            p_to_remove = random.choice(list(current_set))
            current_set.remove(p_to_remove)
            available.add(p_to_remove)
            
            # Re-fill
            greedy_fill(current_set, available)
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
        else:
            # If set is empty, just do a fresh greedy fill
            current_set = set()
            available = set(all_points)
            greedy_fill(current_set, available)
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)

    return list(best_set)
