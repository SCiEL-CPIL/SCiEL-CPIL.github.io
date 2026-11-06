import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Try to find better n
    # We try n from 29 down to 12
    # We allocate time for each
    # If we find a valid coloring, we update best_n and best_colors
    
    # To save time, we try higher n first.
    # If we find 26, we are happy.
    
    # We will use a helper function to run local search for a specific n
    # and return the colors if valid.
    
    for target_n in range(29, 11, -1):
        if time.time() > deadline - 5:
            break
            
        # Allocate remaining time to this target_n
        # But we need to ensure we don't spend all time on 29 and fail.
        # Let's give each target_n a fixed time budget, e.g., 5 seconds.
        # If it fails, move to next.
        
        colors = local_search(target_n, 5.0)
        if colors is not None:
            best_n = target_n
            best_colors = colors
            # We can continue to see if we can find higher?
            # No, we are iterating downwards.
            # So if we found 26, we already tried 29, 28, 27.
            # So we can break if we want?
            # No, maybe 27 was just unlucky.
            # But we already spent time on it.
            # Let's just continue the loop. It will try 25, 24... which are easier.
            # But we already have 26.
            # So we can break if we found a "good" n like 26?
            # Let's break if best_n >= 20 to save time?
            # No, let's just let it run.
            
            # Actually, if we found 26, we can stop trying lower n.
            # But we are iterating down.
            # So if we find 26, the next is 25.
            # We don't need 25.
            # So we should break if we find a high n.
            # Let's break if best_n >= 20.
            if best_n >= 20:
                break

    return {'n': best_n, 'colors': best_colors}

def local_search(n, time_limit):
    # Random initialization
    colors = np.random.randint(0, 5, size=(n, n))
    # Make it symmetric
    colors = (colors + colors.T) // 2
    # Diagonal should be 0 (or any, doesn't matter)
    np.fill_diagonal(colors, 0)
    
    # Precompute edge list
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Function to calculate violations
    def calc_violations():
        # For each color c, count C4s
        # A C4 is a pair of vertices (u,v) with >=2 common neighbors in color c.
        # Let A_c be the adjacency matrix for color c.
        # Common neighbors count is (A_c @ A_c)[u,v].
        # We need sum over u<v of comb(common, 2).
        
        total_viol = 0
        for c in range(5):
            A = np.zeros((n, n), dtype=int)
            # Extract color c
            # This is slow.
            # Better: maintain A_c matrices.
            
        # Let's maintain A_c matrices outside.
        pass

    # Let's use a different representation:
    # A_c[c, i, j] = 1 if color(i,j) == c
    
    A = np.zeros((5, n, n), dtype=int)
    for i, j in edges:
        A[colors[i, j], i, j] = 1
        A[colors[i, j], j, i] = 1
        
    def get_violations():
        v = 0
        for c in range(5):
            A_c = A[c]
            # A_c @ A_c gives common neighbors
            common = A_c @ A_c
            # For i < j, count comb(common[i,j], 2)
            # We can sum all and divide by 2?
            # comb(k, 2) = k*(k-1)/2
            # Sum over i<j.
            # Since common is symmetric, sum over all i!=j is 2 * sum over i<j.
            # So sum(comb(common[i,j], 2)) for i<j = 0.5 * sum_{i!=j} comb(common[i,j], 2)
            # = 0.5 * sum_{i!=j} common[i,j]*(common[i,j]-1)/2
            # = 0.25 * (sum(common^2) - sum(common))
            # Note: common[i,i] is degree of i.
            # We only care about i != j.
            # So subtract diagonal.
            diag_sum = np.sum(np.diag(common))
            total_sum = np.sum(common)
            sq_sum = np.sum(common**2)
            # Sum over i!=j of comb(k,2) = 0.5 * ( (sq_sum - diag_sum) - (total_sum - diag_sum) )
            # = 0.5 * (sq_sum - total_sum)
            # Wait, comb(k,2) = k(k-1)/2.
            # Sum_{i!=j} k(k-1)/2 = 0.5 * (Sum k^2 - Sum k)
            # Sum k^2 over i!=j = sq_sum - diag_sum
            # Sum k over i!=j = total_sum - diag_sum
            # So result = 0.5 * (sq_sum - diag_sum - total_sum + diag_sum) = 0.5 * (sq_sum - total_sum)
            
            viol_c = 0.5 * (sq_sum - total_sum)
            v += viol_c
        return int(v)
    
    current_viol = get_violations()
    if current_viol == 0:
        return colors.flatten()[::2] # Need to return list of colors for i<j
        
    start = time.time()
    while time.time() - start < time_limit:
        if current_viol == 0:
            break
            
        # Pick a random edge
        i, j = random.choice(edges)
        old_c = colors[i, j]
        
        best_c = old_c
        best_v = current_viol
        
        # Try other colors
        for new_c in range(5):
            if new_c == old_c:
                continue
                
            # Update A
            A[old_c, i, j] = 0
            A[old_c, j, i] = 0
            A[new_c, i, j] = 1
            A[new_c, j, i] = 1
            
            # Calculate new violations
            # This is O(5 * n^2) which is 5 * 26^2 = 3380 ops.
            # Very fast.
            new_v = get_violations()
            
            if new_v < best_v:
                best_v = new_v
                best_c = new_c
                
            # Revert
            A[new_c, i, j] = 0
            A[new_c, j, i] = 0
            A[old_c, i, j] = 1
            A[old_c, j, i] = 1
            
        # Apply best change
        if best_c != old_c:
            A[old_c, i, j] = 0
            A[old_c, j, i] = 0
            A[best_c, i, j] = 1
            A[best_c, j, i] = 1
            colors[i, j] = best_c
            colors[j, i] = best_c
            current_viol = best_v
        else:
            # No improvement, maybe random move?
            # To escape local minima, we can accept a worse move with some probability.
            # Or just pick a random edge and random color.
            new_c = random.randint(0, 4)
            if new_c != old_c:
                A[old_c, i, j] = 0
                A[old_c, j, i] = 0
                A[new_c, i, j] = 1
                A[new_c, j, i] = 1
                colors[i, j] = new_c
                colors[j, i] = new_c
                current_viol = get_violations()

    if current_viol == 0:
        # Return list of colors for i<j
        res = []
        for i in range(n):
            for j in range(i+1, n):
                res.append(int(colors[i, j]))
        return res
    return None
