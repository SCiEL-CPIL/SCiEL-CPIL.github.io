import numpy as np
import random
import time

N = 64
# Pre-compute grid coordinates
xs, ys = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
# Flatten to 1D for easier indexing
X = xs.flatten()
Y = ys.flatten()
# Total number of points
M = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 290.0

    # We will maintain a list of selected points (indices into the flattened grid)
    # and a boolean array `forbidden` where True means the point cannot be added.
    
    # To allow for local search (swaps), we need to be able to "un-forbid" points
    # when a point is removed. This is tricky with a simple cumulative mask.
    # However, for a greedy construction, a simple mask works.
    # For local search, we can re-compute the mask from scratch or use a more complex reference counting.
    # Given the time limit, let's try a fast greedy first, then a simple local search 
    # that re-validates the whole set if a swap is proposed (using vectorized checks).
    
    # Actually, a better strategy for local search with vectorization:
    # 1. Greedy phase with mask.
    # 2. Local search: 
    #    - Pick a point to remove.
    #    - Re-compute the forbidden mask for the remaining set (vectorized).
    #    - Pick a random candidate from the un-forbidden set.
    #    - If valid, swap.
    
    # Re-computing the mask for a set of size K takes O(K * M) vectorized ops.
    # With K~20, M=4096, that's 80k ops. Very fast in NumPy.
    
    def compute_forbidden_mask(points_indices):
        """
        Returns a boolean array of length M where True indicates the point is forbidden.
        points_indices: list of indices into the flattened grid.
        """
        if not points_indices:
            return np.zeros(M, dtype=bool)
        
        forbidden = np.zeros(M, dtype=bool)
        
        # Get coordinates of selected points
        px = X[points_indices]
        py = Y[points_indices]
        
        # Distances between selected points (squared)
        # Shape: (K, K)
        dx = px[:, None] - px[None, :]
        dy = py[:, None] - py[None, :]
        d2_sel = dx**2 + dy**2
        
        # For every candidate point q in the grid (M points),
        # we need to check if it forms an isosceles triangle with any pair (i, j) in selected points.
        # Condition 1: d(q, i) == d(q, j)
        # Condition 2: d(q, i) == d(i, j)
        # Condition 3: d(q, j) == d(i, j)
        
        # Let's vectorize over all candidates q and all selected points i.
        # qx, qy are (M, 1)
        qx = X[:, None]
        qy = Y[:, None]
        
        # Distance from all q to all selected points i
        # dx_qi: (M, K), dy_qi: (M, K)
        dx_qi = qx - px[None, :]
        dy_qi = qy - py[None, :]
        d2_qi = dx_qi**2 + dy_qi**2
        
        # d2_qi has shape (M, K)
        # d2_sel has shape (K, K)
        
        # We want to mark q as forbidden if there exists i != j such that:
        # 1. d2_qi[q, i] == d2_qi[q, j]
        # 2. d2_qi[q, i] == d2_sel[i, j]
        # 3. d2_qi[q, j] == d2_sel[i, j]
        
        # This is hard to vectorize directly as a single boolean operation over (M, K, K) 
        # without blowing up memory or time, but K is small.
        
        # Let's iterate over pairs (i, j) of selected points. K is small (<= 30ish).
        # For each pair (i, j):
        #   d_ij = d2_sel[i, j]
        #   Forbidden if:
        #     (d2_qi[:, i] == d2_qi[:, j])  # q on perp bisector
        #     (d2_qi[:, i] == d_ij)          # q on circle centered i
        #     (d2_qi[:, j] == d_ij)          # q on circle centered j
        
        for i in range(len(points_indices)):
            for j in range(i + 1, len(points_indices)):
                d_ij = d2_sel[i, j]
                
                # Vectorized checks for all M candidates
                # Note: We should exclude i and j themselves from being "forbidden" by themselves?
                # No, the mask is for *new* points. The points i and j are already in the set.
                # We just need to know if a *candidate* q is valid.
                
                mask_1 = (d2_qi[:, i] == d2_qi[:, j])
                mask_2 = (d2_qi[:, i] == d_ij)
                mask_3 = (d2_qi[:, j] == d_ij)
                
                forbidden |= (mask_1 | mask_2 | mask_3)
                
        # The selected points themselves should not be considered "forbidden" for the purpose of 
        # adding *new* points, but they are already in the set. 
        # When we pick a new point, we must ensure it's not in the current set.
        # So we return the mask. The caller must also check if the point is already in the set.
        
        return forbidden

    # --- Phase 1: Greedy Construction ---
    points = []
    forbidden_mask = np.zeros(M, dtype=bool)
    
    # Randomize candidate order
    candidates = list(range(M))
    random.shuffle(candidates)
    
    for c in candidates:
        if time.time() > deadline:
            break
            
        # Check if c is already in points
        if c in points:
            continue
            
        # Check if c is forbidden
        if forbidden_mask[c]:
            continue
            
        # Add c
        points.append(c)
        
        # Update forbidden mask
        # We can update incrementally or recompute. 
        # Recomputing is O(K^2 * M). With K~20, M=4096, K^2=400. 400*4096 = 1.6M ops. 
        # Vectorized, this is very fast (< 1ms).
        # So we just recompute for simplicity and robustness.
        forbidden_mask = compute_forbidden_mask(points)

    best_points = points[:]
    best_len = len(points)
    
    # --- Phase 2: Local Search (Swaps) ---
    # Try to improve by swapping a point in the set with a point not in the set.
    # To check if a swap (remove p_remove, add p_add) is valid:
    # 1. Remove p_remove from the set.
    # 2. Compute forbidden mask for the new set.
    # 3. Check if p_add is not in the new set and not forbidden.
    
    in_set = set(points)
    
    # We will use the recomputation strategy for the mask during local search too.
    # Since K is small, this is fast.
    
    no_improvement_count = 0
    max_no_improvement = 5000
    
    while time.time() < deadline:
        if no_improvement_count > max_no_improvement:
            break
            
        # Pick a random point to remove
        i = random.randint(0, len(points) - 1)
        p_remove_idx = points[i]
        
        # Create new set without p_remove
        new_points = points[:i] + points[i+1:]
        
        # Compute forbidden mask for new_points
        # If new_points is empty, mask is all False
        if not new_points:
             new_mask = np.zeros(M, dtype=bool)
        else:
             new_mask = compute_forbidden_mask(new_points)
             
        # Find valid candidates for addition
        # A candidate c is valid if:
        # 1. c is not in new_points
        # 2. c is not in forbidden_mask (for new_points)
        
        # We want to pick a c that is NOT in new_points and NOT in new_mask.
        # Let's create a boolean array of "available" points.
        available = ~new_mask
        # Mark points in new_points as unavailable
        available[new_points] = False
        
        # Count available points
        num_available = np.sum(available)
        
        if num_available == 0:
            no_improvement_count += 1
            continue
            
        # Pick a random available point
        # np.where returns indices where condition is True
        avail_indices = np.where(available)[0]
        p_add_idx = avail_indices[random.randint(0, len(avail_indices) - 1)]
        
        # Perform the swap
        # We assume p_add_idx is valid by construction.
        # But we must ensure p_add_idx != p_remove_idx (it can't be, as p_remove is not in new_points, 
        # and we marked new_points as unavailable, but p_remove is NOT in new_points. 
        # Wait. p_remove was removed. So p_remove is NOT in new_points.
        # Is p_remove forbidden by new_points? 
        # If we remove p_remove, the constraints it imposed are gone. 
        # It might be valid to add it back? That would be a no-op swap.
        # We should avoid adding the same point back if possible, or it doesn't matter.
        # Actually, if we add p_remove back, the set is identical. 
        # To ensure progress, we should prefer p_add != p_remove.
        
        if p_add_idx == p_remove_idx:
            # Try to pick another one if possible
            if len(avail_indices) > 1:
                # Pick a different one
                mask_diff = avail_indices != p_add_idx
                other_indices = avail_indices[mask_diff]
                if len(other_indices) > 0:
                    p_add_idx = other_indices[random.randint(0, len(other_indices) - 1)]
                else:
                    # Only option is to add it back. No improvement.
                    no_improvement_count += 1
                    continue
            else:
                no_improvement_count += 1
                continue
        
        # Update points
        points[i] = p_add_idx
        in_set.remove(p_remove_idx)
        in_set.add(p_add_idx)
        
        # Update global forbidden mask for future checks?
        # We recompute it in the next iteration's "remove" step anyway?
        # No, we recompute for the *new* set when we try to remove *another* point.
        # But we need the mask for the *current* set if we want to do additions?
        # The local search here is strictly swaps. Size is constant.
        
        # We can try to increase size occasionally.
        # If we can add a point, we should.
        # Let's check if we can add a point to the current set.
        
        # Compute mask for current points
        current_mask = compute_forbidden_mask(points)
        available_add = ~current_mask
        available_add[points] = False
        
        num_avail_add = np.sum(available_add)
        if num_avail_add > 0:
            avail_add_indices = np.where(available_add)[0]
            p_new = avail_add_indices[random.randint(0, len(avail_add_indices) - 1)]
            points.append(p_new)
            in_set.add(p_new)
            best_len = len(points)
            no_improvement_count = 0
        else:
            no_improvement_count += 1

    # Final verification
    # The local search assumes validity by construction via the mask.
    # But just in case, we can return the points.
    
    return [(int(X[i]), int(Y[i])) for i in points]

# Quick test to ensure it runs
if __name__ == "__main__":
    import time
    start = time.time()
    pts = construct()
    end = time.time()
    print(f"Points found: {len(pts)}, Time: {end-start:.2f}s")
    # Verify
    def is_valid(pts):
        n = len(pts)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    d12 = (x1-x2)**2 + (y1-y2)**2
                    d13 = (x1-x3)**2 + (y1-y3)**2
                    d23 = (x2-x3)**2 + (y2-y3)**2
                    if d12 == d13 or d12 == d23 or d13 == d23:
                        return False
        return True
    print(f"Valid: {is_valid(pts)}")
