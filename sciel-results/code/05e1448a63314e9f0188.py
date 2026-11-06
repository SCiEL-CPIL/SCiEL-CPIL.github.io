import time
import random
import numpy as np

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 270.0
    
    # Precompute all grid points as a numpy array
    xs = np.arange(N)
    ys = np.arange(N)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    all_points = np.column_stack([XX.ravel(), YY.ravel()])  # shape (4096, 2)
    
    best_set = []
    best_size = 0
    
    def check_and_add(p, points_arr, dist_matrix):
        """
        Check if point p can be added to the set represented by points_arr (k x 2).
        Returns (valid, new_points_arr, new_dist_matrix) or (False, None, None).
        """
        k = len(points_arr)
        if k == 0:
            new_points = np.array([[p[0], p[1]]], dtype=np.int32)
            new_dist = np.zeros((1, 1), dtype=np.int64)
            return True, new_points, new_dist
        
        # Compute distances from p to all existing points
        dx = points_arr[:, 0].astype(np.int64) - p[0]
        dy = points_arr[:, 1].astype(np.int64) - p[1]
        dists = dx * dx + dy * dy  # shape (k,)
        
        # Check 1: all distances from p must be distinct
        if len(np.unique(dists)) != k:
            return False, None, None
        
        # Check 2: for each i, dists[i] must not be in dist_matrix[i, :] (excluding self)
        # dist_matrix[i,j] = squared distance between point i and point j
        # We need: dists[i] not in {dist_matrix[i,j] for j != i}
        for i in range(k):
            # Check if dists[i] appears in row i of dist_matrix (excluding diagonal)
            row = dist_matrix[i, :i]
            row2 = dist_matrix[i, i+1:]
            if (row == dists[i]).any() or (row2 == dists[i]).any():
                return False, None, None
        
        # Build new arrays
        new_points = np.vstack([points_arr, [[p[0], p[1]]]])
        new_dist = np.zeros((k+1, k+1), dtype=np.int64)
        new_dist[:k, :k] = dist_matrix
        new_dist[:k, k] = dists
        new_dist[k, :k] = dists
        new_dist[k, k] = 0
        return True, new_points, new_dist
    
    def greedy_from_order(order, points_arr, dist_matrix, time_limit):
        """Try to add points from order to the set."""
        for idx in order:
            if time.time() > time_limit:
                break
            p = all_points[idx]
            valid, new_pts, new_dm = check_and_add(p, points_arr, dist_matrix)
            if valid:
                points_arr = new_pts
                dist_matrix = new_dm
        return points_arr, dist_matrix
    
    def local_search(points_arr, dist_matrix, time_limit):
        """Try to swap points to improve the set size."""
        k = len(points_arr)
        in_set = np.zeros(N*N, dtype=bool)
        for pt in points_arr:
            in_set[pt[0] * N + pt[1]] = True
        
        # Try removing a point and adding two others
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            k = len(points_arr)
            for remove_idx in range(k):
                if time.time() > time_limit:
                    break
                # Try removing point remove_idx
                if k == 0:
                    break
                remaining_idx = [i for i in range(k) if i != remove_idx]
                rem_points = points_arr[remaining_idx]
                rem_dm = dist_matrix[np.ix_(remaining_idx, remaining_idx)]
                
                # Try to add two points
                candidates = np.where(~in_set)[0]
                random.shuffle(candidates)
                added = 0
                for c_idx in candidates:
                    if added >= 2:
                        break
                    p = all_points[c_idx]
                    valid, new_pts, new_dm = check_and_add(p, rem_points, rem_dm)
                    if valid:
                        rem_points = new_pts
                        rem_dm = new_dm
                        in_set[c_idx] = True
                        added += 1
                
                if added >= 1 and len(rem_points) > k - 1:
                    # We improved or maintained
                    if len(rem_points) > k:
                        points_arr = rem_points
                        dist_matrix = rem_dm
                        improved = True
                        break
                # Reset in_set for next iteration
                in_set[:] = False
                for pt in points_arr:
                    in_set[pt[0] * N + pt[1]] = True
        return points_arr, dist_matrix
    
    num_restarts = 5000
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        points_arr = np.zeros((0, 2), dtype=np.int32)
        dist_matrix = np.zeros((0, 0), dtype=np.int64)
        
        # Strategy: mix of random and structured initializations
        if restart % 5 == 0:
            # Random order
            order = np.random.permutation(N*N)
        elif restart % 5 == 1:
            # Row-major order
            order = np.arange(N*N)
        elif restart % 5 == 2:
            # Column-major order
            order = np.arange(N*N).reshape(N, N).T.ravel()
        elif restart % 5 == 3:
            # Diagonal order
            order = []
            for d in range(2*N - 1):
                for x in range(max(0, d-N+1), min(N, d+1)):
                    y = d - x
                    if 0 <= y < N:
                        order.append(x * N + y)
            order = np.array(order)
        else:
            # Random with bias toward spread
            order = np.random.permutation(N*N)
        
        points_arr, dist_matrix = greedy_from_order(order, points_arr, dist_matrix, deadline)
        
        # Second pass with different order
        if time.time() < deadline - 5:
            order2 = np.random.permutation(N*N)
            points_arr, dist_matrix = greedy_from_order(order2, points_arr, dist_matrix, deadline)
        
        # Local search
        if time.time() < deadline - 10 and len(points_arr) >= 5:
            points_arr, dist_matrix = local_search(points_arr, dist_matrix, deadline - 5)
        
        if len(points_arr) > best_size:
            best_size = len(points_arr)
            best_set = [(int(pt[0]), int(pt[1])) for pt in points_arr]
    
    return best_set
