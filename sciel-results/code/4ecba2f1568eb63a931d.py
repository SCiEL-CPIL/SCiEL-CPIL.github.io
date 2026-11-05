import time
import random
import math
import numpy as np
from collections import Counter

def construct():
    deadline = time.time() + 50.0
    random.seed(42)
    N = 30

    # Precompute all triplets
    triplets = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                triplets.append((i, j, k))
    triplets = np.array(triplets) # Shape (4060, 3)
    
    # For each index, find the indices of the triplets it belongs to
    # This allows fast delta calculation
    idx_triplet_indices = [[] for _ in range(N)]
    for t_idx, (i, j, k) in enumerate(triplets):
        idx_triplet_indices[i].append(t_idx)
        idx_triplet_indices[j].append(t_idx)
        idx_triplet_indices[k].append(t_idx)
    # Convert to numpy arrays for fast indexing
    idx_triplet_indices = [np.array(lst, dtype=int) for lst in idx_triplet_indices]

    def compute_areas(points):
        """
        Compute doubled areas for all triples.
        points: (N, 2) numpy array
        returns: (num_triplets,) numpy array of areas
        """
        i, j, k = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        
        # (p2 - p1) x (p3 - p1)
        dx1 = p2[:, 0] - p1[:, 0]
        dy1 = p2[:, 1] - p1[:, 1]
        dx2 = p3[:, 0] - p1[:, 0]
        dy2 = p3[:, 1] - p1[:, 1]
        
        areas = np.abs(dx1 * dy2 - dx2 * dy1)
        return areas

    def full_count(points):
        areas = compute_areas(points)
        pos_areas = areas[areas > 0]
        if len(pos_areas) == 0:
            return Counter(), 0, 0
        c = Counter(pos_areas.tolist())
        max_area, max_count = c.most_common(1)[0]
        return c, max_area, max_count

    def count_target(points, target):
        areas = compute_areas(points)
        return np.sum(areas == target)

    def get_delta(points, idx, new_pt, target):
        """
        Calculate change in count of target area if point idx moves to new_pt.
        """
        t_indices = idx_triplet_indices[idx]
        if len(t_indices) == 0:
            return 0
        
        # Get the triplets involving idx
        t = triplets[t_indices]
        
        # Identify which of i,j,k is idx to extract the other two
        # t[:, 0], t[:, 1], t[:, 2]
        is_i = (t[:, 0] == idx)
        is_j = (t[:, 1] == idx)
        is_k = (t[:, 2] == idx)
        
        # Extract the other two points
        # For each triplet, we need the coordinates of the two points NOT at idx
        # We can create arrays of the other indices
        other_idx_1 = np.where(is_i, t[:, 1], np.where(is_j, t[:, 0], t[:, 2]))
        other_idx_2 = np.where(is_i, t[:, 2], np.where(is_j, t[:, 2], t[:, 1]))
        
        p_other1 = points[other_idx_1]
        p_other2 = points[other_idx_2]
        
        # Current areas for these triples
        # We need to be careful with the formula. The area formula is symmetric.
        # Let's just recompute the area for these specific triples using the standard formula
        # but we know one point is fixed (new_pt) and the other two are p_other1, p_other2.
        
        # Vectorized area calculation for new_pt, p_other1, p_other2
        # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
        # Here p1=new_pt, p2=p_other1, p3=p_other2
        
        dx1 = p_other1[:, 0] - new_pt[0]
        dy1 = p_other1[:, 1] - new_pt[1]
        dx2 = p_other2[:, 0] - new_pt[0]
        dy2 = p_other2[:, 1] - new_pt[1]
        
        new_areas = np.abs(dx1 * dy2 - dx2 * dy1)
        
        # Get old areas for these specific triples from the full area array?
        # It's faster to recompute old areas for these specific triples too, or store them.
        # Since we are in a loop, storing the full area array is efficient.
        # Let's pass the current areas array to avoid recomputation.
        
        # Wait, the function signature doesn't have areas. 
        # I will modify the search loop to maintain `current_areas`.
        
        # For now, this function is a placeholder for the logic. 
        # I will implement the search loop inline to use `current_areas`.
        pass

    def generate_grid_config():
        return [(x, y) for x in range(-2, 4) for y in range(-2, 3)]

    def generate_dense_config():
        return [(x, y) for x in range(-2, 3) for y in range(-3, 3)]

    def generate_lattice_config(scale=1):
        points = []
        for q in range(-4, 5):
            for r in range(-4, 5):
                x = 2*q + r
                y = r
                if -30 <= x <= 30 and -30 <= y <= 30:
                    points.append((x, y))
        random.shuffle(points)
        return points[:30]

    initial_configs = [
        generate_grid_config(),
        generate_dense_config(),
        generate_lattice_config(1),
        generate_lattice_config(2),
        [(x, y) for x in range(-3, 3) for y in range(-3, 3) if abs(x) + abs(y) <= 4][:30],
    ]

    best_initial = None
    best_initial_count = 0

    for config in initial_configs:
        seen = set()
        filtered = []
        for p in config:
            if p not in seen and -30 <= p[0] <= 30 and -30 <= p[1] <= 30:
                seen.add(p)
                filtered.append(p)
        while len(filtered) < 30:
            p = (random.randint(-30, 30), random.randint(-30, 30))
            if p not in seen:
                seen.add(p)
                filtered.append(p)
        filtered = filtered[:30]
        
        pts_np = np.array(filtered, dtype=float)
        counter, ba, bc = full_count(pts_np)
        if bc > best_initial_count:
            best_initial_count = bc
            best_initial = filtered

    coords = list(best_initial)
    pts_np = np.array(coords, dtype=float)
    counter, target, current_count = full_count(pts_np)
    best_overall = current_count
    best_points = list(coords)
    best_target = target
    current_areas = compute_areas(pts_np)

    # Main Simulated Annealing
    T = 10.0
    cooling_rate = 0.99
    no_improve = 0
    max_no_improve = 200
    restarts = 0
    max_restarts = 5

    while time.time() < deadline - 5:
        if no_improve > max_no_improve:
            restarts += 1
            if restarts > max_restarts:
                break

            counter, ba, bc = full_count(pts_np)
            if bc > best_overall:
                best_overall = bc
                best_points = list(coords)
                best_target = ba
                best_areas = current_areas.copy()

            target = ba
            current_count = bc
            no_improve = 0
            max_no_improve += 50
            T = 10.0
            
            # Perturb
            for _ in range(3):
                idx = random.randint(0, N-1)
                new_pt = (random.randint(-30, 30), random.randint(-30, 30))
                if new_pt not in coords:
                    coords[idx] = new_pt
                    pts_np = np.array(coords, dtype=float)
                    current_areas = compute_areas(pts_np)
                    current_count = np.sum(current_areas == target)
                    break
            continue

        idx = random.randint(0, N-1)
        old_pt = coords[idx]
        
        # Generate candidates
        candidates = []
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                new_pt = (old_pt[0] + dx, old_pt[1] + dy)
                if new_pt != old_pt and new_pt not in coords and -30 <= new_pt[0] <= 30 and -30 <= new_pt[1] <= 30:
                    candidates.append(new_pt)
        
        if not candidates:
            no_improve += 1
            T *= cooling_rate
            continue

        # Evaluate best candidate
        best_delta = -999
        best_new_pt = None
        
        # Precompute old matches count for this idx? 
        # Actually, we can just compute the delta for each candidate.
        # To speed up, we can compute the new areas for the affected triples for all candidates at once?
        # No, candidates are different points.
        
        # Let's just iterate candidates. 
        # Optimization: If we have many candidates, we can vectorize the area computation for all of them?
        # Candidates ~ 40. Affected triples ~ 406.
        # We can create a (40, 406) array of new areas?
        # That's 16000 floats. Very fast.
        
        cand_array = np.array(candidates, dtype=float) # (C, 2)
        
        # Get indices of affected triples
        t_indices = idx_triplet_indices[idx]
        t = triplets[t_indices]
        
        is_i = (t[:, 0] == idx)
        is_j = (t[:, 1] == idx)
        is_k = (t[:, 2] == idx)
        
        other_idx_1 = np.where(is_i, t[:, 1], np.where(is_j, t[:, 0], t[:, 2]))
        other_idx_2 = np.where(is_i, t[:, 2], np.where(is_j, t[:, 2], t[:, 1]))
        
        p_other1 = pts_np[other_idx_1] # (T, 2)
        p_other2 = pts_np[other_idx_2] # (T, 2)
        
        # Compute areas for all candidates and all affected triples
        # cand_array: (C, 1, 2) -> (C, 1, 2)
        # p_other1: (1, T, 2) -> (1, T, 2)
        
        # We want (C, T) areas.
        # dx1 = p_other1 - cand_array
        # dy1 = p_other1 - cand_array
        # dx2 = p_other2 - cand_array
        # dy2 = p_other2 - cand_array
        
        dx1 = p_other1[0] - cand_array[:, 0, None] # Wait, broadcasting
        # p_other1 is (T, 2). cand_array is (C, 2).
        # We want (C, T, 2) - (C, 1, 2) -> (C, T, 2)
        
        dx1 = p_other1[None, :, 0] - cand_array[:, 0, None] # (C, T)
        dy1 = p_other1[None, :, 1] - cand_array[:, 1, None] # (C, T)
        dx2 = p_other2[None, :, 0] - cand_array[:, 0, None] # (C, T)
        dy2 = p_other2[None, :, 1] - cand_array[:, 1, None] # (C, T)
        
        new_areas_matrix = np.abs(dx1 * dy2 - dx2 * dy1) # (C, T)
        
        # Count matches for target in each row
        new_counts = np.sum(new_areas_matrix == target, axis=1)
        
        # Get old counts for these triples
        old_areas_subset = current_areas[t_indices]
        old_count = np.sum(old_areas_subset == target)
        
        # Delta for each candidate
        deltas = new_counts - old_count
        
        best_delta_idx = np.argmax(deltas)
        best_delta = deltas[best_delta_idx]
        best_new_pt = candidates[best_delta_idx]
        
        if best_delta > 0:
            accept = True
        else:
            if T > 0.1:
                accept_prob = math.exp(best_delta / T)
                accept = random.random() < accept_prob
            else:
                accept = False
        
        if accept:
            coords[idx] = best_new_pt
            pts_np = np.array(coords, dtype=float)
            
            # Update current_areas
            # Only need to update the affected triples
            current_areas[t_indices] = new_areas_matrix[best_delta_idx]
            
            current_count += best_delta
            no_improve = 0
            
            if current_count > best_overall:
                best_overall = current_count
                best_points = list(coords)
                best_target = target
                best_areas = current_areas.copy()
        else:
            no_improve += 1
            
        T *= cooling_rate

    # Final verification
    if len(best_points) != N or len(set(best_points)) != N:
        return [(x, y) for x in range(6) for y in range(5)]
    for p in best_points:
        if not (-30 <= p[0] <= 30 and -30 <= p[1] <= 30):
            return [(x, y) for x in range(6) for y in range(5)]

    return best_points
