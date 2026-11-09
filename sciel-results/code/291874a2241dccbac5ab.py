# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
import random

def nikodym_set() -> np.ndarray:
    deadline = time.time() + 620
    p = 7
    q = p * p
    n_points = q * q

    # Field arithmetic helpers
    def to_int(a, b):
        return a + b * p

    def to_pair(x):
        return (x % p, (x // p) % p)

    def f_add(x, y):
        xa, xb = to_pair(x)
        ya, yb = to_pair(y)
        return to_int((xa + ya) % p, (xb + yb) % p)

    def f_sub(x, y):
        xa, xb = to_pair(x)
        ya, yb = to_pair(y)
        return to_int((xa - ya) % p, (xb - yb) % p)

    def f_mul(x, y):
        xa, xb = to_pair(x)
        ya, yb = to_pair(y)
        real = (xa * ya + xb * yb * 3) % p
        imag = (xa * yb + xb * ya) % p
        return to_int(real, imag)

    # Directions: (1, m) for m in F_q and (0, 1)
    directions = [(1, m) for m in range(q)]
    directions.append((0, 1))
    n_dir = len(directions)

    # Precompute point indices and direction offsets
    # point_index maps (x, y) -> idx
    point_index = {}
    idx_to_point = {}
    for x in range(q):
        for y in range(q):
            idx = x * q + y
            point_index[(x, y)] = idx
            idx_to_point[idx] = (x, y)

    # Precompute for each point and each direction, the list of other points on that line
    # line_neighbors[pt][d] = list of indices of other points on the line through pt in direction d
    # There are q-1 such points for each direction.
    line_neighbors = np.zeros((n_points, n_dir, q-1), dtype=np.int32)
    
    # Also precompute the "affects" structure: for a point p, which (x, d) pairs are affected if p is removed?
    # If p is removed, for any line L containing p, the count of "missing points" on L increases by 1 for all x in L \ {p}.
    # Actually, it's simpler to just update counts directly during the search.
    
    # To speed up the initial setup and the search, we can precompute the lines.
    # lines[d][pt] = list of all q points on the line through pt in direction d.
    # But we only need the neighbors (excluding pt).
    
    for d_idx, (vx, vy) in enumerate(directions):
        for pt_idx in range(n_points):
            px, py = idx_to_point[pt_idx]
            for t in range(1, q):
                tx = f_add(px, f_mul(t, vx))
                ty = f_add(py, f_mul(t, vy))
                t_pt = point_index[(tx, ty)]
                line_neighbors[pt_idx, d_idx, t-1] = t_pt

    # Function to check if a set is a valid Nikodym set
    # in_N: boolean array of size n_points
    # Returns: True if valid, False otherwise
    def is_valid(in_N):
        # For each point x and direction d, count how many points in N are missing from the line through x in direction d.
        # Let missing_count[x][d] be the number of points y in line(x, d) \ {x} such that y is NOT in N.
        # A point x is "covered" by direction d if missing_count[x][d] == 0.
        # A point x is "valid" if it is covered by at least one direction.
        # The set is a Nikodym set if every point x in F_q^2 is valid.
        
        # We can compute missing_count efficiently.
        # missing_count[x][d] = sum( 1 for y in line(x, d) \ {x} if not in_N[y] )
        
        # Note: If x is in N, we still need to check if x is covered? 
        # The definition says: "every point x ... must lie on a line ... whose points other than x all belong to N".
        # This condition must hold for ALL x, including those in N.
        # So yes, every x must be covered by at least one direction.
        
        # Optimization: 
        # If we maintain `missing_count` array, we can check validity by seeing if for every x, 
        # there exists a d such that missing_count[x][d] == 0.
        
        # However, calculating missing_count from scratch is O(n_points * n_dir * q) = 2401 * 49 * 48 ~ 5.6M ops.
        # This is fast enough to do periodically or even at every step if we update incrementally.
        
        # Let's assume we have `missing_count` precomputed or updated.
        # For the check itself, we just need to ensure no point has missing_count[x][d] > 0 for ALL d.
        
        # Actually, a faster check:
        # A point x is "bad" if for all d, missing_count[x][d] > 0.
        # We can maintain a count of "how many directions cover x".
        # cover_count[x] = sum( 1 for d in directions if missing_count[x][d] == 0 )
        # If min(cover_count) > 0, it's valid.
        
        pass # We will implement incremental updates in the search loop instead of calling this from scratch.

    def run_local_search(initial_in_N, seed):
        random.seed(seed)
        np.random.seed(seed)
        
        in_N = initial_in_N.copy()
        
        # Initialize missing_count and cover_count
        # missing_count[x][d] = number of points y in line(x, d) \ {x} such that not in_N[y]
        missing_count = np.zeros((n_points, n_dir), dtype=np.int32)
        cover_count = np.zeros(n_points, dtype=np.int32)
        
        # Calculate initial missing_count
        # For each direction d, and each point x, count missing neighbors.
        # We can do this by iterating over lines.
        # For a fixed direction d, the points are partitioned into lines.
        # For each line L, for each x in L, missing_count[x][d] = (len(L) - 1) - (number of points in L \ {x} that are in N)
        # Actually, it's easier to just sum over neighbors.
        
        # Efficient initialization:
        # For each d, create an array of in_N values.
        # For each x, missing_count[x][d] = sum(1 for y in neighbors[x][d] if not in_N[y])
        
        for d in range(n_dir):
            for x in range(n_points):
                # neighbors are in line_neighbors[x, d, :]
                # Count how many are NOT in N
                # line_neighbors[x, d, :] contains indices.
                # We can use numpy sum.
                # But doing this in a loop is slow. Let's vectorize per direction if possible?
                # The neighbors are not contiguous in memory, so direct vectorization is hard without gathering.
                # However, 2401 * 49 * 48 is ~5.6M operations. In Python loop it might take ~1-2 seconds.
                # Let's try to optimize the loop.
                
                # Alternative: 
                # For each direction d, we can iterate through lines.
                # But let's stick to the simple loop first, maybe with some numpy tricks.
                
                # Actually, we can compute this faster by noting that:
                # sum_{y in N} 1_{y in line(x,d)} is the number of points in N on the line.
                # Let line_sum[x][d] = number of points in N on the line through x in direction d (including x if x in N).
                # Then missing_count[x][d] = (q - 1) - (line_sum[x][d] - (1 if in_N[x] else 0))
                
                # Computing line_sum is also O(N * D * Q) if done naively.
                
                # Let's just do the direct count for now, it's likely fast enough for 600s budget with few restarts.
                cnt = 0
                for y in line_neighbors[x, d, :]:
                    if not in_N[y]:
                        cnt += 1
                missing_count[x, d] = cnt
                
        # Calculate cover_count
        for x in range(n_points):
            cover_count[x] = np.sum(missing_count[x, :] == 0)
            
        # Check initial validity
        if np.min(cover_count) <= 0:
             # If invalid, we might need to add points back? 
             # The prompt implies we start with a set and prune.
             # If we start with a random set, it's likely invalid.
             # We need a strategy to handle invalid starts.
             # Option 1: Start with a valid set (e.g., all points) and prune.
             # Option 2: Start with random, then "fix" it by adding points until valid, then prune.
             # Option 3: Use a "repair" phase.
             
             # Let's try starting with a dense set (e.g., 80% random) which is likely valid, 
             # or just start with ALL points (which is always valid) and prune.
             # The "Randomized Start" idea should be "Randomized Pruning Order" or "Randomized Initial Exclusion".
             # Let's stick to starting with ALL points (valid) and using randomized greedy pruning.
             # This is safer.
             
             # If we really want to explore different structures, we can start with a random subset 
             # and if it's invalid, we can try to add points to make it valid? That's hard.
             
             # Let's refine the strategy:
             # 1. Start with All Points.
             # 2. Use a randomized priority queue for pruning.
             # 3. Try multiple restarts with different random seeds for the priority order.
             
             # Wait, if we start with All Points, the first few removals are easy.
             # The bottleneck is the "hard core".
             # Randomizing the order might help escape local minima?
             
             # Let's implement the "All Points + Randomized Greedy Pruning" with multiple restarts.
             
             pass
        
        # Priority Queue: Points with high cover_count are safer to remove.
        # We want to remove points that are "redundant".
        # A point x is redundant if removing it doesn't make any point y invalid.
        # Removing x increases missing_count[y][d] by 1 for all y, d such that x is in line(y, d) \ {y}.
        # This makes y invalid if cover_count[y] was 1 and the direction d that covered y now has missing_count > 0.
        
        # We maintain a heap of ( -cover_count, point_idx ).
        # We pop the point with highest cover_count.
        
        heap = [(-cover_count[i], i) for i in range(n_points) if in_N[i]]
        heapq.heapify(heap)
        
        removed_count = 0
        
        while heap:
            if time.time() > deadline - 30:
                break
                
            neg_cc, p_idx = heapq.heappop(heap)
            
            if not in_N[p_idx]:
                continue
                
            # Try to remove p_idx
            in_N[p_idx] = False
            
            # Update missing_count for all affected (y, d)
            # Affected pairs: y such that p_idx is in line(y, d) \ {y}.
            # This is equivalent to: for each direction d, and for each t in 1..q-1,
            # y = p_idx - t * v_d.
            # We precomputed line_neighbors, but that's neighbors OF x.
            # We need "predecessors" or "points that see p_idx".
            # Due to symmetry, the set of points y that see p_idx in direction d is the same as 
            # the set of points p_idx sees in direction d (shifted).
            # Actually, line(y, d) = line(p_idx, d). So the neighbors are the same set.
            # So the affected y's are exactly the points in line(p_idx, d) \ {p_idx}.
            
            affected = []
            for d in range(n_dir):
                for y in line_neighbors[p_idx, d, :]:
                    old_mc = missing_count[y, d]
                    missing_count[y, d] = old_mc + 1
                    affected.append((y, d, old_mc))
                    
                    # Update cover_count for y
                    # If old_mc was 0, it was covered by d. Now it's 1, so not covered by d.
                    # So cover_count[y] decreases by 1.
                    # If old_mc was > 0, it was not covered by d. Now it's old_mc+1, still not covered.
                    # So cover_count[y] only changes if old_mc == 0.
                    
            invalid = False
            for y, d, old_mc in affected:
                if old_mc == 0:
                    cover_count[y] -= 1
                    if cover_count[y] <= 0:
                        invalid = True
                        break
                        
            if invalid:
                # Rollback
                in_N[p_idx] = True
                for y, d, old_mc in affected:
                    missing_count[y, d] = old_mc
                    if old_mc == 0:
                        cover_count[y] += 1
            else:
                removed_count += 1
                # Update heap for affected points?
                # Their cover_count decreased, so they are LESS likely to be removed.
                # We don't need to push them back with higher priority.
                # However, other points might now be removable?
                # Removing a point makes the set smaller, so it's harder to cover others.
                # So removing a point never makes another point *more* removable (in terms of cover_count).
                # It only makes things worse.
                # So we don't need to update the heap with new candidates?
                # Wait, if we remove p, the cover_count of neighbors decreases.
                # So they are less safe.
                # The points that are safe are those with high cover_count.
                # Since cover_counts only decrease, the heap order is somewhat preserved,
                # but the values in the heap are stale (too high).
                # This is fine for a greedy heuristic; we just might pop a point that is no longer as safe,
                # but we check validity before committing.
                
                # However, we should push back the neighbors with their new (lower) cover_count?
                # No, that would lower their priority. We want to remove high cover_count points.
                # The points that are still high cover_count are the ones NOT affected, or affected but still high.
                # We don't need to push affected points back immediately unless their cover_count increased (which it doesn't).
                
                # Actually, we should probably re-heapify or just let the stale values be.
                # With stale values, we might pick a point that is now cover_count=0 (invalid to remove) 
                # but was cover_count=10 in the heap. We will try to remove it, fail, and rollback.
                # This is inefficient but correct.
                
                # To improve efficiency, we could update the heap entries for affected points.
                # But heapq doesn't support update. We'd have to push new entries and let old ones be ignored.
                # Pushing new entries for all affected points (up to 49*48=2352 points) for each removal 
                # would be expensive.
                
                # Let's just rely on the validity check.
                
        return in_N

    # Strategy:
    # 1. Start with all points.
    # 2. Run local search with a specific random seed.
    # 3. Repeat with different seeds, keeping the best result (smallest |N|).
    
    best_in_N = np.ones(n_points, dtype=bool)
    best_size = n_points
    
    # We have ~600s. 
    # One run takes ~2-5 seconds? 
    # Let's estimate: 
    # Initialization: 5.6M ops ~ 0.5s.
    # Pruning loop: 
    # Each removal involves ~2352 updates. 
    # If we remove 1000 points, that's 2.3M updates. 
    # Total time per run should be well under 1 second.
    # So we can do many restarts.
    
    num_restarts = 100
    
    for i in range(num_restarts):
        if time.time() > deadline - 60:
            break
            
        # Randomize the initial order? 
        # The heap is built based on cover_count. 
        # To introduce randomness, we can add a small random jitter to the priority.
        # Or we can start with a random subset of points removed?
        # Let's try starting with all points, but randomizing the tie-breaking in the heap.
        # Or, better: Randomize the *initial* set slightly?
        # If we remove a random 10% of points initially, it might be invalid.
        # Let's stick to All Points, but randomize the heap insertion order or priorities.
        
        # Actually, the greedy choice is deterministic given the cover_counts.
        # To explore different paths, we can randomize the *order* in which we consider points with the same cover_count.
        # Or, we can add a random noise to the cover_count used for priority.
        
        seed = i * 12345 + 6789
        random.seed(seed)
        
        # Modified run_local_search to accept a random seed for jitter
        # We'll create a new priority: -cover_count + random_noise
        
        in_N = np.ones(n_points, dtype=bool)
        
        # Initialize missing_count and cover_count (same as before)
        missing_count = np.zeros((n_points, n_dir), dtype=np.int32)
        cover_count = np.zeros(n_points, dtype=np.int32)
        
        for d in range(n_dir):
            for x in range(n_points):
                cnt = 0
                for y in line_neighbors[x, d, :]:
                    if not in_N[y]:
                        cnt += 1
                missing_count[x, d] = cnt
                
        for x in range(n_points):
            cover_count[x] = np.sum(missing_count[x, :] == 0)
            
        # Build heap with jitter
        heap = []
        for i in range(n_points):
            if in_N[i]:
                # Jitter to break ties and explore different orders
                jitter = random.uniform(0, 1.0)
                priority = -cover_count[i] + jitter
                heap.append((priority, i))
        heapq.heapify(heap)
        
        while heap:
            if time.time() > deadline - 30:
                break
                
            priority, p_idx = heapq.heappop(heap)
            
            if not in_N[p_idx]:
                continue
                
            # Check if it's still worth trying? 
            # The priority might be stale. 
            # We'll just try to remove.
            
            in_N[p_idx] = False
            
            affected = []
            for d in range(n_dir):
                for y in line_neighbors[p_idx, d, :]:
                    old_mc = missing_count[y, d]
                    missing_count[y, d] = old_mc + 1
                    affected.append((y, d, old_mc))
                    
            invalid = False
            for y, d, old_mc in affected:
                if old_mc == 0:
                    cover_count[y] -= 1
                    if cover_count[y] <= 0:
                        invalid = True
                        break
                        
            if invalid:
                in_N[p_idx] = True
                for y, d, old_mc in affected:
                    missing_count[y, d] = old_mc
                    if old_mc == 0:
                        cover_count[y] += 1
            else:
                # Successful removal
                pass
                
        current_size = np.sum(in_N)
        if current_size < best_size:
            best_size = current_size
            best_in_N = in_N.copy()
            
    # Convert best_in_N to the required format
    points = []
    for i in range(n_points):
        if best_in_N[i]:
            x, y = idx_to_point[i]
            xa, xb = to_pair(x)
            ya, yb = to_pair(y)
            points.append([[xa, xb], [ya, yb]])
            
    if not points:
        # Fallback: return all points
        for x in range(q):
            for y in range(q):
                xa, xb = to_pair(x)
                ya, yb = to_pair(y)
                points.append([[xa, xb], [ya, yb]])
                
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
