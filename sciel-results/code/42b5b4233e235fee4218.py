import time
import random
import numpy as np

N = 64
TIME_LIMIT = 320.0

def construct():
    start_time = time.time()
    
    def time_left():
        return TIME_LIMIT - (time.time() - start_time)
    
    # Precompute all grid points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    all_points_set = set(all_points)
    
    def sq_dist(p1, p2):
        return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

    def get_pair_dists(pts):
        """Get set of squared distances between all pairs in pts."""
        d = set()
        n = len(pts)
        for i in range(n):
            for j in range(i+1, n):
                d.add(sq_dist(pts[i], pts[j]))
        return d

    def is_valid_candidate(q, pts, pair_dists):
        """
        Check if q can be added to pts without forming isosceles triangles.
        pts: list of points
        pair_dists: set of squared distances between all pairs in pts
        """
        # 1. Check if q is equidistant to any two points in pts (apex at q)
        dists_from_q = []
        for p in pts:
            d = sq_dist(q, p)
            dists_from_q.append(d)
        
        if len(set(dists_from_q)) < len(pts):
            return False
        
        # 2. Check if any distance from q to a point in pts matches any pair distance in pts
        # This covers cases where apex is at a point in pts
        d_set = set(dists_from_q)
        if d_set & pair_dists:
            return False
            
        return True

    def greedy_initial(seed):
        """Greedy construction to get a starting set."""
        rng = random.Random(seed)
        pts = []
        pair_dists = set()
        
        candidates = list(all_points)
        rng.shuffle(candidates)
        
        for q in candidates:
            if is_valid_candidate(q, pts, pair_dists):
                pts.append(q)
                # Update pair_dists
                for p in pts[:-1]:
                    pair_dists.add(sq_dist(q, p))
        
        return pts, pair_dists

    def simulated_annealing(initial_pts, initial_pair_dists, time_budget):
        """
        Simulated Annealing to improve the set.
        """
        pts = list(initial_pts)
        pair_dists = set(initial_pair_dists)
        
        current_size = len(pts)
        best_size = current_size
        best_pts = list(pts)
        
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.999
        
        rng = random.Random(42)
        
        while time_budget > 0 and T > T_min:
            # Pick a random point to remove
            if not pts:
                break
            i = rng.randint(0, len(pts) - 1)
            p_remove = pts[i]
            
            # Pick a random candidate to add
            # We can pick from all grid points not in pts
            # To be efficient, we can just pick a random grid point and check if it's in pts
            # Or maintain a list of available points.
            # For N=64, 4096 points. Checking membership in a set is fast.
            
            # Create a set of current points for fast lookup
            pts_set = set(pts)
            
            # Pick random candidate
            # We can bias towards points far from current set? Or just random.
            q_add = (rng.randint(0, N-1), rng.randint(0, N-1))
            
            if q_add in pts_set:
                # If we picked a point already in the set, we effectively just removed p_remove
                # This is a "deletion" move.
                new_pts = [p for j, p in enumerate(pts) if j != i]
                # Recalculate pair_dists for new_pts is expensive O(k^2).
                # We can update pair_dists by removing distances involving p_remove.
                # But we need to know which distances to remove.
                # It's easier to just recalculate if k is small, or maintain a count.
                # Let's just recalculate for simplicity if k < 50, else use a counter map.
                # Actually, let's stick to swaps to keep size constant or growing.
                # If we allow deletions, we might shrink.
                
                # Let's try a swap: remove p_remove, add q_add (if q_add not in pts)
                # If q_add is in pts, we just removed p_remove.
                # Let's enforce q_add not in pts for the swap logic to be clean.
                # We'll retry if q_add is in pts.
                continue

            # Perform swap: remove p_remove, add q_add
            # New set S' = (S \ {p_remove}) U {q_add}
            
            # 1. Check validity of q_add in (S \ {p_remove})
            # We need pair_dists of (S \ {p_remove})
            # And we need to check if q_add creates conflicts with (S \ {p_remove})
            
            # Efficient update:
            # Remove p_remove from pts
            pts_no_remove = [p for j, p in enumerate(pts) if j != i]
            
            # Recalculate pair_dists for pts_no_remove?
            # Or calculate the "new" pair_dists?
            # Let's just calculate validity of q_add against pts_no_remove.
            
            # We need pair_dists of pts_no_remove.
            # We can derive it from current pair_dists by removing distances involving p_remove.
            # But we need to be careful about duplicates in the set.
            # Using a set loses count info.
            
            # Alternative: Just check validity directly.
            # is_valid_candidate(q_add, pts_no_remove, get_pair_dists(pts_no_remove))
            # This is O(k^2) to get pair_dists.
            
            # To speed this up, let's maintain a Counter for pair_dists.
            # But for now, let's see if O(k^2) per step is fast enough.
            # k ~ 20. k^2 = 400.
            # 400 ops per step. 300s / (400 * 100ns) = 750,000 steps.
            # This might be too slow in Python if not optimized.
            
            # Let's optimize is_valid_candidate to take pts and compute pair_dists internally?
            # No, we already have a version that takes pair_dists.
            # We need pair_dists of pts_no_remove.
            
            # Let's try a different approach:
            # Precompute all pairwise squared distances for the grid? No, too big.
            
            # Let's use the "Counter" approach for pair_dists to allow efficient removal.
            # But is_valid_candidate needs a set.
            
            # Let's rewrite the loop to use a fast validity check that computes pair_dists on the fly
            # but only for the subset.
            
            # Actually, if we just do random swaps, we can check validity of the NEW set directly.
            # New set = pts_no_remove + [q_add]
            # We know pts_no_remove is valid (subset of valid set).
            # We only need to check if q_add conflicts with pts_no_remove.
            # This requires pair_dists of pts_no_remove.
            
            # Let's just recalculate pair_dists for pts_no_remove.
            # It's O(k^2).
            
            pd_no_remove = get_pair_dists(pts_no_remove)
            
            if is_valid_candidate(q_add, pts_no_remove, pd_no_remove):
                # Valid swap
                new_pts = pts_no_remove + [q_add]
                new_pair_dists = pd_no_remove
                # Add distances from q_add to all in pts_no_remove
                for p in pts_no_remove:
                    new_pair_dists.add(sq_dist(q_add, p))
                
                # Accept move (always if valid, since it maintains size)
                # Actually, we might want to allow invalid moves in SA.
                # But for simplicity, let's stick to valid swaps first.
                # If we can't find a valid swap, we are stuck.
                # Then we might need to allow "invalid" states or restart.
                
                pts = new_pts
                pair_dists = new_pair_dists
                
                if len(pts) > best_size:
                    best_size = len(pts)
                    best_pts = list(pts)
            else:
                # Invalid swap. In standard SA, we might accept with prob exp(-DeltaE/T).
                # Here, "energy" is number of conflicts.
                # We can count conflicts.
                # For now, let's just skip.
                pass
            
            # Cooling
            T *= cooling_rate
            
            # Check time
            if time.time() - start_time > TIME_LIMIT - 5:
                break

        return best_pts

    # Strategy:
    # 1. Run several greedy initializations.
    # 2. Pick the best one.
    # 3. Run SA on it.
    
    best_initial = []
    best_initial_pd = set()
    max_initial_size = 0
    
    # Try a few greedy seeds
    for seed in range(10):
        if time.time() - start_time > TIME_LIMIT - 60:
            break
        pts, pd = greedy_initial(seed)
        if len(pts) > max_initial_size:
            max_initial_size = len(pts)
            best_initial = pts
            best_initial_pd = pd

    # If greedy didn't find anything, start with empty
    if not best_initial:
        best_initial = []
        best_initial_pd = set()

    # Run SA
    if time.time() - start_time < TIME_LIMIT - 10:
        remaining_time = TIME_LIMIT - 10 - (time.time() - start_time)
        final_pts = simulated_annealing(best_initial, best_initial_pd, remaining_time)
        if len(final_pts) > len(best_initial):
            best_initial = final_pts

    return best_initial

if __name__ == "__main__":
    construct()
