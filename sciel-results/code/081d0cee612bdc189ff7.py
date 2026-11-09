import time
import sys
import random
import math

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Keep 50s margin for 330s limit
    
    # Precompute all points
    points = [(x, y) for x in range(N) for y in range(N)]
    num_points = len(points)
    
    # Precompute squared distances between all pairs
    # dist[i][j] is squared distance between points[i] and points[j]
    # Using a list of lists for speed, though memory is 64*64*64*64 ~ 16M floats? 
    # 4096 points. 4096*4096 = 16M entries. Python ints are heavy.
    # Better: compute on the fly but cache? Or use numpy?
    # Let's use a list of lists of ints. 16M ints might be too slow to initialize or memory heavy in pure Python lists.
    # Let's stick to on-the-fly calculation but optimize the check.
    # Actually, 4096 points. 
    # Let's try to precompute distance sets for each point relative to all others?
    # No, the constraint is dynamic.
    
    # Alternative: Represent the set of "forbidden" distances for each point.
    # If we have a set S, for each p in S, we have a set D_p of distances to other points in S.
    # A new point q can be added if:
    # 1. dist(q, p) is unique for all p in S (i.e., dist(q, p) != dist(q, p') for p != p' in S)
    # 2. For each p in S, dist(q, p) is not in D_p (i.e., it doesn't create an isosceles triangle with vertex p)
    
    # This is exactly the logic in the seed. The seed does greedy.
    # CSP approach:
    # Variables: points (0 to 4095)
    # Domains: {True, False} (include or exclude)
    # Constraints: For any triplet (i, j, k), not (dist(i,j) == dist(i,k) or dist(i,j) == dist(j,k) or dist(j,k) == dist(i,k))
    # Wait, the constraint is "no three points form an isosceles triangle".
    # This means for any distinct i, j, k in S:
    # dist(i,j) != dist(i,k)
    # dist(i,j) != dist(j,k)
    # dist(i,k) != dist(j,k)
    
    # Backtracking search:
    # We select points one by one.
    # State: Current set S (list of indices), and for each p in S, the set of distances D_p.
    # Also, we need to ensure that for any new point q, the distances to all p in S are distinct.
    
    # Heuristic: MRV (Minimum Remaining Values) is hard to implement efficiently here without heavy data structures.
    # Let's try a different heuristic: "Most Constrained" or "Degree".
    # Or simply: Try to add points in an order that maximizes the chance of success?
    # Randomized Restarting Backtracking.
    
    # To make it different from simple greedy:
    # Greedy picks the *first* valid point.
    # Backtracking picks a point, and if it leads to a dead end (no more points can be added), it backtracks and tries the *next* valid point.
    
    # Optimization: 
    # If we just do backtracking on a 4096-node graph, the branching factor is huge.
    # We need a strong pruning strategy.
    
    # Let's use a hybrid: 
    # 1. Use the greedy approach to find a good initial solution quickly.
    # 2. Use local search / hill climbing? No, that's not CSP.
    # 3. Use Backtracking with Randomized Ordering and Early Pruning.
    
    # Key Insight for Speed:
    # Checking if a point q can be added to S requires O(|S|) distance calcs and set lookups.
    # If |S| is large (~100+), this is fast.
    # The bottleneck is the depth of the search.
    
    # Let's implement a simple Backtracking Solver.
    
    best_set = []
    
    # Precompute coordinates for fast access
    coords = points
    
    def check_add(S, dist_from, q_idx):
        q = coords[q_idx]
        qx, qy = q
        d2s = set()
        for p_idx in S:
            p = coords[p_idx]
            px, py = p
            dx = qx - px
            dy = qy - py
            d2 = dx*dx + dy*dy
            if d2 in d2s:
                return False
            d2s.add(d2)
            if d2 in dist_from[p_idx]:
                return False
        return True

    def do_add(S, dist_from, q_idx):
        S.append(q_idx)
        q = coords[q_idx]
        qx, qy = q
        dist_from[q_idx] = set()
        for p_idx in S[:-1]:
            p = coords[p_idx]
            px, py = p
            dx = qx - px
            dy = qy - py
            d2 = dx*dx + dy*dy
            dist_from[p_idx].add(d2)
            dist_from[q_idx].add(d2)

    def do_remove(S, dist_from, q_idx):
        # This is tricky because we modified dist_from for all other points.
        # We need to undo those changes.
        # For backtracking, it's safer to copy the state or use a persistent data structure?
        # Copying dist_from (dict of sets) is O(|S| * avg_degree).
        # If |S| is ~100, and we copy, it's manageable.
        # Let's just pass copies of S and dist_from to the recursive function.
        pass # Not needed if we use copies in recursion

    def backtrack(S, dist_from, remaining_indices, depth, max_depth, current_best):
        nonlocal best_set
        if time.time() > deadline:
            return
        
        if len(S) > len(current_best):
            current_best = S[:]
            if len(current_best) > len(best_set):
                best_set = current_best[:]
        
        if depth >= max_depth:
            return
            
        if not remaining_indices:
            return
            
        # Heuristic: Pick a random point from remaining to try first?
        # Or pick the one that is "most compatible"?
        # Let's just iterate through a shuffled list of remaining indices.
        
        # To avoid deep recursion overhead, we can limit depth or use iterative deepening?
        # Given the time limit, deep backtracking might be too slow if branching is high.
        
        # Strategy: 
        # 1. Shuffle remaining indices.
        # 2. For each candidate q:
        #    a. Check if q can be added.
        #    b. If yes, add it, recurse with remaining - q.
        #    c. If we find a solution better than current best, we might still want to explore others?
        #       No, we want the MAX. So we must explore all valid branches?
        #       That's exponential.
        
        # We need a smarter search.
        # Let's use "Limited Discrepancy Search" or just "Restarting Random Greedy" but with a twist:
        # The seed program does "Restarting Random Greedy".
        # My structural change: **Local Search / Hill Climbing on the set of points**.
        
        # Wait, the prompt asks for a MATERIAL DIFFERENCE.
        # Greedy is "constructive". 
        # Local Search is "destructive/constructive".
        # CSP Backtracking is "complete" (in theory).
        
        # Let's try **Simulated Annealing** or **Hill Climbing with Random Restarts**.
        # State: A set of points S.
        # Move: Remove a point p from S, or Add a point q to S, or Swap p for q.
        # Constraint: No isosceles triangles.
        # Objective: Maximize |S|.
        
        # Problem: Maintaining validity during moves.
        # If I remove a point, the set remains valid.
        # If I add a point, I must check validity.
        
        # Algorithm:
        # 1. Start with a valid set S (found by greedy).
        # 2. While time remains:
        #    a. Try to add a random point not in S. If valid, keep it.
        #    b. If cannot add any, try to remove a random point from S, then try to add two different points?
        #       This allows escaping local optima where no single point can be added, but a swap or replacement could.
        
        # Let's implement **Greedy + Local Search (Add/Remove/Swap)**.
        
        # Actually, a simpler structural change that is very different from pure greedy:
        # **Genetic Algorithm**? Too complex to implement robustly in 300s.
        
        # Let's go with **Backtracking with Heuristics (Most Constrained Variable)**.
        # But to make it fast, we will limit the depth of the backtracking to a small number of "decisions" after a greedy seed, or use it to refine.
        
        # Refined Strategy:
        # 1. Generate a set S using the greedy algorithm (fast, gets ~9-10 points).
        # 2. Use **Hill Climbing**:
        #    - Identify "bottleneck" points: points that prevent many other points from being added.
        #    - Remove one such point.
        #    - Run greedy again on the remaining space, potentially adding more points.
        #    - If the new set is larger, keep it.
        
        # This is a "Greedy with Backtracking" or "Iterative Deepening A*" style.
        
        # Let's implement **Iterative Greedy with Randomized Removal**.
        
        def greedy_construct(allowed_mask, order):
            """
            allowed_mask: set of indices allowed to be added.
            order: list of indices to consider in this order.
            Returns: list of indices in S.
            """
            S = []
            dist_from = {}
            
            for q_idx in order:
                if q_idx not in allowed_mask:
                    continue
                if check_add(S, dist_from, q_idx):
                    do_add(S, dist_from, q_idx)
                    
            return S

        # Main Loop
        # 1. Get initial solution with random greedy.
        # 2. Try to improve by removing points and re-greedy-ing.
        
        # Pre-shuffle indices for randomization
        indices = list(range(num_points))
        
        # Initial Greedy
        random.shuffle(indices)
        allowed = set(indices)
        S = greedy_construct(allowed, indices)
        
        if len(S) > len(best_set):
            best_set = S[:]
            
        # Local Search
        # We will try to "shake" the solution.
        # Strategy: Remove k random points from S, then re-run greedy on the remaining allowed points (excluding the removed ones? No, including them? 
        # If we remove them, we can potentially add different points that were previously blocked).
        
        # Actually, if we remove points from S, the constraints are relaxed. 
        # We should re-run greedy on the *entire* set of points, but with a bias or different order?
        # Or, we can just re-run greedy on the points NOT in the new S?
        
        # Better: 
        # 1. Start with S.
        # 2. Pick a point p in S to remove.
        # 3. New S' = S \ {p}.
        # 4. Run Greedy on all points not in S', starting from scratch? 
        #    No, we want to keep the other points of S.
        #    So, we fix S \ {p} as "already added" and then greedily add the rest.
        
        def greedy_extension(fixed_S, allowed_mask):
            """
            fixed_S: list of indices already in the set.
            allowed_mask: set of indices that can be added.
            Returns: extended list of indices.
            """
            S = fixed_S[:]
            dist_from = {}
            # Rebuild dist_from for fixed_S
            for i, p_idx in enumerate(S):
                dist_from[p_idx] = set()
            for i in range(len(S)):
                for j in range(i):
                    p1 = coords[S[i]]
                    p2 = coords[S[j]]
                    dx = p1[0] - p2[0]
                    dy = p1[1] - p2[1]
                    d2 = dx*dx + dy*dy
                    dist_from[S[i]].add(d2)
                    dist_from[S[j]].add(d2)
            
            # Now greedily add from allowed_mask
            # We need an order. Let's use random shuffle of allowed_mask.
            candidates = [idx for idx in allowed_mask if idx not in fixed_S]
            random.shuffle(candidates)
            
            for q_idx in candidates:
                if check_add(S, dist_from, q_idx):
                    do_add(S, dist_from, q_idx)
                    
            return S

        # Loop
        while time.time() < deadline:
            # Try to improve best_set
            current_S = best_set[:]
            if not current_S:
                break
                
            # Pick a random point to remove
            remove_idx = random.choice(current_S)
            fixed_S = [p for p in current_S if p != remove_idx]
            
            # The allowed mask is all points. 
            # We want to see if we can find a larger set by removing this point.
            # We run greedy_extension on fixed_S with all points allowed.
            # Note: greedy_extension will try to add points. It might re-add remove_idx?
            # Yes, it might. If it does, we are back to the same or similar set.
            # To force exploration, we might want to forbid re-adding the removed point for a few steps?
            # Or just let it be. If it finds a larger set, great.
            
            new_S = greedy_extension(fixed_S, set(indices))
            
            if len(new_S) > len(best_set):
                best_set = new_S[:]
                
            # Additional strategy: Remove 2 points?
            # Or swap?
            # Let's try removing 2 random points.
            if len(current_S) > 2:
                r1, r2 = random.sample(current_S, 2)
                fixed_S2 = [p for p in current_S if p not in (r1, r2)]
                new_S2 = greedy_extension(fixed_S2, set(indices))
                if len(new_S2) > len(best_set):
                    best_set = new_S2[:]

            # Also, try a fresh random greedy restart occasionally to escape local optima entirely
            if random.random() < 0.1:
                random.shuffle(indices)
                allowed = set(indices)
                S_restart = greedy_construct(allowed, indices)
                if len(S_restart) > len(best_set):
                    best_set = S_restart[:]

        return [(coords[i][0], coords[i][1]) for i in best_set]

# Test
if __name__ == "__main__":
    result = construct()
    print(len(result))
