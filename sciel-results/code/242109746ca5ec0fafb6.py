"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import random
import time

# Precompute coordinates and squared distances
# Points are indexed by 0 to N*N-1
# point index i corresponds to (i // N, i % N)
num_points = N * N
coords = [(i // N, i % N) for i in range(num_points)]

# dist[i][j] is squared distance between point i and point j
dist = [[0] * num_points for _ in range(num_points)]

for i in range(num_points):
    x1, y1 = coords[i]
    for j in range(i + 1, num_points):
        x2, y2 = coords[j]
        d = (x1 - x2) ** 2 + (y1 - y2) ** 2
        dist[i][j] = d
        dist[j][i] = d

def is_isosceles(i, j, k):
    """Check if points i, j, k form an isosceles triangle (including degenerate)."""
    d1 = dist[i][j]
    d2 = dist[i][k]
    d3 = dist[j][k]
    return d1 == d2 or d1 == d3 or d2 == d3

def can_add(p, S_set):
    """Check if point p can be added to set S_set without creating an isosceles triangle."""
    # S_set is a set of point indices
    # We need to check if p forms an isosceles triangle with any pair in S_set
    # Note: If S_set has < 2 points, it's always valid.
    if len(S_set) < 2:
        return True
        
    # Optimization: iterate over pairs in S_set
    # Since S_set is a set, we can convert to list once outside if called frequently, 
    # but here it's called with a set. 
    # For efficiency in the search loop, we will keep S as a list and a set.
    s_list = list(S_set)
    for i in range(len(s_list)):
        a = s_list[i]
        for j in range(i + 1, len(s_list)):
            b = s_list[j]
            if is_isosceles(p, a, b):
                return False
    return True

def construct():
    start_time = time.time()
    time_limit = 295  # 5s margin
    
    best_size = num_points
    best_set = None
    
    # Random seed for reproducibility within the run, but different each call? 
    # The task doesn't specify, but standard practice is to use a fixed seed or system time.
    # Using a fixed seed makes debugging easier, but for "best found", randomness is good.
    # We'll use a fixed seed for the main logic to be deterministic if needed, 
    # but the "search" aspect implies trying multiple random paths.
    random.seed(42)
    
    while time.time() - start_time < time_limit:
        # 1. Generate a random maximal set
        # Use a randomized greedy approach
        order = list(range(num_points))
        random.shuffle(order)
        
        current_set = set()
        
        # Greedy add
        for p in order:
            # Check if p can be added
            if can_add(p, current_set):
                current_set.add(p)
        
        # 2. Repair to Maximal
        # Find points that can be added and add them.
        # We need to iterate until no more points can be added.
        # To avoid infinite loops, we pick one point, add it, and restart the check.
        # Or simply iterate through all points repeatedly.
        
        while True:
            added = False
            for p in range(num_points):
                if p in current_set:
                    continue
                # Check if p can be added
                if can_add(p, current_set):
                    current_set.add(p)
                    added = True
                    # Since adding a point might block other points, 
                    # we must restart the scan to ensure maximality is properly checked
                    # for the new state. However, adding a point only makes it 
                    # *harder* for others to be added. It doesn't make previously 
                    # invalid points valid. So we can just continue scanning 
                    # remaining points, but we must be careful.
                    # Actually, if p was invalid before, it stays invalid.
                    # If p was valid, we add it.
                    # So a single pass might not be enough if we skipped a point 
                    # because it was valid, but then added another point which 
                    # blocked it? No, if it was valid, we would have added it.
                    # Wait, the loop `for p in range...` adds p if valid.
                    # If we add p, it might block a later point q.
                    # But q is checked later, so it will see p in current_set.
                    # So a single forward pass is sufficient to reach a state 
                    # where no remaining point in the list can be added?
                    # No. Consider p1, p2.
                    # p1 is valid. Add p1.
                    # p2 was valid wrt old set, but now invalid wrt new set.
                    # We check p2 after p1, so we see p1 and reject p2.
                    # This is correct.
                    # What if p2 is valid, we add p2.
                    # What if p1 was invalid initially, but becomes valid?
                    # Adding points never makes an invalid point valid.
                    # So a single pass is sufficient to find a "locally maximal" 
                    # set? No, "maximal" means NO point can be added.
                    # If we do a single pass, we might miss a point that was 
                    # valid at the start but we didn't check it?
                    # We check ALL points in the range.
                    # So if a point is valid at any time during the pass, 
                    # and it hasn't been checked yet, it will be added.
                    # If it was checked and rejected, it was invalid at that time.
                    # It remains invalid.
                    # So one pass is enough?
                    # Let's trace:
                    # Set S.
                    # Pass 1:
                    # Check p1. Invalid.
                    # Check p2. Valid. Add p2. S = S + {p2}.
                    # Check p3. Valid wrt S? Yes. Add p3. S = S + {p2, p3}.
                    # ...
                    # End of pass.
                    # Is S maximal?
                    # Suppose p4 could be added.
                    # When we checked p4, was it valid?
                    # If p4 was valid at the time we checked it, we would have added it.
                    # Since we didn't add it, it was invalid.
                    # Since we only added points since then, it remains invalid.
                    # So yes, one pass is sufficient to reach a maximal set 
                    # (specifically, a set where no point *later* in the iteration 
                    # order can be added, but points *earlier* that were rejected 
                    # remain rejected).
                    # Wait, if p1 was invalid, it stays invalid.
                    # So the loop `while True` with `added` flag is actually 
                    # redundant if the logic holds, but it's safe.
                    # However, we can break early if we want to save time, 
                    # but correctness is paramount.
                    # Actually, the `while True` loop will terminate because 
                    # `added` will be False when no more points can be added.
                    # And since we only add points, the set grows monotonically.
                    # So it will terminate.
                    # But we can optimize by breaking the inner loop if we 
                    # don't need to restart? No, the inner loop is a single pass.
                    # The outer loop repeats the pass if any point was added.
                    # This is correct but potentially slow.
                    # However, for N=16, it's fast enough.
                    pass
            
            if not added:
                break
        
        # 3. Local Search: Remove and Refill
        # Try to remove a point and see if we can refill with more points? 
        # No, we want to minimize size. So we want to REMOVE points.
        # But removing a point makes it easier to add others.
        # So we can try to remove a point, then greedily add points, and see if the size is smaller.
        # This is a form of local search.
        
        # Let's try a simpler local search: 
        # For each point in the set, try to remove it.
        # Then try to add points greedily.
        # If the resulting set is smaller, keep it.
        
        # But this is expensive. Let's just do the randomized greedy and repair for now.
        # The inspiration says "local removal-and-refill search".
        
        # Let's implement a simple version:
        # Try to remove a random point, then refill greedily.
        # If the size is smaller, keep the new set.
        
        # But we need to be careful with the time limit.
        
        # For now, let's just use the randomized greedy + repair.
        # The inspiration's score of 1.4286 suggests this is already good.
        
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
            # Optional: Try local search on the best set found so far
            # But for now, let's just keep searching with new random orders.
    
    if best_set is None:
        # Fallback: return an empty set? No, must be maximal.
        # This should not happen.
        return []
    
    # Convert best_set to list of tuples
    return [coords[i] for i in best_set]
# EVOLVE-BLOCK-END
