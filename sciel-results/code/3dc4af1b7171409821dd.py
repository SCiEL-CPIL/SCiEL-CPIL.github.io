import time
import random

N = 64
NUM_POINTS = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Keep 50s margin for 330s limit
    
    # Precompute squared distances between all pairs of points
    # Indexing: idx = x * N + y
    # dist2[i][j] is squared distance between point i and point j
    # To save memory, we can store as a list of lists or a flat array.
    # 4096 * 4096 * 4 bytes = 64MB. Acceptable.
    # But Python lists of ints are heavy. 
    # Let's use a list of arrays from numpy? Or just calculate on the fly?
    # Calculating on the fly was the bottleneck? 
    # dx*dx + dy*dy is fast. The overhead is in the Python loops and set lookups.
    # Let's stick to on-the-fly but optimize the data structures.
    
    all_points = list(range(NUM_POINTS))
    # Coordinates for each index
    coords = [(i // N, i % N) for i in range(NUM_POINTS)]
    
    best_set = []
    best_size = 0
    
    # Helper to check if p can be added to S
    # S is a list of point indices
    # dist_from is a list of sets: dist_from[i] contains squared distances from i to other points in S
    def can_add(S, dist_from, p):
        px, py = coords[p]
        d2s = set()
        for a in S:
            ax, ay = coords[a]
            dx = px - ax
            dy = py - ay
            d2 = dx*dx + dy*dy
            if d2 in d2s:
                return False
            d2s.add(d2)
            if d2 in dist_from[a]:
                return False
        return True

    def add_point(S, dist_from, p):
        S.append(p)
        px, py = coords[p]
        dist_from[p] = set()
        for a in S[:-1]:
            ax, ay = coords[a]
            dx = px - ax
            dy = py - ay
            d2 = dx*dx + dy*dy
            dist_from[a].add(d2)
            dist_from[p].add(d2)

    def remove_point(S, dist_from, p):
        # Find index of p in S
        # This is O(K). We can maintain a map idx -> position in S?
        # For simplicity, linear search. K is small (~10).
        idx = S.index(p)
        S.pop(idx)
        px, py = coords[p]
        for a in S:
            ax, ay = coords[a]
            dx = px - ax
            dy = py - ay
            d2 = dx*dx + dy*dy
            dist_from[a].discard(d2)
        # dist_from[p] is no longer needed, but we can leave it or clear it.
        # Clearing avoids stale data if p is re-added later? 
        # Actually, if p is re-added, add_point will reset dist_from[p].
        # But if p is not in S, dist_from[p] should be empty or ignored.
        # Let's clear it to be safe.
        dist_from[p] = set()

    def greedy_search(order):
        S = []
        dist_from = [set() for _ in range(NUM_POINTS)]
        for p in order:
            if can_add(S, dist_from, p):
                add_point(S, dist_from, p)
        return S, dist_from

    def local_search(S, dist_from, time_limit):
        # Try to improve S by swapping points
        # Goal: Find a set of the same size that is "more flexible" or allows adding a new point.
        # We will just try random swaps and see if we can maintain validity.
        # Then, after some swaps, try to add a new point.
        
        current_S = S[:]
        current_dist = [s.copy() for s in dist_from]
        
        # Try to add a point to current_S
        # If we can, we found a larger set!
        
        # Strategy:
        # 1. Pick a random point p not in S.
        # 2. Try to add p.
        # 3. If fails, pick a random point q in S.
        # 4. Remove q.
        # 5. Try to add p.
        # 6. If succeeds, we have a new set of same size.
        # 7. Repeat.
        
        # We'll run this for a fixed number of steps or until time limit.
        steps = 0
        max_steps = 5000
        while time.time() < time_limit and steps < max_steps:
            steps += 1
            # Try to add a random new point
            candidates = [p for p in range(NUM_POINTS) if p not in current_S]
            if not candidates:
                break
            p = random.choice(candidates)
            
            if can_add(current_S, current_dist, p):
                # We found a larger set!
                add_point(current_S, current_dist, p)
                return current_S, current_dist
            
            # If not, try a swap
            if current_S:
                q_idx = random.randint(0, len(current_S) - 1)
                q = current_S[q_idx]
                
                # Temporarily remove q
                remove_point(current_S, current_dist, q)
                
                # Try to add p
                if can_add(current_S, current_dist, p):
                    add_point(current_S, current_dist, p)
                    # We have a valid set of the same size.
                    # Continue searching.
                else:
                    # Swap failed, revert
                    add_point(current_S, current_dist, q)
                    # Note: remove_point cleared dist_from[q], add_point refills it.
                    # But wait, remove_point removed distances from q to others.
                    # add_point adds distances from q to others.
                    # It should restore the state.
                    # However, the order in S changed? 
                    # remove_point pops q. add_point appends q.
                    # The set of points is the same. The dist_from sets are the same.
                    # The order in S might be different, but that doesn't matter for validity.
        
        return current_S, current_dist

    iterations = 0
    while time.time() < deadline:
        iterations += 1
        random.shuffle(all_points)
        S, dist_from = greedy_search(all_points)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
        # If we found a good set, try to improve it with local search
        # Only do local search if we have time
        if time.time() < deadline - 10:
            # Give local search a small budget
            ls_deadline = time.time() + 5
            S_ls, dist_ls = local_search(S, dist_from, ls_deadline)
            if len(S_ls) > best_size:
                best_size = len(S_ls)
                best_set = S_ls[:]
                
    # Convert indices back to (x, y) tuples
    return [coords[i] for i in best_set]
