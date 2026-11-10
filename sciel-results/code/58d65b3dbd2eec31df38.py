"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 64
TIME_LIMIT = 290  # Safety margin below 330s

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    best_pts = []
    best_len = 0
    
    def elapsed():
        return time.time() - start_time
    
    def dist2(a, b):
        return (a[0] - b[0])**2 + (a[1] - b[1])**2
    
    def is_valid_addition(p, pts):
        """Check if adding p to pts creates any isosceles triangle."""
        k = len(pts)
        if k < 2:
            return True
        # Optimization: Check against all existing points
        # We need to ensure p does not form an isosceles triangle with any pair (a,b) in pts
        # This means:
        # 1. d(p,a) != d(p,b) for all a,b in pts
        # 2. d(p,a) != d(a,b) for all a,b in pts
        # 3. d(p,b) != d(a,b) for all a,b in pts
        # Note: 2 and 3 are symmetric if we iterate all pairs.
        
        # Pre-compute distances from p to all points in pts?
        # Or just iterate pairs.
        # Iterating pairs is O(k^2).
        # If k is large, this is slow.
        # Let's stick to O(k^2) for now, but optimize constants.
        
        for i in range(k):
            pi = pts[i]
            d_pi = dist2(p, pi)
            for j in range(i + 1, k):
                pj = pts[j]
                d_pj = dist2(p, pj)
                d_ij = dist2(pi, pj)
                if d_pi == d_pj or d_pi == d_ij or d_pj == d_ij:
                    return False
        return True

    def is_valid_swap(p_out, p_in, pts):
        """Check if swapping p_out with p_in in pts results in a valid set."""
        # Remove p_out, add p_in
        # We need to check if p_in is valid with respect to (pts - {p_out})
        # This is equivalent to is_valid_addition(p_in, pts_without_p_out)
        # But we can optimize by checking conflicts directly.
        
        # A swap is valid if:
        # 1. p_in does not form isosceles with any pair in (pts - {p_out})
        # 2. (pts - {p_out}) is already valid (it is, since pts is valid)
        
        # So we just need to check if p_in conflicts with any pair in pts, EXCLUDING pairs involving p_out?
        # No, if p_in conflicts with a pair (a,b) where a,b are in pts and a!=p_out, b!=p_out, then invalid.
        # If p_in conflicts with a pair (a, p_out), it doesn't matter because p_out is removed.
        
        # So: Check all pairs (a,b) in pts such that a != p_out and b != p_out.
        # If d(p_in, a) == d(p_in, b) or d(p_in, a) == d(a,b) or d(p_in, b) == d(a,b), return False.
        
        # We can iterate through all points in pts, skip p_out, and check pairs.
        # To avoid double counting and complexity, let's just build the list of other points.
        # This is O(k) to build list, O(k^2) to check.
        
        # Alternative:
        # Check if p_in is valid in the set `pts` ignoring p_out.
        # We can reuse is_valid_addition if we pass a list without p_out.
        # But creating a new list is O(k).
        
        # Let's write a specific check for speed.
        k = len(pts)
        # Find index of p_out
        idx_out = -1
        for i in range(k):
            if pts[i] == p_out:
                idx_out = i
                break
        
        if idx_out == -1:
            return False # Should not happen
        
        # Check p_in against all pairs in pts that do not involve p_out
        for i in range(k):
            if i == idx_out: continue
            pi = pts[i]
            d_pi = dist2(p_in, pi)
            for j in range(i + 1, k):
                if j == idx_out: continue
                pj = pts[j]
                d_pj = dist2(p_in, pj)
                d_ij = dist2(pi, pj)
                if d_pi == d_pj or d_pi == d_ij or d_pj == d_ij:
                    return False
        return True

    def local_search(pts, seed=None):
        """
        Local search with Add and Swap moves.
        """
        if seed is not None:
            random.seed(seed)
            
        all_points = [(x, y) for x in range(N) for y in range(N)]
        pts_set = set(pts)
        current_pts = pts[:]
        
        # Track time
        while elapsed() < TIME_LIMIT - 2:
            # 1. Try to Add
            # Pick a random point from the complement
            # To avoid scanning all_points every time, we can maintain a list of candidates
            # Or just pick a random grid point and check if it's in pts_set.
            
            added = False
            for _ in range(100): # Limit attempts per iteration to avoid infinite loops if no add possible
                p = random.choice(all_points)
                if p in pts_set:
                    continue
                if is_valid_addition(p, current_pts):
                    current_pts.append(p)
                    pts_set.add(p)
                    added = True
                    break
            
            if not added:
                # 2. Try to Swap
                # Try to find a valid swap to escape local optimum
                swapped = False
                for _ in range(50):
                    if len(current_pts) == 0:
                        break
                    p_out = random.choice(current_pts)
                    p_in = random.choice(all_points)
                    if p_in in pts_set:
                        continue
                    if is_valid_swap(p_out, p_in, current_pts):
                        # Perform swap
                        current_pts.remove(p_out)
                        pts_set.remove(p_out)
                        current_pts.append(p_in)
                        pts_set.add(p_in)
                        swapped = True
                        break
                
                if not swapped:
                    # Stuck? Maybe restart or kick.
                    # Let's do a "kick": remove a few random points to change landscape
                    if len(current_pts) > 5:
                        for _ in range(3):
                            if current_pts:
                                p = random.choice(current_pts)
                                current_pts.remove(p)
                                pts_set.remove(p)
                    else:
                        break # Can't kick much, break out
            
        return current_pts

    # Phase 1: Greedy Initialization
    def greedy_init(seed=None):
        if seed is not None:
            random.seed(seed)
        pts = []
        all_points = [(x, y) for x in range(N) for y in range(N)]
        candidates = all_points[:]
        random.shuffle(candidates)
        
        for p in candidates:
            if elapsed() > TIME_LIMIT - 10:
                break
            if is_valid_addition(p, pts):
                pts.append(p)
        return pts

    # Run multiple restarts of Local Search
    num_restarts = 20
    for i in range(num_restarts):
        if elapsed() > TIME_LIMIT - 5:
            break
            
        # Initialize with greedy
        pts = greedy_init(seed=i)
        
        # Refine with local search
        pts = local_search(pts, seed=i)
        
        if len(pts) > best_len:
            best_len = len(pts)
            best_pts = pts[:]
            
    return best_pts
# EVOLVE-BLOCK-END
