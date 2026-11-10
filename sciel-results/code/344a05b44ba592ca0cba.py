"""
Strategy: Local Search (Simulated Annealing) to minimize the largest 3-collinear-free subset.
The current program uses a static parabola (score 20). The observed 'island' uses two shifted
parabolas (score ~16). This program searches the space of valid configurations (no 4 collinear)
to find one with a smaller maximum 3-collinear-free subset.
"""
import time
import random
import math
from itertools import combinations

# Deadline guard: 150s limit, use 140s for safety
DEADLINE = 140.0

def construct():
    start_time = time.time()
    
    # 1. Generate initial valid configuration
    # We need 20 distinct points in [-100, 100]^2 with no 4 collinear.
    # A simple way is to pick points on a parabola, but we want to start
    # somewhat random to allow the search to escape local minima.
    # However, generating a valid config randomly is hard. 
    # Let's start with the two-layer parabola idea as a strong seed, 
    # but perturb it.
    
    def generate_seed():
        # Two layers of parabola: y = x^2 and y = x^2 + 1
        # x in -10..9 -> 20 points.
        # Check validity: no 4 collinear.
        pts = []
        for x in range(-10, 10):
            pts.append([x, x * x])
            pts.append([x, x * x + 1])
        # This specific config might have 4-collinearities? 
        # Let's check. If invalid, we need a different seed.
        # Actually, the prompt says the island found a score of 1.25 (which is 16? No, score is value. 
        # Wait, the score in the task is "largest subset with no three collinear". 
        # The "search score" is seed/value. 
        # If value=16, search score = 20/16 = 1.25.
        # So the island found a config with value 16.
        # The parabola has value 20.
        # We want to minimize value.
        
        # Let's just use the two-layer parabola as the start, but verify validity.
        if not is_valid(pts):
             # Fallback: random points, repair if invalid
             pts = random_valid_points()
        return pts

    def random_valid_points():
        while True:
            pts = []
            for _ in range(20):
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                if [x, y] not in pts:
                    pts.append([x, y])
            if is_valid(pts):
                return pts

    def is_valid(pts):
        # Check: 20 distinct points, |x|,|y| <= 100, no 4 collinear.
        if len(pts) != 20:
            return False
        seen = set()
        for p in pts:
            if p in seen:
                return False
            seen.add(tuple(p))
            if abs(p[0]) > 100 or abs(p[1]) > 100:
                return False
        
        # Check no 4 collinear
        # For each pair, check if there are 2 other points on the same line.
        # O(N^4) is too slow for frequent checks? N=20, 20^4 = 160k. 
        # 160k ops is fast in Python? Maybe ~0.01s. 
        # We can optimize: For each pair (i, j), count points k such that collinear(i, j, k).
        # If count >= 4 (including i, j), invalid.
        for i in range(20):
            for j in range(i + 1, 20):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                # Check all other points
                count = 2
                for k in range(20):
                    if k == i or k == j:
                        continue
                    x3, y3 = pts[k]
                    # Collinear if cross product is 0
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        count += 1
                if count >= 4:
                    return False
        return True

    def score(pts):
        # Calculate largest subset with no 3 collinear.
        # This is the exact branch and bound required.
        # N=20.
        # We can use a recursive search with pruning.
        # State: set of points to consider, current subset size.
        # Pruning: if current_size + remaining_points <= best_found, prune.
        
        best = 0
        n = 20
        
        # Precompute collinearity?
        # coll[i][j][k] = True if i, j, k are collinear.
        # Memory: 20*20*20 = 8000 bools. Fast.
        coll = [[[False]*20 for _ in range(20)] for _ in range(20)]
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                for k in range(j+1, n):
                    x3, y3 = pts[k]
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        coll[i][j][k] = True
                        coll[j][i][k] = True
                        coll[k][i][j] = True
                        coll[i][k][j] = True
                        coll[j][k][i] = True
                        coll[k][j][i] = True

        # Branch and Bound
        # We want to find the maximum size of a subset S such that no 3 points in S are collinear.
        # This is equivalent to finding the maximum independent set in a 3-uniform hypergraph where edges are collinear triples.
        
        def bnb(idx, current_size, current_set_mask):
            nonlocal best
            if current_size > best:
                best = current_size
            
            # Pruning: even if we add all remaining points, can we beat best?
            if current_size + (n - idx) <= best:
                return
            
            if idx == n:
                return
            
            # Option 1: Skip point idx
            bnb(idx + 1, current_size, current_set_mask)
            
            # Option 2: Take point idx
            # Check if adding idx to current_set_mask creates a collinear triple.
            # We need to check if there exist i, j in current_set_mask such that coll[i][j][idx] is True.
            # If yes, we cannot take idx.
            # If no, we can take it.
            
            can_take = True
            # Iterate over pairs in current_set_mask
            # current_set_mask is an integer bitmask or a set. 
            # Using a set of indices is easier for N=20.
            # But for speed, let's use a bitmask.
            # N=20 fits in 32-bit int.
            
            # Let's re-implement BNB with bitmask for speed.
            
            pass # Placeholder, will rewrite below

        # Rewrite BNB with bitmask
        best = 0
        def bnb_bitmask(idx, current_size, mask):
            nonlocal best
            if current_size > best:
                best = current_size
            
            if current_size + (n - idx) <= best:
                return
            
            if idx == n:
                return
            
            # Skip
            bnb_bitmask(idx + 1, current_size, mask)
            
            # Take
            # Check if idx is compatible with mask
            # We need to ensure no triple (a, b, idx) with a, b in mask is collinear.
            # This is equivalent to: for all a in mask, is there a b in mask such that coll[a][b][idx]?
            # We can precompute for each idx, a bitmask of "conflicts"?
            # Conflict for idx is a set of pairs? No.
            # If we add idx, we must ensure that no pair (a, b) in mask forms a line with idx.
            # This means for each a in mask, the set of b in mask must not contain any b such that coll[a][b][idx].
            
            # Let's precompute conflict_mask[idx][a] = bitmask of b such that coll[a][b][idx].
            # Then, if we add idx, we need to check if (mask & conflict_mask[idx][a]) is non-empty for any a in mask?
            # Actually, if we add idx, the new mask is mask | (1 << idx).
            # The condition is: there do NOT exist a, b in mask such that coll[a][b][idx].
            
            # Let's precompute: for each idx, a list of pairs (a, b) such that coll[a][b][idx].
            # Or simpler: for each idx, and each a, compute a bitmask of b's that conflict with a and idx.
            
            pass # Will implement in code

        # Let's implement the precomputation and BNB properly.
        
        # Precompute conflict masks
        # conflict[idx][a] is a bitmask of b such that coll[a][b][idx] is True.
        # Note: coll is symmetric.
        conflict = [[0]*20 for _ in range(20)]
        for i in range(n):
            for j in range(n):
                if i == j: continue
                mask = 0
                for k in range(n):
                    if i == k or j == k: continue
                    if coll[i][j][k]:
                        mask |= (1 << k)
                conflict[i][j] = mask

        def bnb_fast(idx, current_size, mask):
            nonlocal best
            if current_size > best:
                best = current_size
            
            if current_size + (n - idx) <= best:
                return
            
            if idx == n:
                return
            
            # Skip idx
            bnb_fast(idx + 1, current_size, mask)
            
            # Take idx
            # Check if idx can be added to mask
            # We need to ensure no a, b in mask form a collinear triple with idx.
            # i.e., for all a in mask, (conflict[idx][a] & mask) == 0?
            # Wait, conflict[idx][a] contains b such that coll[idx][a][b].
            # If we add idx, we are adding a new point. The triples involving idx are (idx, a, b) where a, b are in the current mask.
            # So we need to check if there exists a in mask such that there is a b in mask with coll[idx][a][b].
            # This is equivalent to: for all a in mask, (conflict[idx][a] & mask) == 0.
            
            # Optimization: if we iterate a in mask, we can break early.
            # Extract indices from mask
            # Since N=20, we can just loop a from 0 to 19.
            ok = True
            # Iterate over bits in mask
            m = mask
            while m:
                a = (m & -m).bit_length() - 1
                if (conflict[idx][a] & mask) != 0:
                    ok = False
                    break
                m &= m - 1
            
            if ok:
                bnb_fast(idx + 1, current_size + 1, mask | (1 << idx))

        bnb_fast(0, 0, 0)
        return best

    # Initialize
    pts = generate_seed()
    current_score = score(pts)
    best_pts = [p[:] for p in pts]
    best_score = current_score
    
    # Simulated Annealing
    temp = 10.0
    cool_rate = 0.95
    
    while time.time() - start_time < DEADLINE:
        # Generate neighbor
        new_pts = [p[:] for p in pts]
        idx = random.randint(0, 19)
        dx = random.randint(-5, 5)
        dy = random.randint(-5, 5)
        new_pts[idx][0] += dx
        new_pts[idx][1] += dy
        
        # Check bounds
        if abs(new_pts[idx][0]) > 100 or abs(new_pts[idx][1]) > 100:
            continue
            
        # Check validity
        if not is_valid(new_pts):
            continue
            
        new_score = score(new_pts)
        
        # Acceptance
        if new_score < current_score:
            pts = new_pts
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_pts = [p[:] for p in pts]
        else:
            # Accept with probability
            delta = current_score - new_score # negative
            # Probability = exp(delta / temp)
            # If delta is 0, prob = 1
            if random.random() < math.exp(delta / temp):
                pts = new_pts
                current_score = new_score
        
        temp *= cool_rate
        if temp < 0.1:
            break
            
    return best_pts

if __name__ == "__main__":
    # Simple test
    res = construct()
    print(res)
