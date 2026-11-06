import random
import time
import itertools

N = 16
M = N * N
ALL_MASK = (1 << M) - 1
DEADLINE = time.time() + 290

# Precompute block masks
# block_mask[a][b] is a bitmask of points p such that (p, a, b) is isosceles.
# Note: (p,a,b) is isosceles if pa==pb or pa==ab or pb==ab.
# We represent points as indices 0..M-1.
# point i corresponds to (i//N, i%N).

def _dist2(i, j):
    x1, y1 = divmod(i, N)
    x2, y2 = divmod(j, N)
    return (x1 - x2) ** 2 + (y1 - y2) ** 2

# Precompute all pair distances
dist2 = [[0] * M for _ in range(M)]
for i in range(M):
    for j in range(M):
        dist2[i][j] = _dist2(i, j)

# Precompute block masks
# For each pair (a, b), which points p are blocked?
# p is blocked if dist2[p][a] == dist2[p][b] or dist2[p][a] == dist2[a][b] or dist2[p][b] == dist2[a][b]
block_mask = [[0] * M for _ in range(M)]

for a in range(M):
    for b in range(M):
        if a == b:
            continue
        mask = 0
        dab = dist2[a][b]
        for p in range(M):
            dpa = dist2[p][a]
            dpb = dist2[p][b]
            if dpa == dpb or dpa == dab or dpb == dab:
                mask |= (1 << p)
        block_mask[a][b] = mask

def _is_maximal(S, covered):
    # S is a set of points
    # covered is the bitmask of points blocked by pairs in S
    # Maximal if all points not in S are in covered
    S_mask = 0
    for p in S:
        S_mask |= (1 << p)
    # Points not in S: ALL_MASK ^ S_mask
    # If (covered & (ALL_MASK ^ S_mask)) == (ALL_MASK ^ S_mask), then all are covered
    return (covered & (ALL_MASK ^ S_mask)) == (ALL_MASK ^ S_mask)

def _get_covered(S):
    covered = 0
    S_list = list(S)
    for i in range(len(S_list)):
        for j in range(i + 1, len(S_list)):
            a = S_list[i]
            b = S_list[j]
            covered |= block_mask[a][b]
    return covered

def _greedy_construct(rng):
    S = set()
    covered = 0
    S_list = []
    
    while True:
        # Check if maximal
        if _is_maximal(S, covered):
            break
        
        best_x = -1
        best_gain = -1
        best_new_covered = 0
        
        # Try all points not in S
        for x in range(M):
            if x in S:
                continue
            
            # Calculate new covered if we add x
            new_covered = covered
            for a in S_list:
                new_covered |= block_mask[x][a]
            
            # Calculate gain: number of new points in V\S covered
            # Actually, we just want to maximize the total covered points
            # Or specifically, points in V \ (S U {x})
            S_mask = 0
            for p in S:
                S_mask |= (1 << p)
            S_mask |= (1 << x)
            
            # Gain is popcount of (new_covered & ~S_mask) - popcount(covered & ~S_mask)
            # Since covered is a subset of new_covered, gain is popcount(new_covered & ~S_mask & ~covered)
            # Simpler: just maximize popcount(new_covered)
            # But we should prioritize covering points not in S
            # Let's just use popcount(new_covered)
            
            # Optimization: if S is empty, gain is 0 for all? No, block_mask is 0 for single points.
            # If S is empty, adding any x results in covered=0.
            # So we need to handle the first step specially or just pick random.
            
            if len(S_list) == 0:
                gain = 0
            else:
                gain = bin(new_covered).count('1')
            
            if gain > best_gain:
                best_gain = gain
                best_x = x
                best_new_covered = new_covered
        
        if best_x == -1:
            # Should not happen if not maximal
            break
            
        # Random tie-breaking
        # Find all x with best_gain
        candidates = []
        for x in range(M):
            if x in S:
                continue
            if len(S_list) == 0:
                candidates.append(x)
            else:
                new_covered = covered
                for a in S_list:
                    new_covered |= block_mask[x][a]
                if bin(new_covered).count('1') == best_gain:
                    candidates.append(x)
        
        if not candidates:
            candidates = [best_x]
            
        x = rng.choice(candidates)
        
        S.add(x)
        S_list.append(x)
        
        # Update covered
        for a in S_list[:-1]: # a is in old S
            covered |= block_mask[x][a]
            
    return S

def _local_search(S, rng):
    S = set(S)
    covered = _get_covered(S)
    
    while time.time() < DEADLINE - 5:
        if not S:
            break
            
        # Pick a random point to remove
        y = rng.choice(list(S))
        S.remove(y)
        
        # Recalculate covered
        new_covered = _get_covered(S)
        
        if _is_maximal(S, new_covered):
            # Accepted, size decreased
            covered = new_covered
            continue
        
        # Not maximal. Find uncovered points
        S_mask = 0
        for p in S:
            S_mask |= (1 << p)
        uncovered = (ALL_MASK ^ S_mask) & ~new_covered
        
        if uncovered == 0:
            # Should be maximal
            covered = new_covered
            continue
            
        # Try to find a point z to add that covers the most uncovered points
        best_z = -1
        best_coverage = -1
        
        for z in range(M):
            if z in S:
                continue
            
            # Calculate how many uncovered points are covered by adding z
            # z blocks p if (p, z, a) is isosceles for some a in S
            # So we need to check block_mask[z][a] for all a in S
            z_blocks = 0
            for a in S:
                z_blocks |= block_mask[z][a]
            
            # Count how many of 'uncovered' are in 'z_blocks'
            coverage = bin(uncovered & z_blocks).count('1')
            
            if coverage > best_coverage:
                best_coverage = coverage
                best_z = z
        
        if best_z == -1 or best_coverage == 0:
            # Can't fix it with 1 point. Restore y.
            S.add(y)
            covered = _get_covered(S)
            break
            
        # Add best_z
        S.add(best_z)
        covered = _get_covered(S)
        
        # If we added a point, size is same as before removal.
        # We are just exploring the space of maximal sets.
        
    return S

def construct():
    best_set = None
    best_size = float('inf')
    
    rng = random.Random(42)
    
    # Run multiple restarts
    for i in range(100):
        if time.time() > DEADLINE - 10:
            break
            
        S = _greedy_construct(rng)
        S = _local_search(S, rng)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = S
            print(f"Found size {best_size} at restart {i}")
            
        if best_size <= 10:
            break
            
    if best_set is None:
        # Fallback: just return a valid maximal set
        best_set = _greedy_construct(rng)
        
    return list(best_set)
