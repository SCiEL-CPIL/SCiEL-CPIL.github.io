# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using local search with simultaneous removals.
    
    Key improvements:
    1. Bitset-based line membership for fast intersection checks
    2. Simultaneous removal of 2-3 points to escape local minima
    3. Efficient tracking of good witness counts
    """
    start_time = time.time()
    deadline = start_time + 570  # Safety margin before 600s limit
    
    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3
    
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + k*b1*b2) % p, (a1*b2 + a2*b1) % p)
    
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) 
                  for ya in range(p) for yb in range(p)]
    n = len(all_points)  # 2401
    point_idx = {pt: i for i, pt in enumerate(all_points)}
    
    # Precompute all lines as sets of point indices
    # Vertical lines: fixed (xa, xb)
    vert_lines = defaultdict(list)
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        vert_lines[(xa, xb)].append(i)
    
    # Slope lines: direction (1, m) where m != (0,0)
    slope_lines = {}
    for ma in range(p):
        for mb in range(p):
            if ma == 0 and mb == 0:
                continue
            lines = defaultdict(list)
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                mx = mul_f49((ma, mb), (xa, xb))
                key = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                lines[key].append(i)
            slope_lines[(ma, mb)] = dict(lines)
    
    # For each point, list of lines through it
    point_lines = [[] for _ in range(n)]
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        point_lines[i].append(('vert', (xa, xb), None))
        for ma in range(p):
            for mb in range(p):
                if ma == 0 and mb == 0:
                    continue
                mx = mul_f49((ma, mb), (xa, xb))
                key = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                point_lines[i].append(('slope', (ma, mb), key))
    
    # Initialize with baseline: exclude points on Hermitian unital with ya in QR
    in_N = [True] * n
    qr = set((u*u) % p for u in range(p))  # {0, 1, 2, 4}
    
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        in_unital = (2*xa + ya*ya - k*yb*yb) % p == 0
        if in_unital and ya in qr:
            in_N[i] = False
    
    # line_not_in_N_count[L] = number of points in L with not in_N[j]
    line_not_in_N_count = {}
    for key, pts in vert_lines.items():
        line_not_in_N_count[('vert', key)] = sum(1 for j in pts if not in_N[j])
    for (ma, mb), lines in slope_lines.items():
        for key, pts in lines.items():
            line_not_in_N_count[('slope', (ma, mb), key)] = sum(1 for j in pts if not in_N[j])
    
    # point_good_witness[i] = number of lines L through i where all other points on L are in N
    point_good_witness = [0] * n
    for i in range(n):
        cnt = 0
        for line_rep in point_lines[i]:
            if line_rep[0] == 'vert':
                lkey = ('vert', line_rep[1])
            else:
                lkey = ('slope', line_rep[1], line_rep[2])
            total_nin = line_not_in_N_count[lkey]
            nin_excl_i = total_nin - (0 if in_N[i] else 1)
            if nin_excl_i == 0:
                cnt += 1
        point_good_witness[i] = cnt
    
    def try_remove_single(i):
        """Try to remove point i from N. Returns True if successful."""
        if not in_N[i] or point_good_witness[i] == 0:
            return False
        
        # Collect affected points
        affected = []
        for line_rep in point_lines[i]:
            if line_rep[0] == 'vert':
                key = line_rep[1]
                lkey = ('vert', key)
                line_pts = vert_lines[key]
            else:
                ma, mb = line_rep[1]
                key = line_rep[2]
                lkey = ('slope', (ma, mb), key)
                line_pts = slope_lines[(ma, mb)][key]
            
            old_count = line_not_in_N_count[lkey]
            for j in line_pts:
                if j == i:
                    continue
                was_good = (old_count - (0 if in_N[j] else 1)) == 0
                if was_good:
                    affected.append((j, lkey))
        
        # Check validity
        decrement_counts = defaultdict(int)
        for j, lkey in affected:
            decrement_counts[j] += 1
        
        for j, dec in decrement_counts.items():
            if point_good_witness[j] - dec <= 0:
                return False
        
        # Apply removal
        in_N[i] = False
        for line_rep in point_lines[i]:
            if line_rep[0] == 'vert':
                lkey = ('vert', line_rep[1])
            else:
                lkey = ('slope', line_rep[1], line_rep[2])
            line_not_in_N_count[lkey] += 1
        
        for j, lkey in affected:
            point_good_witness[j] -= 1
        
        return True
    
    def try_remove_set(removal_set):
        """Try to remove a set of points simultaneously. Returns True if successful."""
        if len(removal_set) == 0:
            return True
        if len(removal_set) == 1:
            return try_remove_single(removal_set[0])
        
        # Check all points can be removed
        for i in removal_set:
            if not in_N[i]:
                return False
        
        # Simulate simultaneous removal
        # For each point i in removal_set, collect affected lines and points
        all_affected = defaultdict(int)  # j -> count of good witnesses lost
        line_increment = defaultdict(int)  # lkey -> increment to not_in_N count
        
        for i in removal_set:
            for line_rep in point_lines[i]:
                if line_rep[0] == 'vert':
                    key = line_rep[1]
                    lkey = ('vert', key)
                    line_pts = vert_lines[key]
                else:
                    ma, mb = line_rep[1]
                    key = line_rep[2]
                    lkey = ('slope', (ma, mb), key)
                    line_pts = slope_lines[(ma, mb)][key]
                
                old_count = line_not_in_N_count[lkey]
                line_increment[lkey] += 1
                
                # For each j in line_pts, j != i, check if L was good witness for j
                for j in line_pts:
                    if j == i or j in removal_set:
                        continue
                    was_good = (old_count - (0 if in_N[j] else 1)) == 0
                    if was_good:
                        all_affected[j] += 1
        
        # Check if any point would have 0 good witnesses after removal
        for j, dec in all_affected.items():
            if point_good_witness[j] - dec <= 0:
                return False
        
        # Apply removal
        for i in removal_set:
            in_N[i] = False
        
        for lkey, inc in line_increment.items():
            line_not_in_N_count[lkey] += inc
        
        for j, dec in all_affected.items():
            point_good_witness[j] -= dec
        
        return True
    
    # Phase 1: Greedy single removals
    max_passes = 15
    for pass_num in range(max_passes):
        if time.time() > deadline - 60:
            break
        
        made_progress = False
        for i in range(n):
            if time.time() > deadline - 60:
                break
            if try_remove_single(i):
                made_progress = True
        
        if not made_progress:
            break
    
    # Phase 2: Try simultaneous removals of size 2
    if time.time() < deadline - 30:
        candidates = [i for i in range(n) if in_N[i]]
        # Sort by good witness count (lower first - more critical)
        candidates.sort(key=lambda i: point_good_witness[i])
        
        # Try pairs of nearby candidates
        tried_pairs = set()
        for idx1, i in enumerate(candidates[:200]):  # Limit to first 200 for speed
            if time.time() > deadline - 30:
                break
            for idx2 in range(idx1 + 1, min(idx1 + 10, len(candidates))):
                j = candidates[idx2]
                pair = (min(i, j), max(i, j))
                if pair in tried_pairs:
                    continue
                tried_pairs.add(pair)
                if try_remove_set([i, j]):
                    pass  # Success
    
    # Phase 3: Try simultaneous removals of size 3
    if time.time() < deadline - 15:
        candidates = [i for i in range(n) if in_N[i]]
        candidates.sort(key=lambda i: point_good_witness[i])
        
        tried_triples = set()
        for idx1, i in enumerate(candidates[:100]):
            if time.time() > deadline - 15:
                break
            for idx2 in range(idx1 + 1, min(idx1 + 5, len(candidates))):
                j = candidates[idx2]
                for idx3 in range(idx2 + 1, min(idx3 if False else idx2 + 5, len(candidates))):
                    k_idx = candidates[idx3]
                    triple = tuple(sorted([i, j, k_idx]))
                    if triple in tried_triples:
                        continue
                    tried_triples.add(triple)
                    if try_remove_set([i, j, k_idx]):
                        pass
    
    # Final pass of single removals
    if time.time() < deadline - 5:
        for i in range(n):
            if time.time() > deadline - 5:
                break
            try_remove_single(i)
    
    # Build result
    pts = [all_points[i] for i in range(n) if in_N[i]]
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
