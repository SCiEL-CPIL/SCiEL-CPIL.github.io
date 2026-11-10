# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    
    Structural change: Instead of a greedy deletion heuristic that scans all points
    repeatedly (O(n^2 * lines) per iteration), we model the problem as finding the
    maximum set of points that can be simultaneously excluded (complement of N)
    subject to the constraint that every point in the plane has at least one
    "witness" line where all other points on that line are in N.
    
    We use a batch-removal approach:
    1. Start from a valid baseline (Hermitian unital exclusion with QR y-coords).
    2. Identify a maximal set of points that can be removed "simultaneously"
       without invalidating any point's witness property. This is done by
       checking if the union of their removal effects still leaves every point
       with at least one good witness.
    3. If the batch is valid, apply it; otherwise, fall back to single-point removal.
    
    This batch approach is a structural change from iterative single-point greedy
    to a parallel/batch optimization step, which should allow more points to be
    removed in fewer iterations.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 600s limit
    
    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3
    
    # F_49 arithmetic
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + k*b1*b2) % p, (a1*b2 + a2*b1) % p)
    
    # All points
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) 
                  for ya in range(p) for yb in range(p)]
    n = len(all_points)  # 2401
    
    # Precompute lines
    vert_lines = defaultdict(list)
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        vert_lines[(xa, xb)].append(i)
    
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
    
    # Start with baseline: exclude points on Hermitian unital with y_a in QR
    qr = set((u*u) % p for u in range(p))  # {0, 1, 2, 4}
    
    in_N = [True] * n
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        in_unital = (2*xa + ya*ya - k*yb*yb) % p == 0
        if in_unital and ya in qr:
            in_N[i] = False
    
    # line_not_in_N_count[L] = number of points in L not in N
    line_not_in_N_count = {}
    for key, pts in vert_lines.items():
        cnt = sum(1 for j in pts if not in_N[j])
        line_not_in_N_count[('vert', key)] = cnt
    for (ma, mb), lines in slope_lines.items():
        for key, pts in lines.items():
            cnt = sum(1 for j in pts if not in_N[j])
            line_not_in_N_count[('slope', (ma, mb), key)] = cnt
    
    # point_good_witness[i] = number of lines L through i where all other points in L are in N
    def compute_good_witness(in_N_state, line_nin_count):
        witness = [0] * n
        for i in range(n):
            cnt = 0
            for line_rep in point_lines[i]:
                if line_rep[0] == 'vert':
                    lkey = ('vert', line_rep[1])
                else:
                    lkey = ('slope', line_rep[1], line_rep[2])
                
                total_nin = line_nin_count[lkey]
                nin_excl_i = total_nin - (0 if in_N_state[i] else 1)
                if nin_excl_i == 0:
                    cnt += 1
            witness[i] = cnt
        return witness
    
    point_good_witness = compute_good_witness(in_N, line_not_in_N_count)
    
    # Batch removal strategy
    # In each round, identify a set of points that can all be removed at once.
    # A point i is "removable" if removing it doesn't cause any point to lose
    # its last good witness. For a batch, we need to check the combined effect.
    
    max_rounds = 50
    for rnd in range(max_rounds):
        if time.time() > deadline - 60:
            break
        
        # Find candidate points for removal: points in N whose removal
        # would not immediately break any point's last witness (individually)
        candidates = []
        for i in range(n):
            if not in_N[i]:
                continue
            # Check if i can be removed individually
            safe = True
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
                new_count = old_count + 1
                
                for j in line_pts:
                    if j == i:
                        continue
                    was_good = (old_count - (0 if in_N[j] else 1)) == 0
                    still_good = (new_count - (0 if in_N[j] else 1)) == 0
                    if was_good and not still_good:
                        if point_good_witness[j] <= 1:
                            safe = False
                            break
                if not safe:
                    break
            if safe:
                candidates.append(i)
        
        if not candidates:
            break
        
        # Try to remove a large batch of candidates simultaneously
        # Start with all candidates, check validity, remove problematic ones
        batch = list(candidates)
        
        # Check if the entire batch can be removed
        def check_batch_validity(batch_pts):
            # Simulate removing all points in batch_pts
            # For each line, compute new not-in-N count
            # For each point, check if it still has at least one good witness
            new_nin = dict(line_not_in_N_count)
            for i in batch_pts:
                if not in_N[i]:
                    continue
                for line_rep in point_lines[i]:
                    if line_rep[0] == 'vert':
                        lkey = ('vert', line_rep[1])
                    else:
                        lkey = ('slope', line_rep[1], line_rep[2])
                    new_nin[lkey] = new_nin.get(lkey, 0) + 1
            
            # Check all points
            new_in_N = list(in_N)
            for i in batch_pts:
                new_in_N[i] = False
            
            for i in range(n):
                has_witness = False
                for line_rep in point_lines[i]:
                    if line_rep[0] == 'vert':
                        lkey = ('vert', line_rep[1])
                    else:
                        lkey = ('slope', line_rep[1], line_rep[2])
                    
                    total_nin = new_nin.get(lkey, 0)
                    nin_excl_i = total_nin - (0 if new_in_N[i] else 1)
                    if nin_excl_i == 0:
                        has_witness = True
                        break
                if not has_witness:
                    return False
            return True
        
        # Binary search or greedy reduction to find largest valid batch
        # For efficiency, try removing in chunks
        if len(batch) > 100:
            # Try a large chunk first
            chunk_size = min(len(batch), 200)
            valid_chunk = []
            for i in range(0, len(batch), chunk_size):
                chunk = batch[i:i+chunk_size]
                # Check if this chunk can be added to valid_chunk
                test_batch = valid_chunk + chunk
                if check_batch_validity(test_batch):
                    valid_chunk = test_batch
                else:
                    # Try adding one by one
                    for pt in chunk:
                        test_batch = valid_chunk + [pt]
                        if check_batch_validity(test_batch):
                            valid_chunk.append(pt)
            batch = valid_chunk
        else:
            if not check_batch_validity(batch):
                # Reduce batch
                valid_batch = []
                for pt in batch:
                    test_batch = valid_batch + [pt]
                    if check_batch_validity(test_batch):
                        valid_batch.append(pt)
                batch = valid_batch
        
        if not batch:
            break
        
        # Apply the batch removal
        for i in batch:
            in_N[i] = False
            for line_rep in point_lines[i]:
                if line_rep[0] == 'vert':
                    lkey = ('vert', line_rep[1])
                else:
                    lkey = ('slope', line_rep[1], line_rep[2])
                line_not_in_N_count[lkey] += 1
        
        # Recompute good witness counts
        point_good_witness = compute_good_witness(in_N, line_not_in_N_count)
    
    # Build result
    pts = [all_points[i] for i in range(n) if in_N[i]]
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
