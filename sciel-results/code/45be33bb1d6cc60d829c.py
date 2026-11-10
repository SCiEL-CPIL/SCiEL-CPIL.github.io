# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    Starts with all points in N and greedily removes points.
    """
    start_time = time.time()
    deadline = start_time + 570  # 30s safety margin
    
    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3
    
    # F_49 arithmetic
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + k*b1*b2) % p, (a1*b2 + a2*b1) % p)
    
    # All points as (xa, xb, ya, yb)
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) for ya in range(p) for yb in range(p)]
    n = len(all_points)  # 2401
    point_idx = {pt: i for i, pt in enumerate(all_points)}
    
    # Precompute lines for each direction
    direction_lines = []
    
    # Vertical direction: v = (0, 1)
    vert_lines = defaultdict(list)
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        vert_lines[(xa, xb)].append(i)
    direction_lines.append(list(vert_lines.values()))
    
    # Slope directions: v = (1, m) for m in F_49
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            lines_dict = defaultdict(list)
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                mx = mul_f49((xa, xb), (ma, mb))
                c = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                lines_dict[c].append(i)
            direction_lines.append(list(lines_dict.values()))
    
    # point_line[point_idx][dir_idx] = line_idx within that direction
    point_line = [[None] * 50 for _ in range(n)]
    
    for dir_idx in range(50):
        for line_idx, line in enumerate(direction_lines[dir_idx]):
            for pt_idx in line:
                point_line[pt_idx][dir_idx] = line_idx

    # Initialize: All points in N
    in_N = [True] * n
    
    # not_in_N_count[dir_idx][line_idx] = number of points on that line not in N
    not_in_N_count = [[0] * len(direction_lines[d]) for d in range(50)]
    
    # witness_count[pt_idx] = number of directions for which pt has a witness
    # Initially, all counts are 0, so all points (in N) have all 50 directions as witnesses.
    witness_count = [50] * n
    
    # Greedy removal
    # Order: Random or specific? 
    # Let's try removing points in a fixed order, or maybe random to escape local optima?
    # Fixed order is deterministic. Let's stick to index order for simplicity and speed.
    
    # To potentially improve, we could try multiple passes or different orders.
    # But 1 pass of 2401 points * 2400 checks is ~6M ops. Very fast.
    # We can try to remove more by doing multiple passes until no more can be removed.
    
    removed_any = True
    while removed_any:
        removed_any = False
        for cand in range(n):
            if time.time() > deadline - 10:
                break
            
            if not in_N[cand]:
                continue
                
            # Check if cand can be removed
            # 1. Check if cand itself will have a witness
            # Cand is currently In. 
            # After removal, Cand is Out.
            # Cand has witness in dir d iff not_in_N_count[d][line] == 0 (before removal)
            # Because new_count = old_count + 1. Out needs new_count == 1.
            # So old_count must be 0.
            
            cand_has_witness = False
            for dir_idx in range(50):
                line_idx = point_line[cand][dir_idx]
                if not_in_N_count[dir_idx][line_idx] == 0:
                    cand_has_witness = True
                    break
            
            if not cand_has_witness:
                continue
                
            # 2. Check if any other point loses its last witness
            broken = False
            # Store changes to revert if broken
            # Changes: 
            # - in_N[cand] = False
            # - not_in_N_count[d][l] += 1 for d in 50
            # - witness_count[Q] changes for Q on lines through cand
            
            # We need to simulate the changes.
            # To avoid deep copying, we can just check and revert.
            
            # List of (Q, delta_witness)
            changes = []
            
            for dir_idx in range(50):
                line_idx = point_line[cand][dir_idx]
                line = direction_lines[dir_idx][line_idx]
                old_count = not_in_N_count[dir_idx][line_idx]
                new_count = old_count + 1
                
                for Q in line:
                    if Q == cand:
                        continue
                    
                    # Determine witness status change for Q
                    # Q is either In or Out currently.
                    
                    if in_N[Q]:
                        # Q is In
                        # Old witness: old_count == 0
                        # New witness: new_count == 0 (False)
                        had_witness = (old_count == 0)
                        has_witness = False
                    else:
                        # Q is Out
                        # Old witness: old_count == 1
                        # New witness: new_count == 1
                        had_witness = (old_count == 1)
                        has_witness = (new_count == 1)
                    
                    if had_witness and not has_witness:
                        # Q loses witness
                        changes.append((Q, -1))
                        if witness_count[Q] == 1:
                            broken = True
                            break
                    elif not had_witness and has_witness:
                        # Q gains witness
                        changes.append((Q, 1))
                    
                if broken:
                    break
            
            if broken:
                continue
            
            # Apply changes
            in_N[cand] = False
            removed_any = True
            
            for dir_idx in range(50):
                line_idx = point_line[cand][dir_idx]
                not_in_N_count[dir_idx][line_idx] += 1
            
            for Q, delta in changes:
                witness_count[Q] += delta
            
    # Build result
    pts = [all_points[i] for i in range(n) if in_N[i]]
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
