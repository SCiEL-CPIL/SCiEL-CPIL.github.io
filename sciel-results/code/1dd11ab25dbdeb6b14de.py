# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    Starts with the full space and greedily removes points to maximize the complement,
    ensuring the Nikodym property is maintained.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
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
    
    # Precompute lines for each direction
    # direction_lines[dir_idx] = list of lines, each line is a list of point indices
    direction_lines = []
    
    # Direction 0: vertical v = (0,1)
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

    # Initialize N as all points
    in_N = [True] * n
    
    # witness_count[pt_idx] = number of directions for which pt has a witness
    # A point y has witness d if all other points on L_{y,d} are in N.
    # Initially, all points are in N, so all points have all 50 directions as witnesses.
    witness_count = [50] * n
    
    # Killers[x] = number of points y that would lose their ONLY witness if x is removed.
    # y loses its only witness if witness_count[y] == 1 and x is on the critical line for y.
    # Initially, witness_count is 50, so no one is critical. Killers are all 0.
    Killers = [0] * n
    
    # To efficiently find critical points and update Killers, we need to know
    # for each point y, if it becomes critical (count drops to 1), which direction is it?
    # We can just scan the 50 directions to find the one that is still a witness.
    # This is O(50) per point that becomes critical.
    
    # Main Loop: Greedy removal
    # We maintain a list of "safe" points (Killers[x] == 0).
    # To avoid O(N^2) checks, we just iterate.
    
    # Optimization: Use a priority queue or just a list of candidates.
    # Since N is small, a simple loop is fine.
    
    removed_count = 0
    # We will iterate until no more points can be removed.
    # To make it faster, we can try to remove multiple points per iteration if they are independent.
    # But for safety, let's do one by one, picking the first available safe point.
    # Or, pick the point that "unblocks" the most? No, just any safe point.
    
    # To speed up, we can check all points and remove all that are safe in one pass,
    # but we must be careful about the "multiple loss" issue.
    # If we remove x1 and x2, and y is neighbor to both, y loses 2 witnesses.
    # If y had 2 witnesses, it drops to 0. Invalid.
    # So we can only remove a set S if for all y, losses(y) < witness_count[y].
    
    # Simple approach:
    # 1. Find all x in N with Killers[x] == 0.
    # 2. If none, break.
    # 3. Pick one such x (e.g., the first one).
    # 4. Remove x.
    # 5. Update witness_count for neighbors.
    # 6. Update Killers for points that become critical.
    
    # This is O(N * 50 * 49) per removal.
    # Max 2401 removals. 2401 * 2450 ~ 6M ops. Very fast.
    
    while time.time() < deadline - 10:
        # Find a safe point
        safe_point = -1
        for i in range(n):
            if in_N[i] and Killers[i] == 0:
                safe_point = i
                break
        
        if safe_point == -1:
            break
        
        # Remove safe_point
        in_N[safe_point] = False
        removed_count += 1
        
        # Update neighbors
        for dir_idx in range(50):
            line_idx = point_line[safe_point][dir_idx]
            line = direction_lines[dir_idx][line_idx]
            
            for Q in line:
                if Q == safe_point:
                    continue
                
                # Q loses dir_idx as witness if it had it.
                # How do we know if Q had it?
                # Q had it if all other points on L_{Q, dir_idx} were in N.
                # Before removal, safe_point was in N.
                # If safe_point was the ONLY point in N on L_{Q, dir_idx} \ {Q}?
                # No, the condition is: L \ {Q} subset N.
                # If safe_point is in L \ {Q}, and safe_point was in N, it contributed to the "all in N" condition.
                # After removal, safe_point is NOT in N.
                # So if there are any OTHER points in L \ {Q} that are NOT in N, Q already didn't have the witness.
                # Q only LOSES the witness if it PREVIOUSLY HAD it.
                # Q previously had it iff L_{Q, dir_idx} \ {Q} was a subset of N.
                # This implies safe_point was in N (which it was) AND all others were in N.
                # So we need to check if L_{Q, dir_idx} \ {Q} \ {safe_point} is a subset of N.
                # If yes, Q had the witness, and now loses it.
                # If no, Q didn't have the witness, and status doesn't change.
                
                # Efficient check:
                # We can maintain a count of "missing points" on each line for each point?
                # That's too much state.
                
                # Alternative:
                # We know that Q loses the witness IF AND ONLY IF
                # (1) Q currently has witness_count[Q] > 0 (obviously)
                # (2) The line L_{Q, dir_idx} \ {Q} contained NO points outside N BEFORE the removal.
                #     This is equivalent to: The only point outside N on L_{Q, dir_idx} \ {Q} was safe_point?
                #     No, if there were other points outside N, Q didn't have the witness.
                #     So Q loses the witness iff safe_point was the ONLY point in N on L_{Q, dir_idx} \ {Q}?
                #     No, iff L_{Q, dir_idx} \ {Q} was fully in N.
                
                # Let's use a helper:
                # had_witness = True
                # for R in line:
                #    if R == Q or R == safe_point: continue
                #    if not in_N[R]:
                #       had_witness = False
                #       break
                # if had_witness:
                #    witness_count[Q] -= 1
                #    if witness_count[Q] == 1:
                #       # Q becomes critical
                #       # Find the critical direction
                #       # Update Killers for points on that line
                #       pass
                
                # This inner loop is O(49).
                # Total per removal: 50 directions * 49 points * 49 checks = ~120k ops.
                # 2401 removals * 120k = 288M ops. Might be slow in Python (10-30s).
                # We need to optimize this check.
                
                # Optimization:
                # Maintain `missing_count[dir_idx][line_idx]` = number of points in that line NOT in N.
                # Q has witness d iff missing_count[d][line_idx(Q,d)] == 0 (if Q in N)
                # or missing_count[d][line_idx(Q,d)] == 1 (if Q not in N, and the missing point is Q).
                # Since Q is in N (we are removing from N), Q has witness d iff missing_count[d][line_idx(Q,d)] == 0.
                
                # Let's maintain `missing_count`.
                # Initially 0.
                # When we remove safe_point, for each dir, missing_count[dir][line_idx(safe_point, dir)] += 1.
                
                # So, Q loses witness d iff:
                # BEFORE: missing_count[d][line_idx(Q,d)] == 0
                # AFTER: missing_count[d][line_idx(Q,d)] == 1 (since safe_point is on that line)
                
                # This is a simple check!
                
                # We need `missing_count` array.
                
                # Let's restructure the loop to use missing_count.
                
                # I'll implement this in the next version of the code block.
                
                # For now, to ensure correctness and speed, I will use the missing_count approach.
                
                pass # Placeholder for logic
            
            # Update missing_count for safe_point
            missing_count[dir_idx][line_idx] += 1
            
            # Update witness counts and Killers for Qs
            # We need to iterate Qs again to update witness_count.
            
            # This logic is getting complex to inline. I will rewrite the loop properly below.
            
    # Re-implementing the loop with missing_count for efficiency

    missing_count = [[0] * len(direction_lines[d]) for d in range(50)]
    
    # Re-initialize state for the optimized loop
    in_N = [True] * n
    witness_count = [50] * n
    Killers = [0] * n
    missing_count = [[0] * len(direction_lines[d]) for d in range(50)]
    
    while time.time() < deadline - 10:
        safe_point = -1
        for i in range(n):
            if in_N[i] and Killers[i] == 0:
                safe_point = i
                break
        
        if safe_point == -1:
            break
        
        in_N[safe_point] = False
        
        for dir_idx in range(50):
            line_idx = point_line[safe_point][dir_idx]
            
            # Update missing count
            old_missing = missing_count[dir_idx][line_idx]
            new_missing = old_missing + 1
            missing_count[dir_idx][line_idx] = new_missing
            
            # Points Q on this line lose witness dir_idx if they had it.
            # Q had it if old_missing == 0.
            if old_missing == 0:
                # All Qs on this line (except safe_point) lose this witness
                line = direction_lines[dir_idx][line_idx]
                for Q in line:
                    if Q == safe_point:
                        continue
                    
                    witness_count[Q] -= 1
                    
                    if witness_count[Q] == 1:
                        # Q becomes critical. Find its critical direction.
                        # It must be a direction d' where missing_count[d'][line_idx(Q,d')] == 0.
                        # Since Q is in N.
                        crit_dir = -1
                        for d in range(50):
                            li = point_line[Q][d]
                            if missing_count[d][li] == 0:
                                crit_dir = d
                                break
                        
                        if crit_dir != -1:
                            # Q now kills all points on its critical line
                            crit_line_idx = point_line[Q][crit_dir]
                            crit_line = direction_lines[crit_dir][crit_line_idx]
                            for Z in crit_line:
                                if Z != Q:
                                    Killers[Z] += 1

    # Build result
    pts = [all_points[i] for i in range(n) if in_N[i]]
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
