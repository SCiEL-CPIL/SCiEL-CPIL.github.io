# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict
from heapq import heappush, heappop

def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2.
    Starts from the baseline Hermitian-unital construction, then uses a
    priority-queue guided greedy search to exclude as many additional points
    as possible without breaking the Nikodym property.
    """
    start_time = time.time()
    deadline = start_time + 570  # safety margin

    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3

    # F_49 multiplication: (a1+b1*alpha)*(a2+b2*alpha) = (a1*a2 + k*b1*b2) + (a1*b2 + a2*b1)*alpha
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1 * a2 + k * b1 * b2) % p, (a1 * b2 + a2 * b1) % p)

    # All 2401 points indexed
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p)
                  for ya in range(p) for yb in range(p)]
    n = len(all_points)  # 2401
    point_idx = {pt: i for i, pt in enumerate(all_points)}

    # Precompute lines for each of the 50 directions
    # Direction 0: vertical v=(0,1)
    # Directions 1..49: v=(1, m) for m in F_49
    direction_lines = []

    # Vertical
    vert_lines = defaultdict(list)
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        vert_lines[(xa, xb)].append(i)
    direction_lines.append(list(vert_lines.values()))

    # Slope directions
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            lines_dict = defaultdict(list)
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                mx = mul_f49((xa, xb), m)
                c = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                lines_dict[c].append(i)
            direction_lines.append(list(lines_dict.values()))

    # point_line[pt][dir] = line index
    point_line = [[None] * 50 for _ in range(n)]
    for dir_idx in range(50):
        for line_idx, line in enumerate(direction_lines[dir_idx]):
            for pt_idx in line:
                point_line[pt_idx][dir_idx] = line_idx

    # Baseline: exclude points on Hermitian unital U with y NOT in special set Y
    y_special = set((u * u % p, v) for u in range(p) for v in range(p))
    in_N = [True] * n
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        if (2 * xa + ya * ya - yb * yb * k) % p == 0 and (ya, yb) not in y_special:
            in_N[i] = False

    # not_in_N_count[dir][line] = count of points not in N on that line
    not_in_N_count = [[0] * len(direction_lines[d]) for d in range(50)]
    for dir_idx in range(50):
        for line_idx, line in enumerate(direction_lines[dir_idx]):
            cnt = 0
            for pt_idx in line:
                if not in_N[pt_idx]:
                    cnt += 1
            not_in_N_count[dir_idx][line_idx] = cnt

    # witness_count[pt] = number of directions for which pt has a valid witness
    witness_count = [0] * n
    for pt_idx in range(n):
        for dir_idx in range(50):
            line_idx = point_line[pt_idx][dir_idx]
            cnt = not_in_N_count[dir_idx][line_idx]
            if cnt == 0 or (cnt == 1 and not in_N[pt_idx]):
                witness_count[pt_idx] += 1

    # For each point, precompute the list of (dir_idx, line_idx) pairs where
    # the point is the SOLE non-N point on the line (critical for that direction)
    # i.e., not_in_N_count[dir][line] == 1 and not in_N[pt]
    # Actually, we need: for each point P in N, which directions/lines would be
    # broken if P is excluded? A direction d through P is "safe" to lose only if
    # every other point Q on that line still has another witness direction.
    #
    # Better approach: for each point P in N, compute the set of (dir, line) pairs
    # where P is the ONLY point in N that "covers" some other point.
    #
    # Let's define: for a point Q and direction d, Q is "covered" by d if all other
    # points on the line through Q in direction d are in N.
    # If we remove P from N, then for any Q on the same line as P in direction d,
    # Q may lose direction d as a witness if P was the only non-N point (or one of few).
    #
    # Actually the condition is: Q has direction d as witness iff the number of non-N
    # points on the line through Q in direction d (excluding Q itself if Q not in N)
    # is 0. More precisely:
    #   Let L be the line through Q in direction d.
    #   Q has d as witness iff all points in L \ {Q} are in N.
    #   Equivalently: the count of points in L \ {Q} that are NOT in N is 0.
    #
    # If Q is in N: Q has d as witness iff not_in_N_count[d][line] == 0
    #   (since Q itself is in N, it doesn't contribute to the count)
    #   Wait, not_in_N_count includes Q if Q is not in N. Let me re-derive.
    #
    # not_in_N_count[d][line] = number of points in the line that are NOT in N.
    # For Q in N: Q has d as witness iff not_in_N_count[d][line] == 0
    # For Q not in N: Q has d as witness iff not_in_N_count[d][line] == 1 (only Q itself)
    #
    # When we exclude P (P was in N, now not in N):
    #   For each direction d, let L be the line through P in direction d.
    #   not_in_N_count[d][L] increases by 1.
    #   For each Q in L \ {P}:
    #     If Q is in N: Q had d as witness iff old_count == 0. After: new_count = old_count+1.
    #       Q loses d as witness if old_count == 0 (now new_count == 1 != 0).
    #     If Q is not in N: Q had d as witness iff old_count == 1. After: new_count = old_count+1.
    #       Q loses d as witness if old_count == 1 (now new_count == 2 != 1).
    #
    # So the points that lose a witness are those Q in L\{P} where:
    #   (Q in N and old_count == 0) or (Q not in N and old_count == 1)
    #
    # Precompute for each point P and direction d, the list of Q in L\{P} that would
    # lose direction d if P is excluded. This is expensive to precompute fully, so
    # we compute on the fly but use witness_count to quickly check.

    # Build a priority queue of candidate points to exclude.
    # Priority: points with higher witness_count are safer to exclude (they have more
    # directions to fall back on). Actually, we want to exclude points that cause
    # the least damage. Let's use a different heuristic: exclude points that are
    # "redundant" - i.e., for every other point Q on lines through P, Q has >= 2 witnesses.
    
    # Initialize heap with all points currently in N
    # Use negative witness_count as priority (we want to try high witness_count first? 
    # Or low? Let's try points where exclusion is most likely safe.)
    
    # Actually, let's try a simpler approach: iterate over all points in N,
    # and for each, check if excluding it is safe. Use the witness_count check.
    
    # For efficiency, let's precompute for each point P and direction d:
    #   the line through P in direction d
    #   and for each Q on that line, whether Q would lose direction d if P is excluded.
    
    # This is 2401 * 50 * 49 ~ 5.8M entries, which is feasible.
    
    # lost_witness[P][d] = list of Q in line_through_P_in_dir_d \ {P} that would lose d
    # But we only care about Q that would drop to 0 witnesses. So we can just check
    # witness_count[Q] > 1 (or >= 2) at the time of exclusion.
    
    # Let's just do the greedy loop with on-the-fly checks, but optimize the inner loop.
    
    # Sort candidates: try to exclude points that are on the unital first (they're
    # "structurally" more likely to be removable), then others.
    
    candidates = []
    for i in range(n):
        if in_N[i]:
            candidates.append(i)
    
    # Sort by witness_count descending (higher witness count = more robust, safer to remove)
    candidates.sort(key=lambda i: -witness_count[i])
    
    excluded_count = 0
    for cand in candidates:
        if time.time() > deadline - 30:
            break
        
        if not in_N[cand]:
            continue
        
        # Tentatively exclude cand
        in_N[cand] = False
        excluded_count += 1
        
        # Check if this breaks any point's witness
        broken = False
        affected_points = set()
        
        for dir_idx in range(50):
            line_idx = point_line[cand][dir_idx]
            line = direction_lines[dir_idx][line_idx]
            old_count = not_in_N_count[dir_idx][line_idx]
            new_count = old_count + 1
            
            # Update not_in_N_count
            not_in_N_count[dir_idx][line_idx] = new_count
            
            # Check each Q in line \ {cand}
            for Q in line:
                if Q == cand:
                    continue
                
                # Did Q lose direction dir_idx as witness?
                # Q in N: had witness iff old_count == 0; has witness iff new_count == 0
                # Q not in N: had witness iff old_count == 1; has witness iff new_count == 1
                
                if in_N[Q]:
                    had_witness = (old_count == 0)
                    has_witness = (new_count == 0)
                else:
                    had_witness = (old_count == 1)
                    has_witness = (new_count == 1)
                
                if had_witness and not has_witness:
                    # Q loses this direction
                    witness_count[Q] -= 1
                    affected_points.add(Q)
                    if witness_count[Q] == 0:
                        broken = True
                        break
            
            if broken:
                break
        
        if broken:
            # Revert
            in_N[cand] = True
            excluded_count -= 1
            
            for dir_idx in range(50):
                line_idx = point_line[cand][dir_idx]
                not_in_N_count[dir_idx][line_idx] -= 1
            
            for Q in affected_points:
                witness_count[Q] += 1
        else:
            # Keep the exclusion
            # witness_count for affected points is already updated
            pass
    
    # Build result
    points = []
    for i in range(n):
        if in_N[i]:
            xa, xb, ya, yb = all_points[i]
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
