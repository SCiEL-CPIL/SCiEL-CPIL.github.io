# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict, deque
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    Strategy:
    1. Greedy construction of the complement set S (points NOT in N).
       We add points to S one by one. A point x can be added to S if for every
       line L passing through x, there is some other point y on L (y != x) such
       that the line L still contains a "witness" for y that does not pass through x.
       Actually, the condition for N = All \ S to be a Nikodym set is:
       For every x in All, there exists a line L through x such that (L \ {x}) \cap S is empty.
       This is equivalent to: For every x, there is a direction d such that the line L(x, d)
       contains no points from S other than possibly x.
       
       Let's rephrase the constraint on S:
       If x is in S, then for every direction d, the line L(x, d) must contain at least one
       point y (y != x) such that y is NOT "covered" by x's presence in S in a way that breaks
       y's Nikodym property? No.
       
       Let's stick to the direct definition.
       N is valid iff: for all x in All, exists d such that (L(x,d) \ {x}) subset N.
       This means: for all x in All, exists d such that (L(x,d) \ {x}) \cap S is empty.
       
       Let's define a point x as "blocked" in direction d if S \cap (L(x,d) \ {x}) is not empty.
       x is "safe" if there exists at least one direction d where x is NOT blocked.
       We want to build S such that every x in All is safe.
       
       Greedy Add Strategy:
       Start with S = empty. All points are safe (any direction works).
       Try to add points to S.
       When we add a point p to S:
         For any other point q, in any direction d, if p lies on L(q, d) and p != q,
         then q becomes blocked in direction d (if it wasn't already).
         If q loses its last "safe" direction, then adding p is invalid.
         
       So, for each point q, let safe_dirs[q] be the set of directions d where L(q,d) \ {q} does not intersect S.
       Initially safe_dirs[q] = {0..49}.
       If we add p to S:
         For each q != p, and each direction d such that p in L(q,d):
           safe_dirs[q].remove(d)
         If any safe_dirs[q] becomes empty, we cannot add p.
       
       This is a greedy independent set problem on a hypergraph or similar.
       We can randomize the order of points to try to maximize |S|.
    """
    start_time = time.time()
    deadline = start_time + 590  # 40s safety margin
    
    p = 7
    q_field = p * p  # 49
    k = 3  # alpha^2 = 3
    
    # F_49 multiplication
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + k*b1*b2) % p, (a1*b2 + a2*b1) % p)
    
    # All points
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) for ya in range(p) for yb in range(p)]
    n = len(all_points)  # 2401
    
    # Precompute lines for all 50 directions
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
                mx = mul_f49((xa, xb), (ma, mb))
                c = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                lines_dict[c].append(i)
            direction_lines.append(list(lines_dict.values()))
    
    # point_line[pt][dir] = line_idx
    point_line = [[0] * 50 for _ in range(n)]
    for dir_idx in range(50):
        for line_idx, line in enumerate(direction_lines[dir_idx]):
            for pt_idx in line:
                point_line[pt_idx][dir_idx] = line_idx
    
    # For each point p, and each direction d, we need to know which other points q are on the same line.
    # Actually, we need to know: if we add p to S, which points q lose which directions?
    # q loses direction d if p is on L(q, d) and p != q.
    # This is symmetric: p is on L(q, d) iff q is on L(p, d).
    # So for a fixed p, the set of (q, d) pairs affected is:
    # For each d in 0..49:
    #   L = L(p, d)
    #   For each q in L, q != p:
    #     q loses direction d.
    
    # Precompute for each point p, the list of (q, d) pairs that would be affected if p is added to S.
    # affected[p] = list of (q, d)
    # Note: if multiple points are added, the effect accumulates.
    
    # To make the greedy step fast, we can maintain safe_count[q] = number of safe directions for q.
    # Initially safe_count[q] = 50.
    # When adding p:
    #   For each (q, d) in affected[p]:
    #     if safe_count[q] > 0 and d is currently safe for q:
    #       safe_count[q] -= 1
    #       if safe_count[q] == 0:
    #         adding p is invalid (unless q == p? No, q != p).
    #   Wait, we need to check if d is *currently* safe.
    #   We can maintain a bitmask or a set of safe directions for each q.
    #   With 50 directions, a 64-bit integer bitmask is efficient.
    
    # safe_mask[q]: bit d is 1 if direction d is safe for q.
    # Initially all 1s: (1 << 50) - 1.
    
    # Let's precompute affected[p] as a list of (q, d)
    # But note: if p is added, it affects q only if d is currently safe for q.
    # We can just iterate through affected[p] and check the bit.
    
    # affected[p] is a list of tuples (q, d).
    # Size: for each p, sum over d of (line_size - 1).
    # line_size is 49. So 50 * 48 = 2400 entries per p.
    # Total memory: 2401 * 2400 ~ 5.7M entries. Acceptable.
    
    # Optimization: We don't need to store the list explicitly if we can regenerate it,
    # but precomputing might be faster for the inner loop.
    # Let's precompute a list of lists.
    
    # Actually, we can just use the point_line and direction_lines structures directly.
    # For a given p:
    #   for d in 0..49:
    #     li = point_line[p][d]
    #     line = direction_lines[d][li]
    #     for q in line:
    #       if q != p:
    #         # q loses direction d
    #         if (safe_mask[q] >> d) & 1:
    #            safe_mask[q] &= ~(1 << d)
    #            if safe_mask[q] == 0:
    #               invalid
    
    # This is O(50 * 49) = 2450 operations per point addition.
    # With 2401 points, total ops ~ 5.8M. Very fast.
    
    # We will perform multiple random restarts with different orderings.
    # Also, we can try to remove points from N (greedy removal) as a post-processing step,
    # or just rely on the greedy addition of S.
    
    # The greedy addition of S is equivalent to finding a maximal independent set in a conflict graph,
    # where two points conflict if one blocks the last safe direction of the other.
    # The order matters. Random shuffling helps.
    
    best_S = [False] * n
    best_size = 0
    
    num_restarts = 200  # Increase restarts since each run is fast
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Randomize order
        order = list(range(n))
        random.shuffle(order)
        
        in_S = [False] * n
        safe_mask = [(1 << 50) - 1] * n
        current_size = 0
        
        for idx in order:
            # Check if we can add idx to S
            # We need to verify that no other point q loses its last safe direction.
            # We can simulate the addition and rollback if invalid.
            # Or just check first.
            
            # To check:
            # For each d in 0..49:
            #   li = point_line[idx][d]
            #   line = direction_lines[d][li]
            #   for q in line:
            #     if q != idx:
            #       if (safe_mask[q] >> d) & 1 and safe_mask[q] == (1 << d): # only d is safe
            #          # then adding idx makes q unsafe
            #          invalid = True
            
            can_add = True
            # Optimization: only check q where safe_mask[q] is a power of 2 (only one safe dir left)
            # and that dir is d.
            
            for d in range(50):
                li = point_line[idx][d]
                line = direction_lines[d][li]
                for q in line:
                    if q == idx:
                        continue
                    # If q has exactly one safe direction, and it is d, then adding idx blocks it.
                    if safe_mask[q] == (1 << d):
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                # Add idx to S
                in_S[idx] = True
                current_size += 1
                
                # Update safe masks
                for d in range(50):
                    li = point_line[idx][d]
                    line = direction_lines[d][li]
                    for q in line:
                        if q == idx:
                            continue
                        if (safe_mask[q] >> d) & 1:
                            safe_mask[q] &= ~(1 << d)
        
        if current_size > best_size:
            best_size = current_size
            best_S = in_S[:]
            
            # Early stop if we reach a very good size?
            # Theoretical max is unknown, but let's say if we get > 1000, it's good.
            # Current score 0.9524 corresponds to complement size ~ 114.
            # 2401 * 0.9524 = 2286.7 -> |N| = 2287, |S| = 114.
            # If we can get |S| > 114, we improve.
            if best_size > 150:
                # Keep searching a bit more, but we have a good candidate.
                pass
    
    # Build result N = complement of S
    pts = [all_points[i] for i in range(n) if not best_S[i]]
    if not pts:
        pts = all_points
        
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
