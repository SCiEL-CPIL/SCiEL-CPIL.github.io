import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = 550.0  # Safety margin before 600s kill

    p = 7
    # Field F_49 = F_7[alpha] with alpha^2 = 3
    # Element represented as (a, b) for a + b*alpha
    # Point in F_49^2: ((xa, xb), (ya, yb))
    # We can flatten a point to an integer index 0..2400 for faster set operations
    # index = ((xa * p + xb) * p + ya) * p + yb
    
    def field_mul(a1, b1, a2, b2):
        ra = (a1 * a2 + 3 * b1 * b2) % p
        rb = (a1 * b2 + a2 * b1) % p
        return ra, rb
    
    def field_add(a1, b1, a2, b2):
        return (a1 + a2) % p, (b1 + b2) % p
    
    def field_sub(a1, b1, a2, b2):
        return (a1 - a2) % p, (b1 - b2) % p
    
    def field_neg(a, b):
        return (-a) % p, (-b) % p

    # Precompute all points and their flat indices
    all_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    all_points.append(((xa, xb), (ya, yb)))
    
    # Map flat index to point and vice versa
    point_to_idx = {}
    for i, pt in enumerate(all_points):
        point_to_idx[pt] = i
        
    # Precompute lines
    # A line is defined by a point and a direction (slope m or vertical)
    # Directions:
    # 1. Vertical: x = constant. There are p^2 = 49 such lines.
    # 2. Slope m: y = m*x + c. There are p^2 slopes and p^2 intercepts -> p^4 lines?
    #    No, in F_{49}^2, there are (49+1)*49 lines? 
    #    Number of points = 49^2 = 2401.
    #    Number of directions = 49 + 1 = 50.
    #    Number of lines = 2401 * 50 / 49 = 49 * 50 = 2450 lines.
    
    # Let's enumerate all lines and their points.
    lines = [] # List of sets of point indices
    
    # 1. Vertical lines x = (xa, xb)
    for xa in range(p):
        for xb in range(p):
            line_pts = set()
            for ya in range(p):
                for yb in range(p):
                    pt = ((xa, xb), (ya, yb))
                    line_pts.add(point_to_idx[pt])
            lines.append(line_pts)
            
    # 2. Non-vertical lines y = m*x + c
    # m, c in F_49
    for ma in range(p):
        for mb in range(p):
            for ca in range(p):
                for cb in range(p):
                    line_pts = set()
                    for xa in range(p):
                        for xb in range(p):
                            # y = m*x + c
                            # (ya, yb) = (ma, mb) * (xa, xb) + (ca, cb)
                            mu_a, mu_b = field_mul(ma, mb, xa, xb)
                            ya = (mu_a + ca) % p
                            yb = (mu_b + cb) % p
                            pt = ((xa, xb), (ya, yb))
                            line_pts.add(point_to_idx[pt])
                    lines.append(line_pts)
                    
    # lines is a list of 2450 sets of integers (point indices)
    # len(lines) should be 2450.
    
    # Objective: Find a set of lines L such that the union of points in L is N.
    # Validity: For every point x, there exists a line L_x containing x such that (L_x \ {x}) subset of N.
    # Note: If x is in N, it might be covered by many lines. We just need ONE line for each x.
    # If x is NOT in N, it must be on a line L_x such that all OTHER points on L_x are in N.
    # This implies that if x is not in N, the line L_x must be "almost" contained in N.
    # Specifically, L_x \ {x} subset N.
    
    # Strategy: Greedy Set Cover to minimize |N|.
    # We want to cover all points with lines, but with a twist:
    # The condition "for every x, there is a line L_x with L_x\{x} subset N" is equivalent to:
    # For every x, either:
    # 1. x is in N, and there is a line L containing x such that L\{x} subset N.
    # 2. x is not in N, and there is a line L containing x such that L\{x} subset N.
    
    # Actually, if we pick a set of lines \mathcal{L}, and let N = \cup_{L \in \mathcal{L}} L,
    # is it automatically valid?
    # Let x be any point.
    # If x \in N, then x is on some line L \in \mathcal{L}.
    # Does L\{x} \subset N?
    # Not necessarily. Other points on L might not be in N if they are not covered by any line in \mathcal{L}.
    # But wait, if L is in \mathcal{L}, then ALL points on L are in N (since N is the union of lines in \mathcal{L}).
    # So if x is on L \in \mathcal{L}, then L \subset N, so L\{x} \subset N.
    # So any x in N is satisfied by ANY line in \mathcal{L} containing it.
    
    # What if x \notin N?
    # Then x is not on any line in \mathcal{L}.
    # We need a line L_x containing x such that L_x\{x} \subset N.
    # L_x is NOT in \mathcal{L} (because if it were, x would be in N).
    # So we need a line L_x that is NOT in \mathcal{L}, passes through x, and all other points on it are in N.
    # This means L_x \cap N = L_x \setminus \{x\}.
    # In other words, L_x contains exactly one point outside N, which is x.
    
    # So the problem is:
    # Find a set of lines \mathcal{L} such that:
    # 1. N = \cup_{L \in \mathcal{L}} L.
    # 2. For every x \notin N, there exists a line L_x \notin \mathcal{L} containing x such that L_x \setminus \{x\} \subset N.
    
    # This is a complex constraint.
    # However, if we choose \mathcal{L} to be ALL lines, then N = All Points.
    # Then x \notin N is empty set. Valid.
    # But |N| = 2401. Complement 0. Score 0.
    
    # We want to MINIMIZE |N|.
    # If we choose \mathcal{L} = \emptyset, N = \emptyset.
    # Then for every x, we need a line L_x such that L_x\{x} \subset \emptyset.
    # This implies L_x has only 1 point? No, lines have 49 points.
    # So L_x\{x} has 48 points. They must be in \emptyset. Impossible.
    # So \mathcal{L} cannot be empty.
    
    # Let's try a different perspective.
    # The condition "L_x\{x} \subset N" means that x is an "isolated" point of the complement S = All \ N on the line L_x.
    # Actually, it means x is the ONLY point of S on L_x.
    # So for every x in S, there is a line L_x passing through x such that L_x \cap S = \{x\}.
    # This is exactly the definition of a "blocking set" or something similar?
    # No, it's the dual of a blocking set.
    # A set S is such that every point in S is on a line that intersects S only at that point.
    # This means S is a set of points such that no two points in S lie on a line L where L \cap S has size > 1?
    # No. If x, y in S are on a line L, then for x, the line L_x must have L_x \cap S = \{x\}.
    # So L cannot be L_x.
    # But there must be SOME line through x with only x in S.
    
    # Let's try to construct S (the complement) directly.
    # We want to MAXIMIZE |S|.
    # S must satisfy: For every x in S, there exists a line L_x through x such that L_x \cap S = \{x\}.
    # This means x is "visible" from infinity? No.
    # It means x is not in the "interior" of S with respect to lines.
    # In fact, if S is a set of points no three of which are collinear?
    # If no three are collinear, then for any x in S, pick any line through x.
    # If the line contains another point y in S, then L \cap S has at least 2.
    # But we can pick a line that does NOT contain any other point of S.
    # Since there are 50 lines through x, and S is small, we can likely find a line through x that contains no other point of S.
    # If such a line exists, then L \cap S = \{x\}.
    # So if S is a set of points such that for every x in S, there is a line through x containing no other point of S, then S is a valid complement.
    # This is always true if S is finite and small enough?
    # Actually, for ANY set S, if we pick a random line through x, the probability it contains another point of S is low if S is sparse.
    # But we need it to hold for ALL x in S.
    
    # Consider S = All Points.
    # For x in S, any line through x contains 48 other points in S.
    # So L \cap S = L. Size 49. Not \{x\}.
    # So S = All Points is INVALID.
    
    # Consider S = 1 point.
    # x in S. Any line through x contains no other point of S.
    # So L \cap S = \{x\}. Valid.
    # N = All \ {x}. |N| = 2400. Complement 1.
    
    # Consider S = 2 points x, y.
    # For x, we need a line through x not containing y.
    # There are 49 such lines. Valid.
    # For y, we need a line through y not containing x.
    # There are 49 such lines. Valid.
    # So S of size 2 is valid.
    
    # Consider S = 3 points x, y, z.
    # If x, y, z are collinear on line L.
    # For x, any line through x:
    # - If L is chosen, L \cap S = {x, y, z}. Invalid for x.
    # - If another line M is chosen, M \cap S = {x} (assuming M doesn't hit y or z).
    # So x is satisfied by M.
    # Similarly y and z are satisfied by lines not L.
    # So even if collinear, it's valid as long as we don't HAVE to use L.
    # The condition is "there EXISTS a line".
    # So for any finite set S, is it always valid?
    # Let x in S.
    # We need a line L through x such that L \cap S = \{x\}.
    # This fails ONLY if EVERY line through x contains some other point of S.
    # How many points of S can block all lines through x?
    # Each other point y in S blocks exactly 1 line (the line xy).
    # There are 49 lines through x.
    # So if |S \ {x}| >= 49, it is possible that all lines are blocked.
    # Specifically, if S \ {x} contains a point on every line through x, then x is not satisfied.
    # A set of points that intersects every line through x is a "blocking set" for the pencil at x.
    # The minimum size of such a set is 49? No.
    # In F_q^2, a set of points that meets every line through a fixed point P.
    # The lines through P are the "directions".
    # If we pick one point from each line, we need 49 points.
    # So if |S| > 49, it is possible that some x in S is blocked.
    # But we want to MAXIMIZE |S| while keeping it valid.
    # So we want S such that for every x in S, S does NOT contain a blocking set for x.
    # i.e., for every x in S, there is at least one line through x that contains no other point of S.
    
    # This suggests that if we pick S to be "sparse" in a geometric sense, it will be valid.
    # For example, if S is a set of points in general position (no 3 collinear), then for any x, the lines through x and other points are distinct.
    # There are |S|-1 such lines.
    # If |S|-1 < 49, then there is a line through x not connecting to any other point in S.
    # So if |S| <= 50, S is always valid?
    # If |S| = 50, and no 3 collinear, then for any x, 49 other points block 49 lines.
    # Are there exactly 49 lines through x? Yes.
    # So if the 49 other points lie on distinct lines through x, then ALL lines are blocked.
    # So x would be invalid.
    # So we need |S| such that for every x, the other points do NOT cover all directions.
    
    # To maximize |S|, we should try to make S as large as possible while avoiding this "blocking" configuration for any x.
    # If we pick S randomly, the probability that a specific x is blocked is low.
    # But we need it for ALL x.
    
    # Let's try a heuristic:
    # Start with S = All Points.
    # Check validity.
    # If invalid, remove points that are "bad" (blocked).
    # But removing points makes S smaller, which is the opposite of what we want?
    # No, we want to MAXIMIZE |S|.
    # So we want to ADD points to S?
    # Start with S = \emptyset. Valid.
    # Add points one by one.
    # Check if S remains valid.
    # If adding x makes S invalid, skip x.
    
    # This is a greedy algorithm to maximize |S|.
    
    # Validity check for a set S:
    # For every x in S, check if there exists a line L through x such that L \cap S = \{x\}.
    # This is equivalent to: The set of directions from x to other points in S does NOT cover all 50 directions?
    # No, 49 non-vertical + 1 vertical?
    # Directions in F_49^2: 49 slopes + 1 vertical = 50 directions.
    # So for x, let D_x be the set of directions to other points in S.
    # If |D_x| < 50, then there is a free direction.
    # So x is valid if |D_x| < 50.
    # Since there are only 49 other points max if S is small, |D_x| <= 49.
    # So if |S| <= 50, it is ALWAYS valid?
    # Wait. If |S| = 50, and for some x, the other 49 points lie on distinct lines through x, then |D_x| = 49.
    # But there are 50 directions. So one direction is free.
    # So x is valid.
    # So ANY set S with |S| <= 50 is valid?
    # Yes, because max directions blocked is |S|-1.
    # If |S|-1 < 50, i.e., |S| <= 50, then there is always a free direction.
    
    # So we can have |S| = 50.
    # Can we have |S| > 50?
    # If |S| = 51, then for some x, it is possible that the other 50 points block all 50 directions.
    # This happens if the other points form a "blocking set" for x.
    # A blocking set for x is a set of points such that every line through x contains at least one point from the set.
    # The minimum size of a blocking set for x in F_49^2 is 49?
    # No, a line not through x is a blocking set for the lines through x? No.
    # A set of points B blocks x if for every line L through x, L \cap B != \emptyset.
    # This is equivalent to B being a blocking set for the dual pencil.
    # In F_q^2, the minimum size of a blocking set for a pencil of lines through a point is q.
    # Here q=49.
    # So if the other points contain a blocking set of size 49, then x is blocked.
    # So if |S| is large, we must ensure that for no x in S does S \ {x} contain a blocking set for x.
    
    # Heuristic:
    # Start with S = \emptyset.
    # Try to add points.
    # When adding x, check if S \cup {x} is valid.
    # To check validity of S \cup {x}:
    # 1. Check x: Does S contain a blocking set for x?
    #    If yes, x is invalid. Don't add.
    # 2. Check existing points y in S:
    #    Adding x might block y.
    #    y was valid before, meaning there was a free direction for y.
    #    Adding x blocks the direction y->x.
    #    If the free direction for y was y->x, then y becomes invalid.
    #    So we need to check if y still has a free direction.
    
    # This is expensive to check for all y.
    # But if we keep S small, it's fast.
    # We can just compute the set of blocked directions for each y in S.
    
    # Let's implement this greedy.
    
    # Precompute directions for each pair of points?
    # Or just compute on the fly.
    
    # Directions from p1 to p2:
    # If p1 == p2, undefined.
    # If x1 == x2, vertical.
    # Else slope m = (y2-y1)/(x2-x1).
    
    def get_direction(p1, p2):
        # p1, p2 are flat indices
        # Convert to coords
        # This is slow. Let's precompute.
        pass
        
    # Precompute directions matrix? 2401x2401 is ~5.7M entries.
    # Each entry is a direction code (0..49).
    # 5.7M * 1 byte = 5.7 MB. Feasible.
    
    # Direction codes:
    # 0: vertical
    # 1..49: slopes m.
    # Map slope m to code.
    # m is (ma, mb).
    # code = ma * p + mb + 1?
    # There are 49 slopes.
    
    slope_to_code = {}
    code_to_slope = {}
    for ma in range(p):
        for mb in range(p):
            code = ma * p + mb + 1
            slope_to_code[(ma, mb)] = code
            code_to_slope[code] = (ma, mb)
            
    # Vertical code = 0
    
    # Precompute direction from i to j
    # dir[i][j]
    # Use numpy array of shape (2401, 2401), dtype uint8
    
    dir_matrix = np.zeros((2401, 2401), dtype=np.uint8)
    
    for i in range(2401):
        p1 = all_points[i]
        for j in range(2401):
            if i == j:
                dir_matrix[i][j] = 255 # Self
            else:
                p2 = all_points[j]
                # p1 = ((xa1, xb1), (ya1, yb1))
                # p2 = ((xa2, xb2), (ya2, yb2))
                xa1, xb1 = p1[0]
                ya1, yb1 = p1[1]
                xa2, xb2 = p2[0]
                ya2, yb2 = p2[1]
                
                if xa1 == xa2 and xb1 == xb2:
                    # Vertical
                    dir_matrix[i][j] = 0
                else:
                    # dx = x2 - x1
                    # dy = y2 - y1
                    # m = dy / dx
                    # dx = (xa2-xa1, xb2-xb1)
                    # dy = (ya2-ya1, yb2-yb1)
                    # m = dy * dx^{-1}
                    # dx^{-1} = (dx_a^2 - 3 dx_b^2, -dx_a dx_b) / (dx_a^2 + 3 dx_b^2)^2 ?
                    # In F_7[alpha], (a+b alpha)^{-1} = (a-b alpha) / (a^2 - 3 b^2)
                    # Denom = a^2 - 3 b^2 = a^2 + 4 b^2 (mod 7)
                    
                    dxa = (xa2 - xa1) % p
                    dxb = (xb2 - xb1) % p
                    dya = (ya2 - ya1) % p
                    dyb = (yb2 - yb1) % p
                    
                    # Inverse of dx
                    # denom = dxa^2 + 4 dxb^2
                    denom = (dxa*dxa + 4*dxb*dxb) % p
                    # Since dx != 0, denom != 0 in F_49?
                    # a^2 + 4 b^2 = 0 => a^2 = -4 b^2 = 3 b^2.
                    # (a/b)^2 = 3.
                    # 3 is not a square in F_7?
                    # Squares in F_7: 0, 1, 4, 2 (3^2=2), 4 (4^2=2? No 16=2), 2 (5^2=4), 4 (6^2=1).
                    # Wait.
                    # 1^2=1, 2^2=4, 3^2=2, 4^2=2, 5^2=4, 6^2=1.
                    # Squares: 0, 1, 2, 4.
                    # 3 is NOT a square.
                    # So a^2 = 3 b^2 has no solution with b != 0.
                    # If b=0, a=0.
                    # So dx != 0 implies denom != 0.
                    
                    inv_denom = pow(denom, -1, p) # Python 3.8+ supports pow with mod
                    # Actually, just use pow(denom, 6, 7) for F_7 inverse? No, denom is in F_7.
                    # Yes, denom is an element of F_7.
                    
                    inv_dxa = (dxa * inv_denom) % p
                    inv_dxb = (-dxb * inv_denom) % p
                    
                    # m = dy * dx^{-1}
                    # m = (dya + dyb alpha) * (inv_dxa + inv_dxb alpha)
                    ma = (dya * inv_dxa + 3 * dyb * inv_dxb) % p
                    mb = (dya * inv_dxb + dyb * inv_dxa) % p
                    
                    code = slope_to_code[(ma, mb)]
                    dir_matrix[i][j] = code
                    
    # Now, greedy S construction.
    S = set()
    # We want to maximize |S|.
    # Order of points? Random?
    # Maybe points on a line are "worse" because they block each other?
    # No, if we add x, y on same line, they block one direction for each other.
    # But they don't block other directions.
    
    # Let's try adding points in a specific order.
    # Maybe all points?
    # If we add all points, S = All.
    # Then for x, D_x covers all 50 directions?
    # No, S \ {x} has 2400 points.
    # It definitely covers all directions.
    # So x is invalid.
    
    # So we must stop adding when validity is lost.
    
    # To check if adding x to S is valid:
    # 1. Check x: Does S contain a blocking set for x?
    #    i.e., do the points in S cover all 50 directions from x?
    #    We can maintain a bitmask of blocked directions for each point?
    #    No, S changes.
    #    We can just compute the set of directions from x to points in S.
    #    If size < 50, x is ok.
    # 2. Check y in S:
    #    For each y, the new direction y->x is added to blocked directions.
    #    If y was valid before, it had at least 1 free direction.
    #    If the free direction was y->x, then y becomes invalid.
    #    So we need to know if y has ANY other free direction.
    #    This requires knowing the full set of blocked directions for y.
    
    # This is getting complex.
    # Approximation:
    # Only check x.
    # If x is valid (S doesn't block x), add x.
    # This might make some y invalid later, but we don't check y.
    # This will produce an invalid S.
    
    # We need to ensure S is valid.
    # So we must check all y.
    
    # Optimization:
    # Maintain for each y in S the set of blocked directions B_y.
    # Initially B_y = \emptyset.
    # When adding x:
    # For each y in S:
    #   d = direction(y, x)
    #   B_y.add(d)
    #   If B_y covers all 50 directions, y is invalid.
    # Also for x:
    #   B_x = { direction(x, y) for y in S }
    #   If B_x covers all 50 directions, x is invalid.
    
    # If any is invalid, reject x.
    
    # This is O(|S|) per insertion.
    # Total O(|S|^2).
    # If |S| ~ 100, 10000 ops. Fast.
    
    # Let's implement this.
    
    # Directions are 0..49.
    # Bitmask for 50 bits fits in int.
    # all_dirs_mask = (1 << 50) - 1
    
    all_dirs_mask = (1 << 50) - 1
    
    blocked_dirs = {} # y -> int bitmask
    
    # Try to add points in random order?
    # Or sorted order?
    # Let's try random order with a seed.
    random.seed(42)
    point_indices = list(range(2401))
    random.shuffle(point_indices)
    
    for idx in point_indices:
        if time.time() > deadline:
            break
            
        # Check if adding idx to S is valid
        valid = True
        
        # 1. Check idx
        mask_x = 0
        for y in S:
            d = dir_matrix[idx][y]
            mask_x |= (1 << d)
        if mask_x == all_dirs_mask:
            valid = False
            break # Not needed, just break inner loop? No, just set valid=False and skip
            
        if not valid:
            continue
            
        # 2. Check existing y in S
        # We need to simulate adding x to B_y
        # If any B_y | (1 << dir(y,x)) == all_dirs_mask, then invalid
        
        new_blocked = {}
        for y in S:
            d = dir_matrix[y][idx]
            new_mask = blocked_dirs[y] | (1 << d)
            new_blocked[y] = new_mask
            if new_mask == all_dirs_mask:
                valid = False
                break
                
        if valid:
            S.add(idx)
            blocked_dirs[idx] = mask_x
            for y in S:
                if y != idx:
                    blocked_dirs[y] = new_blocked[y]
                    
    # S is a valid complement.
    # N = All \ S
    # Construct N
    
    N = []
    for i in range(2401):
        if i not in S:
            pt = all_points[i]
            xa, xb = pt[0]
            ya, yb = pt[1]
            N.append([[xa, xb], [ya, yb]])
            
    # Ensure shape (k, 2, 2)
    if not N:
        # Fallback: S = All Points is invalid, but S = 1 point is valid.
        # Just return a valid small N.
        # S = { (0,0,0,0) }
        # N = All \ { (0,0,0,0) }
        for i in range(2401):
            if i != 0:
                pt = all_points[i]
                xa, xb = pt[0]
                ya, yb = pt[1]
                N.append([[xa, xb], [ya, yb]])
                
    return np.array(N, dtype=np.int64)
