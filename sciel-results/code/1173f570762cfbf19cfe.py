import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    # Field elements are pairs (a, b)
    # Point is ((xa, xb), (ya, yb))
    
    # Precompute field operations? Not strictly needed for line membership if we use vector arithmetic.
    # Line through x with direction v: { x + t*v | t in F_q }
    # We need to map points to indices 0..2400.
    
    # Map point to index
    def point_to_idx(pt):
        xa, xb = pt[0]
        ya, yb = pt[1]
        return ((xa * p + xb) * p + ya) * p + yb
        
    def idx_to_point(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p*p)) % p
        xa = (idx // (p*p*p)) % p
        return [[xa, xb], [ya, yb]]
        
    # Generate all points
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    points.append([[xa, xb], [ya, yb]])
                    
    # Precompute lines
    # Directions: (1, m) for m in F_q and (0, 1)
    # We represent direction as a pair of pairs ((1, 0), (ma, mb)) or ((0, 0), (0, 1))
    # Actually, direction v = (vx, vy).
    # We can just iterate through all non-zero vectors v in F_q^2.
    # There are q^2 - 1 directions. But lines are undirected? No, v and -v define the same set of lines?
    # A line is a set of points. The direction defines the slope.
    # In F_q^2, there are q+1 distinct directions (slopes) if we consider projective geometry?
    # No, the problem states: v = (1, m) for m in F_q OR v = (0, 1).
    # This covers all 1D subspaces of F_q^2.
    # There are q+1 = 8 directions.
    
    directions = []
    # v = (1, m)
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))
    # v = (0, 1)
    directions.append(((0, 0), (0, 1)))
    
    # For each direction, generate all lines
    # A line is determined by a direction v and a point x on it.
    # To avoid duplicates, we can fix the "first" point on the line in some ordering.
    # Or simply: for each direction v, and each point x, the line is L = { x + t*v }.
    # Two points x, y define the same line for direction v if y - x is a multiple of v.
    # So for a fixed v, the number of lines is q^2 / q = q.
    # Total lines = (q+1) * q = 56.
    
    # We need to identify unique lines.
    # Let's generate all lines and store them as frozensets of point indices.
    lines_set = set()
    lines_list = []
    lines_through = [set() for _ in range(q*q)]
    
    # Helper to add line
    def add_line(pts):
        # pts is a list of point indices
        # Check if this line already exists?
        # We can just add it to lines_list if it's new.
        # But we need lines_through to be accurate.
        pass
        
    # Iterate through directions
    for vx, vy in directions:
        # vx, vy are pairs (a, b)
        # We need to iterate through all points x
        # To generate unique lines for this direction, we can pick a representative point.
        # A point x is a representative if x is the "smallest" index on the line.
        # Or simpler: just generate all q lines for this direction.
        # How to pick the q lines?
        # Pick q points that are not collinear with v?
        # Actually, just pick the first q points in the list that form a basis?
        # Let's just iterate through all points, and for each, generate the line.
        # Then use a set to deduplicate lines (by their content).
        
        for x_idx in range(q*q):
            x_pt = idx_to_point(x_idx)
            # Generate line points
            line_pts = []
            for t_a in range(p):
                for t_b in range(p):
                    # t = t_a + t_b * alpha
                    # x + t*v
                    # x = (xa, xb), v = (vx_a, vx_b), (vy_a, vy_b)
                    # t*v = (t_a*vx_a, t_b*vx_a + t_a*vx_b, ... wait.
                    # Multiplication in F_q: (a,b)*(c,d) = (ac + bd*k, ad + bc)
                    # k = 3 (smallest non-residue mod 7 is 3? 1^2=1, 2^2=4, 3^2=2, 4^2=2, 5^2=4, 6^2=1.
                    # Squares: 1, 4, 2. Non-squares: 3, 5, 6. Smallest is 3.
                    k = 3
                    
                    # t * vx
                    # t = (ta, tb), vx = (vxa, vxb)
                    # t*vx = (ta*vxa + tb*vxb*k, ta*vxb + tb*vxa)
                    
                    # This is getting slow to compute on the fly.
                    # Precompute multiplication tables?
                    pass
            
            # This approach is too slow to generate lines on the fly in the loop.
            # We should precompute the lines.
            pass
            
    # Let's precompute lines properly.
    # Field multiplication table
    mul_table = [[None]*q for _ in range(q)]
    for a in range(q):
        a_a = a % p
        a_b = a // p
        for b in range(q):
            b_a = b % p
            b_b = b // p
            # (a_a + a_b*alpha) * (b_a + b_b*alpha)
            # = a_a*b_a + (a_a*b_b + a_b*b_a)*alpha + a_b*b_b*alpha^2
            # = a_a*b_a + a_b*b_b*k + (a_a*b_b + a_b*b_a)*alpha
            res_a = (a_a*b_a + a_b*b_b*k) % p
            res_b = (a_a*b_b + a_b*b_a) % p
            mul_table[a][b] = res_a + res_b*p
            
    add_table = [[0]*q for _ in range(q)]
    for a in range(q):
        a_a = a % p
        a_b = a // p
        for b in range(q):
            b_a = b % p
            b_b = b // p
            res_a = (a_a + b_a) % p
            res_b = (a_b + b_b) % p
            add_table[a][b] = res_a + res_b*p
            
    # Precompute lines
    lines = []
    lines_set = set()
    lines_through = [set() for _ in range(q*q)]
    
    for vx_idx in range(q):
        for vy_idx in range(q):
            if vx_idx == 0 and vy_idx == 0: continue
            # Check if this direction is a representative?
            # We want q+1 directions.
            # We can just use all non-zero vectors, but then we generate each line multiple times.
            # To avoid this, we can restrict to specific representatives.
            # But it's easier to just generate all and deduplicate.
            # There are (q^2-1) directions. Each line has q points.
            # Total line-point incidences: (q^2-1) * q * q = q^3(q^2-1).
            # This is 7^3 * 48 = 343 * 48 ~ 16000. Very small.
            
            # Generate line through point 0 with this direction?
            # No, generate line through each point?
            # Let's just generate lines by picking a point and a direction, and taking the "canonical" line.
            # A line is canonical if it contains the smallest index point among its members.
            
            pass
            
    # Efficient Line Generation:
    # For each direction v (represented by a non-zero vector), the lines are the cosets of the subspace spanned by v.
    # We can find the q lines by taking q points that are not in the same line.
    # Actually, we can just iterate through all points, and for each point x, and each direction v,
    # generate the line L = {x + t*v}.
    # Then add L to a set of lines (frozenset of indices).
    # Then for each point in L, add line_id to lines_through[point].
    
    all_lines = []
    seen_lines = set()
    
    # Directions: all non-zero vectors
    # We can just use the q+1 specific directions from the problem statement to be safe and fast.
    # v = (1, m) and v = (0, 1).
    # m is an element of F_q.
    
    for m_idx in range(q):
        # v = (1, m)
        vx = 1 # (1, 0) -> index 1? No, (1,0) is a + b*alpha = 1 + 0*alpha -> a=1, b=0 -> idx 1.
        # Wait, idx = a + b*p.
        # (1, 0) -> 1 + 0*7 = 1.
        # (0, 1) -> 0 + 1*7 = 7.
        # v = (1, m). m is an index.
        # We need to add points.
        
        # To get unique lines for this direction, we can just pick one line, then shift by basis vectors?
        # Simpler: Iterate through all points x. Generate line L_x. Add to set.
        # This will generate the same line q times for each direction.
        # Total iterations: (q+1) directions * q^2 points = 8 * 49 * 49 = 19208.
        # This is fast enough.
        
        # Direction v = (1, m)
        v_x = 1 # (1,0)
        v_y = m_idx
        for x_idx in range(q*q):
            line_pts = []
            for t_idx in range(q):
                # x + t*v
                # x = (xa, xb) -> x_idx
                # t = t_idx
                # v = (v_x, v_y)
                # t*v = (t*v_x, t*v_y)
                tv_x = mul_table[t_idx][v_x]
                tv_y = mul_table[t_idx][v_y]
                
                # x + tv
                # x_idx = xa + xb*p
                # tv_x = tva + tvb*p
                # sum = (xa+tva) + (xb+tvb)*p
                # We need to add indices in F_q.
                # add_table[x_idx][tv_x] gives the x-coordinate sum.
                # add_table[x_idx // p? No, x_idx is the index of the point in F_q^2?
                # No, x_idx is the index of the point in F_q^2.
                # x = (x1, x2) where x1, x2 in F_q.
                # x1_idx = x_idx % q? No.
                # My point representation: [[xa, xb], [ya, yb]].
                # x = (xa + xb*alpha, ya + yb*alpha).
                # x1 = xa + xb*alpha. Index = xa + xb*p.
                # x2 = ya + yb*alpha. Index = ya + yb*p.
                # Point index = (x1_idx * q + x2_idx).
                
                x1_idx = x_idx % q
                x2_idx = x_idx // q
                
                new_x1 = add_table[x1_idx][tv_x]
                new_x2 = add_table[x2_idx][tv_y]
                new_idx = new_x1 * q + new_x2
                line_pts.append(new_idx)
                
            line_tuple = tuple(sorted(line_pts))
            if line_tuple not in seen_lines:
                seen_lines.add(line_tuple)
                line_id = len(all_lines)
                all_lines.append(line_tuple)
                for pt in line_pts:
                    lines_through[pt].add(line_id)
                    
    # Direction v = (0, 1)
    v_x = 0
    v_y = 7 # (0,1) -> 0 + 1*7 = 7
    for x_idx in range(q*q):
        line_pts = []
        for t_idx in range(q):
            tv_x = mul_table[t_idx][v_x] # 0
            tv_y = mul_table[t_idx][v_y]
            
            x1_idx = x_idx % q
            x2_idx = x_idx // q
            
            new_x1 = add_table[x1_idx][tv_x]
            new_x2 = add_table[x2_idx][tv_y]
            new_idx = new_x1 * q + new_x2
            line_pts.append(new_idx)
            
        line_tuple = tuple(sorted(line_pts))
        if line_tuple not in seen_lines:
            seen_lines.add(line_tuple)
            line_id = len(all_lines)
            all_lines.append(line_tuple)
            for pt in line_pts:
                lines_through[pt].add(line_id)

    # Greedy Algorithm
    start_time = time.time()
    deadline = start_time + 600 # 10 minutes
    
    best_S = set()
    
    def greedy_randomize():
        S = set()
        # Order of points
        order = list(range(q*q))
        random.shuffle(order)
        
        # clean_lines[y] = set of lines through y that are disjoint from S \ {y}
        # Initially S is empty, so all lines are clean.
        clean_lines = [set(lines_through[y]) for y in range(q*q)]
        
        for x in order:
            # Check if x can be added
            # 1. x must have a clean line
            if not clean_lines[x]:
                continue
            
            # 2. Adding x must not make any y in S invalid
            # y becomes invalid if all its clean lines contain x.
            valid = True
            for y in S:
                # Check if clean_lines[y] is a subset of lines containing x
                # Lines containing x: lines_through[x]
                # If clean_lines[y] is subset of lines_through[x], then all clean lines for y pass through x.
                # Since x is not in S yet, clean_lines[y] are disjoint from S.
                # If we add x, these lines will intersect S at x.
                # So they will no longer be clean for y (unless y=x, but y in S, x not in S).
                # So y loses these clean lines.
                # If y loses ALL its clean lines, it becomes invalid.
                if clean_lines[y].issubset(lines_through[x]):
                    valid = False
                    break
            
            if valid:
                # Add x
                S.add(x)
                # Update clean_lines for all y in S
                # For any y in S, if a line L in clean_lines[y] contains x, remove L from clean_lines[y]
                # Because L will now intersect S at x (and y), so L is no longer clean for y.
                for y in S:
                    if y == x: continue
                    # Remove lines from clean_lines[y] that contain x
                    # These are lines in lines_through[x]
                    # Intersect with clean_lines[y]
                    to_remove = clean_lines[y].intersection(lines_through[x])
                    clean_lines[y] -= to_remove
                
                # For x itself, clean_lines[x] should be updated?
                # x is in S. Clean lines for x are lines L through x such that L \ {x} is disjoint from S.
                # Currently S contains x.
                # The lines in clean_lines[x] were disjoint from S_old.
                # Since x was not in S_old, they are disjoint from S_old.
                # Now x is in S.
                # A line L through x is clean for x if L \ {x} is disjoint from S.
                # L \ {x} is disjoint from S_old (since L was clean for x before adding x, meaning L \cap S_old = empty).
                # And x is the only point of S on L?
                # Wait, if L was clean for x before, L \cap S_old = empty.
                # After adding x, L \cap S = {x}.
                # So L is clean for x.
                # So clean_lines[x] remains valid?
                # But we need to remove lines that contain other points of S?
                # No, clean_lines[x] was computed based on S_old.
                # The condition for x to be clean is L \cap S = {x}.
                # We know L \cap S_old = empty.
                # So L \cap S = {x}.
                # So clean_lines[x] is correct.
                
                # However, we must ensure that for any other point z in S, we have updated its clean_lines.
                # We did that in the loop above.
                
    # Run multiple restarts
    num_restarts = 0
    while time.time() < deadline - 10:
        greedy_randomize()
        if len(S) > len(best_S):
            best_S = S.copy()
        num_restarts += 1
        # Maybe stop if we find a very large set?
        # Max possible size?
        # If we find 1000+, we can be confident.
        
    # Convert best_S to points
    points_in_S = [idx_to_point(i) for i in best_S]
    
    # The Nikodym set N is the complement of S
    # N = { p in F_q^2 | p not in S }
    # We need to return N.
    # The task asks for the points of N.
    # "listing the points of a set N"
    # "Metric: complement size 2401 - |N|"
    # So we return the array of points in N.
    
    S_indices = set(best_S)
    N_points = []
    for i in range(q*q):
        if i not in S_indices:
            N_points.append(idx_to_point(i))
            
    return np.array(N_points, dtype=np.int64)

