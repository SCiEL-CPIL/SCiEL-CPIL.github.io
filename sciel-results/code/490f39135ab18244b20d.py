# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    
    Strategy:
    1. Start with S = F_49^2 (N = empty). This is a valid Nikodym set?
       No. If N is empty, for any x, we need a line L through x such that L\{x} is subset of N=empty.
       So L\{x} must be empty. But L has 49 points, so L\{x} has 48 points.
       So N=empty is NOT a Nikodym set.
       
       Wait, the definition: "every point x of F_49^2 (in N or not) must lie on a line {x + t*v : t in F_49}, with v = (1, m) for some m in F_49 or v = (0, 1), whose points other than x all belong to N."
       
       So if N is empty, for any x, we need a line L through x such that L \ {x} is subset of empty set.
       This implies L \ {x} is empty, which is impossible for a line of size 49.
       So N must be "large" enough.
       
       Actually, let's re-read carefully.
       "whose points other than x all belong to N"
       So L \ {x} \subseteq N.
       
       If we start with S = empty, N = F_49^2.
       For any x, pick any line L through x. L \ {x} is subset of F_49^2 = N.
       So S = empty is a valid complement. Score = 0.
       
       We want to maximize |S|, i.e., minimize |N|.
       
       Let's use the greedy removal strategy from the current program, but maybe optimize it or use the Hermitian unital idea from the inspiration?
       
       The inspiration (score 1.5374) uses a Hermitian unital U.
       It starts with N = U (or S = F^2 \ U?).
       Let's look at the inspiration code more closely.
       It identifies points on the Hermitian Unital U: Tr(x) + Norm(y) = 0.
       Size of U: q^2 + 1 = 49^2 + 1 = 2402? No, in F_q^2, a Hermitian curve has q^2 + 1 points?
       Wait, Hermitian curve in PG(2, q) has q^2+1 points. In affine plane, it has q^2 points?
       Equation: Tr(x) + Norm(y) = 0.
       Tr(x) = x + x^q = x + x_bar. In F_{49}, x_bar = x^7.
       Norm(y) = y * y^q = y * y^7.
       
       The inspiration code calculates:
       trace(x) = 2 * x[0] (since x = a + b*alpha, x^7 = a - b*alpha, so x+x^7 = 2a).
       norm(y) = y[0]^2 - 3*y[1]^2.
       Condition: 2*x[0] + y[0]^2 - 3*y[1]^2 == 0 mod 7.
       
       Let's check the size of this set U.
       For each x (49 choices), y must satisfy y[0]^2 - 3*y[1]^2 = -2*x[0].
       This is a conic in y-space. For a non-zero RHS, a conic in F_7^2 has 7+1=8 points? Or 7-1=6?
       Actually, number of solutions to X^2 - D Y^2 = K in F_q.
       If D is a square, it's equivalent to (X-Y)(X+Y)=K.
       Here D=3. Is 3 a square in F_7?
       Squares in F_7: 0, 1, 4, 2 (3^2=2), 4 (4^2=2? No 16=2), 2 (5^2=25=4), 4 (6^2=36=1).
       1^2=1, 2^2=4, 3^2=2, 4^2=2, 5^2=4, 6^2=1.
       So 3 is NOT a square in F_7.
       So the conic X^2 - 3Y^2 = K is anisotropic if K != 0?
       Number of points on a non-degenerate conic in F_q^2 is q+1 if q is odd?
       Wait, in affine plane, a conic has q+1 points if it's an oval?
       Actually, the number of solutions to X^2 - D Y^2 = K with D non-square:
       If K=0, only (0,0).
       If K != 0, there are no solutions? Or 2q?
       Let's check. X^2 = 3Y^2 + K.
       Since 3 is non-square, 3Y^2 is non-square (if Y!=0).
       So we are looking for squares of the form 3Y^2 + K.
       
       The inspiration code found `unital_points`.
       Let's assume the inspiration's construction of U is correct and leads to a good starting point.
       The inspiration then performs a greedy set cover to remove points from U (adding them to S? No, S = U \ R).
       Wait, the inspiration says: "S = U \ R".
       So S is a subset of U.
       And N = F^2 \ S.
       The inspiration's score is 1.5374, which is better than the current 1.1565.
       So the Hermitian Unital approach is promising.
       
       Let's combine:
       1. Compute U (Hermitian unital).
       2. Start with S = U.
       3. Check if N = F^2 \ S is a valid Nikodym set.
          If not, we need to adjust S.
          The inspiration uses a "greedy set cover" to find R (points to remove from U) such that the condition holds?
          Actually, the inspiration's comment says:
          "We want to find R subset of U such that for all x in (F^2 \ U) U R, there is a line through x disjoint from S."
          This implies that for x in N, we need a line L through x with L \ {x} subset N.
          Since N = F^2 \ S, L \ {x} subset N means L \ {x} is disjoint from S.
          So L \cap S = {x} if x in S, or L \cap S = empty if x not in S.
          Wait, if x is in N, then x not in S. So L \cap S = empty.
          So for every x in N, there is a line through x disjoint from S.
          
          If we start with S = U, then for x in F^2 \ U (which is in N), we need a line through x disjoint from U.
          Is that true?
          The tangent line at u in U is the only line through u that intersects U only at u.
          But for x not in U, we need a line disjoint from U.
          In PG(2, q), every point outside a conic lies on 2 tangents.
          In the affine plane, does every point outside U lie on a line disjoint from U?
          A line disjoint from U is a line that doesn't intersect U.
          In PG(2, q), a line either intersects the conic in 2 points (secant), 1 point (tangent), or 0 points (external).
          So external lines exist.
          Does every point outside the conic lie on an external line?
          Yes, in PG(2, q), every point outside a conic lies on exactly 2 tangents and (q+1)/2 external lines?
          Actually, the number of external lines through a point outside a conic is (q+1)/2 if q is odd?
          So yes, there are lines disjoint from U passing through any point outside U.
          
          So if S = U, then for any x not in U, there is a line L through x disjoint from U, so L \cap S = empty.
          Condition for x in N (x not in S): satisfied.
          
          What about x in S (x in U)?
          We need a line L through x such that L \ {x} is subset of N.
          i.e., L \cap S = {x}.
          This is exactly the tangent line at x!
          So the tangent line at x intersects U only at x.
          So L \cap S = {x}.
          Condition for x in S: satisfied.
          
          So S = U is a valid complement!
          |S| = |U|.
          Score = 2401 - |U|.
          
          How large is U?
          The equation is 2*x_a + y_a^2 - 3*y_b^2 = 0.
          For each x (49 choices), we need to find y.
          Let K = -2*x_a.
          We need y_a^2 - 3*y_b^2 = K.
          If K=0 (x_a=0), only solution is y=(0,0) since 3 is non-square?
          Wait, y_a^2 = 3*y_b^2.
          If y_b != 0, (y_a/y_b)^2 = 3. No solution.
          So if x_a=0, only y=(0,0). 1 solution.
          If x_a != 0, K != 0.
          Number of solutions to X^2 - D Y^2 = K with D non-square, K != 0.
          In F_q, the number of solutions is 0?
          Let's check for F_7.
          X^2 - 3Y^2 = K.
          Squares: 0, 1, 2, 4.
          3Y^2: 0, 3, 6, 5.
          X^2 - 3Y^2:
          Y=0: X^2 = 0,1,2,4.
          Y=1 (3Y^2=3): X^2-3 = 0,1,2,4 => X^2 = 3,4,5,0. 3,5 not squares. 4,0 are.
             So X^2=4 (X=2,5), X^2=0 (X=0).
             Values: 4-3=1, 0-3=4.
          Let's list all pairs (Y_a, Y_b) and compute V = Y_a^2 - 3Y_b^2.
          Y_b=0: V = Y_a^2. Values: 0, 1, 4, 2.
             Y_a=0: V=0.
             Y_a=1: V=1.
             Y_a=2: V=4.
             Y_a=3: V=2.
             Y_a=4: V=2.
             Y_a=5: V=4.
             Y_a=6: V=1.
          Y_b=1 (3Y_b^2=3): V = Y_a^2 - 3.
             Y_a=0: -3=4.
             Y_a=1: 1-3=-2=5.
             Y_a=2: 4-3=1.
             Y_a=3: 2-3=-1=6.
             Y_a=4: 2-3=6.
             Y_a=5: 4-3=1.
             Y_a=6: 1-3=5.
          Y_b=2 (3Y_b^2=3*4=12=5): V = Y_a^2 - 5.
             Y_a=0: -5=2.
             Y_a=1: 1-5=-4=3.
             Y_a=2: 4-5=-1=6.
             Y_a=3: 2-5=-3=4.
             Y_a=4: 2-5=4.
             Y_a=5: 4-5=6.
             Y_a=6: 1-5=3.
          
          Let's count frequencies of V.
          V=0: (0,0) -> 1
          V=1: (1,0), (6,0), (2,1), (5,1) -> 4
          V=2: (3,0), (4,0), (0,2) -> 3
          V=3: (1,2), (6,2) -> 2
          V=4: (2,0), (5,0), (0,1), (3,2), (4,2) -> 5
          V=5: (1,1), (6,1) -> 2
          V=6: (3,1), (4,1), (2,2), (5,2) -> 4
          
          Sum: 1+4+3+2+5+2+4 = 21? No, 7*7=49 points.
          Let's re-count.
          Y_b=0: 0,1,4,2,2,4,1. (0:1, 1:2, 2:2, 4:2)
          Y_b=1: 4,5,1,6,6,1,5. (1:2, 4:1, 5:2, 6:2)
          Y_b=2: 2,3,6,4,4,6,3. (2:1, 3:2, 4:2, 6:2)
          
          Total counts:
          0: 1
          1: 2+2=4
          2: 2+1=3
          3: 2
          4: 2+1+2=5
          5: 2
          6: 2+2=4
          Sum: 1+4+3+2+5+2+4 = 21.
          Missing 28?
          Ah, Y_b can be 0..6. I only did 0,1,2.
          Y_b=3 (3Y_b^2 = 3*2 = 6): V = Y_a^2 - 6.
             Y_a=0: 1.
             Y_a=1: 2.
             Y_a=2: 5.
             Y_a=3: 3.
             Y_a=4: 3.
             Y_a=5: 5.
             Y_a=6: 2.
             Counts: 1:1, 2:2, 3:2, 5:2.
          Y_b=4 (3Y_b^2 = 3*2 = 6): Same as Y_b=3.
             Counts: 1:1, 2:2, 3:2, 5:2.
          Y_b=5 (3Y_b^2 = 3*4 = 5): Same as Y_b=2.
             Counts: 2:1, 3:2, 4:2, 6:2.
          Y_b=6 (3Y_b^2 = 3*1 = 3): Same as Y_b=1.
             Counts: 1:2, 4:1, 5:2, 6:2.
          
          Total counts:
          0: 1
          1: 4 (from Yb=0) + 1 (Yb=3) + 1 (Yb=4) + 2 (Yb=6) = 8?
             Wait, Yb=1 had 1:2. Yb=6 is same as Yb=1. So 2+2=4.
             Yb=3,4 have 1:1 each.
             Total 1: 4 (from 0,6) + 1 (3) + 1 (4) = 6?
             Let's sum properly.
             Yb=0: 0:1, 1:2, 2:2, 4:2
             Yb=1: 1:2, 4:1, 5:2, 6:2
             Yb=2: 2:1, 3:2, 4:2, 6:2
             Yb=3: 1:1, 2:2, 3:2, 5:2
             Yb=4: 1:1, 2:2, 3:2, 5:2
             Yb=5: 2:1, 3:2, 4:2, 6:2
             Yb=6: 1:2, 4:1, 5:2, 6:2
             
             0: 1
             1: 2+2+1+1+2 = 8
             2: 2+1+2+2+1 = 8
             3: 2+2+2 = 6
             4: 2+1+2+2+1 = 8
             5: 2+2+2 = 6
             6: 2+2+2 = 6
             
             Sum: 1+8+8+6+8+6+6 = 43.
             Still not 49.
             I must have missed something.
             1+8+8+6+8+6+6 = 43.
             49-43=6.
             Where are the missing 6?
             Let's check Y_b=0 again.
             Y_a=0..6.
             0:0, 1:1, 2:4, 3:2, 4:2, 5:4, 6:1.
             0:1, 1:2, 2:2, 4:2. Sum 7. Correct.
             
             Maybe I should just run the code to count.
             But the inspiration code worked and got a good score.
             So |U| is likely around 49*?
             If |U| is small, |S| is small, score is low.
             If |U| is large, |S| is large, score is high.
             Max possible |S| is 2401.
             If U is a curve, |U| is ~49.
             Then score is ~2352.
             The inspiration score 1.5374 corresponds to |S| = 1.5374 * seed.
             If seed is 1.0, |S| is 1.5374? No, score is 2401 - |N| = |S|.
             Wait, the metric is "complement size 2401 - |N|".
             So score = |S|.
             If score is 1.5374, that means |S| = 1.5374? That's impossible, |S| is an integer.
             Ah, the score is "value/seed".
             So value = |S|.
             If seed is 1.0, value = 1.5374? No, value must be integer.
             Maybe the seed score is not 1.0.
             Or maybe the score is normalized?
             "The evaluator measures the raw value (higher is better). The search score is value/seed".
             So if seed program has value V_seed, and this program has value V, score = V/V_seed.
             If score is 1.5374, then V = 1.5374 * V_seed.
             If V_seed is small, V can be large.
             
             Let's assume the inspiration's S is large.
             The inspiration starts with S = U and then removes points?
             "S = U \ R". So S is a subset of U.
             So |S| <= |U|.
             If |U| is small, |S| is small.
             This contradicts the high score if V_seed is small.
             
             Wait, maybe the inspiration's S is NOT a subset of U.
             Let's re-read the inspiration code snippet.
             "S = U \ R".
             "universe = [i for i in range(q*q) if i not in unital_indices]"
             "uncovered = set(universe)"
             "R = []"
             "Greedy Set Cover"
             "uncovered" is the set of points in F^2 \ U.
             The sets are "points in N covered by T_u".
             N = F^2 \ S.
             If S = U \ R, then N = (F^2 \ U) U R.
             The condition for x in N: exists line L through x disjoint from S.
             For x in F^2 \ U (which is in N), we need L disjoint from U \ R.
             If L is disjoint from U, it is disjoint from U \ R.
             So points in F^2 \ U are automatically covered by external lines.
             So why the set cover?
             Maybe the external lines don't exist for all points?
             Or maybe the inspiration is trying to ADD points to S?
             "S = U \ R" means R is removed from U.
             So S is smaller than U.
             This would DECREASE the score.
             
             Unless... the inspiration's "S" is actually N?
             "returning an integer array of shape (k, 2, 2), k >= 1, listing the points of a set N."
             So the function returns N.
             The metric is 2401 - |N| = |S|.
             So we want to maximize |S|, i.e., minimize |N|.
             
             If the inspiration returns N, and N is small, then S is large.
             If the inspiration starts with S = U, then N = F^2 \ U.
             |N| = 2401 - |U|.
             Score = |U|.
             If |U| is ~50, score is ~50.
             If seed score is ~30, then 50/30 = 1.66.
             This matches 1.5374.
             
             So the inspiration simply returns N = F^2 \ U.
             And U is the Hermitian unital.
             
             So the strategy is:
             1. Compute U (Hermitian unital).
             2. Return N = F^2 \ U.
             
             Is this valid?
             We established that S = U is a valid complement.
             So N = F^2 \ U is a valid Nikodym set.
             
             Can we do better?
             We want to maximize |S|.
             S = U is one such set.
             Can we add points to S?
             If we add a point x to S, we must ensure the condition still holds.
             Condition: For every y in F^2, exists line L through y such that L \ {y} subset N.
             If we add x to S (remove from N), we might break the condition for some y.
             Specifically, if x was the "witness" for some y, i.e., x was in L \ {y} for the line L witnessing y.
             If we remove x from N, then L \ {y} is no longer subset of N.
             So y might lose its witness.
             
             So we can try to greedily add points to S (starting from U) as long as the condition holds.
             This is the same as the current program's logic, but starting from S=U instead of S=empty.
             The current program starts from S=empty and adds points.
             It ends up with a small S.
             If we start from S=U (which is already valid), we can try to add more points.
             This should yield a larger S.
             
             So the plan:
             1. Compute U.
             2. Set S = U.
             3. Use the greedy addition algorithm from the current program to add more points to S.
             4. Return N = F^2 \ S.
             
             This combines the structural idea of the inspiration (Hermitian unital as a valid starting point) with the refinement machinery of the current program (greedy addition with validity check).

    """
    start_time = time.time()
    deadline = start_time + 550.0  # Safety margin

    p = 7
    q = p * p
    k = 3  # alpha^2 = 3

    # Field operations
    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)

    def mul(a, b):
        return ((a[0] * b[0] + a[1] * b[1] * k) % p, (a[0] * b[1] + a[1] * b[0]) % p)

    def trace(a):
        return (2 * a[0]) % p

    def norm(a):
        return (a[0] * a[0] - a[1] * a[1] * k) % p

    # Generate all points in F_49^2
    # Index: x = (xa, xb), y = (ya, yb)
    # idx = ((xa * p + xb) * p + ya) * p + yb
    
    num_points = q * q
    
    # Precompute lines
    lines = []
    point_to_lines = {i: [] for i in range(num_points)}
    
    seen_lines = set()
    
    # Directions
    # v = (vx, vy)
    # (1, m) for m in F_49
    # (0, 1)
    
    # For (1, m): y = m*x + c
    # For (0, 1): x = c
    
    # Iterate over all lines
    # For (1, m):
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            for ca in range(p):
                for cb in range(p):
                    c = (ca, cb)
                    line_points = []
                    for xa in range(p):
                        for xb in range(p):
                            x = (xa, xb)
                            mx = mul(m, x)
                            y = add(mx, c)
                            idx = ((xa * p + xb) * p + y[0]) * p + y[1]
                            line_points.append(idx)
                    
                    line_tuple = tuple(sorted(line_points))
                    if line_tuple not in seen_lines:
                        seen_lines.add(line_tuple)
                        line_set = set(line_points)
                        lines.append(line_set)
                        line_id = len(lines) - 1
                        for idx in line_points:
                            point_to_lines[idx].append(line_id)
    
    # For (0, 1): x = c
    for ca in range(p):
        for cb in range(p):
            c = (ca, cb)
            line_points = []
            for ya in range(p):
                for yb in range(p):
                    idx = ((ca * p + cb) * p + ya) * p + yb
                    line_points.append(idx)
                    
            line_tuple = tuple(sorted(line_points))
            if line_tuple not in seen_lines:
                seen_lines.add(line_tuple)
                line_set = set(line_points)
                lines.append(line_set)
                line_id = len(lines) - 1
                for idx in line_points:
                    point_to_lines[idx].append(line_id)

    # Identify points on the Hermitian Unital U
    # Tr(x) + Norm(y) = 0
    unital_indices = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    x = (xa, xb)
                    y = (ya, yb)
                    if (trace(x) + norm(y)) % p == 0:
                        idx = ((xa * p + xb) * p + ya) * p + yb
                        unital_indices.add(idx)
    
    # Initialize S with U
    S = set(unital_indices)
    
    # Initialize valid_count
    # For x in S: valid_count[x] = number of lines L through x such that L ∩ S = {x}
    # For x not in S: valid_count[x] = number of lines L through x such that L ∩ S = ∅
    
    valid_count = [0] * num_points
    
    for x in range(num_points):
        count = 0
        for line_id in point_to_lines[x]:
            L = lines[line_id]
            # Check if L ∩ S is {x} (if x in S) or ∅ (if x not in S)
            inter = []
            for y in L:
                if y in S:
                    inter.append(y)
                    if len(inter) > 1:
                        break
            
            if x in S:
                if inter == [x]:
                    count += 1
            else:
                if len(inter) == 0:
                    count += 1
        
        valid_count[x] = count
    
    # Candidates: points not in S with valid_count > 0
    candidates = set(range(num_points))
    candidates -= S
    
    while candidates:
        if time.time() > deadline:
            break
            
        # Pick a candidate
        # Heuristic: pick one with smallest valid_count
        best_x = None
        best_count = float('inf')
        for x in candidates:
            if valid_count[x] > 0 and valid_count[x] < best_count:
                best_count = valid_count[x]
                best_x = x
                
        if best_x is None:
            break
            
        x = best_x
        
        # Check if adding x is valid
        valid = True
        for line_id in point_to_lines[x]:
            L = lines[line_id]
            s_points = []
            for y in L:
                if y in S:
                    s_points.append(y)
                    if len(s_points) > 1:
                        break
            
            if len(s_points) == 0:
                # L ∩ S = ∅
                for y in L:
                    if y == x:
                        continue
                    if valid_count[y] == 1:
                        valid = False
                        break
                if not valid:
                    break
                    
            elif len(s_points) == 1:
                y0 = s_points[0]
                if valid_count[y0] == 1:
                    valid = False
                    break
            else:
                pass
        
        if not valid:
            candidates.discard(x)
            continue
            
        # Commit: add x to S
        S.add(x)
        candidates.discard(x)
        
        # Update valid_count
        for line_id in point_to_lines[x]:
            L = lines[line_id]
            s_old_points = [y for y in L if y in S and y != x]
            
            if len(s_old_points) == 0:
                for y in L:
                    if y != x:
                        valid_count[y] -= 1
            elif len(s_old_points) == 1:
                y0 = s_old_points[0]
                valid_count[y0] -= 1
            else:
                pass
            
        # Recompute valid_count[x]
        count_x = 0
        for line_id in point_to_lines[x]:
            L = lines[line_id]
            has_other = False
            for y in L:
                if y != x and y in S:
                    has_other = True
                    break
            if not has_other:
                count_x += 1
        valid_count[x] = count_x

    # Construct N = F^2 \ S
    N = [i for i in range(num_points) if i not in S]
    
    # Convert indices to points
    points = []
    for idx in N:
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p * p)) % p
        xa = (idx // (p * p * p)) % p
        points.append([[xa, xb], [ya, yb]])
        
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
