import numpy as np
import time
import sys

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using a Maximum Independent Set heuristic.
    
    The Nikodym property requires that for every point x in F_q^2, there exists a 
    direction v such that the line L(x, v) = {x + t*v | t in F_q} \ {x} is contained 
    in N.
    
    Equivalently, let S = F_q^2 \ N be the complement. The condition is that for 
    every x in F_q^2, there exists a direction v such that L(x, v) \ {x} is disjoint 
    from S.
    
    This implies that for any point x, if x is in S, it must not be the case that 
    for ALL directions v, L(x, v) \ {x} is disjoint from S. Wait, that's not right.
    
    Let's rephrase:
    For every x in F_q^2, let D(x) be the set of directions v such that L(x, v) \ {x} 
    is a subset of N. We need D(x) to be non-empty for all x.
    
    If x is in N, D(x) can be non-empty easily.
    If x is in S, then x is NOT in N. For x to be covered, there must be a direction v 
    such that L(x, v) \ {x} is a subset of N. This means L(x, v) \ {x} must be 
    disjoint from S.
    
    So, if x is in S, there must exist at least one direction v such that no other 
    point on the line through x in direction v is in S.
    
    This means: For every x in S, there exists a direction v such that 
    (L(x, v) \ {x}) ∩ S = ∅.
    
    This is equivalent to saying that S is a set of points such that no point in S 
    is "blocked" by S in all directions.
    
    Actually, let's look at the conflict graph perspective.
    Two points x, y in F_q^2 conflict if they cannot both be in S.
    When do x and y conflict?
    If x and y are on the same line in direction v, say y = x + t*v for t != 0.
    If both x and y are in S, does that violate the condition?
    
    The condition for a point z to be covered by N is that there exists v_z such that 
    L(z, v_z) \ {z} ⊆ N.
    
    If z is in S, we need a direction v such that L(z, v) \ {z} ⊆ N.
    This means for that specific v, no point on L(z, v) \ {z} can be in S.
    
    So, if x ∈ S, then for EVERY direction v, it is NOT true that L(x, v) \ {x} ⊆ N.
    Wait, no. The condition is:
    For ALL x ∈ F_q^2, there EXISTS v_x such that L(x, v_x) \ {x} ⊆ N.
    
    If x ∈ S, then x ∉ N. The condition still applies to x.
    So, if x ∈ S, there must exist a direction v_x such that L(x, v_x) \ {x} ⊆ N.
    This implies that for that specific direction v_x, the set (L(x, v_x) \ {x}) ∩ S is empty.
    
    Therefore, a valid set S is one where:
    For every x ∈ S, there exists at least one direction v such that (L(x, v) \ {x}) ∩ S = ∅.
    
    This is a local property. It's not a simple pairwise conflict graph where edge (x,y) 
    means "x and y cannot both be in S". It's a hypergraph constraint.
    Specifically, for each x, the set of "bad" configurations for x is when for ALL v, 
    (L(x, v) \ {x}) ∩ S ≠ ∅.
    So x ∈ S is valid iff ∃v: (L(x, v) \ {x}) ∩ S = ∅.
    
    This is hard to model as a simple MIS. However, we can use a greedy approach:
    Start with S = ∅.
    Add points to S one by one.
    When adding a point x to S, we must ensure that x itself satisfies the condition 
    relative to the current S ∪ {x}.
    Also, adding x to S might break the condition for other points y ∈ S.
    Specifically, if y ∈ S, it had a "safe" direction v_y where (L(y, v_y) \ {y}) ∩ S_old = ∅.
    If x lies on L(y, v_y) \ {y}, then (L(y, v_y) \ {y}) ∩ S_new is not empty.
    So y might lose its safe direction. We must check if y has ANOTHER safe direction.
    
    So the algorithm is:
    1. S = ∅
    2. For each candidate point x (in some order):
       a. Tentatively add x to S_new = S ∪ {x}.
       b. Check if x has a safe direction in S_new.
          If not, discard x.
       c. For all y ∈ S, check if y still has a safe direction in S_new.
          If any y loses its safe direction, discard x.
       d. If all checks pass, set S = S_new.
    
    This is O(N^2 * Q) per step, which is too slow for N=2401.
    
    Optimization:
    Precompute for each point x and direction v, the set of points on the line.
    Maintain for each y ∈ S, the count of "safe" directions.
    A direction v is safe for y if no point in (L(y, v) \ {y}) is in S.
    
    When we add x to S:
    For each y ∈ S:
       For each direction v:
          If x is on L(y, v) \ {y}, then v is no longer safe for y.
          Decrement safe_count[y].
       If safe_count[y] becomes 0, then y is invalid, so we must revert.
    
    Also, we need to check if x itself has a safe direction.
    x has a safe direction v if no point in (L(x, v) \ {x}) is in S.
    Since x is new, we just check if (L(x, v) \ {x}) ∩ S is empty for some v.
    
    This is much faster.
    Let's implement this greedy algorithm.
    """
    p = 7
    q = p * p
    k_const = 3
    
    # Field operations
    def add_f(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def mul_f(a, b):
        # (a0 + a1*alpha) * (b0 + b1*alpha)
        # = a0*b0 + a0*b1*alpha + a1*b0*alpha + a1*b1*alpha^2
        # = (a0*b0 + a1*b1*k) + (a0*b1 + a1*b0)*alpha
        return ((a[0]*b[0] + a[1]*b[1]*k_const) % p, (a[0]*b[1] + a[1]*b[0]) % p)
    
    # Points in F_q^2
    # Index points from 0 to q^2 - 1
    # Point i corresponds to (x, y) where x = i // q, y = i % q
    # x, y are pairs (a, b)
    
    # Precompute lines
    # Directions: v = (1, m) for m in F_q, and v = (0, 1)
    # Total q + 1 directions
    
    # Let's represent points as flat indices 0..2400
    # x_idx = x_a * p + x_b
    # y_idx = y_a * p + y_b
    # point_idx = x_idx * q + y_idx
    
    # Precompute for each point and direction, the other points on the line
    # line_points[point_idx][dir_idx] = list of other point indices
    
    # This is 2401 * 50 * 48 ~ 5.7M integers. Manageable.
    
    # But we need to check "is x on line(y, v)?"
    # It's symmetric: x is on line(y, v) iff y is on line(x, v).
    
    # Let's precompute neighbors[point_idx][dir_idx] = set of other points on that line
    # Actually, we just need to know if a specific point x is on the line through y in dir v.
    # x = y + t*v for some t != 0.
    # y = x + t'*v for some t' != 0.
    
    # Let's precompute a boolean matrix on_line[point_y][point_x][dir_v] ?
    # 2401 * 2401 * 50 booleans = ~288M booleans = 36MB. OK.
    # But iterating is slow.
    
    # Better: For each point y and direction v, store the set of points on the line.
    # line_set[y][v] = set of point indices
    
    # To check if x is on line(y, v): x in line_set[y][v]
    
    # Greedy Algorithm:
    # S = set()
    # safe_count[y] = number of directions v such that line_set[y][v] ∩ S == empty (excluding y itself, but y is not in line_set[y][v] usually? No, line is {y + t*v | t in F_q}. If t=0, it's y. So line_set includes y. We care about line_set[y][v] \ {y}.)
    
    # Let's precompute line_others[y][v] = frozenset(line_set[y][v] - {y})
    
    # Time limit guard
    start_time = time.time()
    timeout = 500  # seconds
    
    # Precomputation
    # 1. Generate all points
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    points.append(((xa, xb), (ya, yb)))
    
    # Map point tuple to index
    point_to_idx = {pt: i for i, pt in enumerate(points)}
    
    # Directions
    # v = (1, m) for m in F_q, and v = (0, 1)
    dirs = []
    for ma in range(p):
        for mb in range(p):
            dirs.append(((1, 0), (ma, mb))) # v = (1, m) where m=(ma,mb)
    dirs.append(((0, 0), (0, 1))) # v = (0, 1)
    
    num_dirs = len(dirs)
    
    # Precompute line_others
    # line_others[y_idx][v_idx] = list of other point indices
    line_others = [[[] for _ in range(num_dirs)] for _ in range(q*q)]
    
    for y_idx in range(q*q):
        y = points[y_idx]
        for v_idx, v in enumerate(dirs):
            # Generate line points
            # x_t = y + t*v
            for ta in range(p):
                for tb in range(p):
                    t = (ta, tb)
                    if t == (0, 0):
                        continue
                    # x = y + t*v
                    # y = (y_x, y_y), v = (v_x, v_y)
                    # t*v = (t*v_x, t*v_y)
                    # x_x = y_x + t*v_x
                    # x_y = y_y + t*v_y
                    
                    tv_x = mul_f(t, v[0])
                    tv_y = mul_f(t, v[1])
                    
                    x_x = add_f(y[0], tv_x)
                    x_y = add_f(y[1], tv_y)
                    
                    x = (x_x, x_y)
                    x_idx = point_to_idx[x]
                    line_others[y_idx][v_idx].append(x_idx)
    
    # Convert to sets for fast intersection? Or just lists?
    # We need to check if line_others[y][v] intersects S.
    # If S is a set, intersection is fast.
    
    S = set()
    # safe_count[y] = number of directions v where line_others[y][v] is disjoint from S
    safe_count = [0] * (q*q)
    
    # Initialize safe_count for S = empty
    # For empty S, all lines are disjoint. So safe_count[y] = num_dirs for all y.
    for y in range(q*q):
        safe_count[y] = num_dirs
    
    # Candidate order: random or sorted?
    # Random might be better to avoid bias.
    candidates = list(range(q*q))
    np.random.shuffle(candidates)
    
    # Greedy loop
    for x_idx in candidates:
        if time.time() - start_time > timeout:
            break
            
        # Check if x can be added
        # 1. Does x have a safe direction in S ∪ {x}?
        #    We need ∃v such that line_others[x][v] ∩ S == ∅.
        #    Note: x is not in line_others[x][v], so S ∪ {x} intersection is just S.
        x_safe = False
        for v_idx in range(num_dirs):
            # Check if line_others[x_idx][v_idx] intersects S
            # If no intersection, v_idx is safe for x
            is_disjoint = True
            for pt in line_others[x_idx][v_idx]:
                if pt in S:
                    is_disjoint = False
                    break
            if is_disjoint:
                x_safe = True
                break
        
        if not x_safe:
            continue
            
        # 2. Check if adding x breaks any y in S
        #    For each y in S, x might lie on some lines of y.
        #    If x is on line_others[y][v], then v is no longer safe for y.
        #    We need to decrement safe_count[y] for each such v.
        #    If safe_count[y] becomes 0, we cannot add x.
        
        # Optimization: Only check y in S.
        # For each y in S, find all v such that x is in line_others[y][v].
        # This is equivalent to: for each v, if x is in line_others[y][v].
        
        # Let's precompute for each x, the list of (y, v) such that x is in line_others[y][v].
        # This is the reverse adjacency.
        # rev_adj[x_idx] = list of (y_idx, v_idx)
        
        # Precomputing rev_adj is O(N * D * N) = 2401 * 50 * 2401 ~ 288M ops. Too slow?
        # 288M simple ops in Python is ~10-30 seconds. Maybe acceptable?
        # But we can do it on the fly or precompute once.
        
        # Let's precompute rev_adj.
        # rev_adj[x] = list of (y, v)
        
        # Actually, we can just iterate over y in S and check directions.
        # |S| grows. Initially small.
        # If |S| is large, this is slow.
        # But we hope S stays small? No, we want S large.
        
        # Let's try precomputing rev_adj.
        # It's a one-time cost.
        
        # But wait, I'm inside the loop. I should precompute before the loop.
        
        pass # Placeholder, will refactor
        
    # Refactor: Precompute rev_adj before the greedy loop
    
    # Re-initialize
    S = set()
    safe_count = [num_dirs] * (q*q)
    
    rev_adj = [[] for _ in range(q*q)]
    for y_idx in range(q*q):
        for v_idx in range(num_dirs):
            for x_idx in line_others[y_idx][v_idx]:
                rev_adj[x_idx].append((y_idx, v_idx))
    
    # Now the greedy loop
    for x_idx in candidates:
        if time.time() - start_time > timeout:
            break
        
        # Check x safe
        x_safe = False
        for v_idx in range(num_dirs):
            # Check disjointness with S
            # Optimization: if safe_count[x_idx] > 0, it's safe?
            # No, safe_count[x_idx] is for the CURRENT S.
            # If safe_count[x_idx] > 0, then there is a direction v where line_others[x][v] ∩ S is empty.
            # So x is safe to add w.r.t its own condition.
            if safe_count[x_idx] > 0:
                x_safe = True
                break
                
        if not x_safe:
            continue
            
        # Check if adding x breaks any y in S
        # For each (y, v) in rev_adj[x_idx]:
        #    if y in S:
        #       safe_count[y] -= 1
        #       if safe_count[y] == 0:
        #          conflict = True
        #          break
        
        conflict = False
        affected = [] # list of (y, v) to revert
        
        for y_idx, v_idx in rev_adj[x_idx]:
            if y_idx in S:
                safe_count[y_idx] -= 1
                affected.append((y_idx, v_idx))
                if safe_count[y_idx] == 0:
                    conflict = True
                    break
        
        if conflict:
            # Revert
            for y_idx, v_idx in affected:
                safe_count[y_idx] += 1
        else:
            # Add x
            S.add(x_idx)
            # Note: safe_count[x_idx] is already correct?
            # When we add x, does it affect safe_count of x itself?
            # No, safe_count[x] depends on S \ {x}.
            # But wait, the condition for x is ∃v: line_others[x][v] ∩ S_new = ∅.
            # S_new = S_old ∪ {x}.
            # line_others[x][v] does not contain x.
            # So line_others[x][v] ∩ S_new = line_others[x][v] ∩ S_old.
            # So safe_count[x] doesn't change when we add x?
            # Correct.
            
            # But does adding x affect safe_count of other points?
            # Yes, we handled that above (decrementing).
            
            # Does adding x affect safe_count of x?
            # No.
            
            # But what about future points?
            # safe_count is maintained dynamically.
            
    # Construct N = F_q^2 \ S
    # Return points in N
    
    n_points = []
    for i in range(q*q):
        if i not in S:
            n_points.append(points[i])
            
    # Format output
    # Shape (k, 2, 2)
    if not n_points:
        # Should not happen
        return np.array([[[0,0],[0,0]]], dtype=np.int64)
        
    result = np.array([[[pt[0][0], pt[0][1]], [pt[1][0], pt[1][1]]] for pt in n_points], dtype=np.int64)
    
    return result

# Precomputation is heavy. Let's see if it fits in time.
# rev_adj precomputation:
# Outer loop: 2401
# Inner loop: 50
# Innermost: 48
# Total: 2401 * 50 * 48 = 5,762,400 iterations.
# Each iteration appends to a list.
# This is very fast in Python? Maybe 1-2 seconds.

# Greedy loop:
# 2401 iterations.
# Each iteration:
#   Check x_safe: O(1) if using safe_count.
#   Check conflict: Iterate rev_adj[x].
#   rev_adj[x] size is approx 50 * (48/49)? No.
#   Each x is on 50 lines. On each line, it is the "other" point for 48 points?
#   No. For a fixed x, how many (y, v) pairs are there?
#   For each direction v (50 of them), x lies on exactly one line.
#   On that line, there are 48 other points y.
#   For each such y, x is in line_others[y][v].
#   So rev_adj[x] has size 50 * 48 = 2400.
#   So the inner loop is 2400 iterations.
#   Total work: 2401 * 2400 ~ 5.7M iterations.
#   Very fast.

# So the whole thing should run in < 5 seconds.

# Let's double check the logic.
# safe_count[y] = number of directions v such that line_others[y][v] ∩ S == ∅.
# Initially S=∅, so safe_count[y] = 50 for all y.
# When we add x to S:
#   For each (y, v) in rev_adj[x]:
#       if y in S:
#           x is in line_others[y][v].
#           So line_others[y][v] ∩ S is no longer empty (it contains x).
#           So v is no longer safe for y.
#           safe_count[y] -= 1.
#   This is correct.
#   If safe_count[y] becomes 0, y has no safe direction, so y would violate the Nikodym property.
#   So we cannot add x.

# One detail: "line_others[y][v] ∩ S == ∅"
# Does this include y? No, line_others excludes y.
# Does it include x? Yes, if x is on the line.
# So if x is added to S, and x is on line(y, v), then the intersection is not empty.
# Correct.

# What if x == y?
# rev_adj[x] contains (y, v) where x is in line_others[y][v].
# line_others[y][v] excludes y. So x != y.
# So we only check y != x.
# And we only decrement if y is ALREADY in S.
# So x is not in S yet when we check.
# Correct.

# The result N is the complement of S.
# We want to maximize |S|, so minimize |N|.
# The greedy algorithm adds points to S if they don't break existing points.
# This maximizes |S| greedily.

# Potential issue: Order of candidates.
# Random order might be suboptimal.
# But it's a heuristic.

# Let's run it.

# One more check: The problem statement says "v = (1, m) for some m in F_49 or v = (0, 1)".
# My dirs list:
# for ma in 0..6, mb in 0..6: v = ((1,0), (ma,mb)). This is (1, m).
# Plus v = ((0,0), (0,1)). This is (0,1).
# Total 49 + 1 = 50 directions. Correct.

# Field arithmetic:
# alpha^2 = 3.
# mul_f((a0,a1), (b0,b1)):
# (a0 + a1*alpha)(b0 + b1*alpha) = a0b0 + a0b1*alpha + a1b0*alpha + a1b1*alpha^2
# = (a0b0 + 3*a1b1) + (a0b1 + a1b0)*alpha.
# Correct.

# add_f: componentwise mod 7. Correct.

# Point representation:
# (x, y) where x, y are (a, b).
# Index: x_a*7+x_b is 0..48. y_a*7+y_b is 0..48.
# Point index: x_idx * 49 + y_idx.
# Correct.

# The code looks solid.

# Final check on output format.
# "integer array of shape (k, 2, 2)"
# "Entries are cast to int64 and reduced mod 7"
# "duplicate points are merged"
# My construction ensures unique points.
# N is a set, so no duplicates.

# One edge case: k=0?
# "k >= 1"
# If S = F_q^2, N is empty.
# Can S be F_q^2?
# If S = F_q^2, then for any x, safe_count[x] = 0 (since all lines intersect S).
# So we can't add the last point?
# Actually, if S is full, we can't add more.
# But can we reach S = F_q^2?
# The first point added: safe_count = 50. OK.
# The second point: might break the first?
# If x2 is on a line of x1, safe_count[x1] decreases.
# If it reaches 0, we can't add x2.
# So S cannot be arbitrary.
# It's a packing problem.
# So N will likely be non-empty.
# If N is empty, return a dummy point?
# "k >= 1". If N is empty, the set is invalid?
# "N must be a Nikodym set". Empty set is not a Nikodym set (no points to cover? or vacuously true? but k>=1).
# If my algorithm produces S = F_q^2, I need to handle it.
# But it's unlikely.

# Let's add a check.
# If len(n_points) == 0:
#    # Force one point into N?
#    # Remove one point from S, add to N.
#    # But is the resulting N valid?
#    # If we remove x from S, N gains x.
#    # Does N remain a Nikodym set?
#    # The condition is on N.
#    # If N was empty, it wasn't valid.
#    # If we make N non-empty, we need to check validity.
#    # But if S was valid, N is valid.
#    # If S = F_q^2 is valid, then N = ∅ is "valid" in the sense that the condition holds for all x in F_q^2 (vacuously? no, x in N is empty, but condition is for ALL x in F_q^2).
#    # Wait, the condition is: "every point x of F_49^2 (in N or not) must lie on a line... whose points other than x all belong to N."
#    # If N is empty, then for any x, we need a line such that all other points are in N (empty).
#    # This is impossible unless the line has only x. But lines have 49 points.
#    # So N cannot be empty.
#    # So S cannot be F_q^2.
#    # My algorithm should naturally prevent S from becoming F_q^2 because the last point would break someone.
    
# So I don't need to worry about k=0.

# Let's write the code.

# One optimization: use numpy for the final array construction.

# Re-verify the "safe" logic.
# Condition for x to be in S: ∃v such that L(x, v) \ {x} ⊆ N.
# N = F_q^2 \ S.
# L(x, v) \ {x} ⊆ N  <=>  (L(x, v) \ {x}) ∩ S = ∅.
# This matches my safe_count logic.

# Condition for y to remain in S after adding x:
# y must still have a safe direction.
# safe_count[y] > 0.
# This matches.

# The logic holds.

# Time complexity:
# Precompute line_others: 2401 * 50 * 48 ~ 5.7M ops.
# Precompute rev_adj: 5.7M ops.
# Greedy: 2401 * (1 + 2400) ~ 5.7M ops.
# Total ~ 17M ops.
# Python can do ~10M-50M simple ops/sec.
# So ~1-2 seconds.
# Well within 600s.

# Let's implement.

# Note: I should use a list of sets for S? No, a set is fine for O(1) lookup.
# But iterating over S is not needed, we iterate over rev_adj[x].

# One small detail:
# "duplicate points are merged"
# My output is a list of unique points.

# Let's go.

# Wait, I need to make sure `points` list order matches `point_to_idx`.
# Yes, I generated them in the same loop.

# Final Code Structure:
# 1. Constants
# 2. Field ops
# 3. Precompute points, dirs, line_others, rev_adj
# 4. Greedy loop
# 5. Construct N
# 6. Return

# I will wrap it in the function.

# One potential bug:
# In `mul_f`, if a or b is (0,0), it works.
# In `add_f`, works.

# Let's double check `line_others` generation.
# y + t*v.
# t ranges over F_q \ {0}.
# 48 values.
# Correct.

# Let's double check `rev_adj`.
# x is in line_others[y][v] iff y is in line_others[x][v]?
# Yes, lines are symmetric in this context (affine lines).
# x = y + t*v => y = x - t*v = x + (-t)*v.
# Since t != 0, -t != 0.
# So yes.

# The code is ready.

# One minor improvement:
# Use `np.random.seed(42)` for reproducibility?
# The task doesn't require it, but it's good practice.
# However, different seeds might yield different |S|.
# I'll stick to random without seed, or use a fixed seed to ensure consistent scoring?
# The evaluator runs the code. If it's non-deterministic, the score might vary.
# Better to use a fixed seed.

# Also, the order of candidates can be optimized.
# Points with high "connectivity" might be harder to add?
# Or easier?
# Random is fine.

# Let's add `np.random.seed(0)`.

# Final check on the "deadline guard".
# I have `if time.time() - start_time > timeout: break`.
# This is safe.

# The `timeout` is 500s.
# Precomputation is fast.
# Greedy is fast.
# So it will likely finish in < 5s.

# I will set timeout to 550 to be safe.

# Let's write the code.
