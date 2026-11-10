# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 by finding a large valid complement set S.
    
    The Nikodym property requires that for every point x in F_q^2, there is a line 
    through x such that all other points on that line are in N. 
    Equivalently, for the complement S = F_q^2 \ N, the condition is:
    For every x in S and every direction v, there exists t != 0 such that x + t*v is in S.
    (This ensures that for x in S, no line through x is "empty" of S except at x, 
    so every line through x has another point in S, meaning no line through x is 
    fully contained in N \ {x}. Wait, the definition is: 
    "every point x ... must lie on a line ... whose points other than x all belong to N".
    This means for each x, there exists a direction v_x such that {x + t*v_x : t!=0} subset N.
    This implies {x + t*v_x : t!=0} is disjoint from S.
    So, for a fixed x, if x in S, we must NOT have a direction v such that the entire punctured line is in N.
    Equivalently, for every direction v, the punctured line through x in direction v must intersect S.
    So: x in S => for all v, (Line(x, v) \ {x}) intersect S != empty.
    If x in N, the condition is that there EXISTS a v such that (Line(x, v) \ {x}) subset N.
    This is equivalent to: there EXISTS a v such that (Line(x, v) \ {x}) intersect S = empty.
    
    So the condition on S is:
    1. For all x in S: For all v, (Line(x, v) \ {x}) intersect S != empty.
    2. For all x in N: There exists v such that (Line(x, v) \ {x}) intersect S = empty.
    
    Note that condition 2 is automatically satisfied if condition 1 holds for all x in S?
    Let's check. If x in N, we need a v with no S on the line.
    If we maximize S subject to condition 1, does condition 2 hold?
    Actually, condition 1 is a "local" constraint on S. Condition 2 is a "global" constraint on N.
    It turns out that a set S satisfying condition 1 is called a "blocking set" for the lines? No.
    A set S is a blocking set if it intersects every line. That would mean for all x, v, Line(x,v) intersect S != empty.
    Here we require for x in S, the punctured line intersects S.
    
    Let's stick to the valid definition:
    N is Nikodym iff for every x, exists v such that (Line(x, v) \ {x}) subset N.
    This is equivalent to: For every x, NOT (for all v, (Line(x, v) \ {x}) intersect S != empty).
    Wait.
    (Line(x, v) \ {x}) subset N  <=>  (Line(x, v) \ {x}) intersect S = empty.
    So: For every x, exists v such that (Line(x, v) \ {x}) intersect S = empty.
    
    This condition must hold for ALL x, whether in N or S.
    So:
    If x in N: exists v s.t. Line(x, v) \ {x} has no S.
    If x in S: exists v s.t. Line(x, v) \ {x} has no S.
    
    So the condition is uniform: For every x in F_q^2, there exists a direction v such that 
    the set of other points on the line through x in direction v is disjoint from S.
    
    This is a much simpler constraint!
    Let's call a direction v "good" for x if (Line(x, v) \ {x}) intersect S = empty.
    We need: For every x, there is at least one good direction v.
    
    To maximize |S| (minimize |N|), we want to add points to S.
    Adding a point y to S might destroy the "good" directions for other points x.
    Specifically, if we add y to S, then for any x such that y is on Line(x, v), 
    v is no longer a "good" direction for x (unless y=x, which is excluded).
    So adding y to S removes v from the set of good directions for all x on Line(y, v) \ {y}.
    
    We want to find a set S such that for every x, the set of good directions G(x, S) is non-empty.
    G(x, S) = { v | (Line(x, v) \ {x}) intersect S = empty }.
    
    Initially S is empty. G(x, empty) = all directions.
    As we add points to S, G(x, S) shrinks.
    We cannot add a point y to S if it causes G(x, S \cup {y}) to become empty for some x.
    
    This is a Constraint Satisfaction Problem.
    We can model this as a search problem.
    State: S (bitmask).
    Transition: Add y to S.
    Constraint: For all x, G(x, S) != empty.
    
    We can precompute for each point y and direction v, the set of points x for which 
    adding y to S would remove v from G(x, S).
    Actually, adding y to S removes v from G(x, S) if and only if y is in Line(x, v) \ {x}.
    This is equivalent to x being in Line(y, v) \ {y}.
    So, if we maintain for each x the count of "blocked" directions, or the set of available directions,
    we can check validity.
    
    Let's use a bitmask for S.
    Let's precompute for each point y and direction v, the list of points x affected.
    Actually, it's easier to check: 
    If we add y to S, which x's lose a good direction?
    For each direction v, the points x that lose v as a good direction are exactly the points 
    on the line through y in direction v, excluding y.
    So, for each v, the set of x's is Line(y, v) \ {y}.
    
    We can maintain an array `avail_dirs` of shape (total_points, num_dirs) as a bitmask or count.
    Actually, since num_dirs = 50, we can use two uint32 masks per point.
    Or simply, for each point x, keep a count of available directions.
    But we need to know WHICH directions are available to check if we can add y?
    No, we just need to know if `count[x] > 0`.
    
    However, when we add y, we need to decrement `count[x]` for all x in Line(y, v) \ {y} for ALL v.
    And we need to ensure that no count[x] becomes 0.
    
    This is efficient!
    Precomputation:
    For each y, and each v, list the x's in Line(y, v) \ {y}.
    This is `line_neighbors` again, but we can just store the indices.
    
    Algorithm:
    1. Initialize S = empty.
    2. Initialize `dir_counts[x] = num_dirs` for all x.
    3. We want to add points to S.
    4. Greedy: Pick a point y not in S.
       Check if adding y is valid:
       For each v, for each x in Line(y, v) \ {y}:
           if dir_counts[x] == 1:
               # This x will have 0 directions left if we add y
               # So adding y is invalid
               return False
       If valid, add y to S, and update dir_counts:
       For each v, for each x in Line(y, v) \ {y}:
           dir_counts[x] -= 1
       
    5. To maximize S, we should choose y carefully.
       A "good" y is one that affects x's with high dir_counts.
       Or, we can use randomized restarts.
    
    Let's implement this.
    """
    p = 7
    q = p * p
    total_points = q * q
    num_dirs = q + 1  # 50
    
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    # Field setup
    k_const = 3  # alpha^2 = 3, 3 is non-residue mod 7 (3^3 = 27 = 6 = -1)
    
    # Directions
    # v = (1, m) for m in 0..48, and v = (0, 1)
    directions = []
    for m in range(q):
        directions.append((1, m))
    directions.append((0, 1))
    
    # Precompute line neighbors
    # neighbors[y_idx, v_idx] = list of x_idx such that x is on line through y in direction v, x != y
    # We store these in a flat array for speed? Or list of lists.
    # 2401 * 50 * 48 = 5.7M entries.
    # Using list of lists is fine.
    
    # Precompute field addition and multiplication tables for speed?
    # Or just compute on the fly in precomputation.
    
    # Let's precompute neighbors
    neighbors = [[[] for _ in range(num_dirs)] for _ in range(total_points)]
    
    for y_idx in range(total_points):
        y_x_idx = y_idx // q
        y_y_idx = y_idx % q
        y_xa, y_xb = divmod(y_x_idx, p)
        y_ya, y_yb = divmod(y_y_idx, p)
        
        for v_idx, (dx_idx, dy_idx) in enumerate(directions):
            dx_a, dx_b = divmod(dx_idx, p)
            dy_a, dy_b = divmod(dy_idx, p)
            
            for t_idx in range(1, q):
                t_a, t_b = divmod(t_idx, p)
                
                # t * dx
                tx_a = (t_a * dx_a + t_b * dx_b * k_const) % p
                tx_b = (t_a * dx_b + t_b * dx_a) % p
                # t * dy
                ty_a = (t_a * dy_a + t_b * dy_b * k_const) % p
                ty_b = (t_a * dy_b + t_b * dy_a) % p
                
                nx_a = (y_xa + tx_a) % p
                nx_b = (y_xb + tx_b) % p
                ny_a = (y_ya + ty_a) % p
                ny_b = (y_yb + ty_b) % p
                
                nx_idx = nx_a * p + nx_b
                ny_idx = ny_a * p + ny_b
                nx_idx_flat = nx_idx * q + ny_idx
                
                neighbors[y_idx][v_idx].append(nx_idx_flat)
    
    # Greedy search with random restarts
    best_S_size = 0
    best_S_mask = np.zeros(total_points, dtype=bool)
    
    max_restarts = 10
    restarts = 0
    
    while restarts < max_restarts and time.time() < deadline:
        restarts += 1
        
        S = np.zeros(total_points, dtype=bool)
        dir_counts = np.full(total_points, num_dirs, dtype=np.int32)
        
        # Order of points to try
        # We can randomize the order
        order = list(range(total_points))
        random.shuffle(order)
        
        added = 0
        for y_idx in order:
            if S[y_idx]:
                continue
            
            # Check if adding y_idx is valid
            valid = True
            for v_idx in range(num_dirs):
                for x_idx in neighbors[y_idx][v_idx]:
                    if dir_counts[x_idx] == 1:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                S[y_idx] = True
                added += 1
                # Update dir_counts
                for v_idx in range(num_dirs):
                    for x_idx in neighbors[y_idx][v_idx]:
                        dir_counts[x_idx] -= 1
        
        if added > best_S_size:
            best_S_size = added
            best_S_mask = S.copy()
        
        # Early stop if we found a perfect set? 
        # Max possible S size is unknown, but if we get close to total, stop.
        if best_S_size > total_points * 0.9:
            break
    
    # Construct N from best_S_mask
    N_indices = np.where(~best_S_mask)[0]
    
    if len(N_indices) == 0:
        # Fallback
        N_indices = np.arange(total_points)
    
    points = np.zeros((len(N_indices), 2, 2), dtype=np.int64)
    for i, idx in enumerate(N_indices):
        x_idx = idx // q
        y_idx = idx % q
        x_a, x_b = divmod(x_idx, p)
        y_a, y_b = divmod(y_idx, p)
        points[i, 0, 0] = x_a
        points[i, 0, 1] = x_b
        points[i, 1, 0] = y_a
        points[i, 1, 1] = y_b
    
    return points

# EVOLVE-BLOCK-END
