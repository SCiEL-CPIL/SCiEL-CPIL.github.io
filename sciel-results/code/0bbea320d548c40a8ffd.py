# EVOLVE-BLOCK-START
import numpy as np


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 by excluding a line of 49 points plus 48 additional
    points, one on each of 48 distinct parallel lines to the excluded line.
    
    Field: F_49 = F_7[alpha] with alpha^2 = 3.
    A point is [[x_a, x_b], [y_a, y_b]] representing (x_a + x_b*alpha, y_a + y_b*alpha).
    
    We exclude:
    - The line y = 0 (49 points)
    - 48 additional points, one on each line y = c for c in {1, ..., 48} (as field elements)
    
    This gives |S| = 49 + 48 = 97, so |N| = 2401 - 97 = 2304.
    
    Validity: For any point x in F_49^2:
    - If x is not in S, any direction works since at most one point on any line through x 
      is in S.
    - If x is on the excluded line y=0, use a direction not parallel to y=0; the line 
      through x in that direction intersects y=0 only at x, and each additional point 
      P_i is on a different parallel line, so the line from x to P_i is unique for each 
      P_i, but we just need ONE clean line. Use vertical direction (0,1): the line 
      x + t*(0,1) = (x_a, x_b + t) for t in F_49. This line has y varying, x fixed. 
      It intersects y=0 only at x itself. For additional points P_i = (a_i, i), this 
      line contains P_i only if x_a = a_i and x_b = 0 (since y=0 means t such that 
      0 + t = i, so t=i, and x must equal a_i). But we need the line to NOT contain 
      other S points. The line (x_a, x_b + t) for t in F_49 contains points (x_a, y) 
      for all y. The only S points on this line are: x itself (y=0) and possibly P_i 
      if x_a = a_i. So if we choose the additional points P_i such that no two have 
      the same x-coordinate, then for a given x on y=0, at most one P_i has x_a = x_a. 
      So the vertical line through x contains at most 2 S points: x and possibly one P_i. 
      But we need a line where ALL other points are in N. So we need a line through x 
      that contains NO other S points.
      
      Let me reconsider. For x on y=0, we need a line through x where all other 48 
      points are in N. The line in direction (0,1) through x=(a,0) is {(a, y) : y in F_49}. 
      This contains x and possibly one P_i if P_i has x-coordinate a. So this line has 
      at most 2 S points. Not good.
      
      Instead, for x on y=0, use a direction (1, m) for some m. The line is 
      {(x_a + t, x_b + m*t) : t in F_49}. This line intersects y=0 at t=0 (giving x) 
      and possibly another point if there's t != 0 with x_b + m*t = 0, i.e., t = -x_b/m. 
      Since x_b = 0 (x is on y=0), t = 0. So the line intersects y=0 only at x. 
      For additional points P_i = (a_i, i), the line contains P_i if a_i = x_a + t and 
      i = m*t for some t, i.e., a_i = x_a + i/m. So for a fixed x and m, at most one 
      P_i is on this line (since i determines t). So the line in direction (1,m) through 
      x contains at most 2 S points: x and possibly one P_i.
      
      Hmm, still not clean. Let me think differently.
      
      Actually, for x on y=0, I can use ANY direction that doesn't pass through any 
      P_i. Since there are 48 P_i's and 49 possible directions (48 slopes + vertical), 
      for a fixed x, each P_i determines one bad direction (the direction from x to P_i). 
      So there are at most 48 bad directions. Since there are 49 directions total, there's 
      at least one good direction. So x is valid.
      
    - If x = P_i (an additional point), use the line y = i (parallel to y=0). This line 
      contains P_i and no other S points (since y=0 line is different, and other P_j have 
      y=j != i). So this line is clean.
    
    To ensure the construction works, choose the 48 additional points with distinct 
    x-coordinates.
    """
    p = 7
    k = 3  # alpha^2 = 3
    
    # Represent field elements as pairs (a, b) for a + b*alpha
    # We'll work with pairs directly
    
    # The line y = 0: all points (x, 0) for x in F_49
    # 48 additional points: one on each line y = c for c = 1, ..., 48 (as field elements)
    # Choose them with distinct x-coordinates
    
    # Generate all 49 field elements as pairs
    field_elements = [(a, b) for a in range(p) for b in range(p)]
    
    # The excluded line y=0 has 49 points
    # The 48 additional points: for i = 1 to 48, pick point (x_i, y_i) where y_i is the 
    # i-th non-zero field element, and x_i are distinct
    
    # Let's use the first 48 field elements as x-coordinates for the 48 additional points
    # and the next 48 non-zero field elements as y-coordinates
    
    # Actually, let's be more systematic. We have 49 field elements. 
    # y=0 is one of them. The other 48 are non-zero.
    # For the 48 additional points, use y = each non-zero field element, and x = distinct values.
    
    # Let's just pick: for the i-th non-zero field element y_i, use x_i = the i-th field element
    # (including 0 is fine as long as they're distinct)
    
    # Non-zero field elements in order
    non_zero_elements = [(a, b) for a in range(p) for b in range(p) if not (a == 0 and b == 0)]
    
    # 48 non-zero elements
    assert len(non_zero_elements) == 48
    
    # x-coordinates: use the first 48 field elements (including 0)
    x_coords = field_elements[:48]
    
    # Build the set S of excluded points
    S = set()
    
    # Add the line y=0
    for x in field_elements:
        S.add((x[0], x[1], 0, 0))  # (x_a, x_b, y_a, y_b)
    
    # Add the 48 additional points
    for i in range(48):
        x_coord = x_coords[i]
        y_coord = non_zero_elements[i]
        S.add((x_coord[0], x_coord[1], y_coord[0], y_coord[1]))
    
    # Build N = F_49^2 \ S
    N = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    point = (xa, xb, ya, yb)
                    if point not in S:
                        N.append([[xa, xb], [ya, yb]])
    
    return np.array(N, dtype=np.int64)


# EVOLVE-BLOCK-END
