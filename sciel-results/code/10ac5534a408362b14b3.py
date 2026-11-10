# EVOLVE-BLOCK-START
import numpy as np


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_q^2 with q = 49.
    
    Strategy:
    We define the excluded set S to be the graph of the identity function in F_49^2.
    S = { (x, x) : x in F_49 }.
    
    Validity:
    For any point x = (x0, x0) in S, consider the vertical line L passing through x,
    defined by X = x0. The points on this line are { (x0, y) : y in F_49 }.
    The intersection of L with S is { (x0, y) in S : y = x0 } = { (x0, x0) }.
    Thus, the line L contains exactly one point of S, which is x itself.
    Therefore, all other points on L belong to N = F_49^2 \ S.
    This satisfies the Nikodym set condition for every x in S.
    
    For any point z not in S, z is in N. The condition requires that z lies on a line
    whose other points are in N. Since z is in N, we just need to ensure there is a line
    through z that doesn't contain any "bad" points? No, the condition is for EVERY point
    of F_49^2.
    Wait, the condition is: "every point x of F_49^2 ... must lie on a line ... whose
    points other than x all belong to N".
    
    If x is in N, we need a line through x such that all other points on that line are in N.
    This means the line can contain at most 1 point from S (which is x if x in S, or 0 if x in N).
    If x in N, we need a line through x that does NOT intersect S at all?
    Or intersects S only at x? But x is not in S.
    So we need a line through x (where x in N) that is disjoint from S.
    
    Does every point in N have a line through it disjoint from S?
    S is a line (the diagonal).
    Any point z not on the diagonal is not in S.
    Consider a line L through z parallel to the diagonal S.
    L is disjoint from S (since they are distinct parallel lines in a vector space over a field, assuming non-vertical/non-horizontal? No, in F_q^2, parallel lines are disjoint).
    So L contains no points from S.
    Thus, all points on L are in N.
    So the condition is satisfied for all z in N.
    
    Therefore, S = { (x, x) : x in F_49 } is a valid excluded set.
    Size of S is 49.
    Size of N is 2401 - 49 = 2352.
    """
    p = 7
    q = p * p  # 49
    
    # We represent points as [[x_a, x_b], [y_a, y_b]]
    # The field element x = x_a + x_b * alpha.
    # The identity map y = x means y_a = x_a and y_b = x_b.
    
    points = []
    for xa in range(p):
        for xb in range(p):
            ya = xa
            yb = xb
            points.append([[xa, xb], [ya, yb]])
            
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
