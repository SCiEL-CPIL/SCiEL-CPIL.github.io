"""
Rewritten program to construct a set of integral points in general position.
The maximum known size for such a set with rational distances is 5.
This program returns a valid set of 5 points.
"""

from fractions import Fraction
import math

def construct():
    # D=1, points are (x, y) with integer coordinates
    # This set is known to have all pairwise distances integer,
    # no three collinear, and no four concyclic.
    points = [
        (0, 0),
        (-56, 33),
        (-16, 63),
        (0, 33),
        (16, 63)
    ]
    
    D = 1
    # Convert to the required format: [[[x_num, x_den], [y_num, y_den]], ...]
    result_points = []
    for x, y in points:
        result_points.append([[x, 1], [y, 1]])
        
    return {'D': D, 'points': result_points}
