import itertools
import time

def construct():
    # 9 points of the 3x3 grid in homogeneous coordinates (x, y, 1)
    grid_points = [(x, y, 1) for x in range(3) for y in range(3)]
    
    # 3 points at infinity: (1, 0, 0), (0, 1, 0), (1, 1, 0)
    # These correspond to vertical, horizontal, and slope 1 directions.
    inf_points = [(1, 0, 0), (0, 1, 0), (1, 1, 0)]
    
    all_points_homo = grid_points + inf_points
    
    # Projective transformation to map to affine plane
    # T = [[1, 0, 0], [0, 1, 0], [1, 1, 1]]
    # X = x, Y = y, Z = x + y + z
    transformed = []
    for x, y, z in all_points_homo:
        X = x
        Y = y
        Z = x + y + z
        if Z == 0:
            # This shouldn't happen for our points
            # (1,0,0) -> Z=1, (0,1,0) -> Z=1, (1,1,0) -> Z=2
            continue
        transformed.append((X, Y, Z))
        
    # Convert to affine coordinates (x/z, y/z) and scale to integers
    # We need to find a common multiple for all Z
    zs = [z for _, _, z in transformed]
    # LCM of zs
    from math import gcd
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    scale = 1
    for z in zs:
        scale = lcm(scale, z)
        
    points = []
    for X, Y, Z in transformed:
        x_int = X * scale // Z
        y_int = Y * scale // Z
        points.append([x_int, y_int])
        
    return points

# Verification (not part of the final output, but for thought)
# def count_ordinary(points):
#     n = len(points)
#     lines = set()
#     for i in range(n):
#         for j in range(i+1, n):
#             p1 = points[i]
#             p2 = points[j]
#             # Normalize line representation
#             a = p2[1] - p1[1]
#             b = p1[0] - p2[0]
#             c = p2[0]*p1[1] - p1[0]*p2[1]
#             # Normalize (a, b, c)
#             g = gcd(gcd(abs(a), abs(b)), abs(c))
#             if g != 0:
#                 a //= g
#                 b //= g
#                 c //= g
#             if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
#                 a, b, c = -a, -b, -c
#             lines.add((a, b, c))
#             
#     ordinary = 0
#     for line in lines:
#         count = 0
#         for p in points:
#             if line[0]*p[0] + line[1]*p[1] + line[2] == 0:
#                 count += 1
#         if count == 2:
#             ordinary += 1
#     return ordinary
