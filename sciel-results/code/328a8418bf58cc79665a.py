import numpy as np
from itertools import combinations

def szemeredi_trotter_incidences():
    # Grid 1..9
    # 9 vertical lines: x = i, i in 1..9 -> (1, 0, -i)
    # 9 horizontal lines: y = j, j in 1..9 -> (0, 1, -j)
    # 9 diagonal lines.
    # Slope 1: y - x = d -> -x + y - d = 0 -> (-1, 1, -d)
    # d in [-8, 8]
    # Slope -1: x + y = c -> x + y - c = 0 -> (1, 1, -c)
    # c in [2, 18]
    
    possible_d = list(range(-8, 9)) # 17 values
    possible_c = list(range(2, 19)) # 17 values
    
    # Precompute lengths of diagonals in the grid
    # L_d: number of points (x,y) in 1..9 with y-x=d
    L_d = {}
    for d in possible_d:
        count = 0
        for x in range(1, 10):
            y = x + d
            if 1 <= y <= 9:
                count += 1
        L_d[d] = count
        
    # L_c: number of points (x,y) in 1..9 with x+y=c
    L_c = {}
    for c in possible_c:
        count = 0
        for x in range(1, 10):
            y = c - x
            if 1 <= y <= 9:
                count += 1
        L_c[c] = count
        
    # Precompute intersection matrix M[d][c]
    # M[d][c] = 1 if the intersection of y-x=d and x+y=c is in the grid, else 0
    # Intersection: x = (c-d)/2, y = (c+d)/2
    # Must be integer and in 1..9
    M = {}
    for d in possible_d:
        M[d] = {}
        for c in possible_c:
            if (c - d) % 2 != 0:
                M[d][c] = 0
            else:
                x = (c - d) // 2
                y = (c + d) // 2
                if 1 <= x <= 9 and 1 <= y <= 9:
                    M[d][c] = 1
                else:
                    M[d][c] = 0

    best_score = -1
    best_d = ()
    best_c = ()
    
    # Search over all k1 from 0 to 9
    for k1 in range(10):
        k2 = 9 - k1
        # Iterate combinations of k1 from possible_d and k2 from possible_c
        for d_comb in combinations(possible_d, k1):
            # Precompute sum of lengths for this d_comb
            sum_L_d = sum(L_d[d] for d in d_comb)
            # Precompute contribution to n4 for each c?
            # n4 = sum_{d in d_comb} sum_{c in c_comb} M[d][c]
            # Let's precompute for each d in d_comb, the vector of M[d][c] for all c?
            # Or just compute inside.
            
            for c_comb in combinations(possible_c, k2):
                sum_L_c = sum(L_c[c] for c in c_comb)
                
                # Calculate n4
                n4 = 0
                for d in d_comb:
                    for c in c_comb:
                        n4 += M[d][c]
                
                # Calculate n3
                # n3 = |A| + |B| - 2*n4
                # |A| = sum_L_d
                # |B| = sum_L_c
                n3 = sum_L_d + sum_L_c - 2 * n4
                
                # Calculate score
                # Points with count 4: n4
                # Points with count 3: n3
                # Points with count 2: 81 - n4 - n3
                # We select top 54.
                
                if n4 >= 54:
                    score = 4 * 54
                else:
                    score = 4 * n4
                    remaining = 54 - n4
                    if n3 >= remaining:
                        score += 3 * remaining
                    else:
                        score += 3 * n3
                        score += 2 * (remaining - n3)
                
                if score > best_score:
                    best_score = score
                    best_d = d_comb
                    best_c = c_comb

    # Construct lines
    lines = []
    for i in range(1, 10):
        lines.append((1, 0, -i))
    for j in range(1, 10):
        lines.append((0, 1, -j))
    for d in best_d:
        lines.append((-1, 1, -d))
    for c in best_c:
        lines.append((1, 1, -c))
        
    lines = np.array(lines, dtype=np.int64)
    
    # Select points
    xs = np.arange(1, 10)
    ys = np.arange(1, 10)
    PX, PY = np.meshgrid(xs, ys, indexing="ij")
    all_points = np.stack([PX.ravel(), PY.ravel()], axis=1).astype(np.int64)
    
    incidence_counts = np.zeros(len(all_points), dtype=np.int64)
    for idx in range(len(all_points)):
        x = all_points[idx, 0]
        y = all_points[idx, 1]
        for A, B, C in lines:
            if A * x + B * y + C == 0:
                incidence_counts[idx] += 1
                
    sorted_indices = np.argsort(-incidence_counts)
    selected_indices = sorted_indices[:54]
    points = all_points[selected_indices]
    
    return points, lines
