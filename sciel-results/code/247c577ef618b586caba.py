import time
import math
from fractions import Fraction
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    t0 = time.time()
    deadline = 145.0  # Leave 5s for cleanup

    # We will search for a 6-point set in Z^2 (D=1).
    # If found, return it.
    # If not, return the best known 5-point set.
    
    # Fallback 5-point set (from current program, verified)
    base5 = [(Fraction(0), Fraction(0)),
             (Fraction(-56), Fraction(33)),
             (Fraction(-16), Fraction(63)),
             (Fraction(0), Fraction(33)),
             (Fraction(16), Fraction(63))]
    
    best = {'D': 1, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in base5]}
    best_count = 5

    # Search for 6-point set
    # Grid size
    L = 60
    
    points = []
    for x in range(-L, L+1):
        for y in range(-L, L+1):
            points.append((x, y))
            
    n = len(points)
    
    # Precompute squared distances and check if integer
    # dist2[i][j] = (xi-xj)^2 + (yi-yj)^2
    # We only care if it's a perfect square.
    
    # To save memory and time, we build adjacency lists.
    # adj[i] = list of j such that dist(i,j) is integer.
    
    adj = [[] for _ in range(n)]
    
    # Optimization: only compute for i < j
    for i in range(n):
        xi, yi = points[i]
        for j in range(i+1, n):
            xj, yj = points[j]
            dx = xi - xj
            dy = yi - yj
            d2 = dx*dx + dy*dy
            s = math.isqrt(d2)
            if s*s == d2:
                adj[i].append(j)
                adj[j].append(i)
                
    if time.time() > deadline - 20:
        return best

    # Helper to check collinearity
    def is_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

    # Helper to check concyclicity of 4 points
    def is_concyclic(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        # Determinant of:
        # x1^2+y1^2, x1, y1, 1
        # ...
        # Using expansion or Bareiss.
        # For integer points, we can use exact arithmetic.
        # Det = 0 iff concyclic.
        
        # To speed up, we can check if they lie on a circle.
        # Circle eq: x^2 + y^2 + Dx + Ey + F = 0
        # Subtract p1 from others to get linear system for D, E, F.
        # Actually, just compute det.
        
        # Optimization: if 3 are collinear, they are not concyclic (unless circle is line, which is not a circle).
        # But we already check no 3 collinear in the clique search.
        
        # Use fractions for safety? Integers are enough for det.
        # Det of 4x4 matrix.
        # [x1^2+y1^2, x1, y1, 1]
        # ...
        
        # Let's use a simple expansion for 4x4.
        # Or better, transform p1 to origin.
        # If we shift by -p1, the new points are q1=0, q2, q3, q4.
        # Circle through 0, q2, q3, q4.
        # Eq: x^2+y^2 + Ax + By = 0 (since F=0).
        # For q2: x2^2+y2^2 + A x2 + B y2 = 0
        # For q3: x3^2+y3^2 + A x3 + B y3 = 0
        # For q4: x4^2+y4^2 + A x4 + B y4 = 0
        # This is 3 eqs for A, B. Consistent iff det of 3x3 is 0?
        # No, it's a system. If unique A,B exists, check if it satisfies the 3rd?
        # Actually, 3 points define a circle (if not collinear). The 4th is on it iff it satisfies the eq.
        
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        # Shift p1 to origin
        x2 -= x1; y2 -= y1
        x3 -= x1; y3 -= y1
        x4 -= x1; y4 -= y1
        
        # Check if 2,3,4 are collinear with origin? No, p1,p2,p3 not collinear is checked.
        # But p1,p2,p4 might be collinear? No, clique check ensures no 3 collinear.
        
        # Circle through 0, (x2,y2), (x3,y3).
        # Eq: x^2+y^2 + Ax + By = 0
        # x2^2+y2^2 + A x2 + B y2 = 0
        # x3^2+y3^2 + A x3 + B y3 = 0
        
        # Solve for A, B
        # A x2 + B y2 = -(x2^2+y2^2)
        # A x3 + B y3 = -(x3^2+y3^2)
        
        det = x2*y3 - x3*y2
        if det == 0:
            return True # Collinear, but we assume not. If collinear, "concyclic" is degenerate.
                        # The problem says "no four concyclic". Usually implies distinct circle.
                        # If 3 are collinear, they don't form a circle.
                        # But we check no 3 collinear. So det != 0.
                        
        A_num = (-(x2^2+y2^2)*y3 + (x3^2+y3^2)*y2)
        B_num = (x2*(x3^2+y3^2) - x3*(x2^2+y2^2))
        
        # A = A_num / det
        # B = B_num / det
        
        # Check p4: x4^2+y4^2 + A x4 + B y4 = 0
        # x4^2+y4^2 + (A_num x4 + B_num y4) / det = 0
        # det * (x4^2+y4^2) + A_num x4 + B_num y4 == 0
        
        return det * (x4*x4 + y4*y4) + A_num * x4 + B_num * y4 == 0

    # Search for clique of size 6
    # We iterate over pairs (i, j)
    # Then common neighbors k such that not collinear(i,j,k)
    # Then common neighbors l of {i,j,k} such that no 3 collinear
    # Then common neighbors m of {i,j,k,l} such that no 3 collinear and not concyclic(i,j,k,l,m)
    # Then common neighbors n of {i,j,k,l,m} such that no 3 collinear and not concyclic(any 4)
    
    # To make this fast, we use bitsets or sorted lists of neighbors.
    # Since n is ~ 12000 (if L=60, 121x121=14641), lists are fine.
    
    # Pre-sort neighbors?
    # For each i, adj[i] is a list.
    
    # To find common neighbors of a set S, we can intersect the adjacency lists.
    
    found_6 = False
    best_set = []
    
    # Optimization: only consider i < j < k < l < m < n
    # This is hard with the "common neighbor" approach directly.
    # Instead, we can just find ANY 6-clique.
    
    # Iterate over i
    for i in range(n):
        if time.time() > deadline - 5:
            break
        # Iterate over j in adj[i] with j > i
        for j in adj[i]:
            if j <= i:
                continue
            # Find common neighbors of i and j
            # common_ij = adj[i] ∩ adj[j]
            # We can do this by iterating over the smaller list and checking membership in the other (using a set).
            set_j = set(adj[j])
            common_ij = [k for k in adj[i] if k in set_j and k > j] # k > j to ensure ordering?
            # Actually, we just need distinct points. Ordering helps avoid duplicates.
            
            for k in common_ij:
                if is_collinear(points[i], points[j], points[i+k if False else k]): # Wait, points is list of tuples
                    pass
                # Correct indexing
                if is_collinear(points[i], points[j], points[k]):
                    continue
                
                # Find common neighbors of i, j, k
                set_k = set(adj[k])
                # Common neighbors of i,j,k must be in adj[i], adj[j], adj[k]
                # We already have common_ij. Intersect with adj[k].
                common_ijk = [l for l in common_ij if l in set_k and l > k]
                
                for l in common_ijk:
                    # Check no 3 collinear among i,j,k,l
                    # We checked (i,j,k). Now check (i,j,l), (i,k,l), (j,k,l)
                    if is_collinear(points[i], points[j], points[l]) or \
                       is_collinear(points[i], points[k], points[l]) or \
                       is_collinear(points[j], points[k], points[l]):
                        continue
                    
                    # Find common neighbors of i,j,k,l
                    set_l = set(adj[l])
                    common_ijkl = [m for m in common_ijk if m in set_l and m > l]
                    
                    for m in common_ijkl:
                        # Check no 3 collinear
                        if is_collinear(points[i], points[j], points[m]) or \
                           is_collinear(points[i], points[k], points[m]) or \
                           is_collinear(points[i], points[l], points[m]) or \
                           is_collinear(points[j], points[k], points[m]) or \
                           is_collinear(points[j], points[l], points[m]) or \
                           is_collinear(points[k], points[l], points[m]):
                            continue
                        
                        # Check no 4 concyclic among i,j,k,l,m
                        # We need to check all 4-subsets of {i,j,k,l,m}
                        # There are 5 such subsets.
                        # If any is concyclic, skip.
                        if is_concyclic(points[i], points[j], points[k], points[l]) or \
                           is_concyclic(points[i], points[j], points[k], points[m]) or \
                           is_concyclic(points[i], points[j], points[l], points[m]) or \
                           is_concyclic(points[i], points[k], points[l], points[m]) or \
                           is_concyclic(points[j], points[k], points[l], points[m]):
                            continue
                        
                        # Find common neighbors of i,j,k,l,m
                        set_m = set(adj[m])
                        common_ijklm = [p for p in common_ijkl if p in set_m and p > m]
                        
                        for p in common_ijklm:
                            # Check no 3 collinear with p
                            if is_collinear(points[i], points[j], points[p]) or \
                               is_collinear(points[i], points[k], points[p]) or \
                               is_collinear(points[i], points[l], points[p]) or \
                               is_collinear(points[i], points[m], points[p]) or \
                               is_collinear(points[j], points[k], points[p]) or \
                               is_collinear(points[j], points[l], points[p]) or \
                               is_collinear(points[j], points[m], points[p]) or \
                               is_collinear(points[k], points[l], points[p]) or \
                               is_collinear(points[k], points[m], points[p]) or \
                               is_collinear(points[l], points[m], points[p]):
                                continue
                            
                            # Check no 4 concyclic with p
                            # Subsets of {i,j,k,l,m,p} of size 4 containing p?
                            # Or just all 4-subsets of the 6 points.
                            # We already checked all 4-subsets of {i,j,k,l,m}.
                            # Now we need to check 4-subsets involving p.
                            # There are C(5,3) = 10 such subsets.
                            
                            if is_concyclic(points[i], points[j], points[k], points[p]) or \
                               is_concyclic(points[i], points[j], points[l], points[p]) or \
                               is_concyclic(points[i], points[j], points[m], points[p]) or \
                               is_concyclic(points[i], points[k], points[l], points[p]) or \
                               is_concyclic(points[i], points[k], points[m], points[p]) or \
                               is_concyclic(points[i], points[l], points[m], points[p]) or \
                               is_concyclic(points[j], points[k], points[l], points[p]) or \
                               is_concyclic(points[j], points[k], points[m], points[p]) or \
                               is_concyclic(points[j], points[l], points[m], points[p]) or \
                               is_concyclic(points[k], points[l], points[m], points[p]):
                                continue
                                
                            # Found a valid 6-point set!
                            best_set = [points[i], points[j], points[k], points[l], points[m], points[p]]
                            found_6 = True
                            break
                            
                        if found_6: break
                    if found_6: break
                if found_6: break
            if found_6: break
        if found_6: break
        
    if found_6:
        best = {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_set]}
        best_count = 6

    return best

# EVOLVE-BLOCK-END
