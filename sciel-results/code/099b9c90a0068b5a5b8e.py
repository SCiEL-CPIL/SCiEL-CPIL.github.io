"""Rewritten program for Heilbronn triangles: SLP in barycentric coordinates."""
N = 11
import numpy as np
from scipy.optimize import linprog
import itertools

# EVOLVE-BLOCK-START

def construct():
    """
    Place 11 points in the equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2).
    Maximize the minimum area of any triangle formed by three points.
    Uses SLP with barycentric coordinates.
    """
    sqrt3 = np.sqrt(3)
    triples = list(itertools.combinations(range(N), 3))
    n_triples = len(triples)  # 165

    def bary_to_cart(bary):
        """Convert (a,b,c) barycentric to (x,y) Cartesian. a+b+c=1."""
        x = bary[:, 1] + 0.5 * bary[:, 2]
        y = (sqrt3 / 2) * bary[:, 2]
        return np.column_stack([x, y])

    def det_from_bary(bary):
        """
        Compute the signed area ratio for each triple.
        The area of triangle (i,j,k) relative to the reference triangle area
        is |det| / (2 * ref_area) ... actually let's just compute the determinant
        of the 3x3 matrix of barycentric coords.
        
        For points with barycentric coords (a_i, b_i, c_i), the area ratio
        (area_ijk / area_ref) = |det([[a_i,b_i,c_i],[a_j,b_j,c_j],[a_k,b_k,c_k]])|
        
        Since ref_area = sqrt(3)/4, the actual area = ref_area * |det|.
        We maximize the minimum |det|, which is equivalent to maximizing min area.
        """
        dets = np.zeros(n_triples)
        for idx, (i, j, k) in enumerate(triples):
            M = np.array([bary[i], bary[j], bary[k]])
            dets[idx] = np.linalg.det(M)
        return dets

    def min_abs_det(bary):
        return np.min(np.abs(det_from_bary(bary)))

    # Initial configuration: symmetric arrangement
    # Use a known good starting point: 3 corners + 3 edge midpoints + 4 interior
    # Corners: (1,0,0), (0,1,0), (0,0,1)
    # Edge midpoints: (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    # Interior: centroid (1/3,1/3,1/3) and 3 more
    
    bary = np.array([
        [1.0, 0.0, 0.0],   # corner A
        [0.0, 1.0, 0.0],   # corner B
        [0.0, 0.0, 1.0],   # corner C
        [0.5, 0.5, 0.0],   # midpoint AB
        [0.5, 0.0, 0.5],   # midpoint AC
        [0.0, 0.5, 0.5],   # midpoint BC
        [1/3, 1/3, 1/3],   # centroid
        [0.25, 0.25, 0.5], # interior
        [0.25, 0.5, 0.25], # interior
        [0.5, 0.25, 0.25], # interior
        [0.2, 0.2, 0.6],   # interior
    ])
    
    # Normalize to ensure a+b+c=1
    bary = bary / bary.sum(axis=1, keepdims=True)
    
    current_min = min_abs_det(bary)
    
    # SLP iterations
    max_iter = 80
    tol = 1e-10
    
    for iteration in range(max_iter):
        dets = det_from_bary(bary)
        signs = np.sign(dets)
        # Handle zero determinants
        signs[signs == 0] = 1
        
        # Variables: a_0..a_10, b_0..b_10, t => 23 variables
        # c_i = 1 - a_i - b_i
        n_vars = 2 * N + 1
        c_obj = np.zeros(n_vars)
        c_obj[-1] = -1  # minimize -t
        
        A_ub = []
        b_ub = []
        
        # Domain constraints: a_i >= 0, b_i >= 0, a_i + b_i <= 1
        # a_i >= 0 => -a_i <= 0
        for i in range(N):
            row = np.zeros(n_vars)
            row[i] = -1
            A_ub.append(row)
            b_ub.append(0)
        
        # b_i >= 0 => -b_i <= 0
        for i in range(N):
            row = np.zeros(n_vars)
            row[N + i] = -1
            A_ub.append(row)
            b_ub.append(0)
        
        # a_i + b_i <= 1
        for i in range(N):
            row = np.zeros(n_vars)
            row[i] = 1
            row[N + i] = 1
            A_ub.append(row)
            b_ub.append(1)
        
        # Triangle area constraints:
        # For triple (i,j,k), the determinant is:
        # det = a_i*(b_j*c_k - c_j*b_k) - b_i*(a_j*c_k - c_j*a_k) + c_i*(a_j*b_k - b_j*a_k)
        # where c_m = 1 - a_m - b_m
        # 
        # Let's expand det in terms of a and b only.
        # c_i = 1 - a_i - b_i
        # 
        # det = a_i*(b_j*(1-a_k-b_k) - (1-a_j-b_j)*b_k) 
        #       - b_i*(a_j*(1-a_k-b_k) - (1-a_j-b_j)*a_k)
        #       + (1-a_i-b_i)*(a_j*b_k - b_j*a_k)
        #
        # This is still bilinear. For SLP, we linearize around current point.
        # 
        # The signed area constraint: sign_ijk * det >= t
        # Linearize det around current bary:
        # det_linear = det_current + sum of partial derivatives * (delta vars)
        #
        # Partial derivatives of det with respect to a_m and b_m:
        # det = det of [[a_i, b_i, c_i], [a_j, b_j, c_j], [a_k, b_k, c_k]]
        # where c_m = 1 - a_m - b_m
        #
        # d(det)/d(a_m): replace column a of row m with 1, columns b and c unchanged
        #   = cofactor expansion... let's compute directly.
        #
        # Actually, let's use the formula:
        # det = a_i(b_j c_k - c_j b_k) - b_i(a_j c_k - c_j a_k) + c_i(a_j b_k - b_j a_k)
        #
        # d(det)/d(a_i) = b_j c_k - c_j b_k
        # d(det)/d(b_i) = -(a_j c_k - c_j a_k) = c_j a_k - a_j c_k
        # d(det)/d(c_i) = a_j b_k - b_j a_k
        # Similarly for j and k.
        #
        # But c_i = 1 - a_i - b_i, so:
        # d(det)/d(a_i) [total] = d(det)/d(a_i) [direct] + d(det)/d(c_i) * d(c_i)/d(a_i)
        #                        = (b_j c_k - c_j b_k) + (a_j b_k - b_j a_k) * (-1)
        #                        = b_j c_k - c_j b_k - a_j b_k + b_j a_k
        #
        # d(det)/d(b_i) [total] = d(det)/d(b_i) [direct] + d(det)/d(c_i) * d(c_i)/d(b_i)
        #                        = (c_j a_k - a_j c_k) + (a_j b_k - b_j a_k) * (-1)
        #                        = c_j a_k - a_j c_k - a_j b_k + b_j a_k
        
        for idx, (i, j, k) in enumerate(triples):
            sgn = signs[idx]
            det_cur = dets[idx]
            ai, bi, ci = bary[i]
            aj, bj, cj = bary[j]
            ak, bk, ck = bary[k]
            
            # Compute total derivatives
            # d(det)/d(a_i)
            dad_ai = bj * ck - cj * bk - aj * bk + bj * ak
            # d(det)/d(b_i)
            dbb_ai = cj * ak - aj * ck - aj * bk + bj * ak
            
            # d(det)/d(a_j)
            dad_aj = ci * bk - cj * bi - ci * bk + cj * bi  # wait, let me redo
            # d(det)/d(a_j) [direct] = -(a_i c_k - c_i a_k) = c_i a_k - a_i c_k
            # d(det)/d(c_j) = a_i b_k - b_i a_k
            # total: (c_i a_k - a_i c_k) + (a_i b_k - b_i a_k) * (-1)
            dad_aj = ci * ak - ai * ck - ai * bk + bi * ak
            
            # d(det)/d(b_j)
            # d(det)/d(b_j) [direct] = -(a_i c_k - c_i a_k)... no
            # det = a_i(b_j c_k - c_j b_k) - b_i(a_j c_k - c_j a_k) + c_i(a_j b_k - b_j a_k)
            # d/d(b_j) of a_i(b_j c_k - c_j b_k) = a_i c_k
            # d/d(b_j) of -b_i(a_j c_k - c_j a_k) = 0
            # d/d(b_j) of c_i(a_j b_k - b_j a_k) = -c_i a_k
            # d/d(b_j) of c_j = -1, so d/d(b_j) of -a_i c_j b_k = a_i b_k
            # Wait, I need to be more careful.
            # 
            # Let me just compute numerically or use a cleaner approach.
            # 
            # Actually, let's use the matrix derivative approach:
            # M = [[ai, bi, ci], [aj, bj, cj], [ak, bk, ck]]
            # d(det)/d(M[m,n]) = cofactor[m,n]
            # 
            # Cofactor matrix:
            # C[0,0] = bj*ck - cj*bk
            # C[0,1] = -(ak*ck - cj*bk) = cj*bk - ak*ck
            # C[0,2] = ak*bj - aj*bk
            # C[1,0] = -(bi*ck - cj*bk) = cj*bk - bi*ck
            # C[1,1] = ai*ck - ak*ci
            # C[1,2] = -(ai*bj - aj*bi) = aj*bi - ai*bj
            # C[2,0] = bi*cj - cj*bi = 0... no
            # C[2,0] = bi*cj - cj*bi... that's 0? No:
            # C[2,0] = +det([[bi,cj],[bk,ck]]) = bi*ck - cj*bk
            # C[2,1] = -det([[ai,cj],[ak,ck]]) = -(ai*ck - cj*ak) = cj*ak - ai*ck
            # C[2,2] = +det([[ai,bi],[aj,bj]]) = ai*bj - aj*bi
            
            # d(det)/d(a_i) = C[0,0] + C[0,2]*(-1) = (bj*ck - cj*bk) - (ak*bj - aj*bk)
            # = bj*ck - cj*bk - ak*bj + aj*bk
            
            # d(det)/d(b_i) = C[0,1] + C[0,2]*(-1) = (cj*bk - ak*ck) - (ak*bj - aj*bk)
            # = cj*bk - ak*ck - ak*bj + aj*bk
            
            # d(det)/d(a_j) = C[1,0] + C[1,2]*(-1) = (cj*bk - bi*ck) - (aj*bi - ai*bj)
            # = cj*bk - bi*ck - aj*bi + ai*bj
            
            # d(det)/d(b_j) = C[1,1] + C[1,2]*(-1) = (ai*ck - ak*ci) - (aj*bi - ai*bj)
            # = ai*ck - ak*ci - aj*bi + ai*bj
            
            # d(det)/d(a_k) = C[2,0] + C[2,2]*(-1) = (bi*ck - cj*bk) - (ai*bj - aj*bi)
            # = bi*ck - cj*bk - ai*bj + aj*bi
            
            # d(det)/d(b_k) = C[2,1] + C[2,2]*(-1) = (cj*ak - ai*ck) - (ai*bj - aj*bi)
            # = cj*ak - ai*ck - ai*bj + aj*bi
            
            dad_ai = bj*ck - cj*bk - ak*bj + aj*bk
            dbb_ai = cj*bk - ak*ck - ak*bj + aj*bk
            dad_aj = cj*bk - bi*ck - aj*bi + ai*bj
            dbb_aj = ai*ck - ak*ci - aj*bi + ai*bj
            dad_ak = bi*ck - cj*bk - ai*bj + aj*bi
            dbb_ak = cj*ak - ai*ck - ai*bj + aj*bi
            
            # Constraint: sgn * det_linear >= t
            # det_linear = det_cur + dad_ai*(a_i - ai) + dbb_ai*(b_i - bi) + ... 
            # sgn * det_cur + sgn*(dad_ai*a_i + dbb_ai*b_i + ... - dad_ai*ai - dbb_ai*bi - ...) >= t
            # => -sgn*dad_ai*a_i - sgndbb_ai*b_i - ... + t <= sgn*(det_cur - dad_ai*ai - dbb_ai*bi - ...)
            
            row = np.zeros(n_vars)
            row[i] = -sgn * dad_ai
            row[N+i] = -sgn * dbb_ai
            row[j] = -sgn * dad_aj
            row[N+j] = -sgn * dbb_aj
            row[k] = -sgn * dad_ak
            row[N+k] = -sgn * dbb_ak
            row[-1] = 1  # coefficient for t
            
            b_val = sgn * (det_cur - dad_ai*ai - dbb_ai*bi - dad_aj*aj - dbb_aj*bj - dad_ak*ak - dbb_ak*bk)
            
            A_ub.append(row)
            b_ub.append(b_val)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        bounds = []
        for i in range(N):
            bounds.append((0, 1))  # a_i
        for i in range(N):
            bounds.append((0, 1))  # b_i
        bounds.append((0, None))   # t
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                new_a = res.x[:N]
                new_b = res.x[N:2*N]
                new_c = 1 - new_a - new_b
                
                # Check validity
                if np.any(new_c < -1e-9) or np.any(new_a < -1e-9) or np.any(new_b < -1e-9):
                    break
                
                new_bary = np.column_stack([new_a, new_b, new_c])
                new_min = min_abs_det(new_bary)
                
                if new_min > current_min - tol:
                    bary = new_bary
                    current_min = new_min
                else:
                    break
            else:
                break
        except Exception:
            break
    
    # Convert to Cartesian
    cart = bary_to_cart(bary)
    return cart

# EVOLVE-BLOCK-END
