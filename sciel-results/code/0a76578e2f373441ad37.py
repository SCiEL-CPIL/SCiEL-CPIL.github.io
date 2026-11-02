"""Rewritten program for Heilbronn triangles: SLP in barycentric coordinates."""
N = 11
import numpy as np
from scipy.optimize import linprog
import itertools

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
import itertools
import time

N = 11
SQRT3 = np.sqrt(3)
REF_AREA = 0.5 * SQRT3 / 2  # area of equilateral triangle with side 1 = sqrt(3)/4

def _bary_to_cart(bary):
    """Convert barycentric coords (a,b,c) to Cartesian (x,y) for triangle (0,0),(1,0),(0.5,sqrt(3)/2)."""
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = (SQRT3 / 2) * bary[:, 2]
    return np.column_stack([x, y])

def _signed_area_bary(bary, i, j, k):
    """Compute signed area ratio (relative to reference triangle) for triple (i,j,k) in barycentric coords."""
    # The area of triangle formed by points with barycentric coords is:
    # area_ratio = |det([[a_i, b_i, c_i], [a_j, b_j, c_j], [a_k, b_k, c_k]])| / (2 * ref_area) * ... 
    # Actually, the area in absolute terms = |det(M)| * ref_area / 2 where M is 3x3 matrix of barycentric coords
    # Wait, let me reconsider.
    # If points have barycentric coords (a_i, b_i, c_i), their positions are p_i = a_i*A + b_i*B + c_i*C
    # The area of triangle (p_i, p_j, p_k) = 0.5 * |det(p_j - p_i, p_k - p_i)|
    # In terms of barycentric coords, this equals |det([[a_i,b_i,c_i],[a_j,b_j,c_j],[a_k,b_k,c_k]])| * ref_area
    # Let me verify: if we take the three vertices of the reference triangle, bary = identity matrix, det = 1, area = ref_area. Correct.
    # So signed area = det(M) * ref_area where M = [[a_i,b_i,c_i],[a_j,b_j,c_j],[a_k,b_k,c_k]]
    
    M = bary[np.array([i, j, k])]
    return np.linalg.det(M) * REF_AREA

def _all_signed_areas(bary):
    """Compute signed areas for all C(11,3)=165 triples."""
    triples = list(itertools.combinations(range(N), 3))
    areas = np.array([_signed_area_bary(bary, i, j, k) for i, j, k in triples])
    return areas, triples

def _min_area(pts):
    """Compute minimum triangle area for Cartesian points."""
    areas = []
    for i, j, k in itertools.combinations(range(N), 3):
        pi, pj, pk = pts[i], pts[j], pts[k]
        area = 0.5 * abs((pj[0] - pi[0]) * (pk[1] - pi[1]) - (pk[0] - pi[0]) * (pj[1] - pi[1]))
        areas.append(area)
    return min(areas)

def _initial_bary():
    """Generate initial barycentric configuration using a symmetric pattern."""
    # Start with a reasonable symmetric arrangement
    # 3 corners, 3 edge midpoints, 5 interior points in a pattern
    bary = np.zeros((N, 3))
    # Corners
    bary[0] = [1, 0, 0]  # vertex A
    bary[1] = [0, 1, 0]  # vertex B
    bary[2] = [0, 0, 1]  # vertex C
    # Edge midpoints
    bary[3] = [0.5, 0.5, 0]  # midpoint AB
    bary[4] = [0.5, 0, 0.5]  # midpoint AC
    bary[5] = [0, 0.5, 0.5]  # midpoint BC
    # Interior points: use a smaller grid
    # Center and 4 points around it
    bary[6] = [1/3, 1/3, 1/3]  # centroid
    # 4 points at 1/4 positions
    bary[7] = [0.25, 0.5, 0.25]
    bary[8] = [0.5, 0.25, 0.25]
    bary[9] = [0.25, 0.25, 0.5]
    bary[10] = [0.5, 0.5, 0.0]  # this overlaps with midpoint, adjust
    bary[10] = [0.35, 0.35, 0.3]
    
    # Normalize to ensure sum = 1
    bary /= bary.sum(axis=1, keepdims=True)
    return bary

def construct():
    """
    Place 11 points in the equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2).
    Maximize the minimum area of any triangle formed by three points.
    Uses SLP with barycentric coordinates.
    """
    bary = _initial_bary()
    triples = list(itertools.combinations(range(N), 3))
    n_triples = len(triples)  # 165
    
    best_min_area = _min_area(_bary_to_cart(bary))
    
    for iteration in range(100):
        # Compute signed areas at current iterate
        signed_areas = np.array([_signed_area_bary(bary, i, j, k) for i, j, k in triples])
        
        # Determine signs (handle near-zero by defaulting to +1)
        signs = np.where(signed_areas >= 0, 1.0, -1.0)
        
        # Build LP: maximize t
        # Variables: 3*N barycentric coords (a_0..a_10, b_0..b_10, c_0..c_10) + t = 34 variables
        n_vars = 3 * N + 1
        
        # Objective: minimize -t
        c_obj = np.zeros(n_vars)
        c_obj[-1] = -1.0
        
        A_ub = []
        b_ub = []
        
        # Domain constraints: a_i + b_i + c_i = 1 (equality, handled separately)
        # a_i >= 0, b_i >= 0, c_i >= 0 (handled via bounds)
        
        # Triangle area constraints:
        # For each triple (i,j,k): sign_ijk * signed_area_ijk >= t
        # signed_area_ijk = det(M) * REF_AREA where M rows are barycentric coords of i,j,k
        # det(M) = a_i*(b_j*c_k - c_j*b_k) - b_i*(a_j*c_k - c_j*a_k) + c_i*(a_j*b_k - b_j*a_k)
        # This is linear in each barycentric coordinate!
        # Let's expand: det = a_i*b_j*c_k - a_i*c_j*b_k - b_i*a_j*c_k + b_i*c_j*a_k + c_i*a_j*b_k - c_i*b_j*a_k
        # Wait, that's not right. Let me redo.
        # det([[a_i, b_i, c_i], [a_j, b_j, c_j], [a_k, b_k, c_k]])
        # = a_i*(b_j*c_k - c_j*b_k) - b_i*(a_j*c_k - c_j*a_k) + c_i*(a_j*b_k - b_j*a_k)
        # = a_i*b_j*c_k - a_i*c_j*b_k - b_i*a_j*c_k + b_i*c_j*a_k + c_i*a_j*b_k - c_i*b_j*a_k
        
        # Hmm, this is cubic in the barycentric coordinates (products of three different points' coords).
        # So it's NOT linear in the full set of variables.
        
        # I need to linearize. At the current iterate, I'll compute the gradient of det(M) with respect to each barycentric coordinate.
        
        # For triple (i,j,k), det(M) as a function of the 6 barycentric coords (a_i,b_i,c_i,a_j,b_j,c_j,a_k,b_k,c_k):
        # d(det)/da_i = b_j*c_k - c_j*b_k
        # d(det)/db_i = -(a_j*c_k - c_j*a_k) = c_j*a_k - a_j*c_k
        # d(det)/dc_i = a_j*b_k - b_j*a_k
        # d(det)/da_j = -(b_i*c_k - c_i*b_k) = c_i*b_k - b_i*c_k
        # d(det)/db_j = a_i*c_k - c_i*a_k
        # d(det)/dc_j = -(a_i*b_k - b_i*a_k) = b_i*a_k - a_i*b_k
        # d(det)/da_k = b_i*c_j - c_i*b_j
        # d(det)/db_k = -(a_i*c_j - c_i*a_j) = c_i*a_j - a_i*c_j
        # d(det)/dc_k = a_i*b_j - b_i*a_j
        
        for idx, (i, j, k) in enumerate(triples):
            sgn = signs[idx]
            sa = signed_areas[idx]
            
            row = np.zeros(n_vars)
            b_val = -sgn * sa * REF_AREA
            
            # Gradient of det(M) with respect to each barycentric coord
            ai, bi, ci = bary[i]
            aj, bj, cj = bary[j]
            ak, bk, ck = bary[k]
            
            # d(det)/da_i
            row[i] = -sgn * (bj * ck - cj * bk)
            # d(det)/db_i
            row[N + i] = -sgn * (cj * ak - aj * ck)
            # d(det)/dc_i
            row[2*N + i] = -sgn * (aj * bk - bj * ak)
            
            # d(det)/da_j
            row[j] = -sgn * (ci * bk - bi * ck)
            # d(det)/db_j
            row[N + j] = -sgn * (ai * ck - ci * ak)
            # d(det)/dc_j
            row[2*N + j] = -sgn * (bi * ak - ai * bk)
            
            # d(det)/da_k
            row[k] = -sgn * (bi * cj - ci * bj)
            # d(det)/db_k
            row[N + k] = -sgn * (ci * aj - ai * cj)
            # d(det)/dc_k
            row[2*N + k] = -sgn * (ai * bj - bi * aj)
            
            # t term: -t on LHS
            row[3*N] = 1.0
            
            A_ub.append(row)
            b_ub.append(b_val)
        
        # Equality constraints: a_i + b_i + c_i = 1 for each point
        A_eq = []
        b_eq = []
        for i in range(N):
            row = np.zeros(n_vars)
            row[i] = 1.0
            row[N + i] = 1.0
            row[2*N + i] = 1.0
            A_eq.append(row)
            b_eq.append(1.0)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)
        
        # Bounds: 0 <= a_i, b_i, c_i <= 1, 0 <= t
        bounds = [(0, 1)] * (3 * N) + [(0, None)]
        
        # Solve LP
        try:
            result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                            bounds=bounds, method='highs')
            if result.success:
                new_bary = np.zeros((N, 3))
                new_bary[:, 0] = result.x[0:N]
                new_bary[:, 1] = result.x[N:2*N]
                new_bary[:, 2] = result.x[2*N:3*N]
                
                # Normalize to handle numerical issues
                sums = new_bary.sum(axis=1, keepdims=True)
                new_bary /= sums
                
                # Clip to [0,1]
                new_bary = np.clip(new_bary, 0, 1)
                # Re-normalize
                new_bary /= new_bary.sum(axis=1, keepdims=True)
                
                new_min = _min_area(_bary_to_cart(new_bary))
                
                if new_min > best_min_area - 1e-10:
                    best_min_area = new_min
                    bary = new_bary
                else:
                    break
            else:
                break
        except Exception:
            break
    
    return _bary_to_cart(bary)

# EVOLVE-BLOCK-END
