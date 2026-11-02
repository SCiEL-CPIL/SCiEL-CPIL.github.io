"""Rewritten program for Heilbronn triangles; SLP with barycentric coordinates."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
import itertools
import time

def construct():
    """
    Place 11 points in the equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2).
    Maximize the minimum area of any triangle formed by three points.
    Uses a Sequence of Linear Programs (SLP) approach in barycentric coordinates.
    
    In barycentric coordinates (u_i, v_i, w_i) with u_i + v_i + w_i = 1, u_i, v_i, w_i >= 0:
    The area of triangle (i,j,k) is proportional to |det of the 3x3 matrix of barycentric coords|.
    Specifically, area_ijk = (sqrt(3)/4) * |E_ijk| where
    E_ijk = u_i*(v_j*w_k - v_k*w_j) + v_i*(w_j*u_k - w_k*u_j) + w_i*(u_j*v_k - u_k*v_j)
    
    This is bilinear, so we linearize at each SLP iteration.
    """
    sqrt3 = np.sqrt(3)
    
    def bary_to_cart(u, v, w):
        """Convert barycentric (u,v,w) to Cartesian (x,y)."""
        x = v + 0.5 * w
        y = (sqrt3 / 2) * w
        return x, y
    
    def det_ijk(uv1, uv2, uv3):
        """Compute the signed determinant for area calculation.
        uv1, uv2, uv3 are (u, v) pairs (w = 1 - u - v)."""
        u1, v1 = uv1
        u2, v2 = uv2
        u3, v3 = uv3
        w1 = 1.0 - u1 - v1
        w2 = 1.0 - u2 - v2
        w3 = 1.0 - u3 - v3
        # det of [[u1,v1,w1],[u2,v2,w2],[u3,v3,w3]]
        # = u1*(v2*w3 - v3*w2) - v1*(u2*w3 - u3*w2) + w1*(u2*v3 - u3*v2)
        return u1 * (v2 * w3 - v3 * w2) - v1 * (u2 * w3 - u3 * w2) + w1 * (u2 * v3 - u3 * v2)
    
    def compute_min_det(uv):
        """Compute minimum absolute determinant over all triples."""
        min_val = np.inf
        for i, j, k in itertools.combinations(range(N), 3):
            d = det_ijk(uv[i], uv[j], uv[k])
            min_val = min(min_val, abs(d))
        return min_val
    
    # Initial configuration: symmetric arrangement
    # Use a grid in barycentric coordinates
    # 11 points: try to spread them out
    # Start with a reasonable symmetric configuration
    uv = np.zeros((N, 2))
    
    # 3 corners: (1,0), (0,1), (0,0) in (u,v) => w = 1-u-v
    # (1,0) => u=1,v=0,w=0 => corner A
    # (0,1) => u=0,v=1,w=0 => corner B  
    # (0,0) => u=0,v=0,w=1 => corner C
    uv[0] = [1.0, 0.0]  # A
    uv[1] = [0.0, 1.0]  # B
    uv[2] = [0.0, 0.0]  # C
    
    # 3 edge midpoints
    uv[3] = [0.5, 0.5]  # AB midpoint (u=0.5, v=0.5, w=0)
    uv[4] = [0.5, 0.0]  # AC midpoint (u=0.5, v=0, w=0.5)
    uv[5] = [0.0, 0.5]  # BC midpoint (u=0, v=0.5, w=0.5)
    
    # 4 interior points - spread out
    # Centroid and three points around it
    uv[6] = [1/3, 1/3]  # centroid (u=1/3, v=1/3, w=1/3)
    # Three more interior points
    uv[7] = [0.4, 0.3]
    uv[8] = [0.3, 0.4]
    uv[9] = [0.35, 0.35]
    uv[10] = [0.25, 0.25]
    
    # Precompute triples
    triples = list(itertools.combinations(range(N), 3))
    
    # SLP iterations
    max_iter = 100
    timeout = time.time() + 110  # Leave 10s buffer
    
    for iteration in range(max_iter):
        if time.time() > timeout:
            break
        
        # Compute signed determinants and signs
        signed_dets = []
        signs = []
        for i, j, k in triples:
            d = det_ijk(uv[i], uv[j], uv[k])
            signed_dets.append(d)
            signs.append(1 if d >= 0 else -1)
        
        # Linearize each determinant around current point
        # E_ijk = u_i*(v_j*w_k - v_k*w_j) - v_i*(u_j*w_k - u_k*w_j) + w_i*(u_j*v_k - u_k*v_j)
        # where w_m = 1 - u_m - v_m
        #
        # Let's expand in terms of u and v only (w = 1 - u - v):
        # This is complex. Let's compute gradients numerically or analytically.
        #
        # Actually, let's use the product rule for the linearization.
        # At current point, E = E0.
        # Linear approximation: E ≈ E0 + sum over variables of (dE/dvar) * (var - var0)
        #
        # Variables: u_0..u_10, v_0..v_10 (22 variables) + t (1 variable) = 23 total
        #
        # For triple (i,j,k), let's compute dE/du_m and dE/dv_m for m in {i,j,k}
        # (other variables have zero derivative for this triple)
        
        n_vars = 2 * N + 1  # u_0..u_10, v_0..v_10, t
        # Variable indices: 0..10 for u, 11..21 for v, 22 for t
        
        c = np.zeros(n_vars)
        c[-1] = -1  # minimize -t => maximize t
        
        A_ub = []
        b_ub = []
        
        # Domain constraints in barycentric coords:
        # u_i >= 0 => -u_i <= 0
        # v_i >= 0 => -v_i <= 0
        # w_i = 1 - u_i - v_i >= 0 => u_i + v_i <= 1
        for i in range(N):
            # -u_i <= 0
            row = np.zeros(n_vars)
            row[i] = -1
            A_ub.append(row)
            b_ub.append(0)
            
            # -v_i <= 0
            row = np.zeros(n_vars)
            row[11 + i] = -1
            A_ub.append(row)
            b_ub.append(0)
            
            # u_i + v_i <= 1
            row = np.zeros(n_vars)
            row[i] = 1
            row[11 + i] = 1
            A_ub.append(row)
            b_ub.append(1)
        
        # Area constraints: sign * E_linear >= t
        # sign * (E0 + sum(grad * delta)) >= t
        # => -sign * E0 - sign * sum(grad * delta) + t <= 0
        
        for idx, (i, j, k) in enumerate(triples):
            E0 = signed_dets[idx]
            sgn = signs[idx]
            
            # Compute gradients for this triple
            # E = u_i*(v_j*w_k - v_k*w_j) - v_i*(u_j*w_k - u_k*w_j) + w_i*(u_j*v_k - u_k*v_j)
            # where w_m = 1 - u_m - v_m
            #
            # Let's compute dE/du_m and dE/dv_m for m in {i,j,k}
            # Use the chain rule. Let f_m = v_m * w_n - v_n * w_m (for appropriate n)
            #
            # Actually, let's just compute the gradient by expanding:
            # E = u_i*(v_j*(1-u_k-v_k) - v_k*(1-u_j-v_j)) - v_i*(u_j*(1-u_k-v_k) - u_k*(1-u_j-v_j)) + (1-u_i-v_i)*(u_j*v_k - u_k*v_j)
            #
            # This is getting complex. Let me compute dE/du_m and dE/dv_m directly.
            
            ui, vi = uv[i]
            uj, vj = uv[j]
            uk, vk = uv[k]
            wi = 1 - ui - vi
            wj = 1 - uj - vj
            wk = 1 - uk - vk
            
            # E = ui*(vj*wk - vk*wj) - vi*(uj*wk - uk*wj) + wi*(uj*vk - uk*vj)
            
            # dE/dui = (vj*wk - vk*wj) + wi * d(wi)/dui * (uj*vk - uk*vj) ... wait
            # Actually: dE/dui = (vj*wk - vk*wj) + (1-ui-vi) * 0 + d(wi)/dui * (uj*vk - uk*vj)
            # d(wi)/dui = -1
            # So dE/dui = (vj*wk - vk*wj) - (uj*vk - uk*vj)
            
            # dE/dvi = -(uj*wk - uk*wj) + d(wi)/dvi * (uj*vk - uk*vj)
            # d(wi)/dvi = -1
            # So dE/dvi = -(uj*wk - uk*wj) - (uj*vk - uk*vj)
            
            # dE/duj: 
            # ui*(vj*wk - vk*wj) contributes: ui * (-vk * d(wj)/duj) = ui * (-vk * (-1)) = ui*vk
            # Wait, let me be more careful.
            # Term 1: ui*(vj*wk - vk*wj). d/duj = ui*(0 - vk*d(wj)/duj) = ui*(0 - vk*(-1)) = ui*vk
            # Term 2: -vi*(uj*wk - uk*wj). d/duj = -vi*(1*wk - uk*d(wj)/duj) = -vi*(wk - uk*(-1)) = -vi*(wk + uk)
            # Term 3: wi*(uj*vk - uk*vj). d/duj = wi*(1*vk - 0) = wi*vk
            # So dE/duj = ui*vk - vi*(wk + uk) + wi*vk
            
            # dE/dvj:
            # Term 1: ui*(vj*wk - vk*wj). d/dvj = ui*(1*wk - vk*d(wj)/dvj) = ui*(wk - vk*(-1)) = ui*(wk + vk)
            # Term 2: -vi*(uj*wk - uk*wj). d/dvj = 0
            # Term 3: wi*(uj*vk - uk*vj). d/dvj = wi*(0 - uk*1) = -wi*uk
            # So dE/dvj = ui*(wk + vk) - wi*uk
            
            # dE/duk:
            # Term 1: ui*(vj*wk - vk*wj). d/duk = ui*(vj*d(wk)/duk - 0) = ui*(vj*(-1)) = -ui*vj
            # Term 2: -vi*(uj*wk - uk*wj). d/duk = -vi*(uj*d(wk)/duk - 1*wj) = -vi*(uj*(-1) - wj) = -vi*(-uj - wj) = vi*(uj + wj)
            # Term 3: wi*(uj*vk - uk*vj). d/duk = wi*(0 - 1*vj) = -wi*vj
            # So dE/duk = -ui*vj + vi*(uj + wj) - wi*vj
            
            # dE/dvk:
            # Term 1: ui*(vj*wk - vk*wj). d/dvk = ui*(0 - 1*wj) = -ui*wj
            # Term 2: -vi*(uj*wk - uk*wj). d/dvk = 0
            # Term 3: wi*(uj*vk - uk*vj). d/dvk = wi*(uj*1 - 0) = wi*uj
            # So dE/dvk = -ui*wj + wi*uj
            
            dE_dui = (vj*wk - vk*wj) - (uj*vk - uk*vj)
            dE_dvi = -(uj*wk - uk*wj) - (uj*vk - uk*vj)
            dE_duj = ui*vk - vi*(wk + uk) + wi*vk
            dE_dvj = ui*(wk + vk) - wi*uk
            dE_duk = -ui*vj + vi*(uj + wj) - wi*vj
            dE_dvk = -ui*wj + wi*uj
            
            # Build constraint: -sgn*E0 - sgn*(sum of grad*delta) + t <= 0
            row = np.zeros(n_vars)
            b_val = -sgn * E0
            
            # u_i
            row[i] = -sgn * dE_dui
            # v_i
            row[11 + i] = -sgn * dE_dvi
            # u_j
            row[j] = -sgn * dE_duj
            # v_j
            row[11 + j] = -sgn * dE_dvj
            # u_k
            row[k] = -sgn * dE_duk
            # v_k
            row[11 + k] = -sgn * dE_dvk
            # t
            row[-1] = 1
            
            A_ub.append(row)
            b_ub.append(b_val)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Bounds: u_i, v_i in [0, 1], t >= 0
        bounds = []
        for i in range(N):
            bounds.append((0, 1))  # u_i
        for i in range(N):
            bounds.append((0, 1))  # v_i
        bounds.append((0, None))  # t
        
        # Solve LP
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if result.success:
                new_uv = np.column_stack([result.x[:N], result.x[N:2*N]])
                
                # Check if this is an improvement in actual minimum determinant
                old_min = compute_min_det(uv)
                new_min = compute_min_det(new_uv)
                
                if new_min > old_min - 1e-10:
                    uv = new_uv
                else:
                    break
            else:
                break
        except Exception:
            break
    
    # Convert final barycentric coordinates to Cartesian
    points = np.zeros((N, 2))
    for i in range(N):
        u, v = uv[i]
        w = 1 - u - v
        x = v + 0.5 * w
        y = (sqrt3 / 2) * w
        points[i, 0] = x
        points[i, 1] = y
    
    return points

# EVOLVE-BLOCK-END
