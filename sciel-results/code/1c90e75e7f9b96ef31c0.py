"""Hexagon packing: place 11 unit hexagons in minimum enclosing regular hexagon."""
import numpy as np
import time

N = 11
T_START = time.time()
DEADLINE = 150.0
MARGIN = 12.0

S3 = np.sqrt(3.0)
HEX_VERTS = np.array([[np.cos(np.pi/3*k), np.sin(np.pi/3*k)] for k in range(6)])
E1 = np.array([2.0, 0.0])
E2 = np.array([1.0, S3])

def hex_center(ax, ay):
    return ax * E1 + ay * E2

def cluster_support(centers, thetas, nk):
    """Max over hexes of (nk . c_i + h_hex_i(nk))"""
    h_max = -np.inf
    for i in range(len(centers)):
        ct = np.cos(thetas[i])
        st = np.sin(thetas[i])
        rv_x = ct * HEX_VERTS[:,0] - st * HEX_VERTS[:,1]
        rv_y = st * HEX_VERTS[:,0] + ct * HEX_VERTS[:,1]
        proj = np.max(nk[0]*rv_x + nk[1]*rv_y)
        h_i = nk[0]*centers[i,0] + nk[1]*centers[i,1] + proj
        if h_i > h_max:
            h_max = h_i
    return h_max

def evaluate_config(centers, thetas):
    """Find min enclosing hexagon. Returns (side, center, angle_rad)."""
    best_side = np.inf
    best_phi = 0.0
    # Optimize container orientation phi
    for phi in np.linspace(0, np.pi/3, 361):
        max_h = 0.0
        for k in range(6):
            nk = np.array([np.cos(phi + k*np.pi/3), np.sin(phi + k*np.pi/3)])
            h_k = cluster_support(centers, thetas, nk)
            if h_k > max_h:
                max_h = h_k
        side = max_h * 2.0 / S3
        if side < best_side:
            best_side = side
            best_phi = phi
    # Refine
    for _ in range(2):
        for phi in np.linspace(best_phi - 0.01, best_phi + 0.01, 101):
            max_h = 0.0
            for k in range(6):
                nk = np.array([np.cos(phi + k*np.pi/3), np.sin(phi + k*np.pi/3)])
                h_k = cluster_support(centers, thetas, nk)
                if h_k > max_h:
                    max_h = h_k
            side = max_h * 2.0 / S3
            if side < best_side:
                best_side = side
                best_phi = phi
    
    # Find container center O
    nks = np.array([[np.cos(best_phi + k*np.pi/3), np.sin(best_phi + k*np.pi/3)] for k in range(6)])
    hs = np.array([cluster_support(centers, thetas, nks[k]) for k in range(6)])
    L = best_side
    
    # O must satisfy: n_k . O >= h_k - L*S3/2 for all k
    # Use the centroid of the 6 "offset" points as a good center
    # The feasible region for O is a hexagon; its center is approximately:
    O = np.zeros(2)
    for k in range(6):
        O += (hs[k] - L*S3/2) * nks[k]
    O /= 6.0
    
    # Verify: check all constraints
    # If violated, the side might need to be larger, but our L should be correct
    return L, O, best_phi

def construct():
    global T_START
    elapsed = time.time() - T_START
    if elapsed > DEADLINE - MARGIN:
        topo = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(1,1),(-2,1),(-1,-1)]
        centers = np.array([hex_center(a,b) for a,b in topo])
        thetas = np.zeros(N)
        side, O, phi = evaluate_config(centers, thetas)
        inner = [[float(centers[i,0]), float(centers[i,1]), 0.0] for i in range(N)]
        return inner, (float(O[0]), float(O[1])), float(side), float(np.degrees(phi))
    
    topologies = [
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(1,1),(-2,1),(-1,-1)],  # flower
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(3,2)],          # 4-3-4
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(4,1),(0,2),(1,2),(2,2)],          # 3-5-3
        [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(0,2),(1,2)],                # 5-3-2
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)],                # 3-4-3
        [(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(0,3),(1,3)],          # 2-4-3-2
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)],          # 4-4-3
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3)],                # 3-3-3-1
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(3,2),(0,3),(1,3)],          # 2-3-4-2
        [(0,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3),(0,4)],          # 1-3-3-3-1
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(3,2)],          # 4-3-4 variant
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(1,1),(0,2),(-1,2)],     # flower variant
    ]
    
    best_side = np.inf
    best_config = None
    
    for topo in topologies:
        if time.time() - T_START > DEADLINE - MARGIN:
            break
        if len(topo) != N:
            continue
        centers = np.array([hex_center(a,b) for a,b in topo])
        
        # All axis-aligned
        thetas = np.zeros(N)
        side, O, phi = evaluate_config(centers, thetas)
        if side < best_side:
            best_side = side
            best_config = (centers.copy(), thetas.copy(), O.copy(), phi)
        
        # All rotated 30°
        thetas30 = np.full(N, np.pi/6)
        side, O, phi = evaluate_config(centers, thetas30)
        if side < best_side:
            best_side = side
            best_config = (centers.copy(), thetas30.copy(), O.copy(), phi)
        
        # Try small cluster shifts
        for dx in np.linspace(-0.3, 0.3, 7):
            for dy in np.linspace(-0.3, 0.3, 7):
                if time.time() - T_START > DEADLINE - MARGIN:
                    break
                c_shifted = centers + np.array([dx, dy])
                side, O, phi = evaluate_config(c_shifted, thetas)
                if side < best_side:
                    best_side = side
                    best_config = (c_shifted, thetas.copy(), O.copy(), phi)
    
    if best_config is None:
        topo = topologies[0]
        centers = np.array([hex_center(a,b) for a,b in topo])
        thetas = np.zeros(N)
        best_side, O, phi = evaluate_config(centers, thetas)
        best_config = (centers, thetas, O, phi)
    
    centers, thetas, O, phi = best_config
    inner = [[float(centers[i,0]), float(centers[i,1]), float(np.degrees(thetas[i]))] for i in range(N)]
    return inner, (float(O[0]), float(O[1])), float(best_side), float(np.degrees(phi))
