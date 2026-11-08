import numpy as np
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    # Positions for 11 hexagons
    # 1 center
    # 6 ring 1 (honeycomb neighbors)
    # 4 ring 2 (corners of a larger hexagon)
    
    pos = []
    pos.append([0.0, 0.0])
    
    # Ring 1: Honeycomb neighbors
    # Flat-top hexagons, neighbors are at 30, 90, 150, 210, 270, 330 degrees
    # Distance sqrt(3)
    # Vector u = (3/2, sqrt(3)/2), v = (0, sqrt(3))
    # Neighbors:
    # (3/2, sqrt(3)/2)
    # (3/2, -sqrt(3)/2)
    # (0, sqrt(3))
    # (-3/2, sqrt(3)/2)
    # (-3/2, -sqrt(3)/2)
    # (0, -sqrt(3))
    
    u = np.array([1.5, np.sqrt(3)/2])
    v = np.array([0.0, np.sqrt(3)])
    
    ring1 = [
        u,
        np.array([1.5, -np.sqrt(3)/2]),
        v,
        -u + v, # (-1.5, sqrt(3)/2)
        -u - v, # (-1.5, -sqrt(3)/2)
        -v
    ]
    
    for p in ring1:
        pos.append([p[0], p[1]])
        
    # Ring 2: Pick 4 corners
    # We want to minimize the bounding box.
    # The honeycomb lattice grows.
    # Let's pick (2,0), (0,2), (-2,0), (0,-2) in lattice coords (i,j)
    # P = i*u + j*v
    # (2,0) -> 2u = (3, sqrt(3))
    # (0,2) -> 2v = (0, 2sqrt(3))
    # (-2,0) -> -2u = (-3, -sqrt(3))
    # (0,-2) -> -2v = (0, -2sqrt(3))
    
    ring2 = [
        2*u,
        2*v,
        -2*u,
        -2*v
    ]
    
    for p in ring2:
        pos.append([p[0], p[1]])
        
    pos = np.array(pos)
    
    # Calculate S
    # Using the logic from before
    normals = np.array([
        [0, 1],
        [0, -1],
        [np.sqrt(3), 1],
        [np.sqrt(3), -1],
        [-np.sqrt(3), 1],
        [-np.sqrt(3), -1]
    ])
    consts = np.array([
        np.sqrt(3)/2,
        np.sqrt(3)/2,
        np.sqrt(3),
        np.sqrt(3),
        np.sqrt(3),
        np.sqrt(3)
    ])
    
    hex_verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    def calc_S(pos):
        verts = pos[:, np.newaxis, :] + hex_verts[np.newaxis, :, :]
        max_s = 0.0
        for m in range(6):
            proj = verts @ normals[m]
            s_vals = proj / consts[m]
            max_s = max(max_s, np.max(s_vals))
        return max_s

    final_S = calc_S(pos)
    
    inner = []
    for i in range(N):
        inner.append([pos[i][0], pos[i][1], 0.0])
        
    return inner, (0.0, 0.0), final_S, 0.0
