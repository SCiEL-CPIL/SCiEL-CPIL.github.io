import numpy as np
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    # Positions for 11 hexagons
    # 1 center
    # 6 ring 1
    # 4 ring 2 (corners)
    
    pos = []
    pos.append([0.0, 0.0])
    
    # Ring 1
    for i in range(6):
        angle = i * np.pi / 3
        pos.append([np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)])
        
    # Ring 2: Pick 4 corners
    # Corners are at angles 0, 60, 120, 180, 240, 300
    # We pick 0, 60, 120, 180
    for i in range(4):
        angle = i * np.pi / 3
        pos.append([2 * np.sqrt(3) * np.cos(angle), 2 * np.sqrt(3) * np.sin(angle)])
        
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
