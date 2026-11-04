import numpy as np
import time

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 100.0

    centers = []
    for i in range(3):
        for j in range(4):
            x = j * np.sqrt(3)
            y = i * 1.5
            if i % 2 == 1:
                x += np.sqrt(3) / 2
            centers.append([x, y])
    
    centers = np.array(centers)
    center = np.mean(centers, axis=0)
    centers = centers - center

    vertices = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])

    best_L = float('inf')
    best_theta = 0.0

    for theta_deg in np.linspace(0, 30, 300):
        if time.time() > deadline:
            break
            
        theta = np.deg2rad(theta_deg)
        angles = theta + np.pi/6 + np.arange(6) * np.pi/3
        normals = np.column_stack((np.cos(angles), np.sin(angles)))
        
        # Precompute support for these 6 normals
        supports = np.array([np.max(vertices @ n) for n in normals])
        
        # Compute dot products for all centers and all normals
        # centers: 12x2, normals: 6x2
        # dots: 12x6
        dots = centers @ normals.T
        
        # Add supports (broadcast 1x6 to 12x6)
        total_projs = dots + supports
        
        # Max over all centers and normals
        max_proj = np.max(total_projs)
        
        L = max_proj * 2 / np.sqrt(3)
        
        if L < best_L:
            best_L = L
            best_theta = theta

    inner = [[c[0], c[1], 0.0] for c in centers]
    
    return inner, (0.0, 0.0), best_L, np.rad2deg(best_theta)
