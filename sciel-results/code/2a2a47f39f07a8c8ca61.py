"""Program for packing 200 equal circles in a unit square using Hexagonal Packing."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # We will try to fit a hexagonal lattice into the square [-0.5, 0.5]^2.
    # The lattice is defined by horizontal spacing dx and vertical spacing dy = dx * sqrt(3) / 2.
    # The nearest neighbor distance is dx, so the radius is r = dx / 2.
    # We want to maximize dx.
    
    # To fit N points, we can arrange them in a grid of R rows and C columns.
    # R * C >= N.
    # The width of the grid is (C - 1) * dx.
    # The height of the grid is (R - 1) * dy.
    # We need (C - 1) * dx <= 1.0 and (R - 1) * dy <= 1.0.
    # Also, the grid must be centered.
    # dx <= 1.0 / (C - 1)
    # dy <= 1.0 / (R - 1) => dx * sqrt(3) / 2 <= 1.0 / (R - 1) => dx <= 2.0 / ((R - 1) * sqrt(3))
    # So dx_max = min(1.0 / (C - 1), 2.0 / ((R - 1) * sqrt(3)))
    # We need to find R, C such that R * C >= 200 and dx_max is maximized.
    
    best_dx = 0
    best_R = 0
    best_C = 0
    
    # Search for optimal R and C
    # R and C should be around sqrt(200) ~ 14
    for R in range(10, 25):
        for C in range(10, 25):
            if R * C < N:
                continue
            if R == 1 or C == 1:
                continue
            
            dx_w = 1.0 / (C - 1)
            dx_h = 2.0 / ((R - 1) * np.sqrt(3))
            dx_candidate = min(dx_w, dx_h)
            
            if dx_candidate > best_dx:
                best_dx = dx_candidate
                best_R = R
                best_C = C
                
    R = best_R
    C = best_C
    dx = best_dx
    dy = dx * np.sqrt(3) / 2
    
    # Generate the hexagonal grid
    # Center the grid in the square
    # x positions: - (C-1)*dx/2 + i*dx
    # y positions: - (R-1)*dy/2 + j*dy
    # For odd j, shift x by dx/2
    
    centers_list = []
    for j in range(R):
        for i in range(C):
            x = -(C - 1) * dx / 2 + i * dx
            y = -(R - 1) * dy / 2 + j * dy
            if j % 2 == 1:
                x += dx / 2
            centers_list.append([x, y])
            
    centers = np.array(centers_list)
    
    # We have R*C points. We need exactly N=200.
    # If R*C > N, we have extra points. We should remove the ones that are least "central" or just take the first N?
    # Taking the first N might bias the distribution.
    # A better strategy: Generate the full grid, then select the N points that are closest to the center (0,0)?
    # Or simply, since the grid is symmetric, we can just take a subset that maintains symmetry if possible.
    # However, for simplicity and robustness, let's just take the first N points from the list.
    # But wait, if we take the first N, we might cut off the right/bottom edges, reducing the effective packing density or causing boundary issues if the grid is larger than the square?
    # No, the grid is constructed to fit within the square.
    # If we have extra points, they are inside the square.
    # We just need to return 200 of them.
    # To maximize the minimum distance, we should probably keep the most "interior" points or just any 200?
    # Actually, if we have a grid of 225 points (15x15) and we drop 25, the radius is determined by the nearest neighbors of the *remaining* points.
    # The nearest neighbors are still at distance dx.
    # So the radius is still dx/2, provided no two remaining points are closer than dx.
    # In a hex grid, all neighbors are at distance dx.
    # So as long as we don't remove points in a way that creates new closer pairs (which is impossible, removing points only increases distances), the radius is at least dx/2.
    # The boundary constraint: min_i min_k (0.5 - |c_ik|).
    # The points are within the square.
    # So the radius is determined by the nearest neighbor distance, which is dx.
    # So r = dx/2.
    
    # We just need to pick 200 points.
    # To be safe, let's pick the 200 points closest to the center (0,0).
    # This ensures the "core" of the packing is preserved.
    dist_to_center = np.sum(centers**2, axis=1)
    sorted_indices = np.argsort(dist_to_center)
    centers = centers[sorted_indices[:N]]
    
    # Final check and clamp
    centers = np.clip(centers, -0.5, 0.5)
    
    return centers
# EVOLVE-BLOCK-END
