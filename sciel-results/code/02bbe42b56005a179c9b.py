import numpy as np
import math

N = 12

def hex_to_cart(q, r):
    a = math.sqrt(3)
    x = a * (q + r / 2.0)
    y = a * (math.sqrt(3) / 2.0) * r
    return x, y

def get_cluster_points(coords):
    return [hex_to_cart(q, r) for q, r in coords]

def calc_min_side(points):
    pts = np.array(points)
    min_s_prime = float('inf')
    best_theta = 0
    for theta_deg in np.linspace(0, 30, 601):
        theta = math.radians(theta_deg)
        c, s = math.cos(theta), math.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        rotated_pts = pts @ rot.T
        
        max_proj = 0
        for k in range(6):
            angle = math.radians(k * 60)
            ux, uy = math.cos(angle), math.sin(angle)
            u = np.array([ux, uy])
            projs = rotated_pts @ u
            width = projs.max() - projs.min()
            if width > max_proj:
                max_proj = width
        
        s_prime = max_proj / math.sqrt(3)
        if s_prime < min_s_prime:
            min_s_prime = s_prime
            best_theta = theta_deg
            
    return min_s_prime, best_theta

# Clusters
cluster1 = [
    (0,0),
    (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
    (2,0), (1,1), (0,2), (-1,2), (-2,1)
]

cluster4 = [
    (0,0),
    (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
    (1,1), (2,-1), (1,-2), (-1,-1), (-2,1)
]

best_s = float('inf')
best_cluster_pts = None
best_theta = 0

for cluster in [cluster1, cluster4]:
    pts = get_cluster_points(cluster)
    s_prime, theta = calc_min_side(pts)
    if s_prime < best_s:
        best_s = s_prime
        best_cluster_pts = pts
        best_theta = theta

final_side = best_s + 1.0

c, s = math.cos(math.radians(best_theta)), math.sin(math.radians(best_theta))
rot = np.array([[c, -s], [s, c]])
final_pts = np.array(best_cluster_pts) @ rot.T

inner = []
for x, y in final_pts:
    inner.append([float(x), float(y), 0.0])

# EVOLVE-BLOCK-START
def construct():
    return inner, (0.0, 0.0), final_side, 0.0
# EVOLVE-BLOCK-END
