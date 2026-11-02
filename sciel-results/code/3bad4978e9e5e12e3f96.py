import numpy as np
import time

N = 12
DEADLINE = 150.0
t0 = time.time()

def hex_vertices(center, angle_deg=0.0):
    # returns 6 vertices of a unit regular hexagon (side 1) centered at center, rotated by angle_deg
    a = np.deg2rad(angle_deg)
    # base vertices at angles 0,60,...,300 (vertex at 0)
    base = np.array([[np.cos(np.pi*k/3), np.sin(np.pi*k/3)] for k in range(6)])
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s],[s, c]])
    return center + base @ R.T  # (6,2)

def lattice_point(i, j):
    e1 = np.array([1.5, np.sqrt(3)/2])
    e2 = np.array([0.0, np.sqrt(3)])
    return i*e1 + j*e2

def build_clusters():
    # returns list of lists of (x,y) centers, all with angle 0
    clusters = []
    # C1: 7 + 5 balanced ring2
    ring1 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
    ring2_bal = [(2,0),(0,2),(-2,2),(-2,0),(0,-2)]
    c1 = ring1 + ring2_bal
    clusters.append([lattice_point(i,j) for (i,j) in c1])
    # C2: 7 + 5 other distribution
    ring2b = [(1,1),(0,2),(-2,2),(-2,0),(1,-2)]
    clusters.append([lattice_point(i,j) for (i,j) in ring1+ring2b])
    # C3: 2x4 parallelogram
    c3 = [(i,j) for i in range(4) for j in range(2)]
    clusters.append([lattice_point(i,j) for (i,j) in c3])
    # C4: 3x4 parallelogram
    c4 = [(i,j) for i in range(3) for j in range(4)]
    clusters.append([lattice_point(i,j) for (i,j) in c4])
    # C5: (2,1,1) hexagon - construct
    # ... 
    return clusters
