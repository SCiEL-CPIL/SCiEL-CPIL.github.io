import numpy as np

N = 12

def construct():
    # Lattice parameters
    d = np.sqrt(3)
    e1 = np.array([d, 0])
    e2 = np.array([d/2, d*np.sqrt(3)/2])
    
    centers = []
    for i in range(3):
        for j in range(4):
            c = i * e1 + j * e2
            centers.append(c)
            
    centers = np.array(centers)
    centroid = np.mean(centers, axis=0)
    centers = centers - centroid
    
    # Normals of the outer hexagon (fixed at 0)
    n0 = np.array([1.0, 0.0])
    n1 = np.array([0.5, np.sqrt(3)/2])
    n2 = np.array([-0.5, np.sqrt(3)/2])
    normals = np.array([n0, n1, n2])
    
    def calc_A_prime(theta):
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        rotated_centers = centers @ R.T
        projs = rotated_centers @ normals.T
        max_abs_projs = np.max(np.abs(projs), axis=0)
        return np.max(max_abs_projs)
    
    # Optimization
    # The function has period pi/3?
    # The cluster has 2-fold symmetry. The hexagon has 6-fold.
    # So we search in [0, pi/6] is enough?
    # Let's search [0, pi/3] to be safe.
    
    thetas = np.linspace(0, np.pi/3, 2000)
    A_primes = np.array([calc_A_prime(t) for t in thetas])
    min_idx = np.argmin(A_primes)
    theta_best = thetas[min_idx]
    
    # Refine
    thetas_fine = np.linspace(theta_best - 0.005, theta_best + 0.005, 1000)
    A_primes_fine = np.array([calc_A_prime(t) for t in thetas_fine])
    min_idx_fine = np.argmin(A_primes_fine)
    theta_opt = thetas_fine[min_idx_fine]
    A_prime_opt = A_primes_fine[min_idx_fine]
    
    A_opt = A_prime_opt + np.sqrt(3)/2
    S_opt = 2 * A_opt / np.sqrt(3)
    
    # Final centers
    c = np.cos(theta_opt)
    s = np.sin(theta_opt)
    R = np.array([[c, -s], [s, c]])
    final_centers = centers @ R.T
    
    inner = []
    angle_deg = np.degrees(theta_opt)
    for c_pt in final_centers:
        inner.append([float(c_pt[0]), float(c_pt[1]), float(angle_deg)])
        
    return inner, (0.0, 0.0), float(S_opt), 0.0
