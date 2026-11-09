import numpy as np
import time
import scipy.optimize as opt

N = 50
DEADLINE = 110.0  # Safety margin from 120s

def construct():
    start_time = time.time()
    
    def get_ring_initial():
        counts = [1, 6, 12, 18, 13]
        points = []
        points.append([0.0, 0.0])
        
        # Radii estimates for r ~ 0.126
        radii = [0.0, 0.25, 0.45, 0.65, 0.85]
        
        for k, count in enumerate(counts):
            r_ring = radii[k]
            if count == 0:
                continue
            angles = np.linspace(0, 2*np.pi, count, endpoint=False)
            if k % 2 == 1:
                angles += np.pi / count
            for a in angles:
                points.append([r_ring * np.cos(a), r_ring * np.sin(a)])
        
        return np.array(points)

    def penalty(c, r):
        centers = c.reshape(N, 2)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1e10)
        
        overlap_viol = np.maximum(0, 2*r - dists)
        overlap_pen = np.sum(overlap_viol**2)
        
        norms = np.linalg.norm(centers, axis=1)
        bound_viol = np.maximum(0, norms - (1 - r))
        bound_pen = np.sum(bound_viol**2)
        
        return overlap_pen + bound_pen

    best_centers = None
    best_r = 0.0
    
    low = 0.12
    high = 0.13
    
    n_restarts = 3
    
    for i in range(12):
        if time.time() > DEADLINE - 20:
            break
            
        mid = (low + high) / 2.0
        
        found_feasible = False
        best_pen = np.inf
        best_x = None
        
        for restart in range(n_restarts):
            if time.time() > DEADLINE - 10:
                break
                
            initial_c = get_ring_initial()
            noise_scale = 0.02
            initial_c += np.random.normal(0, noise_scale, initial_c.shape)
            
            norms = np.linalg.norm(initial_c, axis=1)
            scale = np.where(norms > (1 - mid), (1 - mid) / norms, 1.0)
            initial_c *= scale[:, np.newaxis]
            
            c0 = initial_c.flatten()
            
            try:
                res = opt.minimize(
                    penalty, 
                    c0, 
                    args=(mid,), 
                    method='Nelder-Mead', 
                    options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-7}
                )
                
                if res.fun < best_pen:
                    best_pen = res.fun
                    best_x = res.x
                    
                if best_pen < 1e-6:
                    found_feasible = True
                    break
                    
            except Exception:
                pass
        
        if found_feasible and best_x is not None:
            low = mid
            best_r = mid
            best_centers = best_x.reshape(N, 2)
        else:
            high = mid
            
    if best_centers is None:
        return get_ring_initial()
        
    return best_centers

if __name__ == "__main__":
    c = construct()
    if len(c) == N:
        min_dist = 1e10
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(c[i] - c[j])
                if d < min_dist:
                    min_dist = d
        min_bound = 1e10
        for i in range(N):
            d = 1 - np.linalg.norm(c[i])
            if d < min_bound:
                min_bound = d
        r = min(min_dist, min_bound) / 2
        print(f"Radius: {r}")
