import numpy as np
import time

N = 26

def calc_radii(P):
    """
    Calculate maximum feasible radii for given centers P.
    P: (N, 2) array of centers.
    Returns: (N,) array of radii.
    """
    # Distances to walls
    # x, y in [0, 1]
    # Fix: Use np.minimum for element-wise minimum
    dist_walls = np.minimum(P, 1 - P)
    dist_walls = np.min(dist_walls, axis=1) # (N,)
    
    # Pairwise distances
    # D[i, j] = ||P[i] - P[j]||
    D = np.linalg.norm(P[:, np.newaxis, :] - P[np.newaxis, :, :], axis=2)
    
    # Initialize radii with wall constraints
    R = dist_walls.copy()
    
    # Iteratively update to satisfy circle-circle constraints
    # R[i] <= D[i, j] - R[j] for all j != i
    # R[i] = min(R[i], min_j (D[i, j] - R[j]))
    
    for _ in range(20):
        # M[i, j] = D[i, j] - R[j]
        M = D - R[np.newaxis, :]
        # Set diagonal to infinity so it doesn't affect min
        np.fill_diagonal(M, np.inf)
        # New R is min of current R (walls) and min over j
        R_new = np.minimum(R, np.min(M, axis=1))
        if np.allclose(R, R_new):
            R = R_new
            break
        R = R_new
        
    return R

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    best_score = 0.0
    best_P = None
    best_R = None
    
    # Helper to create random initial positions
    def random_init():
        # Start with a grid-like structure for better initial packing
        # 5x5 grid = 25 points. Add 1 random point.
        # Or just random. Let's try random with some structure.
        P = np.random.uniform(0.1, 0.9, (N, 2))
        return P

    # Objective function for Simulated Annealing
    # We want to maximize sum(radii).
    # Penalty for overlaps: sum(max(0, r_i + r_j - d_ij)^2)
    # Penalty for wall violation: sum(max(0, r_i - dist_to_wall)^2)
    def objective(P):
        R = calc_radii(P)
        # If any radius is negative or zero, it's a strong penalty
        if np.any(R <= 0):
            return -1e10
        
        # Calculate overlaps based on the *ideal* radii if they were to touch?
        # No, the standard approach is:
        # For a given P, the max radii are R.
        # The "energy" we want to minimize is the loss from a target radius?
        # Actually, for packing, maximizing sum(R) is the goal.
        # Simulated Annealing usually minimizes energy.
        # Let Energy = -sum(R) + Penalty(Invalidity)
        # But calc_radii already ensures validity (disjoint and inside) by construction.
        # So R is always valid.
        # Therefore, we just maximize sum(R).
        # So SA should minimize -sum(R).
        
        score = np.sum(R)
        return -score

    def local_search_sa(P_init, steps=50000):
        P = P_init.copy()
        current_energy = objective(P)
        best_P_local = P.copy()
        best_energy_local = current_energy
        
        T = 0.05 # Initial temperature
        T_min = 0.001
        cooling_rate = 0.9995
        
        for s in range(steps):
            if time.time() > deadline:
                break
                
            # Pick a random circle
            k = np.random.randint(N)
            P_old = P[k].copy()
            
            # Perturb position
            step_size = 0.05 * (T / 0.05) # Scale step with temp? Or fixed?
            # Let's use a fixed step size for simplicity, maybe adapt later
            P[k] += np.random.normal(0, 0.02, 2)
            P[k] = np.clip(P[k], 0, 1)
            
            new_energy = objective(P)
            
            # Acceptance criterion
            if new_energy < current_energy:
                current_energy = new_energy
                P[k] = P[k].copy() # Keep change
                if new_energy < best_energy_local:
                    best_energy_local = new_energy
                    best_P_local = P.copy()
            else:
                # Accept with probability exp(-delta_E / T)
                delta_E = new_energy - current_energy
                if delta_E > 0:
                    prob = np.exp(-delta_E / T)
                    if np.random.rand() < prob:
                        current_energy = new_energy
                        P[k] = P[k].copy()
                    else:
                        P[k] = P_old
                else:
                    # Should not happen if new_energy < current_energy handled above
                    current_energy = new_energy
                    P[k] = P[k].copy()
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
        return best_P_local, best_energy_local

    # Main loop
    # Run multiple restarts
    num_restarts = 5
    
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        P_init = random_init()
        
        # If we have a previous best, use it as a starting point for later restarts?
        # Or just random. Let's mix.
        if best_P is not None and i > 0:
             # Perturb best slightly
             P_init = best_P + np.random.normal(0, 0.05, (N, 2))
             P_init = np.clip(P_init, 0, 1)
        
        P_candidate, energy_candidate = local_search_sa(P_init)
        
        # Calculate actual score for the candidate
        R_candidate = calc_radii(P_candidate)
        score_candidate = np.sum(R_candidate)
        
        if score_candidate > best_score:
            best_score = score_candidate
            best_P = P_candidate
            best_R = R_candidate
            current_best_P = P_candidate.copy()
            
    # Final polish on the best solution found
    if best_P is not None and time.time() < deadline - 5.0:
        # Do a very fine-grained search around the best
        P_final, energy_final = local_search_sa(best_P, steps=50000)
        R_final = calc_radii(P_final)
        score_final = np.sum(R_final)
        
        if score_final > best_score:
            best_score = score_final
            best_P = P_final
            best_R = R_final

    # Ensure valid output
    # If best_P is None, fallback to a simple grid
    if best_P is None:
        best_P = np.random.uniform(0.1, 0.9, (N, 2))
        best_R = calc_radii(best_P)

    result = np.column_stack((best_P, best_R))
    return result
