import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_score = 0.0
    best_params = None

    # Helper to check validity and get score
    def get_score(params):
        X = params[:2*N].reshape(N, 2)
        R = params[2*N:]
        
        # Check boundaries
        if np.any(R < 0):
            return -np.inf
        if np.any(X[:, 0] - R < -1e-8) or np.any(X[:, 0] + R > 1 + 1e-8):
            return -np.inf
        if np.any(X[:, 1] - R < -1e-8) or np.any(X[:, 1] + R > 1 + 1e-8):
            return -np.inf
            
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(X[i] - X[j])
                if dist < R[i] + R[j] - 1e-8:
                    return -np.inf
        
        return np.sum(R)

    # Local search using gradient descent on penalty
    def local_search(params):
        X = params[:2*N].reshape(N, 2)
        R = params[2*N:].copy()
        
        for step in range(200):
            # Try to grow radii
            R_new = R + 0.001
            params_test = np.concatenate([X.flatten(), R_new])
            
            # Check validity
            valid = True
            # Boundary check
            if np.any(R_new < 0) or np.any(X[:, 0] - R_new < -1e-8) or np.any(X[:, 0] + R_new > 1 + 1e-8) or np.any(X[:, 1] - R_new < -1e-8) or np.any(X[:, 1] + R_new > 1 + 1e-8):
                valid = False
            
            # Overlap check
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        dist = np.linalg.norm(X[i] - X[j])
                        if dist < R_new[i] + R_new[j] - 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                R = R_new
            else:
                # Resolve overlaps by moving X
                # Calculate forces
                F = np.zeros((N, 2))
                # Boundary forces
                for i in range(N):
                    if X[i, 0] < R[i]:
                        F[i, 0] += (R[i] - X[i, 0])
                    if X[i, 0] > 1 - R[i]:
                        F[i, 0] -= (X[i, 0] - (1 - R[i]))
                    if X[i, 1] < R[i]:
                        F[i, 1] += (R[i] - X[i, 1])
                    if X[i, 1] > 1 - R[i]:
                        F[i, 1] -= (X[i, 1] - (1 - R[i]))
                
                # Overlap forces
                for i in range(N):
                    for j in range(i+1, N):
                        d = X[i] - X[j]
                        dist = np.linalg.norm(d)
                        min_dist = R[i] + R[j]
                        if dist < min_dist and dist > 1e-10:
                            force = (min_dist - dist) / dist
                            F[i] += d * force
                            F[j] -= d * force
                
                # Move X
                X = X - F * 0.1
                # Clip to [0,1]
                X = np.clip(X, 0, 1)
                
                # If still invalid after moving, try random perturbation
                # Check if we are stuck
                # Simple check: if no movement happened
                # For simplicity, just continue loop

        # Final validation
        params_final = np.concatenate([X.flatten(), R])
        score = get_score(params_final)
        if score > 0:
            return params_final, score
        return None, 0.0

    # Random restarts
    while time.time() < deadline:
        # Random init
        X = np.random.rand(N, 2)
        R = np.full(N, 0.01)
        params = np.concatenate([X.flatten(), R])
        
        p, s = local_search(params)
        if p is not None and s > best_score:
            best_score = s
            best_params = p
            # Optional: if score is very good, break? No, keep searching.

    if best_params is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Reshape best_params to (N, 3)
    X = best_params[:2*N].reshape(N, 2)
    R = best_params[2*N:]
    return np.column_stack((X, R))
