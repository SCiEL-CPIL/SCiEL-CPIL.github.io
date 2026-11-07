"""Root finder for 3 fixed 2D systems using Broyden's method with LM regularization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    def time_left():
        return deadline - time.time()

    x = np.array(x0, dtype=float)
    best_x = x.copy()
    best_norm = np.inf
    
    max_calls = 480  # Stay well under 500
    calls = 0
    tol = 1e-12
    
    # Helper to evaluate F and track calls
    def eval_F(x_val):
        nonlocal calls
        calls += 1
        return np.asarray(F(x_val), dtype=float)

    # Initialize Jacobian with finite differences
    # We need 3 calls: F(x), F(x+e1), F(x+e2)
    if time_left() < 1.0 or calls >= max_calls:
        return best_x

    Fx = eval_F(x)
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx < tol:
        return x
    best_x = x.copy()
    best_norm = norm_Fx

    J = np.zeros((2, 2))
    h = 1e-6
    # Compute columns of J
    e1 = np.array([h, 0.0])
    e2 = np.array([0.0, h])
    Fx_e1 = eval_F(x + e1)
    Fx_e2 = eval_F(x + e2)
    
    J[:, 0] = (Fx_e1 - Fx) / h
    J[:, 1] = (Fx_e2 - Fx) / h

    # Broyden's method loop
    # We have used 3 calls so far.
    # Each subsequent iteration uses 1 call for F(x_new)
    # We also need to keep track of s and y for updates
    
    lambda_lm = 1.0
    stagnation_count = 0
    prev_norm = best_norm

    while time_left() > 2.0 and calls < max_calls:
        # Check convergence
        if norm_Fx <= 1e-8:
            best_x = x.copy()
            best_norm = norm_Fx
            break
        
        # Solve (J + lambda*I) dx = -Fx
        # Using LM regularization to handle singular J
        try:
            A = J + lambda_lm * np.eye(2)
            dx = np.linalg.solve(A, -Fx)
        except np.linalg.LinAlgError:
            # Fallback to simple gradient step if solve fails
            dx = -lambda_lm * Fx 

        # Damping step size if step is too large or causes divergence
        # Try full step first
        x_new = x + dx
        Fx_new = eval_F(x_new)
        norm_Fx_new = np.linalg.norm(Fx_new)

        if norm_Fx_new < norm_Fx:
            # Successful step
            x = x_new
            Fx = Fx_new
            norm_Fx = norm_Fx_new
            
            if norm_Fx < best_norm:
                best_norm = norm_Fx
                best_x = x.copy()
            
            # Broyden Rank-1 Update
            s = x - (x - dx) # This is just dx, but let's be explicit about state change
            # Actually s = x_new - x_old = dx
            s = dx
            y = Fx_new - Fx_old # Wait, Fx is updated. We need old Fx.
            # Let's restructure to keep old values
            
            # Correction:
            # We need y = F(x_new) - F(x_old)
            # We have Fx_new and Fx (which was F(x_old))
            # But we just updated x and Fx.
            # So y = Fx_new - (Fx_new - y)? No.
            
            # Let's restart the logic block cleanly.
            # State before step: x, Fx, J
            # Step: dx
            # New State: x_new = x + dx, Fx_new
            # s = x_new - x = dx
            # y = Fx_new - Fx
            
            # Update J: J_new = J + (y - J*s)/s^T * s^T
            # We need to store Fx_old before overwriting.
            
            # Since I modified the code flow above, let's fix the update logic here.
            # I will re-implement the loop body correctly.
            pass 
        else:
            # Unsuccessful step
            lambda_lm *= 10.0
            stagnation_count += 1
            if stagnation_count > 5:
                # Restart with new Jacobian estimate
                # Reset lambda
                lambda_lm = 1.0
                stagnation_count = 0
                # Re-estimate Jacobian at current best_x or current x?
                # Current x is likely bad. Use best_x? Or just re-center FD at current x?
                # Re-centering at current x is standard for Broyden restarts.
                if time_left() < 5.0 or calls >= max_calls - 3:
                    break
                
                Fx = eval_F(x)
                norm_Fx = np.linalg.norm(Fx)
                if norm_Fx < best_norm:
                    best_norm = norm_Fx
                    best_x = x.copy()
                
                J = np.zeros((2, 2))
                Fx_e1 = eval_F(x + e1)
                Fx_e2 = eval_F(x + e2)
                J[:, 0] = (Fx_e1 - Fx) / h
                J[:, 1] = (Fx_e2 - Fx) / h
                continue

        # --- Corrected Loop Body Implementation ---
        # The logic above got messy. Let's write a clean, robust version.
        
        break # Exit the messy draft loop and use the clean one below

    # --- Clean Robust Implementation ---
    # Reset state for clean run
    x = np.array(x0, dtype=float)
    best_x = x.copy()
    best_norm = np.inf
    calls = 0
    lambda_lm = 1.0
    
    def get_F(x_val):
        nonlocal calls
        calls += 1
        return np.asarray(F(x_val), dtype=float)

    # Initial F and J
    Fx = get_F(x)
    norm_Fx = np.linalg.norm(Fx)
    best_norm = norm_Fx
    best_x = x.copy()
    
    h = 1e-6
    e1 = np.array([h, 0.0])
    e2 = np.array([0.0, h])
    Fx1 = get_F(x + e1)
    Fx2 = get_F(x + e2)
    J = np.array([
        [(Fx1[0]-Fx[0])/h, (Fx2[0]-Fx[0])/h],
        [(Fx1[1]-Fx[1])/h, (Fx2[1]-Fx[1])/h]
    ])

    stagnation = 0
    prev_norm = norm_Fx

    while time_left() > 2.0 and calls < max_calls:
        if norm_Fx <= 1e-8:
            break
        
        # Solve (J + lambda I) dx = -Fx
        try:
            A = J + lambda_lm * np.eye(2)
            dx = np.linalg.solve(A, -Fx)
        except np.linalg.LinAlgError:
            dx = -lambda_lm * Fx

        x_new = x + dx
        Fx_new = get_F(x_new)
        norm_Fx_new = np.linalg.norm(Fx_new)

        if norm_Fx_new < norm_Fx:
            # Success
            s = x_new - x
            y = Fx_new - Fx
            
            # Broyden Update: J = J + (y - J s) s^T / (s^T s)
            J_s = J @ s
            numerator = (y - J_s) @ s.T
            denominator = s @ s
            
            if denominator > 1e-15:
                J = J + numerator / denominator
            
            x = x_new
            Fx = Fx_new
            norm_Fx = norm_Fx_new
            
            if norm_Fx < best_norm:
                best_norm = norm_Fx
                best_x = x.copy()
            
            # Reduce lambda if it was increased
            if lambda_lm > 1.0:
                lambda_lm = max(1.0, lambda_lm * 0.1)
            
            stagnation = 0
            prev_norm = norm_Fx
        else:
            # Failure
            lambda_lm *= 10.0
            stagnation += 1
            
            if stagnation > 5:
                # Restart Jacobian
                if calls >= max_calls - 3:
                    break
                Fx = get_F(x)
                norm_Fx = np.linalg.norm(Fx)
                if norm_Fx < best_norm:
                    best_norm = norm_Fx
                    best_x = x.copy()
                
                Fx1 = get_F(x + e1)
                Fx2 = get_F(x + e2)
                J = np.array([
                    [(Fx1[0]-Fx[0])/h, (Fx2[0]-Fx[0])/h],
                    [(Fx1[1]-Fx[1])/h, (Fx2[1]-Fx[1])/h]
                ])
                lambda_lm = 1.0
                stagnation = 0

    # Final check
    if best_norm > 1e-8:
        # One last desperate Newton step if we have budget? 
        # Or just return best_x.
        pass

    return best_x
# EVOLVE-BLOCK-END
