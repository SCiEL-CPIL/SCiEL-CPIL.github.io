# EVOLVE-BLOCK-START
import time
import numpy as np

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan exercise policy using global parametric regression 
    (polynomial in log-price) for continuation values.
    
    This approach avoids the grid-clipping bias and interpolation errors 
    of local grid methods by fitting a smooth global function to the 
    value function at each time step.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # Extract parameters from contract
    K = contract.get('strike', contract.get('K', 100.0))
    r = contract.get('rate', contract.get('r', 0.05))
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    
    # Training paths shape: (n_paths, n_cols)
    n_paths, n_cols = training_paths.shape
    
    # Determine which columns correspond to exercise dates
    # Standard assumption: if n_cols == n_dates, cols are t1..tn.
    # If n_cols == n_dates + 1, col 0 is t0, cols 1..n are t1..tn.
    # The task says "12 monthly exercise dates (no time-0 exercise)".
    # We assume the provided paths align with the exercise dates for simplicity,
    # or if there's an extra column, it's t0.
    # Let's check the mean of the first column vs K to see if it's t0.
    # Usually S0 is given as 90/100/110.
    # If n_cols > n_dates, we drop the first column (t0).
    if n_cols > n_dates:
        exercise_data = training_paths[:, 1:n_dates+1]
    else:
        exercise_data = training_paths[:, :n_dates]
        
    # Ensure we have exactly n_dates columns
    if exercise_data.shape[1] < n_dates:
        # Fallback: pad or resize if data is weird
        # But standard contract should provide correct shape.
        # If shorter, we can't do much. Assume valid input.
        pass

    n_paths_eff, n_steps_eff = exercise_data.shape
    if n_steps_eff != n_dates:
        # If mismatch, try to use what we have
        n_dates = n_steps_eff

    # Define the basis functions for the regression.
    # We model V(S) as a polynomial in x = log(S).
    # For a put option, the value is convex in S, but in log(S) it might be smoother.
    # Let's use a quadratic polynomial: V(S) = a0 + a1*log(S) + a2*log(S)^2
    # Actually, for a put, max(K-S, 0) is piecewise linear in S.
    # In log(S), it's not a polynomial. 
    # However, the *continuation* value is smooth.
    # The total value V_t(S) = max(Intrinsic, Continuation).
    # The Intrinsic part is kinked.
    # If we fit V_t(S) directly with a polynomial, we might smooth out the kink,
    # which is bad for the exercise decision (Intrinsic vs Continuation).
    
    # Better approach:
    # 1. Estimate the Continuation Value C_t(S) = E[discount * V_{t+1}(S_{t+1}) | S_t=S].
    # 2. V_t(S) = max(Intrinsic(S), C_t(S)).
    # 3. We need to estimate C_t(S) for any S.
    
    # How to estimate C_t(S)?
    # C_t(S) is the expected value of V_{t+1}(S_{t+1}) given S_t=S.
    # Since S_{t+1} = S_t * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z),
    # S_{t+1} is log-normal given S_t.
    # If V_{t+1}(S) is well-approximated by a function f(S), we can compute the expectation analytically or via quadrature.
    # However, we don't have an analytical form for V_{t+1}.
    
    # Alternative Strategy: "Direct" Regression on the Value Function
    # We can regress the *total* value V_t(S) on the state S_t.
    # But we don't have V_t(S) directly. We have paths.
    
    # Let's use the standard "Longstaff-Schwartz" style but with a global fit.
    # Backward induction:
    # 1. At T, V_T(S) = Intrinsic(S).
    # 2. At t < T, we want to find V_t(S).
    #    We know that V_t(S) = max(Intrinsic(S), E[discount * V_{t+1}(S_{t+1}) | S_t=S]).
    #    Let C_t(S) = E[discount * V_{t+1}(S_{t+1}) | S_t=S].
    #    We can estimate C_t(S) by regressing the *next step's value* V_{t+1}(S_{t+1}) 
    #    on the *current step's state* S_t? No, that's not right.
    #    V_{t+1}(S_{t+1}) depends on S_{t+1}.
    #    The expectation is over S_{t+1} | S_t.
    
    # Let's use a simpler approximation for C_t(S):
    # Assume V_{t+1}(S) can be represented by a basis function expansion: V_{t+1}(S) ≈ sum_j c_j * phi_j(S).
    # Then C_t(S) = discount * E[ sum_j c_j * phi_j(S_{t+1}) | S_t=S ]
    #             = discount * sum_j c_j * E[ phi_j(S_{t+1}) | S_t=S ].
    # If phi_j(S) = S^k (power basis), we can compute E[S_{t+1}^k | S_t] analytically for GBM.
    # E[S_{t+1}^k | S_t] = S_t^k * exp(k*(r - 0.5*sigma^2)*dt + 0.5*k*(k-1)*sigma^2*dt)
    # This is exact for the power basis!
    
    # So, the strategy is:
    # 1. Represent V_{t+1}(S) using a power basis: V_{t+1}(S) ≈ c0 + c1*S + c2*S^2 + ...
    #    Wait, V is bounded for a put? No, V is bounded by K for a put? 
    #    Actually, V(S) <= K for a put. So it doesn't grow with S.
    #    A power basis like S^2 will blow up. 
    #    However, the coefficients for high powers will be near zero or negative?
    #    For a put, V(S) is decreasing in S.
    #    Let's use a basis that decays or is bounded?
    #    Or just use a low-order polynomial in S? 
    #    For S in [0, 2K], a quadratic or cubic in S might work.
    #    
    # Let's try a basis of powers of S: [1, S, S^2].
    # We fit V_{t+1}(S_{t+1}) ~ c0 + c1*S_{t+1} + c2*S_{t+1}^2 using the training paths at t+1.
    # Then we compute C_t(S_t) = discount * (c0 + c1*E[S_{t+1}|S_t] + c2*E[S_{t+1}^2|S_t]).
    # E[S_{t+1}|S_t] = S_t * exp(r*dt).
    # E[S_{t+1}^2|S_t] = S_t^2 * exp((2r + sigma^2)*dt).
    # This gives an *analytical* smooth function for C_t(S_t).
    # Then V_t(S_t) = max(Intrinsic(S_t), C_t(S_t)).
    
    # This avoids grid interpolation entirely and uses the exact GBM moments.
    # The only approximation is the functional form of V_{t+1}.
    # Since V is smooth (except at the exercise boundary), a low-order polynomial in S 
    # might be sufficient, especially if we restrict the domain or use a transformed basis.
    # However, V(S) for a put is flat for S > K (if far out of the money? No, it goes to 0).
    # For S < K, it's roughly linear.
    # A quadratic in S might capture the curvature.
    
    # Let's use basis: [1, S, S^2, S^3] to be safe.
    # Or even better, since we know the domain of S is roughly [0, 2K], 
    # we can normalize S by K to keep numbers stable.
    
    basis_order = 3  # Up to S^3
    
    # Pre-compute GBM moments for the basis functions
    dt = T / n_dates
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    
    # E[S_{t+1}^k | S_t] = S_t^k * M_k
    # M_k = exp(k * drift + 0.5 * k * (k-1) * vol**2)
    # Note: vol**2 = sigma^2 * dt
    # M_k = exp(k*(r-0.5*sigma^2)*dt + 0.5*k*(k-1)*sigma^2*dt)
    #      = exp(k*r*dt - 0.5*k*sigma^2*dt + 0.5*k^2*sigma^2*dt - 0.5*k*sigma^2*dt)
    #      = exp(k*r*dt + 0.5*k*(k-1)*sigma^2*dt)
    
    moments = np.zeros(basis_order + 1)
    for k in range(basis_order + 1):
        if k == 0:
            moments[k] = 1.0
        else:
            moments[k] = np.exp(k * r * dt + 0.5 * k * (k - 1) * sigma**2 * dt)
            
    discount = np.exp(-r * dt)
    
    # V_grids is not needed. We store the coefficients for each time step.
    # coeffs[t] are the coefficients for V_t(S) approximation?
    # No, we need the coefficients for V_{t+1} to compute C_t.
    # Let's store the coefficients of the fitted V_t for each t.
    
    # coeffs[t] = [c0, c1, c2, c3] such that V_t(S) ≈ c0 + c1*S + c2*S^2 + c3*S^3
    # But V_t is the *optimal* value, which is max(Intrinsic, Continuation).
    # We cannot simply fit V_t with a polynomial because of the kink.
    # However, we can fit the *Continuation* part?
    # Or, we can fit V_t on the *continuation* region?
    
    # Alternative: 
    # 1. Fit V_{t+1}(S) with a polynomial using paths at t+1.
    # 2. Compute C_t(S) analytically using the moments.
    # 3. V_t(S) = max(Intrinsic(S), C_t(S)).
    # 4. Now we have V_t(S) as a function. 
    # 5. To fit V_t for the next step, we need to regress V_t(S_t) on S_t?
    #    But V_t(S) is not a polynomial. It's max(linear, polynomial).
    #    We can still fit a polynomial to V_t(S_t) using the training paths at t.
    #    The fit will approximate the max function.
    
    # This seems robust.
    
    V_coeffs = [None] * n_dates
    
    # Initialize at T (last date)
    # V_T(S) = max(K - S, 0)
    # We need to fit this to a polynomial? 
    # Or we can handle the last step specially?
    # For t = n_dates - 1, V is just Intrinsic.
    # We can fit a polynomial to Intrinsic(S) at the last date's states.
    # Or, since we know the form, we can just use the Intrinsic formula directly for the last step?
    # But the "fit" step is what generates the coefficients for the *next* backward step.
    
    # Let's fit a polynomial to V_t(S_t) for each t, starting from T.
    
    # At T:
    S_T = exercise_data[:, -1]
    V_T = np.maximum(K - S_T, 0.0)
    
    # Fit polynomial to V_T ~ S_T
    # Use np.polyfit? Or linear algebra.
    # Design matrix X_T: columns are [1, S_T, S_T^2, S_T^3]
    # We want c such that X_T @ c ≈ V_T
    
    def fit_poly(S, V, order):
        # S: (N,)
        # V: (N,)
        # Returns coefficients [c0, c1, ..., c_order] for c0 + c1*S + ...
        X = np.vander(S, N=order+1, increasing=True)
        # Solve least squares
        # Add small regularization to avoid ill-conditioning
        try:
            c, residuals, rank, s = np.linalg.lstsq(X, V, rcond=None)
            return c
        except:
            # Fallback: just intrinsic?
            return np.zeros(order+1)
            
    V_coeffs[-1] = fit_poly(S_T, V_T, basis_order)
    
    # Backward loop
    for t in range(n_dates - 2, -1, -1):
        S_t = exercise_data[:, t]
        S_t_next = exercise_data[:, t+1]
        
        # 1. Evaluate V_{t+1}(S_{t+1}) using the fitted coefficients
        # V_{t+1}(S) ≈ c0 + c1*S + c2*S^2 + c3*S^3
        coeffs_next = V_coeffs[t+1]
        # Horner's method or matrix mult
        # X_next = np.vander(S_t_next, N=basis_order+1, increasing=True)
        # V_next_on_paths = X_next @ coeffs_next
        
        # Faster:
        V_next_on_paths = np.zeros_like(S_t_next)
        for k in range(basis_order, -1, -1):
            V_next_on_paths = V_next_on_paths * S_t_next + coeffs_next[k]
            
        # 2. Compute Continuation Value C_t(S_t) analytically
        # C_t(S_t) = discount * sum_k coeffs_next[k] * E[S_{t+1}^k | S_t]
        # E[S_{t+1}^k | S_t] = S_t^k * moments[k]
        
        C_t_on_paths = np.zeros_like(S_t)
        for k in range(basis_order + 1):
            C_t_on_paths += coeffs_next[k] * moments[k] * (S_t ** k)
        C_t_on_paths *= discount
        
        # 3. Compute Intrinsic Value
        Intrinsic_t = np.maximum(K - S_t, 0.0)
        
        # 4. Compute V_t(S_t) = max(Intrinsic_t, C_t_on_paths)
        V_t_on_paths = np.maximum(Intrinsic_t, C_t_on_paths)
        
        # 5. Fit V_t(S_t) to a polynomial to get coeffs for the next iteration
        V_coeffs[t] = fit_poly(S_t, V_t_on_paths, basis_order)
        
        # Check time
        if time.time() > deadline:
            # If we run out of time, we have V_coeffs for current t.
            # We need to fill the rest.
            # For remaining t, we can just use the last computed coeffs or intrinsic.
            # Since we are going backwards, if we break at t, we have coeffs for t..T.
            # We need coeffs for 0..t-1.
            # We can fill them with the coeffs from t (constant approximation) or intrinsic.
            # Using coeffs from t is better than intrinsic for early steps?
            # Actually, if we break early, we haven't computed the early steps.
            # We can just break and handle None in the class.
            break
            
    # Fill any remaining None V_coeffs
    for t in range(n_dates):
        if V_coeffs[t] is None:
            # Use the last known coeffs or zero
            # Find next non-None
            for j in range(t+1, n_dates):
                if V_coeffs[j] is not None:
                    V_coeffs[t] = V_coeffs[j]
                    break
            else:
                V_coeffs[t] = np.zeros(basis_order+1)
                
    class ExercisePolicy:
        def __init__(self, coeffs_list, K, basis_order):
            self.coeffs_list = coeffs_list
            self.K = K
            self.basis_order = basis_order
            
        def exercise(self, current_state, time):
            """
            current_state: array of spot prices
            time: index of the exercise date (0 to n_dates-1)
            Returns: boolean vector
            """
            if time >= len(self.coeffs_list):
                return (self.K - current_state) > 0
                
            coeffs = self.coeffs_list[time]
            
            # Evaluate V_t(S) using the polynomial
            V_current = np.zeros_like(current_state)
            for k in range(self.basis_order, -1, -1):
                V_current = V_current * current_state + coeffs[k]
                
            # Intrinsic value
            intrinsic = np.maximum(self.K - current_state, 0.0)
            
            # Exercise if Intrinsic >= V_current
            # Note: V_current is the estimated total value.
            # If Intrinsic >= V_current, it means Intrinsic >= Continuation (since V = max(Intr, Cont)).
            # Actually, V_current is an approximation of max(Intr, Cont).
            # If Intrinsic > V_current, we definitely exercise.
            # If Intrinsic < V_current, we hold.
            # If they are close, it's a tie.
            # Standard: Exercise if Intrinsic >= V.
            
            return intrinsic >= V_current
            
    return ExercisePolicy(V_coeffs, K, basis_order)

# EVOLVE-BLOCK-END
