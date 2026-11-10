"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def fit_exercise_policy(training_paths, contract):
    t_start = time.time()
    deadline = 150.0 - 10.0  # 140s safety margin
    
    try:
        # Extract parameters
        if hasattr(contract, 'params'):
            params = contract.params
        elif isinstance(contract, dict):
            params = contract
        else:
            params = {}
        
        S0_list = params.get('S0', [90, 100, 110])
        K = params.get('K', 100.0)
        r = params.get('r', 0.05)
        sigma = params.get('sigma', 0.2)
        T = params.get('T', 1.0)
        n_exercises = params.get('n_exercises', 12)
        
        # Ensure training_paths is a numpy array
        paths = np.asarray(training_paths, dtype=float)
        
        if paths.ndim == 1:
            paths = paths.reshape(1, -1)
            
        n_paths = paths.shape[0]
        n_timesteps = paths.shape[1]
        
        # The problem states 12 monthly exercise dates, no time-0 exercise.
        # Paths likely include time 0. So indices 1..12 correspond to exercise dates 1..12.
        # If paths shape is (N, 13), indices 1..12 are exercise dates.
        # If paths shape is (N, 12), indices 0..11 are exercise dates.
        
        # Standard assumption: paths include t=0.
        # Let's verify dimensions. T=1, 12 steps.
        # If n_timesteps == n_exercises + 1, then t=0 is included.
        # If n_timesteps == n_exercises, then t=0 is NOT included or the first step is t=1.
        
        # Let's assume the standard GBM simulation includes t=0.
        # Exercise dates are at t = 1/12, 2/12, ..., 12/12.
        
        # We will perform Backward Induction using LSM.
        
        # 1. Initialize Continuation Values
        # V is an array of shape (n_paths, n_exercises)
        # V[:, j] is the value at exercise date j+1 (index 0 corresponds to date 1)
        
        # We need to map path indices to exercise dates.
        # Let's assume paths are indexed 0..n_timesteps-1.
        # If n_timesteps == 13, exercise dates are at indices 1, 2, ..., 12.
        # If n_timesteps == 12, exercise dates are at indices 0, 1, ..., 11.
        
        # Let's determine the offset.
        # If the first column is S0 (which is 90, 100, 110), then offset is 1.
        # Check if first column is constant or matches S0 distribution.
        # Since S0 is different for different paths? No, S0 is a scalar per contract usually, 
        # but the task says "Spot 90/100/110". This implies the contract might be evaluated 
        # for multiple spots, or the training paths are generated for a specific spot.
        # The `contract` object likely specifies which spot is being used for this specific 
        # `fit_exercise_policy` call, OR the training paths are mixed.
        # However, `fit_exercise_policy` returns a single policy object.
        # The `exercise` method receives `current_state` and `time`.
        # This implies the policy must handle any spot.
        
        # Wait, the task says: "Candidate receives current states only during scoring... Spot 90/100/110".
        # This suggests the policy is fit on training paths (which might be for a specific spot or mixed)
        # and then evaluated on paths for spots 90, 100, 110.
        # If training paths are for a single spot, the policy might not generalize well to others 
        # unless the model is robust.
        # However, LSM uses the paths to fit the regression. If the paths are for S0=100, 
        # the fitted regression should be valid for the distribution of states reached.
        
        # Let's assume the training paths are consistent with the contract parameters.
        
        # Determine exercise date indices in the paths array
        if n_timesteps == n_exercises + 1:
            ex_indices = np.arange(1, n_exercises + 1)
        else:
            ex_indices = np.arange(0, n_exercises)
            
        # V_next will store the continuation value for the next step
        # We work backward from the last exercise date.
        
        # At the last exercise date (j = n_exercises - 1), the continuation value is 0 
        # (assuming no further exercise opportunities and the option expires or is held to maturity 
        # but since it's Bermudan, we exercise or let it expire. If it expires, value is 0 if not exercised? 
        # No, for a put, if not exercised at the last date, it expires worthless if S > K? 
        # Actually, standard Bermudan: you can exercise at discrete dates. If you don't exercise at the last date, 
        # the option expires. So the terminal value is 0.)
        
        # However, usually we compare Exercise vs Continuation.
        # Continuation value at the last step is 0.
        
        V_cont = np.zeros(n_paths)
        
        # Store boundaries for each exercise date
        # We will fit a regression model to V_cont to get the continuation value function C(S)
        # Then we compare Exercise Payoff E(S) vs C(S)
        
        # We want to return a policy that takes (current_state, time) and returns a boolean.
        # We can fit a global regression for the continuation value at each step, 
        # or fit boundaries.
        # Fitting boundaries is more direct for the decision rule.
        # But the boundaries depend on the regression coefficients.
        
        # Strategy:
        # 1. Iterate backward from last exercise date to first.
        # 2. At each date j, we have V_cont (which is the discounted expected value of future optimal decisions).
        # 3. We fit a regression model to V_cont as a function of S_j.
        # 4. We compute the exercise payoff E_j(S) = max(K - S, 0).
        # 5. We determine the exercise region where E_j(S) > C_j(S).
        # 6. For a put, this region is typically S < B_j.
        # 7. We can fit a boundary B_j by finding the crossover point.
        #    Alternatively, we can just store the regression coefficients and the decision rule 
        #    "Exercise if E(S) > C(S)" and evaluate C(S) using the fitted coefficients at test time.
        #    This is more robust than finding a single boundary point, especially if the 
        #    regression is not perfectly monotonic or if we want to capture non-linearities.
        
        # Let's store the regression coefficients for each step.
        # Degree of polynomial: 2 or 3 is usually sufficient for GBM.
        poly_degree = 3
        
        coeffs_per_step = []
        
        # We need to be careful with the time index.
        # ex_indices maps to the columns in `paths`.
        
        for j in range(n_exercises - 1, -1, -1):
            # Get current stock prices at this exercise date
            S_j = paths[:, ex_indices[j]]
            
            # Fit regression to V_cont using S_j
            # X = [1, S, S^2, S^3]
            X = np.vander(S_j, N=poly_degree + 1, increasing=True)
            
            try:
                # Solve least squares
                coeffs, _, _, _ = np.linalg.lstsq(X, V_cont, rcond=None)
            except:
                coeffs = np.zeros(poly_degree + 1)
                
            coeffs_per_step.append(coeffs)
            
            # Compute the continuation value estimate for all paths at this step
            C_j = X @ coeffs
            
            # Compute immediate exercise payoff
            E_j = np.maximum(K - S_j, 0.0)
            
            # Determine which paths should exercise
            # Exercise if E_j > C_j
            exercise_mask = E_j > C_j
            
            # Update V_cont for the previous step
            # V_prev(S_{j-1}) = E[ exp(-r*dt) * V_current(S_j) | S_{j-1} ]
            # Where V_current(S_j) = max(E_j, C_j)
            
            V_current = np.maximum(E_j, C_j)
            
            # Discount and roll back
            # dt = T / n_exercises
            dt = T / n_exercises
            discount = np.exp(-r * dt)
            
            # V_cont for the previous step is the discounted value of V_current
            # BUT, this is only valid if we are at the previous step.
            # The paths are already simulated. The value at step j-1 is the expected value of step j.
            # Since we are using Monte Carlo paths, we can just take the discounted V_current 
            # as the "target" for the regression at step j-1?
            # No, that's not right.
            # The value at step j-1 for a specific path is the expected value of the future cash flows 
            # given the state at j-1.
            # In LSM, we use the regression of V_current on S_j to estimate the conditional expectation 
            # at step j-1?
            # Actually, the standard LSM algorithm works like this:
            # 1. Start from the end. V_N = Payoff_N.
            # 2. At step N-1:
            #    - Fit regression of V_N on S_N.
            #    - Compute C_{N-1}(S) using the fitted regression evaluated at S_{N-1}? 
            #      NO. The continuation value at step N-1 is E[ V_N(S_N) | S_{N-1} ].
            #      We estimate this by fitting V_N as a function of S_{N-1} (or S_N? No, S_{N-1} is the state).
            #      Wait, the regression is typically fit on the *current* state to predict the *next* value.
            #      So at step j, we have V_{j+1} (the value at the next step).
            #      We fit a regression of V_{j+1} on S_j.
            #      Then C_j(S_j) = Regressor(S_j).
            #      Then V_j(S_j) = max(E_j(S_j), C_j(S_j)).
            #      Then we move to j-1.
            
            # Let's re-verify the loop logic.
            # My loop above:
            # 1. Takes V_cont (which is the value from the *next* step, i.e., V_{j+1}).
            # 2. Fits regression of V_cont on S_j.
            # 3. Computes C_j = Regression(S_j).
            # 4. Computes V_j = max(E_j, C_j).
            # 5. Sets V_cont = V_j (discounted? No, V_j is the value *at* time j. 
            #    The value at time j-1 is E[ exp(-r dt) V_j | S_{j-1} ].
            #    So for the *next* iteration (j-1), the "V_cont" input should be exp(-r dt) * V_j.
            
            # So, we need to discount V_current before assigning it to V_cont for the next iteration.
            
            V_cont = discount * V_current
            
            # Check time
            if time.time() > deadline - 5.0:
                break
        
        # coeffs_per_step is in reverse order (last step first)
        coeffs_per_step = coeffs_per_step[::-1]
        
        # Now we have coefficients for each exercise date j (0 to n_exercises-1)
        # coeffs_per_step[j] are the coefficients for the regression of the *continuation value* 
        # at step j+1? 
        # Let's trace:
        # j = n_exercises - 1 (last step):
        #   V_cont is 0 (initialization).
        #   Fit regression of 0 on S_last. Coeffs are 0.
        #   C_last = 0.
        #   V_last = max(E_last, 0) = E_last.
        #   V_cont = discount * V_last.
        #   coeffs_per_step.append(coeffs for 0).
        
        # j = n_exercises - 2:
        #   V_cont is discount * V_last.
        #   Fit regression of V_cont on S_{n_exercises-2}.
        #   This regression estimates E[ V_last(S_last) | S_{n_exercises-2} ] (approximately, with discount).
        #   C_{n_exercises-2} = Regression(S_{n_exercises-2}).
        #   V_{n_exercises-2} = max(E_{n_exercises-2}, C_{n_exercises-2}).
        #   coeffs_per_step.append(coeffs for this regression).
        
        # So coeffs_per_step[j] corresponds to the regression used to compute the continuation value 
        # at exercise date j+1 (1-indexed) or j (0-indexed)?
        # Let's index carefully.
        # In the loop, `j` goes from n_exercises-1 down to 0.
        # `ex_indices[j]` is the column for exercise date j+1 (if 1-indexed dates 1..12).
        # The regression fit at iteration `j` uses `S_j` (prices at date j+1) and `V_cont` (values from date j+2).
        # So `coeffs_per_step[j]` (before reversal) is the regression of the value *after* date j+1 
        # on the price *at* date j+1.
        # This regression gives the Continuation Value at date j+1.
        
        # After reversal, `coeffs_per_step[j]` corresponds to exercise date index `j` (0-indexed, where 0 is the first exercise date).
        # Wait, the loop appends coeffs for j=n-1, then j=n-2...
        # So list is [coeffs_j=n-1, coeffs_j=n-2, ..., coeffs_j=0]
        # Reversed: [coeffs_j=0, coeffs_j=1, ..., coeffs_j=n-1]
        
        # So `coeffs_per_step[j]` are the coefficients for the continuation value at exercise date `j` (0-indexed).
        # To decide whether to exercise at date `j`, we compare:
        # Exercise Payoff: max(K - S, 0)
        # Continuation Value: P(S) where P is the polynomial defined by `coeffs_per_step[j]`.
        
        class ExercisePolicy:
            def __init__(self, coeffs_list, n_exercises, K, T):
                self.coeffs_list = coeffs_list
                self.n_exercises = n_exercises
                self.K = K
                self.T = T
                self.dt = T / n_exercises
                
            def exercise(self, current_state, time):
                current_state = np.asarray(current_state, dtype=float)
                time = np.asarray(time)
                
                if np.ndim(time) == 0:
                    time = np.full_like(current_state, time, dtype=float)
                
                n = len(current_state)
                result = np.zeros(n, dtype=bool)
                
                for i in range(n):
                    t = time[i]
                    s = current_state[i]
                    
                    # Determine exercise date index
                    # Exercise dates are at dt, 2dt, ..., T
                    # t is in [0, T]
                    # date_idx = round(t / dt) - 1
                    date_idx = int(round(t / self.dt)) - 1
                    
                    if date_idx < 0:
                        date_idx = 0
                    elif date_idx >= self.n_exercises:
                        date_idx = self.n_exercises - 1
                        
                    # Get coefficients for this date
                    coeffs = self.coeffs_list[date_idx]
                    
                    # Compute continuation value
                    # coeffs are for polynomial of degree poly_degree
                    # P(s) = c0 + c1*s + c2*s^2 + ...
                    power = 1.0
                    cont_val = 0.0
                    for c in coeffs:
                        cont_val += c * power
                        power *= s
                        
                    # Compute exercise payoff
                    ex_val = max(self.K - s, 0.0)
                    
                    # Exercise if payoff > continuation
                    if ex_val > cont_val:
                        result[i] = True
                        
                return result

        return ExercisePolicy(coeffs_per_step, n_exercises, K, T)
        
    except Exception as e:
        # Fallback
        class ExercisePolicyFallback:
            def __init__(self, K):
                self.K = K
            
            def exercise(self, current_state, time):
                current_state = np.asarray(current_state, dtype=float)
                return current_state < self.K
        
        try:
            K = params.get('K', 100.0)
        except:
            K = 100.0
        return ExercisePolicyFallback(K)

# EVOLVE-BLOCK-END
