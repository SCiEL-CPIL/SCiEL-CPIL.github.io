import numpy as np
import time
from scipy.optimize import minimize_scalar

class ExercisePolicy:
    def __init__(self, critical_prices):
        self.critical_prices = critical_prices
        self.exercise_times = np.linspace(1/12.0, 1.0, 12)

    def exercise(self, current_state, time):
        current_state = np.asarray(current_state)
        k_float = time * 12.0
        k_int = int(round(k_float))
        
        if k_int < 1 or k_int > 12:
            return np.zeros_like(current_state, dtype=bool)
            
        crit_price = self.critical_prices[k_int - 1]
        return current_state <= crit_price

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    
    n_ex = 12
    
    paths_raw = np.asarray(training_paths)
    
    if paths_raw.ndim == 1:
        paths_raw = paths_raw.reshape(1, -1)
        
    if paths_raw.shape[1] == 13:
        # Assume columns 0 is S0, 1-12 are exercise dates
        paths = paths_raw[:, 1:13]
    elif paths_raw.shape[1] == 12:
        # Assume columns 0-11 are exercise dates
        paths = paths_raw[:, 0:12]
    else:
        # Fallback: take last 12 columns
        paths = paths_raw[:, -12:]
        
    M = paths.shape[0]
    
    # Discount factors for the 12 exercise dates
    t_steps = np.arange(1, 13) / 12.0
    discount_factors = np.exp(-r * t_steps)
    
    # --- Longstaff-Schwartz Regression Approach ---
    # 1. Split data into in-sample (fit) and out-of-sample (control) to reduce overfitting bias
    # Use a random shuffle for robustness
    rng = np.random.default_rng(42)
    indices = rng.permutation(M)
    split_idx = int(M * 0.5)
    in_sample_idx = indices[:split_idx]
    out_sample_idx = indices[split_idx:]
    
    paths_in = paths[in_sample_idx]
    paths_out = paths[out_sample_idx]
    
    # 2. Backward induction with regression
    # V_cont[k] is the estimated continuation value at step k (discounted to time 0? No, usually undiscounted local value or discounted to step k)
    # Let's define V_cont[k] as the expected value of future payoffs discounted to time t_k.
    # At the last step (k=11, t=1.0), V_cont[11] = 0 (no future after last exercise date? 
    # Wait, if we don't exercise at the last date, the option expires worthless for a Bermudan put? 
    # Yes, "12 monthly exercise dates". If not exercised, it expires. So terminal payoff is 0 if not exercised.)
    
    # Actually, standard Bermudan: if not exercised at T, value is 0.
    # So at step 11 (last date), V_cont = 0.
    # Decision: Exercise if (K - S) > 0.
    
    # We will store the continuation values for each step (1..11). Step 12 has V_cont=0.
    # Let's index steps 0..11 corresponding to t=1/12 .. 12/12.
    # V_cont[11] = 0.
    
    # We need to estimate V_cont[10] using paths at step 11.
    # Regression: Fit a function f(S) such that f(S_{t_{k+1}}) approximates the realized value at t_{k+1}
    # The realized value at t_{k+1} is:
    #   If exercised at t_{k+1}: (K - S_{t_{k+1}})
    #   If not exercised: V_cont[k+1] (which is 0 for k+1=11, or estimated value for earlier)
    # 
    # Wait, the standard LS approach:
    # At step k+1, we know the optimal decision (from backward induction).
    # Let Y_{k+1} be the value of the option at step k+1 under the optimal policy.
    # Y_{k+1} = max( (K - S_{k+1}), V_cont[k+1] )
    # We regress Y_{k+1} on S_k (or features of S_k) to estimate V_cont[k].
    # Then at step k, we compare (K - S_k) with V_cont[k] to decide.
    
    # Let's implement this.
    # V_cont[11] = 0.0
    V_cont = np.zeros(12)
    
    # We need to store the "realized values" Y_k for each step k (1..11) to use in regression for k-1.
    # Actually, we can compute Y_k on the fly during backward pass.
    
    # Basis functions for regression: Polynomial in S, maybe log(S), etc.
    # For a put, the value is convex in S.
    # Let's use a simple polynomial basis: [1, S, S^2, S^3]
    # Or better: [1, log(S), (log(S))^2, ...]
    # Let's stick to simple polynomial in S for stability with small data.
    # Actually, linear regression on [1, S] is too simple.
    # Let's use [1, S, S^2].
    
    def get_basis(S):
        S = np.asarray(S)
        return np.column_stack([np.ones_like(S), S, S**2])
    
    # We will store the critical prices
    critical_prices = np.zeros(12)
    
    # We need to simulate the decisions on the in-sample paths to get Y_k?
    # No, LS uses the "realized" values from the full path, assuming the future decisions are optimal.
    # But the future decisions depend on the regression fits.
    # So:
    # 1. Start at k=11. V_cont[11] = 0.
    #    Decision at 11: Exercise if K - S_11 > 0.
    #    Y_11 = max(K - S_11, 0).
    # 2. Regress Y_11 on S_10 to get V_cont[10].
    # 3. Decision at 10: Exercise if K - S_10 > V_cont[10].
    #    Y_10 = max(K - S_10, V_cont[10]).
    # 4. Regress Y_10 on S_9 to get V_cont[9].
    # ...
    # 12. Decision at 1: Exercise if K - S_1 > V_cont[1].
    #     Y_1 = max(K - S_1, V_cont[1]).
    #     (We don't need to regress Y_1 on S_0 as there is no exercise at t=0).
    
    # Note: The regression should be done on the in-sample set.
    # The decisions Y_k are calculated for ALL paths (in and out) using the V_cont estimates?
    # No, the "realized" value Y_k for path i is determined by the actual future prices of that path and the optimal policy.
    # Since the optimal policy is defined by the V_cont functions, we can compute Y_k for any path.
    # However, to avoid look-ahead bias in the *estimation*, we regress Y_{k+1} (computed using V_cont[k+1]...) on S_k.
    
    # Let's compute Y_k for all steps for the in-sample paths first?
    # No, Y_k depends on V_cont[k], which depends on Y_{k+1}.
    # So we must go backward.
    
    # Let's store Y for each step for the in-sample paths.
    Y_in = np.zeros((len(in_sample_idx), 12))
    
    # Step 11 (last)
    S_11 = paths_in[:, 11]
    Y_in[:, 11] = np.maximum(K - S_11, 0.0)
    V_cont[11] = 0.0
    
    # Regress Y_11 on S_10 to get V_cont[10]
    S_10 = paths_in[:, 10]
    X_10 = get_basis(S_10)
    # Solve least squares
    try:
        coeffs_10, _, _, _ = np.linalg.lstsq(X_10, Y_in[:, 11], rcond=None)
    except:
        coeffs_10 = np.zeros(3)
    
    # V_cont[10] = X_10 @ coeffs_10
    # But we need V_cont as a function of S to evaluate at other steps.
    # So we store the coefficients.
    coeffs_list = [None] * 12
    coeffs_list[10] = coeffs_10
    
    # Now compute Y_10 for in-sample paths
    V_cont_10_in = X_10 @ coeffs_10
    Y_in[:, 10] = np.maximum(K - S_10, V_cont_10_in)
    
    # Continue backward...
    for k in range(10, 0, -1):
        if time.time() > deadline - 20.0:
            break
            
        S_k = paths_in[:, k]
        S_k_minus_1 = paths_in[:, k-1]
        
        # Regress Y_k on S_{k-1}
        X_k_minus_1 = get_basis(S_k_minus_1)
        try:
            coeffs_k_minus_1, _, _, _ = np.linalg.lstsq(X_k_minus_1, Y_in[:, k], rcond=None)
        except:
            coeffs_k_minus_1 = np.zeros(3)
        
        coeffs_list[k-1] = coeffs_k_minus_1
        
        # Evaluate V_cont[k-1] for in-sample paths at step k-1
        V_cont_k_minus_1_in = X_k_minus_1 @ coeffs_k_minus_1
        Y_in[:, k-1] = np.maximum(K - S_k_minus_1, V_cont_k_minus_1_in)
        
        # Determine critical price for step k-1?
        # The critical price is where (K - S) = V_cont(S).
        # We can find this by searching.
        # For now, we just store the coefficients. We will determine critical prices later.
    
    # Now we have coeffs_list[0..10].
    # coeffs_list[k] approximates V_cont[k] (continuation value at step k) as a function of S_k.
    # Note: coeffs_list[11] is None because V_cont[11]=0.
    
    # Determine critical prices for each step 0..11 (corresponding to exercise dates 1..12)
    # For step k (0-indexed), critical price C_k is the solution to K - S = V_cont_k(S).
    # If no solution in (0, K), then either always exercise (C_k = K) or never exercise (C_k = 0).
    
    def find_critical_price(coeffs, K):
        # V_cont(S) = c0 + c1*S + c2*S^2
        # K - S = c0 + c1*S + c2*S^2
        # c2*S^2 + (c1+1)*S + (c0 - K) = 0
        a = coeffs[2]
        b = coeffs[1] + 1.0
        c = coeffs[0] - K
        
        if abs(a) < 1e-10:
            # Linear equation
            if abs(b) < 1e-10:
                # Constant
                if c < 0:
                    return K # K - S > 0 always? No, K - S = c0. If c0 < 0, K-S < 0?
                    # Wait. K - S = V. If V is constant c0.
                    # K - S = c0 => S = K - c0.
                    # If K - c0 is in [0, K], that's the critical price.
                    crit = K - c0
                    if 0 <= crit <= K:
                        return crit
                    elif crit < 0:
                        return 0.0 # S > K - c0 > K? No.
                    else:
                        return K
                else:
                    return 0.0
            else:
                crit = -c / b
                if 0 <= crit <= K:
                    return crit
                elif crit < 0:
                    return 0.0
                else:
                    return K
        else:
            disc = b**2 - 4*a*c
            if disc < 0:
                # No real roots. Check sign at S=0 and S=K.
                # f(S) = K - S - V(S)
                # We want f(S) = 0.
                f0 = K - 0 - (coeffs[0])
                fK = K - K - (coeffs[0] + coeffs[1]*K + coeffs[2]*K**2)
                if f0 > 0 and fK > 0:
                    return 0.0 # Exercise value > Cont always?
                elif f0 < 0 and fK < 0:
                    return K # Cont > Exercise always?
                else:
                    # Should not happen if no roots and signs differ?
                    # If signs differ, there must be a root.
                    return 0.0
            else:
                sqrt_disc = np.sqrt(disc)
                s1 = (-b + sqrt_disc) / (2*a)
                s2 = (-b - sqrt_disc) / (2*a)
                
                candidates = [s for s in [s1, s2] if 0 <= s <= K]
                if not candidates:
                    # Check behavior
                    f0 = K - coeffs[0]
                    fK = -(coeffs[0] + coeffs[1]*K + coeffs[2]*K**2)
                    if f0 > 0:
                        return 0.0
                    else:
                        return K
                else:
                    # We want the critical price where we switch from Exercise to Cont.
                    # For S < C, Exercise (K-S > V). For S > C, Cont (V > K-S).
                    # So f(S) = K - S - V(S) is positive for S < C and negative for S > C.
                    # Let's check the sign of f(S) at a point below the candidate.
                    # Usually the smaller root is the critical price for a put?
                    # Let's just pick the one that separates the regions correctly.
                    # Test a point S_test = candidates[0] - epsilon
                    # If f(S_test) > 0, then S_test is in Exercise region.
                    # We want the boundary.
                    # Let's pick the candidate that is consistent with f(S) > 0 for small S.
                    # f(0) = K - c0.
                    # If f(0) > 0, we are in Exercise region at S=0.
                    # So the critical price is the root where f crosses from + to -.
                    
                    # Let's evaluate f at the candidates' neighbors.
                    best_crit = 0.0
                    for s in candidates:
                        # Check if f(s - 0.1) > 0 and f(s + 0.1) < 0
                        # Or just use the fact that for a put, the exercise region is S < C.
                        # So we want the largest S such that f(S) >= 0?
                        # Actually, if there are two roots, the region between them might be different.
                        # Let's just pick the root that is in [0, K].
                        # Heuristic: For a put, C is usually the smaller root if both are positive?
                        # Let's check the value at S=0.
                        if f0 > 0:
                            # We start in Exercise. We cross to Cont at the first root.
                            # So the smallest positive root is C.
                            best_crit = min([s for s in candidates if s >= 0])
                        else:
                            # We start in Cont. We cross to Exercise at the first root?
                            # But for a put, we don't exercise at high S.
                            # So if f(0) < 0, we are in Cont at S=0? That means V(0) > K.
                            # This is unusual for a put. V(0) should be K (if r=0) or higher?
                            # With interest, V(0) > K. So f(0) = K - V(0) < 0.
                            # Then we might cross to Exercise at some low S?
                            # No, if V(0) > K, then at S=0, Cont > Ex.
                            # As S increases, Ex decreases. Cont decreases.
                            # It's possible they cross.
                            # Let's just pick the largest root in [0, K] as the boundary?
                            # Let's try to be robust: pick the root that maximizes the objective?
                            # For now, pick the median of candidates.
                            best_crit = np.median(candidates)
                    return best_crit
    
    critical_prices = np.zeros(12)
    for k in range(12):
        if k == 11:
            # Last step, V_cont = 0.
            # K - S > 0 => S < K.
            critical_prices[k] = K
        else:
            if coeffs_list[k] is not None:
                critical_prices[k] = find_critical_price(coeffs_list[k], K)
            else:
                critical_prices[k] = K # Fallback
    
    critical_prices = np.clip(critical_prices, 0, K)
    
    # --- Refinement using Out-of-Sample and Coordinate Descent ---
    
    def evaluate_policy(c_prices, paths_eval):
        M_eval = paths_eval.shape[0]
        exercised = np.zeros(M_eval, dtype=bool)
        exercise_step = np.full(M_eval, -1, dtype=int)
        
        for k in range(12):
            not_exercised = ~exercised
            if not np.any(not_exercised):
                break
            cond = (paths_eval[:, k] <= c_prices[k]) & not_exercised
            exercised[cond] = True
            exercise_step[cond] = k
        
        payoffs = np.zeros(M_eval)
        if np.any(exercised):
            idx = np.arange(M_eval)
            ex_idx = idx[exercised]
            ex_steps = exercise_step[exercised]
            S_ex = paths_eval[ex_idx, ex_steps]
            disc_ex = discount_factors[ex_steps]
            payoffs[ex_idx] = disc_ex * (K - S_ex)
        
        return np.mean(payoffs)
    
    # Evaluate initial policy on out-of-sample to check quality
    # But we want to maximize on the *training* set? The task says "Maximize mean discounted payoff".
    # Usually this implies on the evaluation set, but we only have training paths.
    # The evaluator uses independent evaluation paths.
    # So we should optimize on the training paths, but use out-of-sample for validation if needed.
    # Given the small data, overfitting is a risk.
    # Let's do coordinate descent on the in-sample set, but monitor out-of-sample.
    
    best_c = critical_prices.copy()
    best_val_in = evaluate_policy(best_c, paths_in)
    best_val_out = evaluate_policy(best_c, paths_out)
    
    # Coordinate descent on in-sample
    max_iter = 10
    for iteration in range(max_iter):
        if time.time() > deadline - 10.0:
            break
        
        improved = False
        for k in range(12):
            if time.time() > deadline - 5.0:
                break
                
            c_old = best_c[k]
            
            # Bisection search for the best c_k
            lo, hi = 0.0, K
            best_c_k = c_old
            best_val_k = best_val_in
            
            for _ in range(10):
                if time.time() > deadline - 5.0:
                    break
                mid = (lo + hi) / 2
                test_c = best_c.copy()
                test_c[k] = mid
                val = evaluate_policy(test_c, paths_in)
                
                if val > best_val_k:
                    best_val_k = val
                    best_c_k = mid
                    lo = mid
                else:
                    hi = mid
            
            if best_val_k > best_val_in + 1e-10:
                best_c[k] = best_c_k
                best_val_in = best_val_k
                improved = True
        
        if not improved:
            break
    
    critical_prices = best_c
    critical_prices = np.clip(critical_prices, 0, K)
    
    return ExercisePolicy(critical_prices)
