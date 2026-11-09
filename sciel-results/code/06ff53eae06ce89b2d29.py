import numpy as np
import time
import sys
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    start_time = time.time()
    timeout = 580  # Safety margin before 630s limit
    
    best_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    best_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    best_score = 1.0

    # Column generation approach:
    # Instead of constructing full 10K x K matrix, iteratively find violated constraints
    # and add them to the LP.
    
    def solve_with_column_generation(K, max_iterations=100, time_limit=500):
        """
        Solve the LP using column generation (constraint generation).
        
        The full problem has constraints for all x in [1, 10K]:
          sum_{k=2}^{K} {x/k} * f(k) >= -1.0001
        
        We start with a small set of constraints and iteratively add violated ones.
        """
        indices = np.arange(2, K + 1)
        n_vars = K - 1
        c = np.log(indices) / indices  # Objective: minimize sum c_k * f_k
        bounds = [(-10, 10)] * n_vars
        
        # Start with a small set of x values
        # Use x values that are likely to be binding
        initial_x = np.array([1, 2, 3, 5, 10, K, 2*K, 5*K, 10*K], dtype=np.int64)
        initial_x = initial_x[initial_x <= 10*K]
        
        active_constraints_x = set(initial_x.tolist())
        
        for iteration in range(max_iterations):
            if time.time() - start_time > time_limit:
                break
            
            # Build constraint matrix for active constraints
            x_active = np.array(sorted(active_constraints_x), dtype=np.int64)
            if len(x_active) == 0:
                break
            
            # A_{i,k} = {x_i / k} for each active x_i and k in indices
            # We want: A @ f >= -1.0001, i.e., -A @ f <= 1.0001
            A = (x_active[:, None] / indices[None, :]) % 1
            A = -A  # For linprog: A_ub @ x <= b_ub
            b = np.full(len(x_active), 1.0001)
            
            # Solve LP
            try:
                res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
            except Exception as e:
                print(f"LP solve error at iteration {iteration}: {e}", file=sys.stderr)
                return None
            
            if res.status != 0:
                # LP is infeasible or unbounded with current constraints
                # Try adding more constraints
                if res.status == 2:  # Infeasible
                    return None
                continue
            
            f_vals = res.x
            
            # Check if any constraint is violated
            # Compute M(x) = -S*x + sum f(k) floor(x/k) for all x in [1, 10K]
            # where S = sum f(k)/k
            S = np.sum(f_vals / indices)
            
            # Efficiently find violated constraints
            # M(x) = -S*x + sum_{k=2}^{K} f(k) * floor(x/k)
            # We need M(x) <= 1.0001 for all x in [1, 10K]
            
            # Sample x values to find violations
            # Use a combination of uniform sampling and targeted sampling
            x_sample = np.random.choice(10*K, size=min(5000, 10*K), replace=False)
            # Also include some structured samples
            x_structured = np.arange(1, 10*K+1, max(1, 10*K//5000))
            x_check = np.unique(np.concatenate([x_sample, x_structured]))
            
            # Compute M(x) for sampled x values
            # M(x) = -S*x + sum f(k) floor(x/k)
            floor_parts = np.floor(x_check[:, None] / indices[None, :])
            M_vals = -S * x_check + np.dot(f_vals, floor_parts)
            
            # Find violated constraints
            violated = M_vals > 1.0001 + 1e-6
            if not np.any(violated):
                # All sampled constraints satisfied, check all x to be sure
                x_all = np.arange(1, 10*K+1)
                floor_parts_all = np.floor(x_all[:, None] / indices[None, :])
                M_all = -S * x_all + np.dot(f_vals, floor_parts_all)
                if np.max(M_all) <= 1.0001 + 1e-6:
                    # Solution is feasible!
                    score = -np.sum(f_vals * np.log(indices) / indices)
                    return f_vals, score
                else:
                    # Add all violated x values
                    violated_all = M_all > 1.0001 + 1e-6
                    for x_v in x_all[violated_all]:
                        active_constraints_x.add(int(x_v))
            else:
                # Add violated x values to active constraints
                for x_v in x_check[violated]:
                    active_constraints_x.add(int(x_v))
        
        return None
    
    # Try a range of K values with column generation
    K_candidates = [100, 200, 500, 1000, 2000, 5000]
    
    for K in K_candidates:
        if time.time() - start_time > timeout:
            break
        
        try:
            result = solve_with_column_generation(K, max_iterations=50, time_limit=timeout - (time.time() - start_time))
            
            if result is not None:
                f_vals, score = result
                if score > best_score:
                    best_score = score
                    # Construct domain and values
                    domain = np.arange(1, K + 1, dtype=np.int64)
                    values = np.zeros(K, dtype=np.float64)
                    values[0] = 0.0  # f(1) orig, evaluator adjusts
                    values[1:] = f_vals
                    best_domain = domain
                    best_values = values
                    print(f"K={K}: score={score:.4f}", file=sys.stderr)
                    
        except Exception as e:
            print(f"Error for K={K}: {e}", file=sys.stderr)
            continue
    
    return best_domain, best_values
