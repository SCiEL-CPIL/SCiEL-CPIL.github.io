import numpy as np
import time
import random
from scipy.optimize import linprog

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    demands_arr = np.array(demands)
    capacities_arr = np.array(capacities)
    open_cost_arr = np.array(open_cost)
    assign_cost_arr = np.array(assign_cost)
    
    # 1. Solve LP Relaxation for Guidance
    # Variables: x_ij (100*20), y_j (20)
    # Objective: sum c_ij x_ij + sum f_j y_j
    c_obj = np.concatenate([assign_cost_arr.flatten(), open_cost_arr])
    
    # Constraints
    # 1. sum_j x_ij = 1
    A_eq = np.zeros((n_customers, n_customers * n_facilities + n_facilities))
    for i in range(n_customers):
        A_eq[i, i*n_facilities : (i+1)*n_facilities] = 1.0
    b_eq = np.ones(n_customers)
    
    # 2. Capacity: sum_i d_i x_ij <= C_j y_j
    A_ub = np.zeros((n_facilities, n_customers * n_facilities + n_facilities))
    b_ub = np.zeros(n_facilities)
    for j in range(n_facilities):
        for i in range(n_customers):
            A_ub[j, i*n_facilities + j] = demands_arr[i]
        A_ub[j, n_customers * n_facilities + j] = -capacities_arr[j]
        
    # 3. Linking: x_ij <= y_j
    A_link = np.zeros((n_customers * n_facilities, n_customers * n_facilities + n_facilities))
    b_link = np.zeros(n_customers * n_facilities)
    idx = 0
    for i in range(n_customers):
        for j in range(n_facilities):
            A_link[idx, i*n_facilities + j] = 1.0
            A_link[idx, n_customers * n_facilities + j] = -1.0
            idx += 1
            
    A_ub_total = np.vstack([A_ub, A_link])
    b_ub_total = np.concatenate([b_ub, b_link])
    bounds = [(0, 1)] * (n_customers * n_facilities + n_facilities)
    
    x_solution = None
    try:
        res = linprog(c_obj, A_ub=A_ub_total, b_ub=b_ub_total, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.success:
            x_solution = res.x
    except Exception:
        pass
        
    # 2. Generate Initial Solution
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    if x_solution is not None:
        x_vals = x_solution[:n_customers * n_facilities].reshape(n_customers, n_facilities)
        # Sort customers by "confidence" (max x_ij) descending? Or just random?
        # Let's sort by demand descending to place big ones first, using LP guide.
        order = np.argsort(-demands_arr)
        
        for i in order:
            # Get facilities sorted by LP value descending
            fac_order = np.argsort(-x_vals[i])
            assigned = False
            for f in fac_order:
                if load[f] + demands_arr[i] <= capacities_arr[f]:
                    result[i] = f
                    load[f] += demands_arr[i]
                    assigned = True
                    break
            if not assigned:
                # Fallback: assign to facility with most remaining capacity
                remaining = capacities_arr - load
                best_f = np.argmax(remaining)
                result[i] = best_f
                load[best_f] += demands_arr[i]
    else:
        # Fallback Greedy
        order = np.argsort(-demands_arr)
        for i in order:
            feasible = [f for f in range(n_facilities) if load[f] + demands_arr[i] <= capacities_arr[f]]
            if feasible:
                costs = [assign_cost_arr[i, f] for f in feasible]
                best_f = feasible[np.argmin(costs)]
                result[i] = best_f
                load[best_f] += demands_arr[i]
            else:
                best_f = np.argmax(capacities_arr - load)
                result[i] = best_f
                load[best_f] += demands_arr[i]

    # 3. Simulated Annealing
    def get_cost(res):
        used = set(res)
        open_c = sum(open_cost_arr[j] for j in used)
        assign_c = sum(assign_cost_arr[i, res[i]] for i in range(n_customers))
        return open_c + assign_c
        
    current_cost = get_cost(result)
    best_result = result.copy()
    best_cost = current_cost
    
    # Recalculate load for current
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        load[result[i]] += demands_arr[i]
        
    T = 1.0
    T_min = 1e-5
    alpha = 0.9995
    
    steps = 0
    max_steps = 500000 # Cap steps to avoid infinite loop if time check is slow
    
    while time.time() < deadline and steps < max_steps:
        # Check time more frequently
        if steps % 100 == 0 and time.time() > deadline:
            break
            
        # Choose Move Type
        if random.random() < 0.7:
            # Move
            i = random.randint(0, n_customers - 1)
            current_f = result[i]
            # Pick a new facility
            new_f = random.randint(0, n_facilities - 1)
            if new_f == current_f:
                continue
                
            # Check Capacity
            if load[new_f] + demands_arr[i] > capacities_arr[new_f]:
                continue
                
            # Calculate Delta
            delta_assign = assign_cost_arr[i, new_f] - assign_cost_arr[i, current_f]
            
            # Open Cost Delta
            delta_open = 0
            new_load_current = load[current_f] - demands_arr[i]
            new_load_new = load[new_f] + demands_arr[i]
            
            if new_load_current == 0:
                delta_open -= open_cost_arr[current_f]
            if new_load_new == demands_arr[i]: # Was empty
                delta_open += open_cost_arr[new_f]
                
            total_delta = delta_assign + delta_open
            
            if total_delta < 0 or random.random() < np.exp(-total_delta / T):
                # Accept
                load[current_f] -= demands_arr[i]
                load[new_f] += demands_arr[i]
                result[i] = new_f
                current_cost += total_delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_result = result.copy()
        else:
            # Swap
            i = random.randint(0, n_customers - 1)
            k = random.randint(0, n_customers - 1)
            if i == k:
                continue
                
            f_i = result[i]
            f_k = result[k]
            if f_i == f_k:
                continue
                
            # Check Capacity for Swap
            # i moves to f_k, k moves to f_i
            # load[f_k] + d_i <= C[f_k] AND load[f_i] + d_k <= C[f_i]
            if load[f_k] + demands_arr[i] > capacities_arr[f_k]:
                continue
            if load[f_i] + demands_arr[k] > capacities_arr[f_i]:
                continue
                
            # Calculate Delta
            delta_assign = (assign_cost_arr[i, f_k] - assign_cost_arr[i, f_i]) + \
                           (assign_cost_arr[k, f_i] - assign_cost_arr[k, f_k])
            
            # Open Cost Delta
            # Loads change:
            # load[f_i] - d_i + d_k
            # load[f_k] - d_k + d_i
            # Others unchanged.
            
            old_load_i = load[f_i]
            old_load_k = load[f_k]
            
            new_load_i = old_load_i - demands_arr[i] + demands_arr[k]
            new_load_k = old_load_k - demands_arr[k] + demands_arr[i]
            
            delta_open = 0
            # Facility f_i
            if old_load_i == 0: # Was empty, now has d_k (since d_i removed, but wait, if old_load_i==0, then result[i] couldn't be f_i? No, result[i]=f_i implies load>=d_i. So old_load_i >= d_i > 0)
                pass 
            # Actually, if a facility becomes empty, we save cost. If it becomes non-empty, we pay.
            
            # f_i status change
            if old_load_i == demands_arr[i] and new_load_i == 0:
                delta_open -= open_cost_arr[f_i]
            elif old_load_i == demands_arr[i] and new_load_i > 0:
                # Was only i, now has k. Still open. No change in open cost? 
                # Wait, if it was open (load>0) and remains open (load>0), delta is 0.
                # If it was open and becomes empty, delta is -open.
                # If it was empty and becomes open, delta is +open.
                pass
            # f_k status change
            if old_load_k == demands_arr[k] and new_load_k == 0:
                delta_open -= open_cost_arr[f_k]
                
            # Check if f_i becomes empty
            if new_load_i == 0:
                delta_open -= open_cost_arr[f_i]
            # Check if f_k becomes empty
            if new_load_k == 0:
                delta_open -= open_cost_arr[f_k]
                
            # Wait, the logic above is flawed because we subtracted "becomes empty" but didn't add "becomes non-empty".
            # Let's recalculate open cost delta robustly.
            old_open_i = 1 if old_load_i > 0 else 0
            new_open_i = 1 if new_load_i > 0 else 0
            old_open_k = 1 if old_load_k > 0 else 0
            new_open_k = 1 if new_load_k > 0 else 0
            
            delta_open = (new_open_i - old_open_i) * open_cost_arr[f_i] + \
                         (new_open_k - old_open_k) * open_cost_arr[f_k]
            
            total_delta = delta_assign + delta_open
            
            if total_delta < 0 or random.random() < np.exp(-total_delta / T):
                # Accept
                load[f_i] = new_load_i
                load[f_k] = new_load_k
                result[i] = f_k
                result[k] = f_i
                current_cost += total_delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_result = result.copy()
        
        T *= alpha
        if T < T_min:
            T = T_min
        steps += 1
        
    return best_result
