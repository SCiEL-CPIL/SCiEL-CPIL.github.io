import numpy as np
import time
import random

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build successors and calculate structural metrics
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            successors[p].append(i)
            
    # Calculate Bottom Level (BL)
    bl = [0] * n
    def compute_bl(i):
        if bl[i] != 0:
            return bl[i]
        max_succ_bl = 0
        for s in successors[i]:
            s_bl = compute_bl(s)
            if s_bl > max_succ_bl:
                max_succ_bl = s_bl
        bl[i] = 1 + max_succ_bl
        return bl[i]
        
    for i in range(n):
        compute_bl(i)
    bl_arr = np.array(bl, dtype=float)
    
    # Calculate Rank (number of successors, recursive)
    rank = [0] * n
    def compute_rank(i):
        if rank[i] != 0:
            return rank[i]
        if not successors[i]:
            rank[i] = 1
        else:
            rank[i] = 1 + sum(compute_rank(s) for s in successors[i])
        return rank[i]
        
    for i in range(n):
        compute_rank(i)
    rank_arr = np.array(rank, dtype=float)
    
    # Simulation function: Given priority scores, return number of stations
    def simulate(prio_scores):
        # Map task index to priority
        # We need to simulate the greedy assignment
        # State: 
        # - assigned: boolean array
        # - remaining_time: list of remaining times for each task (or just use times)
        # - available: set of tasks whose predecessors are all assigned
        
        assigned = [False] * n
        remaining_time = times.copy()
        # Count predecessors remaining for each task
        pred_count = [len(p) for p in preds]
        # List of available tasks
        available = [i for i in range(n) if pred_count[i] == 0]
        
        stations = 1
        current_station_time = 0
        
        while True:
            # Find best task to assign
            best_task = -1
            best_prio = -np.inf
            
            for i in available:
                if remaining_time[i] <= (cycle_time - current_station_time):
                    if prio_scores[i] > best_prio:
                        best_prio = prio_scores[i]
                        best_task = i
                        
            if best_task == -1:
                # No task fits, start new station
                stations += 1
                current_station_time = 0
                # Re-check available tasks (same set, but now full capacity)
                # The loop will re-evaluate
                continue
            
            # Assign best_task
            current_station_time += remaining_time[best_task]
            assigned[best_task] = True
            available.remove(best_task)
            
            # Update predecessors for successors
            for s in successors[best_task]:
                pred_count[s] -= 1
                if pred_count[s] == 0:
                    available.append(s)
                    
            # Check if all tasks assigned
            if all(assigned):
                break
                
        return stations

    # Optimization: Simulated Annealing
    # Initial priority: BL + Time
    current_prio = bl_arr * 10 + times
    current_cost = simulate(current_prio)
    
    best_prio = current_prio.copy()
    best_cost = current_cost
    
    start_time = time.time()
    deadline = start_time + 100.0 # 100s safety margin
    
    # Simulated Annealing Parameters
    temperature = 10.0
    cooling_rate = 0.95
    min_temperature = 0.01
    
    # To make it more effective, we can perturb individual task priorities
    # Or use a set of weights for linear combination
    
    # Let's try perturbing the priority vector directly
    # The search space is high dimensional, but N=30 is small.
    
    # A better approach: Search over the weights of the linear combination
    # P = w1*BL + w2*Rank + w3*Time
    # This reduces dimensionality to 3.
    
    # Let's try both: direct perturbation and weight perturbation.
    # Weight perturbation is more stable.
    
    w1, w2, w3 = 1.0, 0.0, 1.0 # Initial: BL + Time
    current_prio = w1 * bl_arr + w2 * rank_arr + w3 * times
    current_cost = simulate(current_prio)
    best_prio = current_prio.copy()
    best_cost = current_cost
    
    temperature = 1.0
    while temperature > min_temperature and time.time() < deadline:
        # Perturb weights
        new_w1 = w1 + random.gauss(0, 0.5)
        new_w2 = w2 + random.gauss(0, 0.5)
        new_w3 = w3 + random.gauss(0, 0.5)
        
        # Ensure non-negative weights? Not strictly necessary, but helpful.
        # Let's allow negative to explore more space, but clamp if needed.
        
        new_prio = new_w1 * bl_arr + new_w2 * rank_arr + new_w3 * times
        new_cost = simulate(new_prio)
        
        if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / temperature):
            w1, w2, w3 = new_w1, new_w2, new_w3
            current_prio = new_prio
            current_cost = new_cost
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_prio = new_prio.copy()
                
        temperature *= cooling_rate
        
        # Occasionally restart with different initial weights to escape local minima
        if random.random() < 0.05:
            w1 = random.uniform(0, 2)
            w2 = random.uniform(0, 2)
            w3 = random.uniform(0, 2)
            current_prio = w1 * bl_arr + w2 * rank_arr + w3 * times
            current_cost = simulate(current_prio)
            if current_cost < best_cost:
                best_cost = current_cost
                best_prio = current_prio.copy()
            temperature = 1.0 # Reset temperature on restart
            
    return best_prio
