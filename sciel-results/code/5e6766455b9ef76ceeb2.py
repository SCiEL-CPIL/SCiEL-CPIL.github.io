"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def priority(instance):
    """
    Returns a priority score for each activity to be used in a Serial SGS.
    Uses a combination of critical path length, resource intensity, and local search.
    """
    start_time = time.time()
    # Safety margin: 140s out of 150s limit
    deadline = start_time + 140.0

    # 1. Extract and process instance data
    N = len(instance["durations"])
    durations = np.array(instance["durations"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)  # Shape (N, R)
    capacities = np.array(instance["capacities"], dtype=float) # Shape (R,)
    R = len(capacities)
    
    # Handle predecessors
    preds = instance.get("predecessors", [[] for _ in range(N)])
    if isinstance(preds, np.ndarray):
        if preds.ndim == 2:
             preds = [np.where(preds[i] > 0)[0].tolist() for i in range(N)]
        else:
             preds = [[] for _ in range(N)]
    elif isinstance(preds, list):
         normalized_preds = []
         for p in preds:
             if isinstance(p, (list, tuple, np.ndarray)):
                 normalized_preds.append(list(map(int, p)))
             else:
                 normalized_preds.append([])
         preds = normalized_preds
    else:
         preds = [[] for _ in range(N)]

    # Build successors
    succs = [[] for _ in range(N)]
    for i in range(N):
        for p in preds[i]:
            succs[p].append(i)
            
    # 2. Compute Static Heuristics
    # Topological Sort
    in_degree = [len(p) for p in preds]
    queue = [i for i in range(N) if in_degree[i] == 0]
    topo_order = []
    indeg_temp = in_degree[:]
    q = queue[:]
    while q:
        u = q.pop(0)
        topo_order.append(u)
        for v in succs[u]:
            indeg_temp[v] -= 1
            if indeg_temp[v] == 0:
                q.append(v)
                
    # Compute Critical Path Length (CPL) backwards
    cpl = np.zeros(N)
    for i in range(len(topo_order) - 1, -1, -1):
        u = topo_order[i]
        max_succ_cpl = 0
        for v in succs[u]:
            if cpl[v] > max_succ_cpl:
                max_succ_cpl = cpl[v]
        cpl[u] = durations[u] + max_succ_cpl
        
    # Compute Resource Intensity (RI): sum of demand/capacity ratios
    cap_safe = np.where(capacities > 0, capacities, 1.0)
    ratios = demands / cap_safe  # Shape (N, R)
    ri = np.sum(ratios, axis=1)  # Shape (N,)
    
    # 3. Define Fast SGS Simulation
    def simulate_sgs(priorities):
        """
        Serial SGS simulation.
        Returns makespan.
        """
        scheduled = [False] * N
        # active: list of (finish_time, idx)
        active = []
        remaining_res = capacities.copy()
        time_current = 0.0
        makespan = 0.0
        scheduled_count = 0
        
        while scheduled_count < N:
            # Find eligible
            eligible = []
            for i in range(N):
                if not scheduled[i]:
                    is_eligible = True
                    for p in preds[i]:
                        if not scheduled[p]:
                            is_eligible = False
                            break
                    if is_eligible:
                        eligible.append(i)
                        
            if not eligible:
                # Wait for next event
                if not active:
                    break # Should not happen if valid
                min_finish = min(f for f, _ in active)
                time_current = min_finish
                # Process completions
                new_active = []
                for f, idx in active:
                    if f > time_current:
                        new_active.append((f, idx))
                    else:
                        remaining_res += demands[idx]
                        makespan = max(makespan, f)
                active = new_active
                continue
                
            # Sort eligible by priority (descending)
            eligible.sort(key=lambda x: priorities[x], reverse=True)
            
            # Attempt to schedule
            any_scheduled = False
            for i in eligible:
                # Check resources
                if np.all(remaining_res >= demands[i]):
                    scheduled[i] = True
                    scheduled_count += 1
                    remaining_res -= demands[i]
                    finish_t = time_current + durations[i]
                    active.append((finish_t, i))
                    makespan = max(makespan, finish_t)
                    any_scheduled = True
                    
            if not any_scheduled:
                # Wait
                if not active:
                    break
                min_finish = min(f for f, _ in active)
                time_current = min_finish
                new_active = []
                for f, idx in active:
                    if f > time_current:
                        new_active.append((f, idx))
                    else:
                        remaining_res += demands[idx]
                        makespan = max(makespan, f)
                active = new_active
                
        return makespan

    # 4. Generate Initial Candidates
    candidates = []
    
    # Heuristic 1: CPL
    candidates.append(cpl.copy())
    
    # Heuristic 2: CPL * (1 + RI)
    candidates.append(cpl * (1.0 + ri))
    
    # Heuristic 3: CPL * (1 + 2*RI)
    candidates.append(cpl * (1.0 + 2.0 * ri))
    
    # Heuristic 4: LPT (Duration)
    candidates.append(durations.copy())
    
    # Heuristic 5: Duration * (1 + RI)
    candidates.append(durations * (1.0 + ri))

    # 5. Local Search (Hill Climbing)
    best_prio = None
    best_ms = float('inf')
    
    # Evaluate initial candidates
    for p in candidates:
        ms = simulate_sgs(p)
        if ms < best_ms:
            best_ms = ms
            best_prio = p.copy()
            
    # If we still have time, do local search
    # We will perform swaps or random perturbations
    # Since N=30, we can afford a decent number of iterations.
    
    # Strategy: Randomly swap two elements in the priority vector?
    # Or slightly perturb?
    # Since the SGS sorts by priority, the exact values don't matter, only the order.
    # So we can work with a permutation of indices or just relative priorities.
    # However, the function returns a score vector.
    # Let's stick to perturbing the vector.
    
    # A better local search for priority lists:
    # 1. Take the best permutation.
    # 2. Try swapping adjacent elements in the sorted order?
    # Or just random swaps in the index space.
    
    # Let's try a simple hill climbing on the priority vector itself.
    # If we change a priority, it might change the order.
    
    # Alternative: Generate a permutation, convert to priorities, simulate.
    # This is cleaner.
    
    # Convert best_prio to a permutation
    # Sort indices by priority descending
    best_order = np.argsort(best_prio)[::-1]
    
    def perm_to_prio(order):
        # Assign priorities such that order[0] has highest priority, etc.
        # Just use rank.
        prio = np.zeros(N)
        for rank, idx in enumerate(order):
            prio[idx] = N - rank
        return prio
    
    current_order = best_order.copy()
    current_prio = perm_to_prio(current_order)
    current_ms = best_ms
    
    max_iters = 5000
    iters = 0
    
    while time.time() < deadline and iters < max_iters:
        # Randomly pick two indices in the order and swap them
        i, j = random.sample(range(N), 2)
        new_order = current_order.copy()
        new_order[i], new_order[j] = new_order[j], new_order[i]
        
        new_prio = perm_to_prio(new_order)
        new_ms = simulate_sgs(new_prio)
        
        if new_ms < current_ms:
            current_ms = new_ms
            current_order = new_order
            current_prio = new_prio
        else:
            # Occasionally accept worse moves? (Simulated Annealing style)
            # For simplicity and speed, let's stick to hill climbing or add small probability
            # Given the time constraint, pure hill climbing is safer.
            pass
        iters += 1
        
    return current_prio

# EVOLVE-BLOCK-END
