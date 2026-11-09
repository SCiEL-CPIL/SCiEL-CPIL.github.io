"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def _topo_order(n, preds):
    """Kahn's algorithm; returns list of tasks in topological order and successors."""
    indeg = np.zeros(n, dtype=int)
    succs = [[] for _ in range(n)]
    for i, ps in enumerate(preds):
        for p in ps:
            succs[p].append(i)
            indeg[i] += 1
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in succs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) < n:
        remaining = [i for i in range(n) if i not in set(order)]
        order.extend(remaining)
    return order, succs

def _simulate_greedy(times, preds, ct, priorities):
    """
    Simulates the greedy station filling process.
    Returns the number of stations used.
    """
    n = len(times)
    # Build adjacency for quick eligibility check
    # We need to know which tasks are predecessors of which
    # preds[i] is list of predecessors of i
    
    # To efficiently find eligible tasks, we can maintain a list of "available" tasks.
    # A task is available if all its predecessors are assigned.
    
    assigned = [False] * n
    # Count of unassigned predecessors for each task
    unassigned_preds = [len(preds[i]) for i in range(n)]
    
    # List of currently available tasks
    # We need to pick the one with highest priority that fits in the current station.
    # Since N is small (30), we can just scan all available tasks each time we add to a station.
    
    stations = 0
    current_load = 0.0
    # We need a way to quickly check if a task fits.
    # Eligible if: not assigned, unassigned_preds[i] == 0, and times[i] <= ct - current_load
    
    while True:
        # Find best task to put in current station
        best_task = -1
        best_priority = -np.inf
        
        for i in range(n):
            if not assigned[i] and unassigned_preds[i] == 0:
                if times[i] <= ct - current_load:
                    if priorities[i] > best_priority:
                        best_priority = priorities[i]
                        best_task = i
        
        if best_task == -1:
            # No task fits in current station
            if current_load > 0:
                # Move to next station
                stations += 1
                current_load = 0.0
                # Continue loop to fill next station
            else:
                # If current station is empty and no task fits, it means no task is eligible?
                # This should not happen if graph is valid and we haven't assigned all.
                # If all tasks are assigned, loop should break.
                # Check if all assigned
                if all(assigned):
                    break
                # If not all assigned but none eligible, this is a deadlock or error in logic.
                # But with topological sort, if we pick available tasks, we shouldn't deadlock unless cycle.
                # Assuming valid DAG.
                break
        else:
            # Assign best_task to current station
            assigned[best_task] = True
            current_load += times[best_task]
            
            # Update unassigned_preds for successors of best_task
            # We need successors list. Let's build it once outside or pass it?
            # For simplicity in this function, we'll just re-check unassigned_preds logic.
            # Wait, unassigned_preds is decremented when a predecessor is assigned.
            # So we need to know who are the successors of best_task.
            # Let's build successors list in priority() and pass it or make it global?
            # Better to build it in priority and pass to simulate.
            
    return stations

def _simulate_greedy_optimized(times, preds, succs, ct, priorities):
    n = len(times)
    assigned = [False] * n
    unassigned_preds = [len(preds[i]) for i in range(n)]
    
    stations = 0
    current_load = 0.0
    
    while True:
        best_task = -1
        best_priority = -np.inf
        
        for i in range(n):
            if not assigned[i] and unassigned_preds[i] == 0:
                if times[i] <= ct - current_load:
                    if priorities[i] > best_priority:
                        best_priority = priorities[i]
                        best_task = i
        
        if best_task == -1:
            if current_load > 0:
                stations += 1
                current_load = 0.0
            else:
                if all(assigned):
                    break
                # Safety break
                break
        else:
            assigned[best_task] = True
            current_load += times[best_task]
            
            for s in succs[best_task]:
                unassigned_preds[s] -= 1
                
    return stations

def priority(instance):
    t0 = time.time()
    deadline = 110.0 # Leave 10s margin
    
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    ct = float(instance["cycle_time"])
    
    # Precompute topo order and successors
    order, succs = _topo_order(n, preds)
    
    # Initial Heuristic: Critical Path + Time
    cp = np.zeros(n, dtype=float)
    for i in reversed(order):
        best = 0.0
        for s in succs[i]:
            if cp[s] > best:
                best = cp[s]
        cp[i] = times[i] + best
        
    # Normalize
    def norm(x):
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
        
    cp_n = norm(cp)
    time_n = norm(times)
    
    # Initial scores
    current_scores = 0.8 * cp_n + 0.2 * time_n
    current_stations = _simulate_greedy_optimized(times, preds, succs, ct, current_scores)
    
    # Local Search: Hill Climbing
    # We will try to perturb the scores to see if we can get fewer stations.
    # Since station count is discrete, we might get stuck.
    # We can also try to optimize for "balance" as a tie-breaker if station count is same?
    # The metric is ONLY station count. So we just want to minimize it.
    
    # Strategy: 
    # 1. Try random perturbations.
    # 2. If we find a better score (fewer stations), keep it.
    # 3. If we don't find a better one for a while, maybe try larger steps or different heuristics?
    
    # Let's try a few different base heuristics and pick the best one first.
    candidates = []
    
    # H1: CP + Time
    candidates.append(0.8 * cp_n + 0.2 * time_n)
    
    # H2: Time only
    candidates.append(time_n)
    
    # H3: CP only
    candidates.append(cp_n)
    
    # H4: CP + Successor Count
    trans_succ_count = np.zeros(n, dtype=int)
    for i in reversed(order):
        c = 0
        for s in succs[i]:
            c += 1 + trans_succ_count[s]
        trans_succ_count[i] = c
    tsucc_n = norm(trans_succ_count.astype(float))
    candidates.append(0.5 * cp_n + 0.3 * time_n + 0.2 * tsucc_n)
    
    # H5: LP-inspired? (Approximation)
    # Tasks with high time and many successors are critical.
    candidates.append(0.4 * cp_n + 0.4 * time_n + 0.2 * tsucc_n)
    
    best_scores = None
    best_stations = float('inf')
    
    for cand in candidates:
        stations = _simulate_greedy_optimized(times, preds, succs, ct, cand)
        if stations < best_stations:
            best_stations = stations
            best_scores = cand.copy()
            
    # Now hill climb on best_scores
    # We want to minimize stations.
    # Perturbation: add small noise.
    # If we find same number of stations, we might want to keep the one that has more "slack" in the last station?
    # No, the metric is just station count.
    
    # However, if we have the same number of stations, we are "tied". 
    # The evaluator picks the highest priority eligible task.
    # If two assignments yield the same number of stations, does it matter which one we produce?
    # The evaluator uses OUR scores to make the assignment.
    # So if we output scores A, the evaluator makes assignment X (stations S).
    # If we output scores B, the evaluator makes assignment Y (stations S).
    # Both score S. So it doesn't matter which we output, as long as S is minimal.
    
    # But wait, maybe we can find a score vector that leads to a *better* assignment if we tweak it?
    # Yes, that's what hill climbing is for.
    
    step_size = 0.05
    no_improve_count = 0
    max_no_improve = 5000 # Limit iterations to avoid wasting time
    
    while time.time() - t0 < deadline:
        if no_improve_count > max_no_improve:
            break
            
        # Generate a perturbed score vector
        # Add gaussian noise
        noise = np.random.normal(0, step_size, n)
        new_scores = best_scores + noise
        
        # Optional: Normalize new scores? Not strictly necessary for comparison, but keeps values bounded.
        # The evaluator compares priorities, so relative order matters.
        
        new_stations = _simulate_greedy_optimized(times, preds, succs, ct, new_scores)
        
        if new_stations < best_stations:
            best_stations = new_stations
            best_scores = new_scores
            no_improve_count = 0
            # Increase step size slightly? Or keep it?
            # Let's keep it constant for stability, or decrease if we want to polish.
            # Actually, if we found a new optimum, we might want to explore around it more.
        else:
            no_improve_count += 1
            # If stuck, maybe reduce step size to polish?
            if no_improve_count % 100 == 0:
                step_size *= 0.9
                if step_size < 0.001:
                    break # Stuck in local optimum, stop
            
    return np.asarray(best_scores, dtype=float)

# EVOLVE-BLOCK-END
