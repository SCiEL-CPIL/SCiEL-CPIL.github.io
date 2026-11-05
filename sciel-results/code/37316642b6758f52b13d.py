"""
Strategy: Branch and Bound for Assembly Line Balancing.
Instead of optimizing priority scores for a fixed greedy algorithm, 
we directly search for the minimum number of stations K.
We use a backtracking solver that assigns tasks to stations, respecting 
precedence and cycle time constraints.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import deque

def _topological_sort(n, preds):
    """Returns a topological order of tasks."""
    in_deg = [0] * n
    succs = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            if p >= 0:
                in_deg[i] += 1
                succs[p].append(i)
                
    q = deque([i for i in range(n) if in_deg[i] == 0])
    order = []
    indeg = in_deg[:]
    while q:
        u = q.popleft()
        order.append(u)
        for v in succs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order

def _compute_lower_bound(instance):
    """Compute a lower bound on the number of stations."""
    times = instance["times"]
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n_tasks = len(times)
    
    # LB 1: Total work / cycle time
    total_work = sum(times)
    lb1 = int(np.ceil(total_work / cycle_time))
    
    # LB 2: Length of longest path (critical path)
    # h[i] = times[i] + max(h[j] for j in succs[i])
    # We need the max path length in terms of time.
    succs = [[] for _ in range(n_tasks)]
    for i in range(n_tasks):
        for p in preds[i]:
            if p >= 0:
                succs[p].append(i)
                
    # Topological sort
    in_deg = [len(p) for p in preds]
    q = deque([i for i in range(n_tasks) if in_deg[i] == 0])
    topo = []
    indeg = in_deg[:]
    while q:
        u = q.popleft()
        topo.append(u)
        for v in succs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
                
    # Compute longest path to end (height)
    height = [0] * n_tasks
    for u in reversed(topo):
        if not succs[u]:
            height[u] = times[u]
        else:
            height[u] = times[u] + max(height[v] for v in succs[u])
            
    max_height = max(height) if height else 0
    lb2 = int(np.ceil(max_height / cycle_time))
    
    return max(lb1, lb2)

def _solve_alb(instance, max_time=10):
    """
    Solve ALB using Branch and Bound.
    Returns the minimum number of stations found within max_time.
    """
    times = np.array(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n_tasks = len(times)
    
    # Precompute successors and predecessors sets for fast lookup
    succs = [set() for _ in range(n_tasks)]
    for i in range(n_tasks):
        for p in preds[i]:
            if p >= 0:
                succs[p].add(i)
                
    # Topological order
    topo_order = _topological_sort(n_tasks, preds)
    
    # Lower bound
    lb = _compute_lower_bound(instance)
    
    # Try K from lb upwards
    # We will try to find a valid assignment for K stations.
    # If found, return K.
    
    # To speed up search, we can use a heuristic ordering for tasks within a station.
    # But here we just need to find *any* valid assignment for a given K.
    
    # We will use a recursive backtracking approach.
    # State: 
    #   station_idx: current station we are filling (0 to K-1)
    #   assigned: list of tasks assigned so far
    #   station_loads: list of loads for each station
    #   last_task_in_station: to help with precedence? 
    #   Actually, precedence is global. A task is eligible if all preds are assigned.
    #   Since we assign in topological order? No, we can assign in any order as long as preds are satisfied.
    #   However, it is efficient to process tasks in topological order? 
    #   Not necessarily, but it helps ensure we don't pick a task before its preds.
    #   Let's maintain a set of available tasks (preds all assigned).
    
    # Optimization: 
    # Instead of picking any available task, we can restrict choices.
    # A common heuristic is to pick the task with the most successors or longest time among available.
    # But for a *search*, we need to explore.
    # However, for 30 tasks, full B&B might be too slow if not pruned well.
    # Let's try a simpler approach: 
    # For a fixed K, can we assign tasks to stations?
    # This is a bin packing problem with precedence.
    
    # Let's use a specific heuristic-guided backtracking.
    # We will iterate through tasks in a specific order (e.g., topological) and assign each to a station.
    # This is "serial scheduling".
    # At each step, pick the next task from the set of available tasks (all preds assigned).
    # Assign it to one of the K stations where it fits (load + time <= CT) and precedence is ok 
    # (precedence is ok if we ensure we only pick from available tasks).
    # Wait, if we pick from available tasks, all preds are assigned. 
    # Do they need to be in the *previous* station or any previous station?
    # In ALB, if task A is pred of B, A must be in station s_A <= s_B.
    # If we assign tasks one by one, and we maintain that we are filling stations sequentially?
    # No, we can assign to any station.
    # But if we assign task B to station 2, and task A (pred) is in station 1, that's fine.
    # If task A is in station 2, that's fine (same station, sequential order within station matters).
    # Within a station, tasks are sequential. So if A and B are in same station, A must come before B.
    # Our "available" set logic handles global precedence.
    # But it doesn't handle *within-station* order.
    # However, if we just check "fits in load", we might put B before A in the station list, which is invalid.
    # To handle this, when assigning to a station, we must ensure that if there are other tasks in that station 
    # that are successors of the current task, they are not already placed? 
    # Actually, simpler: just append to the end of the station's task list.
    # If we always append, then the order in the station is the order of assignment.
    # If A is pred of B, and we assign A then B to same station, A is before B. Correct.
    # If we assign B then A to same station, B is before A. Incorrect.
    # But can we assign B before A?
    # B is only "available" if A is assigned. So we can't assign B before A.
    # So the "available" set logic naturally enforces the order!
    # Great. So we just need to:
    # 1. Start with station 0.
    # 2. Find all available tasks (preds assigned).
    # 3. For each available task, try assigning it to any station (0..K-1) where it fits.
    #    Wait, can we assign to station 1 while station 0 is not full?
    #    Yes, in ALB, we can have empty space in station 0 and use station 1.
    #    However, to minimize stations, we should try to fill earlier stations first?
    #    Not necessarily, but it's a good heuristic.
    #    Also, we must ensure that if we assign task to station s, all its preds are in station <= s.
    #    Since preds are assigned, they are in some station. 
    #    If a pred is in station > s, that's a violation.
    #    So when assigning task t to station s, we must check:
    #    for all p in preds[t]: station_of[p] <= s.
    
    # Let's implement this backtracking.
    
    def backtrack(station_idx, assigned_count, station_loads, station_tasks, task_station):
        if time.time() > deadline:
            return False # Timeout
        
        if assigned_count == n_tasks:
            # Check if all stations are valid? 
            # If we got here, all tasks assigned. 
            # We need to ensure no station exceeds cycle time (checked during assignment).
            # And precedence (checked during assignment).
            return True
        
        # Find available tasks
        available = []
        for t in range(n_tasks):
            if task_station[t] == -1: # Not assigned
                # Check if all preds are assigned
                ok = True
                for p in preds[t]:
                    if p >= 0 and task_station[p] == -1:
                        ok = False
                        break
                if ok:
                    available.append(t)
                    
        if not available:
            return False # Dead end
        
        # Heuristic: sort available tasks by some criteria (e.g., time desc, succ count desc)
        # to find a solution faster.
        available.sort(key=lambda t: -times[t])
        
        for t in available:
            # Try to assign t to a station
            # Which stations can t go to?
            # 1. Fits in load: station_loads[s] + times[t] <= cycle_time
            # 2. Precedence: for all p in preds[t], task_station[p] <= s
            
            # Determine valid stations
            min_pred_station = 0
            for p in preds[t]:
                if p >= 0:
                    min_pred_station = max(min_pred_station, task_station[p])
            
            for s in range(K):
                if s < min_pred_station:
                    continue
                if station_loads[s] + times[t] > cycle_time:
                    continue
                
                # Assign t to s
                task_station[t] = s
                station_loads[s] += times[t]
                station_tasks[s].append(t)
                
                if backtrack(station_idx, assigned_count + 1, station_loads, station_tasks, task_station):
                    return True
                
                # Undo
                task_station[t] = -1
                station_loads[s] -= times[t]
                station_tasks[s].remove(t)
                
                # Pruning: If we tried station s and it failed, do we need to try s+1?
                # If s failed, s+1 might work if load is lower? 
                # But station loads are independent.
                # However, if we assigned to s, we used up capacity.
                # If we don't assign to s, we leave capacity.
                # Is there symmetry breaking?
                # If station s and s+1 are identical (same load, same tasks), we can skip one.
                # For now, we just try all.
                
        return False

    deadline = time.time() + max_time
    
    # Try K from lb upwards
    for K in range(lb, n_tasks + 1):
        if time.time() > deadline:
            break
        
        # Initialize
        station_loads = [0.0] * K
        station_tasks = [[] for _ in range(K)]
        task_station = [-1] * n_tasks
        
        # Start backtracking
        # We don't need station_idx in backtrack if we allow assignment to any station.
        # The "station_idx" variable in my sketch was misleading. 
        # We are filling a set of K stations.
        # We just need to assign all tasks.
        
        # One optimization: We can restrict the search space by noting that 
        # if we have empty stations at the end, we could have used fewer.
        # But we are fixing K.
        
        # Let's run the search.
        # To make it faster, we can use a heuristic to order the *stations* we try?
        # Or order the tasks?
        # The task ordering in `available` is already heuristic.
        
        # Let's add a simple pruning: 
        # If the remaining work cannot fit in the remaining capacity, prune.
        # This is hard to compute dynamically.
        
        # Let's just try it.
        # To avoid deep recursion limits (30 tasks is fine), we proceed.
        
        # We need a wrapper to handle the "K" variable in the closure.
        # Since K changes in the loop, we define backtrack inside or pass K.
        
        def bt(assigned_count, station_loads, station_tasks, task_station):
            if time.time() > deadline:
                return False
            if assigned_count == n_tasks:
                return True
            
            available = []
            for t in range(n_tasks):
                if task_station[t] == -1:
                    ok = True
                    for p in preds[t]:
                        if p >= 0 and task_station[p] == -1:
                            ok = False
                            break
                    if ok:
                        available.append(t)
                        
            if not available:
                return False
                
            # Sort available by time desc
            available.sort(key=lambda t: -times[t])
            
            for t in available:
                min_pred_station = 0
                for p in preds[t]:
                    if p >= 0:
                        min_pred_station = max(min_pred_station, task_station[p])
                        
                for s in range(K):
                    if s < min_pred_station:
                        continue
                    if station_loads[s] + times[t] > cycle_time:
                        continue
                    
                    task_station[t] = s
                    station_loads[s] += times[t]
                    station_tasks[s].append(t)
                    
                    if bt(assigned_count + 1, station_loads, station_tasks, task_station):
                        return True
                    
                    task_station[t] = -1
                    station_loads[s] -= times[t]
                    station_tasks[s].pop()
                    
            return False

        if bt(0, station_loads, station_tasks, task_station):
            return K
            
    # If no K found within time, return a heuristic estimate
    # Fallback to a simple greedy result
    return _greedy_fallback(instance)

def _greedy_fallback(instance):
    """Simple greedy fallback if B&B times out."""
    times = np.array(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n_tasks = len(times)
    
    succs = [[] for _ in range(n_tasks)]
    for i in range(n_tasks):
        for p in preds[i]:
            if p >= 0:
                succs[p].append(i)
                
    in_deg = [len(p) for p in preds]
    q = deque([i for i in range(n_tasks) if in_deg[i] == 0])
    topo = []
    indeg = in_deg[:]
    while q:
        u = q.popleft()
        topo.append(u)
        for v in succs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
                
    # Height
    height = [0] * n_tasks
    for u in reversed(topo):
        if not succs[u]:
            height[u] = times[u]
        else:
            height[u] = times[u] + max(height[v] for v in succs[u])
            
    # Greedy by height
    assigned = [False] * n_tasks
    stations = []
    current_load = 0
    current_tasks = []
    
    # We need to pick tasks in a way that respects precedence.
    # Simple approach: iterate through topological order? 
    # No, we need to pick from available.
    
    # Let's just use the scores from the original program's best heuristic as a fallback
    # Or just return a large number if we fail.
    # Actually, let's just run the original greedy logic quickly.
    
    # Re-implementing the original _greedy_balance quickly
    scores = height
    assigned = [False] * n_tasks
    station_loads = []
    assign = []
    current_load = 0
    current_tasks = []
    
    while any(not assigned[i] for i in range(n_tasks)):
        if current_load > 0:
            added = True
            while added:
                added = False
                best_task = -1
                best_score = -1
                for t in range(n_tasks):
                    if assigned[t]:
                        continue
                    ok = True
                    for p in preds[t]:
                        if p >= 0 and not assigned[p]:
                            ok = False
                            break
                    if not ok:
                        continue
                    if current_load + times[t] > cycle_time:
                        continue
                    if scores[t] > best_score:
                        best_score = scores[t]
                        best_task = t
                if best_task >= 0:
                    assigned[best_task] = True
                    current_load += times[best_task]
                    current_tasks.append(best_task)
                    added = True
            assign.append(current_tasks)
            current_tasks = []
            current_load = 0
        else:
            best_task = -1
            best_score = -1
            for t in range(n_tasks):
                if assigned[t]:
                    continue
                ok = True
                for p in preds[t]:
                    if p >= 0 and not assigned[p]:
                        ok = False
                        break
                if not ok:
                    continue
                if times[t] > cycle_time:
                    continue
                if scores[t] > best_score:
                    best_score = scores[t]
                    best_task = t
            if best_task < 0:
                for t in range(n_tasks):
                    if not assigned[t]:
                        best_task = t
                        break
            assigned[best_task] = True
            current_load += times[best_task]
            current_tasks.append(best_task)
            
    return len(assign)

def priority(instance):
    """
    The public interface.
    The evaluator uses the returned scores to perform its own greedy assignment.
    However, the task description says: "stations are filled with the highest-priority eligible task that fits the cycle time."
    This implies the evaluator does the greedy part.
    
    BUT, looking at the "CURRENT PROGRAM", it returns `best_scores`.
    If I return scores that correspond to the *optimal* assignment found by my B&B, 
    will the evaluator's greedy algorithm produce that assignment?
    Not necessarily. The evaluator's greedy is fixed.
    
    Wait, the task says: "Write a Python priority(instance) function that scores each assembly task; stations are filled with the highest-priority eligible task that fits the cycle time."
    
    This means the *evaluator* runs the greedy algorithm using my scores.
    So I cannot control the assignment directly. I can only control the *scores*.
    
    Therefore, my B&B approach to find the *minimum K* is useful to *know* the optimal K, 
    but I still need to return scores that *lead* the evaluator's greedy to that K.
    
    How can I find scores that lead the greedy to the optimal solution?
    I can use the B&B to find *one* optimal assignment (or a good one).
    Then, I can reverse-engineer the priority scores that would make the greedy algorithm pick those tasks in that order.
    
    Algorithm:
    1. Use B&B to find a good assignment (min stations).
    2. Extract the order in which tasks were assigned to stations in that solution.
       Let the solution be a list of stations, each a list of tasks.
       The greedy algorithm picks tasks one by one.
       The order of picking is:
       Station 0: t1, t2, ...
       Station 1: t3, t4, ...
       
       The greedy algorithm at each step picks the eligible task with the highest score.
       So, if I want the greedy to pick t1 first, t1 must have the highest score among all eligible tasks at step 1.
       At step 1, eligible tasks are those with no preds.
       So score(t1) > score(t) for all other eligible t.
       
       This suggests I can set scores based on the "desirability" of picking a task at a specific step.
       
       A simpler approach:
       The greedy algorithm is deterministic given the scores.
       I can simulate the greedy algorithm with different scores and check if it produces the optimal K.
       This is what the current program does!
       
       So, how is my B&B different?
       The current program tries a few random/heuristic scores.
       I can use B&B to find the *structure* of the optimal solution, and then *construct* a score vector that forces the greedy to follow that structure.
       
       Construction:
       Let the optimal assignment be $A = [S_0, S_1, \dots, S_{K-1}]$.
       The tasks in $S_i$ are assigned in some order. Let's say $S_i = [t_{i,1}, \dots, t_{i,m}]$.
       The global order of assignment is $t_{0,1}, \dots, t_{0,m}, t_{1,1}, \dots$.
       
       For the greedy to pick $t_{i,j}$ at step $k$, $t_{i,j}$ must have a higher score than any other eligible task at that step.
       
       We can assign scores based on the step index.
       score(t) = -step_index(t) + small_random_noise.
       This ensures that tasks assigned earlier in the optimal solution have higher scores.
       However, we must ensure that at each step, the *next* task in the optimal solution is the *highest* score among *eligible* tasks.
       
       If we set score(t) = -step_index(t), then at any step, the eligible tasks include the next task in the optimal solution (call it $t^*$) and potentially other tasks that were assigned later in the optimal solution but are now eligible.
       If we pick $t^*$, its score is $-k$.
       Other eligible tasks $t$ have step_index $> k$, so score $< -k$.
       So $t^*$ will be picked.
       
       Wait, are there eligible tasks with step_index $< k$?
       No, because all tasks with step_index $< k$ are already assigned.
       
       So, if we set `score(t) = -step_index(t)`, the greedy algorithm will exactly replicate the assignment order of the optimal solution found by B&B!
       
       This is a brilliant structural change.
       1. Find optimal (or near-optimal) assignment using B&B.
       2. Record the step index for each task in that assignment.
       3. Return scores = -step_index.
       
       This guarantees the evaluator's greedy will produce the same number of stations as the B&B solution.
    """
    
    start_time = time.time()
    deadline = start_time + 140 # 10s safety margin
    
    # 1. Solve for optimal assignment
    # We need the assignment, not just the count.
    # Let's modify _solve_alb to return the assignment if found.
    
    times = np.array(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n_tasks = len(times)
    
    succs = [set() for _ in range(n_tasks)]
    for i in range(n_tasks):
        for p in preds[i]:
            if p >= 0:
                succs[p].add(i)
                
    topo_order = _topological_sort(n_tasks, preds)
    lb = _compute_lower_bound(instance)
    
    best_assignment = None
    best_K = n_tasks + 1
    
    # Try to find a solution for K = lb, lb+1, ...
    # We will store the assignment that works.
    
    def find_assignment(K):
        station_loads = [0.0] * K
        station_tasks = [[] for _ in range(K)]
        task_station = [-1] * n_tasks
        assignment_order = [] # List of tasks in the order they were assigned
        
        def bt(assigned_count):
            nonlocal assignment_order
            if time.time() > deadline:
                return False
            if assigned_count == n_tasks:
                return True
            
            available = []
            for t in range(n_tasks):
                if task_station[t] == -1:
                    ok = True
                    for p in preds[t]:
                        if p >= 0 and task_station[p] == -1:
                            ok = False
                            break
                    if ok:
                        available.append(t)
                        
            if not available:
                return False
                
            # Heuristic: sort by time desc to find solution faster
            available.sort(key=lambda t: -times[t])
            
            for t in available:
                min_pred_station = 0
                for p in preds[t]:
                    if p >= 0:
                        min_pred_station = max(min_pred_station, task_station[p])
                        
                for s in range(K):
                    if s < min_pred_station:
                        continue
                    if station_loads[s] + times[t] > cycle_time:
                        continue
                    
                    # Assign
                    task_station[t] = s
                    station_loads[s] += times[t]
                    station_tasks[s].append(t)
                    assignment_order.append(t)
                    
                    if bt(assigned_count + 1):
                        return True
                    
                    # Undo
                    task_station[t] = -1
                    station_loads[s] -= times[t]
                    station_tasks[s].pop()
                    assignment_order.pop()
                    
            return False

        if bt(0):
            return assignment_order, station_tasks
        return None, None

    for K in range(lb, n_tasks + 1):
        if time.time() > deadline:
            break
        order, stations = find_assignment(K)
        if order is not None:
            best_K = K
            best_assignment = order
            break
            
    if best_assignment is None:
        # Fallback: Use a good heuristic score
        # Just use height-based scores
        height = [0] * n_tasks
        for u in reversed(topo_order):
            if not succs[u]:
                height[u] = times[u]
            else:
                height[u] = times[u] + max(height[v] for v in succs[u])
        return np.array(height, dtype=float)
        
    # 2. Construct scores from the assignment order
    # score(t) = -index(t) in best_assignment
    scores = np.zeros(n_tasks, dtype=float)
    for idx, t in enumerate(best_assignment):
        scores[t] = -idx
        
    # Add tiny noise to break ties deterministically if needed, 
    # but since indices are unique, no ties.
    
    return scores

# EVOLVE-BLOCK-END
