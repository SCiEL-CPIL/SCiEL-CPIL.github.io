"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def priority(instance):
    """
    Returns a priority score for each activity to be used in a Serial SGS.
    
    Strategy:
    Instead of returning a static priority vector derived from heuristics and local search,
    this function performs a full Serial SGS simulation internally. 
    It uses a DYNAMIC priority rule that changes based on the current schedule state.
    
    The dynamic rule prioritizes activities based on:
    1. Critical Path Length (CPL) - static importance.
    2. Marginal Resource Pressure - how much this activity consumes relative to 
       currently available resources and the "tightness" of the resource constraints.
       
    The function returns the priority scores assigned during this simulated run.
    While the SGS in the evaluator uses a static priority list, returning the scores 
    used in a high-quality dynamic simulation serves as a proxy for a "good" ordering.
    However, to maximize the chance of the evaluator's SGS producing a good makespan,
    we actually use the dynamic simulation to find the BEST STATIC ORDER that approximates 
    the dynamic behavior, or simply return the dynamic scores themselves if we assume 
    the evaluator might re-simulate. 
    
    Wait, the contract says: "Write a Python priority(instance) function that scores each activity; 
    a serial schedule-generation scheme starts the highest-priority eligible activity..."
    
    If the evaluator's SGS uses the returned scores STATICALLY (i.e., it sorts eligible activities 
    by these fixed scores at every step), then returning a static vector is required.
    
    The key insight from the "inspiration" and the stagnation directive is to change the 
    representation or optimization method.
    
    Let's try a different optimization method: 
    Instead of Hill Climbing on the priority vector, we will use a **Greedy Constructive Heuristic 
    with Adaptive Resource Weighting** to generate the priority vector.
    
    Actually, looking at the inspirations:
    - Inspiration 1: Greedy Constructive Heuristic with Adaptive Resource Weighting.
    - Inspiration 2: Dynamic priority.
    
    Since the evaluator uses a STATIC SGS (it takes the priority vector and uses it to sort 
    eligible activities at each step), we cannot implement true dynamic priority in the 
    `priority` function itself unless the evaluator calls `priority` repeatedly or passes state. 
    The contract says `priority(instance) -> one score per activity`. This implies a static vector.
    
    Therefore, the "Dynamic" strategy in Inspiration 2 likely means using a dynamic metric 
    to *compute* the static scores, or the evaluator is actually dynamic (unlikely given standard SGS).
    
    Let's look at the "COMPUTE BUDGET" and "STAGNATION" hints.
    "Model the scheduling problem as a maximum flow problem... minimum cut capacity... directly 
    quantifies the resource bottleneck impact".
    
    This suggests computing a score based on network flow/min-cut concepts.
    
    Alternative Structural Change:
    The current program uses `simulate_sgs` to evaluate candidates. This is O(N^2 * R) per evaluation.
    With N=30, this is fast.
    
    Let's try a **Simulated Annealing** approach with a more sophisticated move set and 
    a better initial solution, or **Genetic Algorithm**.
    
    Actually, the most promising structural change for RCPSP is often **Iterated Local Search (ILS)** 
    with a strong perturbation, or using **Tabu Search**.
    
    Let's implement a **Tabu Search** on the priority permutation.
    Neighborhood: Swap two activities in the priority list.
    Tabu List: Recent swaps are forbidden to avoid cycling.
    Aspiration Criterion: If a tabu move leads to a better solution than the best found so far, it is allowed.
    
    This is structurally different from the simple Hill Climbing in the current program.
    
    Also, we will improve the initial solution generation using a **Greedy Heuristic with Dynamic Weights** 
    computed on the fly, which might provide a better starting point than static CPL/RI.
    
    Let's combine:
    1. Compute a base priority using a "Resource-Adjusted Critical Path".
    2. Use Tabu Search to refine the priority order.
    
    Resource-Adjusted Critical Path:
    For each activity i, score = CPL[i] * (1 + sum(demand[i,r] / capacity[r])^alpha)
    This is similar to current.
    
    Let's focus on the **Tabu Search** aspect as the structural change.
    """
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin

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
            
    # 2. Compute Static Heuristics for Initial Solution
    # Topological Sort
    in_degree = [len(p) for p in preds]
    q = [i for i in range(N) if in_degree[i] == 0]
    topo_order = []
    indeg_temp = in_degree[:]
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
        
    # Compute Resource Intensity (RI)
    cap_safe = np.where(capacities > 0, capacities, 1.0)
    ratios = demands / cap_safe
    ri = np.sum(ratios, axis=1)
    
    # Initial Priority: Weighted combination
    # We will create a few diverse initial permutations
    def make_perm_from_scores(scores):
        # Sort indices by score descending
        return np.argsort(scores)[::-1]

    # Candidate 1: CPL
    perm1 = make_perm_from_scores(cpl)
    # Candidate 2: CPL * (1 + RI)
    scores2 = cpl * (1.0 + ri)
    perm2 = make_perm_from_scores(scores2)
    # Candidate 3: Random
    perm3 = np.random.permutation(N)
    # Candidate 4: Duration
    perm4 = make_perm_from_scores(durations)

    candidates = [perm1, perm2, perm3, perm4]

    # 3. Fast SGS Simulation
    def simulate_sgs(order):
        """
        Simulates SGS given a priority order (permutation of 0..N-1).
        Returns makespan.
        """
        scheduled = [False] * N
        active = [] # (finish_time, idx)
        remaining_res = capacities.copy()
        time_current = 0.0
        makespan = 0.0
        scheduled_count = 0
        
        # Pre-compute priority rank for fast lookup
        # rank[i] = position of i in order (0 is highest)
        # Actually, we just need to sort eligible by their position in order.
        # To make it O(1) comparison, we can create a rank array.
        rank = np.zeros(N, dtype=int)
        for pos, idx in enumerate(order):
            rank[idx] = pos
            
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
                continue
                
            # Sort eligible by rank (lower rank = higher priority)
            eligible.sort(key=lambda x: rank[x])
            
            any_scheduled = False
            for i in eligible:
                if np.all(remaining_res >= demands[i]):
                    scheduled[i] = True
                    scheduled_count += 1
                    remaining_res -= demands[i]
                    finish_t = time_current + durations[i]
                    active.append((finish_t, i))
                    makespan = max(makespan, finish_t)
                    any_scheduled = True
                    
            if not any_scheduled:
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

    # 4. Tabu Search
    best_order = None
    best_ms = float('inf')
    
    # Evaluate initial candidates
    for p in candidates:
        ms = simulate_sgs(p)
        if ms < best_ms:
            best_ms = ms
            best_order = p.copy()
            
    current_order = best_order.copy()
    current_ms = best_ms
    
    # Tabu List: store (i, j) pairs or (j, i). Since swap is symmetric, store sorted tuple.
    tabu_list = []
    tabu_tenure = 10 # Length of tabu list
    
    iters = 0
    max_iters = 20000
    
    while time.time() < deadline and iters < max_iters:
        # Generate neighborhood: all possible swaps
        # For N=30, 435 swaps. 20000 iters * 435 = 8.7M sims. Too slow?
        # 1 sim is ~0.1ms? 8.7M * 0.1ms = 870s. Too slow.
        # We need to sample the neighborhood or use a faster heuristic.
        
        # Let's sample a subset of swaps per iteration.
        # Or, reduce max_iters.
        # Let's try sampling 50 random swaps per iteration.
        
        best_neighbor_order = None
        best_neighbor_ms = current_ms
        best_swap = None
        
        # Sample 20 random swaps
        for _ in range(20):
            i, j = random.sample(range(N), 2)
            swap_key = (min(i, j), max(i, j))
            
            # Check Tabu
            if swap_key in tabu_list:
                # Aspiration: only if it improves global best (we don't know ms yet)
                # For simplicity, skip tabu moves unless we are stuck.
                # Or just skip.
                continue
                
            new_order = current_order.copy()
            new_order[i], new_order[j] = new_order[j], new_order[i]
            
            new_ms = simulate_sgs(new_order)
            
            if new_ms < best_neighbor_ms:
                best_neighbor_ms = new_ms
                best_neighbor_order = new_order
                best_swap = swap_key
                
        if best_swap is not None:
            # Accept if better, or if aspiration criterion met (better than global best)
            if best_neighbor_ms < current_ms or best_neighbor_ms < best_ms:
                current_order = best_neighbor_order
                current_ms = best_neighbor_ms
                
                if current_ms < best_ms:
                    best_ms = current_ms
                    best_order = current_order.copy()
                    
                # Update Tabu List
                tabu_list.append(best_swap)
                if len(tabu_list) > tabu_tenure:
                    tabu_list.pop(0)
            else:
                # No improvement, maybe add a random swap to tabu to force exploration?
                # Or just keep going.
                pass
                
        iters += 1
        
    # Convert best_order to priority scores
    # The evaluator sorts by score descending.
    # So highest priority should have highest score.
    # If best_order[0] is the highest priority activity, it should get the highest score.
    prio = np.zeros(N)
    for rank, idx in enumerate(best_order):
        prio[idx] = N - rank
        
    return prio

# EVOLVE-BLOCK-END
