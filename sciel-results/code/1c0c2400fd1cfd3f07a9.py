"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using a PFS-specific TSP decomposition.
    Fixes the first job (highest total processing time) and optimizes the rest via TSP
    on incremental makespan contributions.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Fast fallback for very large instances
    if n_jobs * n_machines > 500000:
        return list(np.argsort(-processing_times.sum(axis=1)))
    
    def makespan(seq):
        """Compute makespan for a given sequence of job indices."""
        if len(seq) == 0:
            return 0.0
        completion = np.zeros(n_machines)
        for job in seq:
            prev = completion[0]
            curr = prev + processing_times[job, 0]
            completion[0] = curr
            for m in range(1, n_machines):
                prev = curr
                curr = max(prev, completion[m]) + processing_times[job, m]
                completion[m] = curr
        return completion[-1]
    
    # Pre-compute the N x N matrix of incremental makespan contributions.
    # cost[i, j] = incremental makespan when job j is appended after job i.
    # We approximate this by the completion time of job j on the last machine
    # minus the completion time of job i on the last machine, assuming the
    # rest of the sequence is "frozen" or by a simpler proxy.
    # Actually, for a TSP-like heuristic, we want a cost that reflects the
    # "delay" or "load" added. A good proxy is the total processing time of j
    # plus the max waiting time it would incur if i were the previous job.
    # However, computing exact incremental costs for all pairs is O(N^2 * M).
    # Let's compute a matrix where cost[i, j] is the makespan increase if j
    # is placed immediately after i in a sequence of just [i, j].
    # This is a valid lower bound / proxy for the incremental cost.
    
    # Compute completion times for single jobs
    # For a pair (i, j), the completion time of j is:
    # C_j = max(C_i, C_{j, m-1}) + p_{j, m}
    # This is complex. Let's use a simpler, well-known heuristic for TSP on FSP:
    # The cost of moving j after i is often approximated by the processing time of j
    # plus the idle time it causes.
    
    # Let's stick to the strategy: "edge weights are approximated by the incremental
    # makespan contribution of inserting job j after job i (computed via a pre-calculated
    # N x N matrix)".
    
    # To compute the exact incremental cost of inserting j after i in a context is hard
    # without the context. But for a TSP-like construction, we can use the cost of the
    # sequence [i, j] minus the cost of [i].
    # Cost([i]) = max(p_i0, p_i1, ..., p_iM) + p_iM? No, makespan of [i] is just the sum of p_i along the path.
    # Actually, makespan of [i] is simply the sum of processing times of i on all machines? No.
    # In a flow shop, job i starts at M1 at time 0. Finishes M1 at p_i0. Starts M2 at p_i0. Finishes M2 at p_i0 + p_i1.
    # So makespan of [i] is sum(p_i).
    
    # Makespan of [i, j]:
    # Job i finishes at sum(p_i).
    # Job j starts M1 at sum(p_i). Finishes M1 at sum(p_i) + p_j0.
    # Job j starts M2 at max(sum(p_i) + p_j0, sum(p_i) + p_i1)? No.
    # Job j can start M2 when M2 is free (after job i finishes M2) AND job j finishes M1.
    # Job i finishes M2 at p_i0 + p_i1.
    # Job j finishes M1 at sum(p_i) + p_j0.
    # So job j starts M2 at max(p_i0 + p_i1, sum(p_i) + p_j0).
    
    # Let's precompute the "tail" and "head" or just use the pairwise makespan increment.
    
    # cost[i, j] = makespan([i, j]) - makespan([i])
    
    cost_matrix = np.zeros((n_jobs, n_jobs))
    for i in range(n_jobs):
        ms_i = makespan([i])
        for j in range(n_jobs):
            if i == j:
                cost_matrix[i, j] = 0
            else:
                ms_ij = makespan([i, j])
                cost_matrix[i, j] = ms_ij - ms_i
    
    # Strategy:
    # 1. Fix the first job as the one with the highest total processing time.
    #    This is a strong heuristic for FSP (NEH starts with the largest job).
    # 2. Treat the remaining jobs as a TSP where we start at the fixed job
    #    and must visit all other jobs exactly once, minimizing the sum of
    #    edge costs.
    
    # However, the sum of pairwise costs is not exactly the makespan of the whole sequence.
    # But it is a good surrogate.
    
    # We will use a Nearest Neighbor heuristic to construct a path, then 2-opt to improve it.
    
    def tsp_nearest_neighbor(start_node, nodes_to_visit, cost_mat):
        """Construct a TSP path using Nearest Neighbor."""
        path = [start_node]
        unvisited = set(nodes_to_visit)
        
        while unvisited:
            current = path[-1]
            # Find nearest unvisited node
            best_node = None
            best_cost = float('inf')
            for node in unvisited:
                c = cost_mat[current, node]
                if c < best_cost:
                    best_cost = c
                    best_node = node
            
            if best_node is None:
                break
                
            path.append(best_node)
            unvisited.remove(best_node)
            
        return path
    
    def tsp_2opt(path, cost_mat, time_budget):
        """Improve a TSP path using 2-opt."""
        n = len(path)
        if n <= 2:
            return path
        
        # Precompute path cost
        def path_cost(p):
            c = 0.0
            for k in range(len(p) - 1):
                c += cost_mat[p[k], p[k+1]]
            return c
        
        best_path = list(path)
        best_cost = path_cost(best_path)
        
        while time.time() < time_budget:
            improved = False
            best_delta = 0.0
            best_i, best_j = -1, -1
            
            for i in range(n - 1):
                for j in range(i + 1, n):
                    # 2-opt move: reverse segment [i+1 : j+1]
                    # New edges: (p[i], p[j+1]) and (p[j], p[i+1]) if j+1 < n
                    # Wait, standard 2-opt on a path (not cycle):
                    # If we reverse segment [i : j], the edges affected are:
                    # (p[i-1], p[i]) and (p[j], p[j+1]) become (p[i-1], p[j]) and (p[i], p[j+1])
                    # Let's use the standard reversal of [i : j] inclusive.
                    
                    # Indices in path: 0 to n-1
                    # Reverse path[i : j+1]
                    
                    # Old cost parts:
                    # Edge (i-1, i) if i > 0
                    # Edge (j, j+1) if j < n-1
                    # Internal edges in [i : j] are reversed but sum is same if symmetric?
                    # Cost matrix is NOT symmetric.
                    # So we must recompute or carefully update.
                    # Given the complexity and small N, let's just recompute path cost for the candidate.
                    
                    candidate = best_path[:i] + best_path[i:j+1][::-1] + best_path[j+1:]
                    c = path_cost(candidate)
                    delta = c - best_cost
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
            
            if best_delta < -1e-10:
                i, j = best_i, best_j
                best_path[i:j+1] = best_path[i:j+1][::-1]
                best_cost += best_delta
                improved = True
            else:
                break
        
        return best_path
    
    # Generate initial solutions
    candidates = []
    
    # 1. TSP-based construction
    # Fix first job
    totals = processing_times.sum(axis=1)
    first_job = int(np.argmax(totals))
    remaining_jobs = [j for j in range(n_jobs) if j != first_job]
    
    # Nearest Neighbor from first_job
    path_nn = tsp_nearest_neighbor(first_job, remaining_jobs, cost_matrix)
    candidates.append(path_nn)
    
    # 2. NEH heuristic (as a strong baseline)
    def neh_heuristic():
        order = list(np.argsort(-totals))
        seq = [order[0]]
        for job in order[1:]:
            best_pos = 0
            best_ms = float('inf')
            for pos in range(len(seq) + 1):
                trial = seq[:pos] + [job] + seq[pos:]
                ms = makespan(trial)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
            seq = seq[:best_pos] + [job] + seq[best_pos:]
        return seq
    
    candidates.append(neh_heuristic())
    
    # 3. Palmer heuristic
    def palmer_heuristic():
        n = n_jobs
        priorities = []
        for job in range(n_jobs):
            p_first = processing_times[job, 0]
            p_last = processing_times[job, -1]
            priorities.append(n * p_first - p_last)
        priorities = np.array(priorities)
        return list(np.argsort(-priorities))
    
    candidates.append(palmer_heuristic())
    
    # 4. LPT
    candidates.append(list(np.argsort(-totals)))
    
    # 5. Random permutations
    n_random = min(3, max(1, n_jobs // 10))
    for _ in range(n_random):
        seq = list(range(n_jobs))
        random.shuffle(seq)
        candidates.append(seq)
    
    best_seq = None
    best_ms = float('inf')
    
    # Apply local search (TSP 2-opt on the surrogate cost, then verify with makespan)
    # Actually, we should apply the TSP 2-opt to the TSP candidate specifically.
    # For other candidates, we can apply a standard 2-opt on makespan or just use them as seeds.
    
    for i, seq in enumerate(candidates):
        if time.time() > deadline:
            break
        
        # If it's the TSP candidate, apply TSP 2-opt
        if i == 0:
            # Apply TSP 2-opt
            remaining = deadline - time.time()
            time_budget = min(remaining * 0.5, 30.0)
            seq_tsp = tsp_2opt(seq, cost_matrix, time_budget)
            # Verify with actual makespan
            ms = makespan(seq_tsp)
            if ms < best_ms:
                best_ms = ms
                best_seq = seq_tsp
        else:
            # For other candidates, apply a simple 2-opt on makespan
            # To save time, we can just evaluate them directly or do a light local search.
            # Let's do a light 2-opt on makespan for the NEH candidate (usually best).
            if i == 1: # NEH
                remaining = deadline - time.time()
                time_budget = min(remaining * 0.3, 20.0)
                
                def local_search_makespan(seq, time_budget):
                    seq = list(seq)
                    n = len(seq)
                    best_ms = makespan(seq)
                    while time.time() < time_budget:
                        improved = False
                        best_delta = 0.0
                        best_i, best_j = -1, -1
                        for i_idx in range(n):
                            for j_idx in range(i_idx + 2, n + 1):
                                seq2 = seq[:i_idx] + seq[i_idx:j_idx][::-1] + seq[j_idx:]
                                ms = makespan(seq2)
                                delta = ms - best_ms
                                if delta < best_delta:
                                    best_delta = delta
                                    best_i, best_j = i_idx, j_idx
                        if best_delta < -1e-10:
                            i_idx, j_idx = best_i, best_j
                            seq[i_idx:j_idx] = seq[i_idx:j_idx][::-1]
                            best_ms += best_delta
                            improved = True
                        else:
                            break
                    return seq, best_ms
                
                seq_ls, ms = local_search_makespan(seq, time_budget)
                if ms < best_ms:
                    best_ms = ms
                    best_seq = seq_ls
            else:
                ms = makespan(seq)
                if ms < best_ms:
                    best_ms = ms
                    best_seq = list(seq)
    
    if best_seq is None:
        best_seq = candidates[0]
        
    return list(best_seq)
# EVOLVE-BLOCK-END
