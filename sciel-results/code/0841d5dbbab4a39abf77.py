import numpy as np
import time
import random

def order_jobs(processing_times):
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We aim to stop at 145s.
    deadline = start_time + 145.0
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Precompute cumulative processing times for each job
    # cum_job[j][m] = sum of processing times for job j on machines 0..m
    # This allows O(1) calculation of job duration on any machine segment
    cum_job = np.zeros((n_jobs, n_machines))
    for j in range(n_jobs):
        cum_job[j, 0] = processing_times[j, 0]
        for m in range(1, n_machines):
            cum_job[j, m] = cum_job[j, m-1] + processing_times[j, m]
            
    # Helper to calculate makespan for a given permutation (list of job indices)
    # Used for initialization and final verification
    def calc_makespan(perm):
        completion = np.zeros(n_machines)
        for j in perm:
            job_times = processing_times[j]
            for m in range(n_machines):
                if m == 0:
                    st = completion[0]
                else:
                    st = max(completion[m], completion[m-1])
                completion[m] = st + job_times[m]
        return completion[-1]

    # 1. NEH Construction (Nawaz, Enscore, Ham)
    # Sort jobs by total processing time in descending order
    total_times = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-total_times)
    
    seq = [sorted_jobs[0]]
    current_ms = calc_makespan(seq)
    
    for j in sorted_jobs[1:]:
        best_pos = 0
        best_makespan = float('inf')
        
        # Try inserting job j at each position in the current sequence
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [j] + seq[pos:]
            ms = calc_makespan(new_seq)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        seq.insert(best_pos, j)
        current_ms = best_makespan
        
        # Time check
        if time.time() > deadline:
            return list(seq)

    best_seq = list(seq)
    best_ms = current_ms
    
    # 2. Fast Local Search with Delta Evaluation
    # We use a greedy local search with two neighborhoods: Swap and Insertion.
    # Instead of recalculating makespan from scratch (O(N*M)), we estimate the delta.
    
    # For a swap of jobs i and k (i < k) in sequence S:
    # The makespan change is complex but can be bounded. 
    # A simpler and very effective approach for FSP is to use the "Insertion" neighborhood
    # with a fast evaluation.
    
    # Let's implement a fast evaluation for Insertion.
    # Moving job at index i to index j (i < j).
    # The jobs between i and j-1 shift right.
    # The job at i shifts to j.
    
    # Actually, for FSP, exact O(1) delta for arbitrary moves is hard.
    # However, we can use a heuristic delta or simply accept that for small N (which synthetic instances likely are, or medium), 
    # the O(N*M) calculation might be acceptable if N is small.
    # But if N is large (e.g., 100), O(N^2 * M) per pass is too slow.
    
    # Let's check the size. If N is small, we can do more restarts.
    # If N is large, we need to be efficient.
    
    # Strategy:
    # 1. Use NEH.
    # 2. Perform Local Search with Swap and Insertion.
    # 3. Use a fast "approximate" makespan update if possible, or just optimize the constant factors.
    
    # Optimization: Vectorize the makespan calculation?
    # NumPy doesn't easily handle the sequential dependency of flow shop.
    
    # Let's stick to the efficient Python loop but add a "stall" detection.
    # If no improvement is found in a full pass, we stop.
    
    # To speed up the local search, we will use a "tabu-like" memory or just random restarts.
    # Given the time limit, multiple restarts with perturbation is often better than one long search.
    
    n_restarts = 10
    # Adjust restarts based on problem size
    if n_jobs > 50:
        n_restarts = 5
    elif n_jobs > 20:
        n_restarts = 10
    else:
        n_restarts = 20
        
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
            
        if restart == 0:
            current_seq = list(best_seq)
            current_ms = best_ms
        else:
            # Perturb the best sequence found so far
            current_seq = list(best_seq)
            # Randomly swap 2-5 pairs
            n_perturb = random.randint(2, min(5, len(current_seq)//2))
            for _ in range(n_perturb):
                i = random.randint(0, len(current_seq) - 2)
                j = random.randint(i + 1, len(current_seq) - 1)
                current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
            current_ms = calc_makespan(current_seq)
            
        # Local Search
        improved = True
        max_iterations = 1000
        iteration = 0
        
        while improved and time.time() < deadline - 2.0 and iteration < max_iterations:
            improved = False
            iteration += 1
            n = len(current_seq)
            
            # Neighborhood 1: Insertion (Or-opt)
            # Try moving each job to a better position
            # To speed up, we can randomly sample moves or iterate sequentially.
            # Sequential is safer for convergence.
            
            for i in range(n - 1):
                if time.time() > deadline - 2.0:
                    break
                job = current_seq[i]
                best_pos = i
                best_ms_local = current_ms
                
                # Check positions to the right
                # We can limit the search window for speed if N is large
                # But for correctness, we should check all.
                for j in range(i + 1, n):
                    # Construct new sequence efficiently?
                    # Creating a new list is O(N).
                    # This makes the inner loop O(N^2) per pass, and O(N^3) total.
                    # For N=100, 10^6 * M operations. If M=10, 10^7 ops. Python is slow.
                    
                    # Optimization: Only evaluate if the move *might* be better?
                    # Hard to know without calculation.
                    
                    new_seq = current_seq[:i] + current_seq[i+1:j+1] + [job] + current_seq[j+1:]
                    ms = calc_makespan(new_seq)
                    
                    if ms < best_ms_local:
                        best_ms_local = ms
                        best_pos = j
                        
                if best_pos != i:
                    # Perform the move
                    job = current_seq[i]
                    current_seq = current_seq[:i] + current_seq[i+1:best_pos+1] + [job] + current_seq[best_pos+1:]
                    current_ms = best_ms_local
                    improved = True
                    if current_ms < best_ms:
                        best_ms = current_ms
                        best_seq = list(current_seq)
                    # Break and restart pass? Or continue?
                    # Restarting the pass ensures we don't miss interactions.
                    break
                    
            if not improved:
                # Neighborhood 2: Swap
                for i in range(n - 1):
                    if time.time() > deadline - 2.0:
                        break
                    for j in range(i + 1, n):
                        new_seq = current_seq[:]
                        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                        ms = calc_makespan(new_seq)
                        if ms < current_ms:
                            current_ms = ms
                            current_seq = new_seq
                            improved = True
                            if current_ms < best_ms:
                                best_ms = current_ms
                                best_seq = list(current_seq)
                            break
                    if improved:
                        break

    return best_seq
