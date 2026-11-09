"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    """
    Main partition function using Parallel Tempering (Replica Exchange Simulated Annealing).
    
    Strategy:
    1. Map binary labels {0, 1} to spin variables s in {-1, 1}.
       Let s_i = 1 if label is 0, s_i = -1 if label is 1.
       An edge (i, j) is cut if s_i != s_j.
       Cut size = 0.5 * sum_{i,j} A_{ij} (1 - s_i s_j)
       Maximizing cut is equivalent to minimizing E = -sum_{i,j} A_{ij} s_i s_j = -s^T A s.
    
    2. Use Parallel Tempering with a set of replicas at different temperatures.
       - Low temperature replicas focus on refining local optima.
       - High temperature replicas explore the global landscape.
       - Periodic swaps between replicas allow good configurations to "diffuse" 
         from high T (where they were found) to low T (where they are refined).
    
    3. Vectorized updates for high throughput within the time budget.
    """
    n = len(adjacency)
    # Ensure adjacency is a numpy array for efficient ops
    A = np.asarray(adjacency)
    
    # Safety margin: evaluator kills at 150s. We target finishing by 145s.
    deadline = time.time() + 145.0
    
    # Initialize spins: random +1 / -1
    # We maintain a 2D array of shape (num_replicas, n)
    num_replicas = 16
    
    # Temperatures: geometric progression from low to high
    # T=0 is greedy/local search. T=high is random.
    # We want the highest T to be "hot enough" to flip many bits.
    # Typical heuristic: T_high ~ n (or max degree)
    # Let's use a range that covers the relevant energy scales.
    # Max possible energy change for one flip is bounded by degree.
    # Let's set T levels: 0.1, 0.2, ..., up to ~10.0
    temps = np.logspace(np.log10(0.1), np.log10(10.0), num=num_replicas)
    
    # Initialize all replicas with the same random start? 
    # No, better to initialize independently or slightly perturbed to break symmetry.
    # But for simplicity and speed, let's start them all with a random state, 
    # then let them diverge.
    initial_spins = np.random.choice([-1, 1], size=(num_replicas, n))
    
    # Precompute degrees for potential use, though not strictly needed for the vectorized update
    # degrees = A.sum(axis=1)
    
    # Track best cut size and corresponding spins
    # Cut size = 0.5 * (sum(A) - sum(s^T A s))
    # sum(A) is constant. We just need to minimize s^T A s.
    total_edges = A.sum() // 2 # Since symmetric, sum(A) is 2 * edges
    
    def calc_energy(spins):
        # Returns s^T A s for all replicas (shape: num_replicas)
        # Efficient: (spins @ A) * spins, then sum axis 1
        return np.sum(spins @ A * spins, axis=1)
    
    # Initial energies
    energies = calc_energy(initial_spins)
    
    # Best tracking
    best_energy = np.min(energies)
    best_spins = initial_spins[np.argmin(energies)].copy()
    best_cut = total_edges - 0.5 * best_energy # Wait, formula:
    # Cut = 0.5 * sum_{i,j} A_{ij} (1 - s_i s_j) = 0.5 * (sum A - sum s_i s_j A_{ij})
    # sum A = 2 * total_edges
    # Cut = 0.5 * (2*total_edges - E) = total_edges - 0.5 * E
    # So if E is negative, Cut > total_edges. Correct.
    
    # If best_energy is positive, it means the random start was bad (aligned spins).
    # We want to minimize E (make it as negative as possible).
    
    swap_interval = 100 # Swap replicas every 100 steps per replica
    
    start_time = time.time()
    steps_per_replica = 0
    
    while time.time() < deadline:
        # 1. Perform one Metropolis step for all replicas
        # We vectorize the selection of vertices to flip.
        
        # For each replica r, pick a random vertex v_r
        # We can do this in a loop over replicas if vectorization of the flip logic is hard,
        # or we can try to vectorize the "try" phase.
        # Given n=100, num_replicas=16, a Python loop over 16 replicas is fine.
        
        for r in range(num_replicas):
            # Pick random vertex
            v = np.random.randint(n)
            
            # Calculate energy change delta E for flipping v in replica r
            # E = s^T A s
            # Flip s_v -> -s_v
            # Change in E:
            # The terms involving s_v are:
            # s_v^2 A_vv (0 if no self loops, usually 0)
            # 2 * s_v * sum_{u != v} A_vu s_u
            # Let K_v = sum_{u} A_vu s_u (interaction field)
            # Old contribution: 2 * s_v * (K_v - A_vv*s_v) + A_vv*s_v^2 ... 
            # Simpler: 
            # E_new - E_old = -2 * s_v * K_v
            # Where K_v = (A @ s)[v]
            
            # We need K for all vertices for this replica to calculate delta for v.
            # Calculating A @ s for each replica in the inner loop is O(N^2).
            # 16 replicas * 100^2 * steps. 
            # Steps ~ 100,000? 16 * 10000 * 100000 = 1.6e10 ops. Too slow in Python/NumPy loop overhead?
            # NumPy is fast. 16 * (100x100 mat-vec) is cheap.
            
            # Let's optimize: Compute interactions K = A @ spins for all replicas at once?
            # spins shape (R, N), A shape (N, N).
            # K = spins @ A.T ? No, (spins @ A) gives (R, N).
            # K_r[v] = sum_u spins[r, u] * A[u, v].
            # This is exactly spins @ A.
            
            # We will compute K for all replicas once per "batch" of steps?
            # Or just compute it inside the loop for the specific replica?
            # Since we are doing 1 step per replica per iteration, we can precompute K for all replicas 
            # at the start of the outer while loop? 
            # No, spins change.
            
            # Let's restructure:
            # Outer loop: Time steps.
            # In each time step, we update ALL replicas.
            
            # Actually, the current loop structure `for r in range(num_replicas)` inside `while time < deadline`
            # means we do 1 step for replica 0, then 1 for 1, etc.
            # This is inefficient for vectorization.
            
            # Better structure:
            # While time < deadline:
            #    For all replicas, pick random vertices.
            #    Compute K = spins @ A (shape R, N)
            #    Compute delta_E for each replica's chosen vertex.
            #    Accept/Reject.
            #    Update spins.
            #    Occasionally swap.
            
            # Let's rewrite the loop structure for vectorization.
            pass # Break out of this flawed inner loop structure

        # The code above inside the while loop was a draft. 
        # I will implement the vectorized version below.
        break # Exit this dummy loop to implement the correct one

    # REIMPLEMENTATION OF THE MAIN LOOP WITH VECTORIZATION
    
    # Reset state if the above loop ran (it didn't, it broke immediately)
    # Let's just write the correct loop from scratch inside the function.
    
    # Re-initialize if needed (it wasn't modified)
    spins = initial_spins
    energies = calc_energy(spins)
    best_energy = np.min(energies)
    best_spins = spins[np.argmin(energies)].copy()
    best_cut = total_edges - 0.5 * best_energy
    
    swap_counter = 0
    swap_interval = 50 # Tune: how often to attempt swaps
    
    while time.time() < deadline:
        # 1. Compute interaction fields K = spins @ A
        # Shape: (num_replicas, n)
        K = spins @ A
        
        # 2. Select random vertices for each replica
        # v_indices shape: (num_replicas,)
        v_indices = np.random.randint(n, size=num_replicas)
        
        # 3. Compute Delta E for each replica
        # s_v is spins[r, v_indices[r]]
        # K_v is K[r, v_indices[r]]
        # delta_E = -2 * s_v * K_v
        s_v = spins[np.arange(num_replicas), v_indices]
        K_v = K[np.arange(num_replicas), v_indices]
        delta_E = -2 * s_v * K_v
        
        # 4. Metropolis Acceptance
        # Accept if delta_E < 0 or random < exp(-delta_E / T)
        # T = temps[r]
        
        # Calculate acceptance probabilities
        # Note: delta_E can be positive or negative.
        # If delta_E < 0, always accept.
        # If delta_E > 0, accept with prob exp(-delta_E / T).
        
        rand_vals = np.random.rand(num_replicas)
        
        # exp_arg = -delta_E / temps
        # To avoid overflow/underflow issues with large negative/positive, 
        # we can just compute. delta_E is bounded by ~2*max_degree. T is > 0.1.
        # exp(-delta_E/T) is safe.
        
        accept_prob = np.exp(-delta_E / temps)
        # Clip to [0, 1] just in case of floating point weirdness (though exp is always > 0)
        # If delta_E is very negative, exp is huge, prob > 1, accept.
        # If delta_E is very positive, exp is tiny, prob ~ 0, reject.
        
        # Vectorized comparison:
        # accept if rand < prob OR delta_E < 0
        # Actually, if delta_E < 0, prob > 1, so rand < prob is True.
        # So just: accept = rand_vals < accept_prob
        # Wait, if delta_E is negative, exp(-neg/pos) = exp(pos) > 1. rand < 1 < exp(...). True.
        # So the condition `rand_vals < accept_prob` handles both cases correctly.
        
        accept_mask = rand_vals < accept_prob
        
        # 5. Apply Updates
        # Only flip spins where accept_mask is True
        if np.any(accept_mask):
            # Flip: spins[r, v] *= -1
            # We need to update spins in place.
            # Indices: rows = np.where(accept_mask)[0], cols = v_indices[accept_mask]
            rows = np.where(accept_mask)[0]
            cols = v_indices[accept_mask]
            spins[rows, cols] *= -1
            
            # Update energies incrementally?
            # E_new = E_old + delta_E
            energies[rows] += delta_E[rows]
            
            # Update best
            # Check if any updated energy is better than best_energy
            # Find min energy in updated set
            # It's possible the global best is in an unupdated replica, but we track global best.
            # Just check the updated ones against current best.
            min_updated_idx = np.argmin(energies[rows])
            if len(rows) > 0:
                candidate_energy = energies[rows[min_updated_idx]]
                if candidate_energy < best_energy:
                    best_energy = candidate_energy
                    global_idx = rows[min_updated_idx]
                    best_spins = spins[global_idx].copy()
                    best_cut = total_edges - 0.5 * best_energy
        
        # 6. Replica Exchange (Swap)
        # Attempt swap every swap_interval steps
        # Simple strategy: Try to swap adjacent replicas (r, r+1)
        # Accept swap if (E_low - E_high) * (1/T_low - 1/T_high) < 0 ... 
        # Standard criterion: Accept swap between i and j (T_i < T_j) with prob:
        # exp( (E_i - E_j) * (1/T_j - 1/T_i) )
        # Note: 1/T_j - 1/T_i is negative.
        # Let's just do a simple random pair swap attempt every interval.
        
        swap_counter += 1
        if swap_counter % swap_interval == 0:
            # Pick two random replicas to swap
            r1, r2 = np.random.choice(num_replicas, size=2, replace=False)
            
            # Ensure r1 is the lower temperature index (just for consistency, formula works either way)
            # T[r1], T[r2]
            T1 = temps[r1]
            T2 = temps[r2]
            E1 = energies[r1]
            E2 = energies[r2]
            
            # Delta for swap acceptance:
            # We want to swap states.
            # Probability ratio: exp( (E1/T2 - E2/T1) - (E1/T1 - E2/T2) )
            # = exp( E1(1/T2 - 1/T1) + E2(1/T1 - 1/T2) )
            # = exp( (E1 - E2) * (1/T2 - 1/T1) )
            
            inv_T_diff = (1.0 / T2) - (1.0 / T1)
            log_accept_prob = (E1 - E2) * inv_T_diff
            
            # Avoid overflow
            if log_accept_prob > 0:
                # Always accept
                do_swap = True
            else:
                # Accept with prob exp(log_accept_prob)
                do_swap = np.random.rand() < np.exp(log_accept_prob)
            
            if do_swap:
                # Swap spins and energies
                spins[r1], spins[r2] = spins[r2].copy(), spins[r1].copy()
                energies[r1], energies[r2] = energies[r2], energies[r1]
                
                # Note: best_spins is a copy, so it's unaffected by the swap in `spins`.
                # However, if the best replica was swapped, its energy value in `energies` moved,
                # but `best_energy` and `best_spins` are stored separately, so they remain valid.
                # We don't need to update best_spins here unless we re-evaluate, which we don't.
                
    # Convert best_spins (-1, 1) to labels (0, 1)
    # s=1 -> label 0
    # s=-1 -> label 1
    # label = 0 if s==1 else 1
    # label = (1 - s) // 2 ?
    # s=1 -> (1-1)/2 = 0.
    # s=-1 -> (1-(-1))/2 = 1.
    labels = ((1 - best_spins) // 2).astype(int)
    
    return labels

# EVOLVE-BLOCK-END
