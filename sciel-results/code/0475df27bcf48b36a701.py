"""
Strategy: Multi-Start Vectorized Simulated Annealing.
The current spectral approach finds a good starting point but often gets stuck in 
local optima during the greedy polish phase. Simulated Annealing (SA) is a 
probabilistic technique that allows the search to escape local optima by accepting 
worse solutions with a temperature-dependent probability. 
By running multiple independent SA instances in parallel (vectorized), we explore 
different regions of the solution space simultaneously, significantly increasing 
the chance of finding a near-global maximum cut.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Uses Multi-Start Vectorized Simulated Annealing.
    """
    adj = np.asarray(adjacency, dtype=np.float64)
    n = len(adj)
    
    if n <= 1:
        return np.zeros(n, dtype=np.int8)
    
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We stop at 140s.
    deadline = start_time + 140.0
    
    # Precompute degrees and initial cut size helper
    degrees = np.sum(adj, axis=1)
    
    # If graph is empty or trivial
    if not np.any(adj):
        return np.zeros(n, dtype=np.int8)
    
    # Number of parallel SA instances
    # n=100 is small. We can afford many instances.
    # 50-100 instances is a good balance.
    n_instances = 100
    
    # Initialize states: (n_instances, n) matrix of 0/1 labels
    # Random initialization
    states = np.random.randint(0, 2, size=(n_instances, n), dtype=np.int8)
    
    # Compute initial cut sizes for all instances
    # Cut size = sum over edges (i,j) where state[i] != state[j]
    # Vectorized cut calculation:
    # For each instance k, cut_k = 0.5 * sum_{i,j} adj[i,j] * (1 - (2*state[k,i]-1)*(2*state[k,j]-1))
    # Or simpler: cut_k = sum_{i} (degrees[i] - sum_{j in neighbors} (1 if state[i]==state[j] else 0))
    # Even simpler vectorized:
    # Let S be states (M, N). Let A be adjacency (N, N).
    # same_side = S @ A @ S.T ? No.
    # Cut = (1/2) * trace(S (I - A) S^T) ? No.
    # Cut = 0.5 * sum_{i,j} A_{ij} (1 - s_i s_j) where s in {-1, 1}
    # Let s_signed = 2*states - 1 (M, N)
    # cut = 0.5 * (sum(signed) @ A @ sum(signed).T ... no, per instance)
    
    def calc_cuts(states, adj, degrees):
        """
        states: (M, N) int8 0/1
        adj: (N, N) float
        degrees: (N,) float
        returns: (M,) float cut sizes
        """
        # s_signed: (M, N) in {-1, 1}
        s_signed = 2 * states.astype(np.float64) - 1
        # For each instance m, cut_m = 0.5 * sum_{i,j} A_{ij} (1 - s_{mi} s_{mj})
        # = 0.5 * (sum_{i,j} A_{ij} - sum_{i,j} A_{ij} s_{mi} s_{mj})
        # sum_{i,j} A_{ij} = 2 * sum(degrees) = constant for all m
        # term_m = sum_{i,j} A_{ij} s_{mi} s_{mj} = s_{m}^T A s_{m}
        
        # Compute s^T A s for all m
        # s_signed @ A -> (M, N)
        # (s_signed @ A) * s_signed -> (M, N)
        # sum over axis 1 -> (M,)
        
        # Optimization: A is symmetric.
        # s^T A s = sum_i s_i (A s)_i
        # Let v = s_signed @ A.T (which is s_signed @ A since symmetric)
        # result = np.sum(s_signed * v, axis=1)
        
        v = states.astype(np.float64) @ adj.T  # Wait, states are 0/1. Need signed.
        s_signed = 2 * states.astype(np.float64) - 1
        v = s_signed @ adj.T
        s_as = np.sum(s_signed * v, axis=1)
        
        total_edges_weight = np.sum(degrees) # This is 2 * E
        # Cut = 0.5 * (2E - s^T A s)
        cuts = 0.5 * (total_edges_weight - s_as)
        return cuts
    
    cuts = calc_cuts(states, adj, degrees)
    best_cuts = cuts.copy()
    best_states = states.copy()
    
    # SA Parameters
    # Temperature schedule: T = T0 * alpha^iter
    # T0 should be high enough to allow exploration.
    # Max possible gain is degree. Avg degree ~ 10-20 for n=100 sparse?
    # Let's assume max degree < 50.
    # T0 = 10.0, alpha = 0.995? 
    # We need to run for a long time. 140 seconds.
    # Each iteration involves:
    # 1. Random flip selection
    # 2. Delta calculation
    # 3. Acceptance check
    # 4. Update state and cut
    
    # Vectorized Delta Calculation:
    # If we flip vertex v in instance m:
    # delta_cut = 2 * (num_same_side_neighbors) - degree[v]
    # num_same_side_neighbors = sum_{u} A[v,u] * (state[m,v] == state[m,u])
    
    # We will perform batches of flips to utilize vectorization.
    # Batch size: 1 (one flip per instance per step) is standard SA.
    # But we have 100 instances. We can do 1 flip per instance per step.
    # Total flips per second?
    # Matrix mult (100, 100) @ (100, 100) is fast.
    # Let's just loop steps.
    
    T0 = 50.0
    alpha = 0.999 # Very slow cooling to allow long exploration
    # Or use time-based cooling?
    # Let's use step-based but estimate steps.
    # 140s / (time per step).
    # Let's run until time limit.
    
    # Precompute neighbor lists for faster delta?
    # For n=100, matrix ops are fine.
    
    # To speed up delta calculation:
    # We need same_side_counts for the vertex being flipped.
    # Let flip_idx be (M,) array of vertex indices to flip.
    # Let s be current states (M, N).
    # For instance m, we flip v = flip_idx[m].
    # old_state = s[m, v]
    # new_state = 1 - old_state
    # 
    # delta = 2 * (sum_{u} A[v,u] * (s[m,u] == s[m,v])) - degree[v]
    # 
    # We can compute this vectorized:
    # 1. Extract rows s[:, v] -> (M,)
    # 2. We need sum of A[v, :] * (s[m, :] == s[m, v])
    # This is tricky to vectorize across different v for different m without loops or advanced indexing.
    # 
    # Alternative:
    # Compute full same_side matrix for all instances?
    # same_mask[m, i, j] = (s[m,i] == s[m,j])
    # This is (M, N, N). M=100, N=100 -> 1,000,000 bools. 1MB. Acceptable.
    # same_counts[m, i] = sum_j A[i,j] * same_mask[m, i, j]
    # 
    # If we flip v, the change in cut is:
    # The edges incident to v change status.
    # If s[m,v] was 0, it becomes 1.
    # Edges (v,u) where s[m,u]==0 become cut (gain +1).
    # Edges (v,u) where s[m,u]==1 become uncut (loss -1).
    # Net change = (# neighbors with 0) - (# neighbors with 1)
    # Note: degree[v] = #neighbors0 + #neighbors1
    # So #neighbors0 - #neighbors1 = 2*#neighbors0 - degree[v]
    # 
    # So if we precompute `same_counts` (M, N) where same_counts[m, i] is # neighbors of i with same label as i in instance m:
    # If we flip i:
    # new_same_counts[i] = degree[i] - same_counts[i] (because neighbors that were same are now different, and vice versa)
    # Wait.
    # Let S_i be set of neighbors with same label.
    # Let D_i be set of neighbors with different label.
    # |S_i| = same_counts[m, i]
    # |D_i| = degree[i] - same_counts[m, i]
    # If we flip i, the new S'_i will be the old D_i.
    # So new_same_counts[m, i] = degree[i] - same_counts[m, i]
    # 
    # The change in cut size:
    # Old cut contribution from edges incident to i: |D_i|
    # New cut contribution: |S_i| (which are now different)
    # Delta = |S_i| - |D_i| = same_counts[m, i] - (degree[i] - same_counts[m, i])
    # Delta = 2 * same_counts[m, i] - degree[i]
    
    # So we need to maintain `same_counts` (M, N).
    # Updating `same_counts` after a flip is O(N) per instance if done naively, or O(N) total if vectorized?
    # When we flip vertex v in instance m:
    # 1. same_counts[m, v] changes to degree[v] - same_counts[m, v]
    # 2. For all neighbors u of v, same_counts[m, u] changes because the label of v changed.
    #    If s[m, u] == s[m, v] (old), then u loses one same-neighbor. same_counts[m, u] -= 1
    #    If s[m, u] != s[m, v] (old), then u gains one same-neighbor. same_counts[m, u] += 1
    
    # This update is sparse. We can use the adjacency matrix.
    # Let flip_mask be (M, N) where flip_mask[m, v] = 1 if v is flipped in m.
    # This is hard to vectorize the neighbor updates efficiently without loops or complex indexing.
    
    # Given n=100, a simple loop over instances for the update might be fast enough?
    # Or we can just recalculate cuts from scratch every K steps?
    # Recalculating cuts from scratch is O(M * N^2) = 100 * 10000 = 1,000,000 ops. Very fast.
    # Let's just recalculate cuts from scratch every step. It's simpler and less error-prone.
    # 100 instances * 100 steps/s = 10,000 recalc/s.
    # 140s * 10,000 = 1.4 million recalcs.
    # Each recalc is ~1M flops. 1.4 T flops? No, 1M flops * 1.4M steps = 1.4 * 10^12 flops.
    # Modern CPU ~ 10-50 GFLOPS. 1.4 TFLOPS / 20 GFLOPS = 70s.
    # This might be too slow if we do it every step.
    
    # Let's optimize:
    # We only need to update the cut for the instance that changed.
    # But we have M instances.
    # If we do 1 flip per instance per step, we update M cuts.
    # Using the Delta formula:
    # We need same_counts for the flipped vertex.
    # We can maintain same_counts incrementally.
    
    # Let's try the incremental update approach.
    # State: states (M, N) int8
    # Aux: same_counts (M, N) float
    
    # Initialize same_counts
    # same_counts[m, i] = sum_j A[i,j] * (states[m,i] == states[m,j])
    
    def init_same_counts(states, adj):
        # states: (M, N)
        # adj: (N, N)
        # same_mask: (M, N, N)
        # same_counts: (M, N)
        
        # To avoid (M, N, N) memory, we can loop over M?
        # M=100. Loop 100 times.
        # Inside: same_mask[i, :] = (states[m, i] == states[m, :])
        # same_counts[m, i] = sum(adj[i, :] * same_mask[i, :])
        
        M, N = states.shape
        sc = np.zeros((M, N), dtype=np.float64)
        
        # Vectorized over M and N?
        # states[:, :, None] == states[:, None, :] -> (M, N, N)
        # This is 100 * 100 * 100 = 1,000,000 booleans. 1 MB.
        # Then multiply by adj[None, :, :] (broadcast)
        # Sum over axis 2.
        
        # Let's try this.
        s_expanded = states.astype(np.int8)[:, :, None]
        s_expanded_T = states.astype(np.int8)[:, None, :]
        
        same_mask = (s_expanded == s_expanded_T)
        
        # adj is (N, N). Broadcast to (1, N, N)
        # We need to sum over neighbors j for each i.
        # same_counts[m, i] = sum_j adj[i, j] * same_mask[m, i, j]
        
        # Use einsum or tensordot?
        # np.einsum('ij,mij->mi', adj, same_mask)
        
        sc = np.einsum('ij,mij->mi', adj, same_mask)
        return sc
    
    same_counts = init_same_counts(states, adj)
    
    # Verify cuts
    # cut = 0.5 * (2E - sum_i (degree[i] - same_counts[i])) ?
    # No.
    # Cut = sum_i (degree[i] - same_counts[i]) / 2
    # Because each cut edge is counted twice (once for i, once for j).
    # And for a cut edge (i,j), it contributes 1 to degree[i]-same_counts[i] and 1 to degree[j]-same_counts[j].
    # Sum = 2 * Cut.
    
    cuts = 0.5 * np.sum(degrees[None, :] - same_counts, axis=1)
    
    # Main SA Loop
    T = T0
    
    # Precompute random numbers? No, generate on fly.
    
    while time.time() < deadline:
        # Select vertices to flip
        # Random integer 0 to n-1 for each instance
        flip_idx = np.random.randint(0, n, size=n_instances)
        
        # Get old states for flipped vertices
        # states[m, flip_idx[m]]
        # Advanced indexing:
        rows = np.arange(n_instances)
        old_states = states[rows, flip_idx]
        
        # Calculate Delta for each instance
        # delta[m] = 2 * same_counts[m, flip_idx[m]] - degrees[flip_idx[m]]
        sc_flip = same_counts[rows, flip_idx]
        deg_flip = degrees[flip_idx]
        deltas = 2 * sc_flip - deg_flip
        
        # Acceptance Probability
        # If delta > 0, always accept.
        # If delta < 0, accept with prob exp(delta / T)
        
        # Vectorized acceptance
        # rand = np.random.rand(n_instances)
        # accept = (deltas > 0) | (np.random.rand(n_instances) < np.exp(deltas / T))
        # Note: exp(negative) is in (0, 1).
        
        # To avoid overflow/underflow issues, we can just compute.
        # T starts at 50, deltas are small (~ -10 to 10). exp(-10/50) is fine.
        
        rand = np.random.rand(n_instances)
        # For delta > 0, exp(delta/T) > 1, so prob is 1.
        # We can just use np.where(deltas > 0, 1.0, np.exp(deltas / T))
        # But np.exp of large positive numbers might overflow if T is small.
        # T is decreasing. Min T?
        # If T becomes very small, exp(delta/T) for delta>0 is huge.
        # We just need to compare with rand < prob.
        # If prob > 1, it's always true.
        
        probs = np.exp(deltas / T)
        # Clip probs to [0, 1] for safety, though mathematically delta>0 -> prob>1
        # Actually, if delta > 0, we accept.
        # So:
        accept_mask = (deltas > 0) | (rand < probs)
        
        # Apply flips where accepted
        # If accepted, flip the state.
        # states[rows[accept_mask], flip_idx[accept_mask]] = 1 - states[...]
        
        accepted_rows = rows[accept_mask]
        accepted_flip_idx = flip_idx[accept_mask]
        
        if len(accepted_rows) > 0:
            # Update states
            states[accepted_rows, accepted_flip_idx] = 1 - states[accepted_rows, accepted_flip_idx]
            
            # Update same_counts
            # For each accepted instance m and flipped vertex v:
            # 1. same_counts[m, v] = degrees[v] - same_counts[m, v]
            # 2. For neighbors u of v:
            #    If old_state == states[m, u] (before flip, which is now flipped, so old_state = 1 - new_state)
            #    Actually, let's use the delta logic.
            #    The change in same_counts for neighbor u depends on whether u had the same label as v BEFORE the flip.
            #    Let old_v = 1 - states[m, v] (current state is new).
            #    If states[m, u] == old_v, then u and v were same. Now they are different. same_counts[m, u] -= 1
            #    If states[m, u] != old_v, then u and v were different. Now they are same. same_counts[m, u] += 1
            
            # We can vectorize this update using the adjacency matrix.
            # Let's gather the necessary info.
            
            # This part is tricky to fully vectorize without loops because the neighbor set is different for each v.
            # However, we can use sparse-like updates or just loop over the accepted instances if the number is small?
            # n_instances=100. Acceptance rate might be 50-100%. So ~50-100 updates per step.
            # Looping 100 times per step is fine.
            
            for m in accepted_rows:
                v = accepted_flip_idx[m]
                old_v = 1 - states[m, v] # State is already flipped. old_v is the previous value.
                
                # Update v's own count
                same_counts[m, v] = degrees[v] - same_counts[m, v]
                
                # Update neighbors
                # neighbors = np.nonzero(adj[v])[0]
                # This is O(N) per update. 100 updates * 100 neighbors = 10,000 ops.
                # 10,000 ops per step.
                # 100 steps/s -> 1,000,000 ops/s. Very fast.
                
                neighbors = np.nonzero(adj[v])[0]
                if len(neighbors) > 0:
                    # Check which neighbors had the same label as old_v
                    # states[m, neighbors] == old_v
                    # If yes, they lost a same-neighbor.
                    # If no, they gained a same-neighbor.
                    
                    # diff = (states[m, neighbors] == old_v).astype(np.int8)
                    # If same: -1. If different: +1.
                    # change = 1 - 2 * (states[m, neighbors] == old_v)
                    # Wait.
                    # If states[m, u] == old_v (Same): change is -1.
                    # If states[m, u] != old_v (Diff): change is +1.
                    # So change = +1 if diff, -1 if same.
                    # change = 1 - 2 * (same_bool)
                    
                    same_bool = (states[m, neighbors] == old_v)
                    changes = 1 - 2 * same_bool.astype(np.float64)
                    
                    same_counts[m, neighbors] += changes
            
            # Update cuts for accepted instances
            # cuts[m] = 0.5 * sum(degrees - same_counts[m, :])
            # We can update cuts incrementally too.
            # delta_cut = 0.5 * sum(changes in (degree - same_count))
            # Change in cut = 0.5 * ( - delta_same_count_sum )
            # But we already calculated `deltas` for the flip.
            # The `deltas` calculated BEFORE the flip is the change in cut.
            # So cuts[accepted_rows] += deltas[accept_mask]
            
            cuts[accepted_rows] += deltas[accept_mask]
            
            # Update bests
            improved_mask = cuts[accepted_rows] > best_cuts[accepted_rows]
            if np.any(improved_mask):
                improved_idx = accepted_rows[improved_mask]
                best_cuts[improved_idx] = cuts[improved_idx]
                best_states[improved_idx] = states[improved_idx]
        
        # Cooling
        T *= alpha
        
        # Optional: Reheat? No, standard SA.
        
        # Check time
        if time.time() > deadline - 5.0:
            break
            
    # Return the best state found
    # best_states is (M, N). We need to pick the single best one.
    best_idx = np.argmax(best_cuts)
    return best_states[best_idx]

# EVOLVE-BLOCK-END
