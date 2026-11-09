"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    # labels are 0/1. Edge is cut if labels differ.
    # diff[i,j] is 1 if labels[i] != labels[j]
    diff = labels[None, :] != labels[:, None]
    # adjacency is bool matrix. We count edges where diff is True.
    # Since graph is undirected, each edge is counted twice in the matrix sum.
    return int((adjacency & diff).sum() // 2)

def _spectral_init(adjacency):
    """Initialize using the sign of the Fiedler vector."""
    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1)
    L = np.diag(degrees) - adjacency.astype(float)
    
    try:
        vals, vecs = np.linalg.eigh(L)
        fiedler = vecs[:, 1]
        labels = (fiedler >= 0).astype(np.int8)
        return labels
    except Exception:
        return np.random.randint(0, 2, size=n, dtype=np.int8)

def _walksat(adjacency, labels, deadline, p=0.5):
    """
    WalkSAT-style stochastic local search for Max Cut.
    
    Formulation:
    Max Cut is equivalent to Max-2-SAT where each edge (u, v) is a clause (x_u XOR x_v).
    We want to satisfy as many clauses as possible.
    
    Algorithm:
    1. Compute the number of unsatisfied clauses (uncut edges) for each variable.
       Let U_i be the number of neighbors j such that label[i] == label[j] (unsatisfied).
    2. If all satisfied, we are done.
    3. Select a variable to flip:
       - With probability (1-p), pick a variable randomly from those with U_i > 0 (or all if none).
       - With probability p, pick a variable randomly from those with the maximum U_i.
         (This biases towards fixing the most violated constraints, but allows randomness to escape.)
       Actually, standard WalkSAT picks a random unsatisfied clause, then a random variable in it.
       For Max Cut, an "unsatisfied clause" is an uncut edge (u, v) where label[u] == label[v].
       Picking a random uncut edge and then a random endpoint is equivalent to:
       Probability of picking node i is proportional to U_i (its number of uncut edges).
       
       Let's implement the "biased random walk":
       - Calculate U (vector of uncut edge counts per node).
       - If sum(U) == 0, break.
       - Generate random weights W_i = U_i * exp(noise) or just pick based on U_i.
       - To allow escaping local optima where single flips might not help immediately or lead to cycles,
         we can introduce a small random perturbation or simply use the standard WalkSAT move:
         Pick a random node i with probability proportional to U_i.
         Flip i.
         If this doesn't improve or gets stuck, the stochastic nature handles it.
         
       However, pure WalkSAT on Max-2-SAT is known to be very effective.
       To enhance exploration, we can occasionally flip a node with U_i=0 (random walk) if stuck?
       No, if U_i=0, flipping it increases unsatisfied clauses (makes cut worse).
       Standard WalkSAT only flips variables in unsatisfied clauses.
       
       Let's stick to: Pick i with prob proportional to U_i. Flip i.
       This is a stochastic local search that naturally performs random walks on the state space
       weighted by the "frustration" (uncut edges).
    """
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    best_cut = current_cut
    best_labels = labels.copy()
    
    adj_float = adjacency.astype(float)
    # Precompute degrees? Not strictly needed for the U calculation if we do it directly.
    # U_i = sum_{j} A_{ij} * (1 if label[i]==label[j] else 0)
    # Let's use a vectorized approach for speed.
    
    start_time = time.time()
    steps = 0
    max_steps = 1000000 # Safety cap, deadline is primary
    
    # To speed up, we can maintain U incrementally.
    # Initial U calculation:
    # diff_matrix = (labels[None, :] != labels[:, None])
    # U = (adjacency & ~diff_matrix).sum(axis=1)
    
    # Let's just recompute U each step for simplicity first, then optimize if too slow.
    # N=100, N^2=10000 ops. 100k steps = 1e9 ops. Might be too slow in pure Python loop if not vectorized.
    # We need to vectorize the U update or the selection.
    
    # Vectorized U calculation:
    # same = (labels[None, :] == labels[:, None])
    # U = (adjacency & same).sum(axis=1)
    
    # Incremental Update:
    # If we flip node k:
    # For all j != k:
    #   If label[k] == label[j] (before flip), edge (k,j) was uncut (unsatisfied). After flip, it becomes cut (satisfied).
    #     So U_j decreases by 1. U_k changes...
    #   If label[k] != label[j] (before flip), edge (k,j) was cut (satisfied). After flip, it becomes uncut (unsatisfied).
    #     So U_j increases by 1.
    #
    # Let's derive U update for flipping k:
    # Let x be labels. Flip k: x_k -> 1-x_k.
    # U_i = sum_j A_{ij} [x_i == x_j]
    #
    # Change in U for node j (j != k):
    # If x_j == x_k (old): Edge (j,k) was uncut. Now x_j != (1-x_k) = 1-x_j? No.
    #   Old: x_j == x_k. Term was 1.
    #   New: x_j vs 1-x_k. Since x_k=1-x_j, 1-x_k=x_j. So x_j == 1-x_k. Term is 1.
    #   Wait. If x_j == x_k, then x_k is 1-x_j. 1-x_k is x_j. So x_j == 1-x_k.
    #   So the edge (j,k) remains uncut?
    #   Let's check: x_j=0, x_k=0. Uncut. Flip k -> x_k=1. x_j=0, x_k=1. Cut.
    #   So if x_j == x_k (old), edge was uncut (contributes to U_j). After flip, it is cut (does not contribute).
    #   So U_j decreases by 1.
    #
    # If x_j != x_k (old): Edge (j,k) was cut.
    #   x_j=0, x_k=1. Cut. Flip k -> x_k=0. x_j=0, x_k=0. Uncut.
    #   So U_j increases by 1.
    #
    # Summary for j != k:
    # delta_U_j = -1 if x_j == x_k else 1
    #
    # Change in U for node k:
    # U_k = sum_j A_{kj} [x_k == x_j]
    # Old U_k = count of neighbors with same label.
    # New U_k = count of neighbors with different label (since x_k flipped, same becomes diff).
    # New U_k = degree_k - Old U_k.
    #
    # So:
    # 1. Identify neighbors of k: mask = adjacency[k]
    # 2. For neighbors j:
    #    if labels[j] == labels[k]: U[j] -= 1
    #    else: U[j] += 1
    # 3. U[k] = degrees[k] - U[k]
    
    degrees = adjacency.sum(axis=1)
    
    # Initialize U
    same_matrix = (labels[None, :] == labels[:, None])
    U = (adjacency & same_matrix).sum(axis=1)
    
    while time.time() < deadline:
        if steps > max_steps: break
        steps += 1
        
        # If no unsatisfied edges, we found a perfect cut (or disconnected components)
        if U.sum() == 0:
            break
            
        # Select node to flip
        # Probability proportional to U_i
        # We can use np.random.choice with p
        # But U_i can be 0. We only want to pick from U_i > 0.
        # If all U_i > 0, just use U.
        # If some are 0, we need to handle it.
        # Standard WalkSAT: Pick random unsatisfied clause (edge), then random endpoint.
        # This is equivalent to picking node i with prob proportional to U_i.
        # If U.sum() > 0, there is at least one node with U_i > 0.
        
        # Optimization: Instead of building p vector every time (O(N)),
        # we can just pick a random edge from the uncut edges.
        # But maintaining list of uncut edges is complex.
        # Let's just use np.random.choice with weights. N=100 is small.
        weights = U.astype(float)
        # Ensure no negative weights (shouldn't happen)
        weights[weights < 0] = 0
        total_w = weights.sum()
        if total_w == 0:
            break
        p = weights / total_w
        k = np.random.choice(n, p=p)
        
        # Flip k
        old_label = labels[k]
        labels[k] = 1 - old_label
        
        # Update U
        # Neighbors of k
        neighbors = np.where(adjacency[k])[0]
        
        # For each neighbor j:
        # if labels[j] == old_label (which was labels[k] before flip): U[j] -= 1
        # else: U[j] += 1
        
        # Vectorized update for neighbors
        # neighbor_labels = labels[neighbors]
        # same_as_old = (neighbor_labels == old_label)
        # U[neighbors] += np.where(same_as_old, -1, 1)
        
        # And update U[k]
        # U[k] = degrees[k] - U[k]
        
        # Let's do it safely:
        neighbor_labels = labels[neighbors] # Note: labels[k] is already flipped, but we compare to old_label
        # Wait, labels[neighbors] does NOT include k (unless self-loop, which we assume none).
        # So labels[neighbors] is current state.
        # We need to compare against old_label of k.
        
        same_as_old = (neighbor_labels == old_label)
        
        # Create delta vector for U
        # If same_as_old, delta is -1. Else +1.
        deltas = np.where(same_as_old, -1, 1)
        
        U[neighbors] += deltas
        
        # Update U[k]
        U[k] = degrees[k] - U[k]
        
        # Calculate current cut size to track best
        # Cut = (M - sum(U)) / 2 ?
        # Total edges M = sum(degrees)/2
        # Uncut edges = sum(U) / 2 (since each uncut edge contributes 1 to U_i and 1 to U_j)
        # Cut edges = M - Uncut_edges
        # Cut = M - sum(U)/2
        
        # Let's just compute cut size periodically or incrementally.
        # Delta in cut size when flipping k:
        # New Cut = Old Cut + (Uncut_edges_becoming_Cut) - (Cut_edges_becoming_Uncut)
        # From above:
        # Neighbors with same label (old): were uncut, now cut. Count = sum(same_as_old)
        # Neighbors with diff label (old): were cut, now uncut. Count = sum(~same_as_old)
        # Delta_Cut = sum(same_as_old) - sum(~same_as_old)
        
        num_same = np.sum(same_as_old)
        num_diff = len(neighbors) - num_same
        delta_cut = num_same - num_diff
        
        current_cut += delta_cut
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
            
    return best_labels, best_cut

def partition(adjacency):
    """
    Main partition function using WalkSAT.
    1. Spectral Init
    2. WalkSAT Local Search
    3. Random Restarts with WalkSAT
    """
    n = len(adjacency)
    start_time = time.time()
    # Evaluator kills at 150s. Use 140s to be safe.
    deadline = start_time + 140.0
    
    best_labels = None
    best_cut = -1
    
    # Phase 1: Spectral Initialization + WalkSAT
    labels = _spectral_init(adjacency)
    
    # Run WalkSAT for a chunk of time
    # Let's give it 50% of the time initially? Or run until a stagnation?
    # WalkSAT is very fast. Let's run for 10 seconds first.
    t1 = time.time() + 10.0
    labels, cut = _walksat(adjacency, labels, t1)
    if cut > best_cut:
        best_cut = cut
        best_labels = labels.copy()
        
    # Phase 2: Random Restarts
    # If we have time left, do random restarts.
    # Each restart: Random init + short WalkSAT.
    remaining = deadline - time.time() - 5.0
    if remaining > 2.0:
        # Estimate time per restart.
        # WalkSAT steps are O(N) roughly (vectorized neighbor update).
        # Should be able to do millions of steps per second?
        # Let's just allocate time.
        num_restarts = int(remaining / 2.0) # 2s per restart
        num_restarts = max(1, min(num_restarts, 20)) # Cap restarts
        
        time_per_restart = remaining / num_restarts
        
        for r in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
                
            # Random Init
            labels = np.random.randint(0, 2, size=n, dtype=np.int8)
            
            sub_deadline = time.time() + time_per_restart - 0.1
            labels, cut = _walksat(adjacency, labels, sub_deadline)
            
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
                
    if best_labels is None:
        best_labels = _spectral_init(adjacency)
        
    return best_labels
# EVOLVE-BLOCK-END
