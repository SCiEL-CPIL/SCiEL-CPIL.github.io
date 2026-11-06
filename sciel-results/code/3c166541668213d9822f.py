"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    """
    Partition graph into two sets to maximize cut size.
    Uses spectral initialization followed by local search.
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to float for eigenvector computation
    A = np.asarray(adjacency, dtype=np.float32)
    
    # Precompute degree vector (sum of each row)
    # For a vertex v, gain of flipping v from label l to 1-l is:
    # 2 * (edges to opposite side) - (edges to same side)
    # Let S be sum of weights to same side, T be sum to opposite side.
    # deg[v] = S + T.
    # gain = 2T - S = 2T - (deg[v] - T) = 3T - deg[v]? No.
    # Let's derive:
    # Current cut contribution involving v: T (edges to other side)
    # New cut contribution involving v: S (edges to same side)
    # Change = S - T
    # S = deg[v] - T
    # Change = deg[v] - 2T
    # So we want to flip if deg[v] - 2T > 0 => 2T < deg[v] => T < deg[v]/2
    # Alternatively, gain = 2 * (edges to opposite) - deg[v] is not quite right for "change in cut size".
    # Let's stick to: new_cut = old_cut - T + S.
    # Improvement = S - T.
    # We can compute T (sum of edges to opposite side) efficiently.
    
    # Vectorized local search function
    def local_search(labels, A, deg, time_limit):
        labels = labels.copy()
        # Convert labels to 0/1 ints for easy arithmetic
        # 0 and 1 are fine.
        # T_v = sum of A[v, u] where labels[u] != labels[v]
        # Let's use a trick:
        # Let x be vector of labels, mapped to +/-1?
        # If we map 0 -> -1, 1 -> 1.
        # Then edge (u,v) is in cut if x[u] != x[v].
        # Sum of weights to opposite side for v:
        # T_v = 0.5 * (sum_u A[v,u] - sum_u A[v,u] x[v] x[u])
        # Because x[v]x[u] is 1 if same, -1 if opposite.
        # sum A[v,u] = deg[v].
        # sum A[v,u] x[v] x[u] = x[v] * (A x)[v].
        # So T_v = 0.5 * (deg[v] - x[v] (Ax)_v).
        # Improvement = S - T = (deg - T) - T = deg - 2T.
        # Improvement = deg - (deg - x[v](Ax)_v) = x[v](Ax)_v.
        # So if x[v](Ax)_v > 0, flipping improves the cut.
        # This is elegant.
        
        x = np.where(labels == 0, -1.0, 1.0).astype(np.float32)
        
        start_time = time.time()
        # Initial Ax
        Ax = A @ x
        
        # We iterate until no improvement or time limit
        # To be fast, we can check for global improvement or individual.
        # Standard local search: flip best vertex? Or first improving?
        # First improving is O(N) per pass.
        # With N=100, N^2 = 10,000 ops per pass. Very fast.
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > time_limit:
                break
            
            # Compute gain for flipping each vertex
            # gain_v = x[v] * Ax[v]
            # If gain_v > 0, flipping v increases the cut.
            gains = x * Ax
            
            # Find vertices with positive gain
            # We want to flip the one with maximum gain to converge faster?
            # Or just flip all positive? (Simultaneous update is not strictly local search, can oscillate)
            # Let's do sequential or batch.
            # Batch: flip all where gain > 0.
            # But flipping multiple at once changes Ax for others.
            # For N=100, let's do a single best flip per iteration to be safe, 
            # or a pass where we flip one by one.
            # Given N is small, a pass of "flip if beneficial" (sequential) is safe and fast.
            
            # However, vectorized "flip all positive" is a common heuristic (like in some spectral methods).
            # Let's try sequential pass for stability.
            
            improved = False
            # We need to update Ax as we flip.
            # Flipping v: x[v] -> -x[v].
            # New Ax = Ax - 2 * x[v] * A[:, v]  (since A is symmetric, A[v,:] = A[:,v])
            # Wait, (A x)_u = sum A[u,w] x[w].
            # If x[v] changes by delta (which is -2x[v]), then Ax changes by A[:, v] * delta.
            
            # To keep it simple and robust within time limit:
            # Just recompute Ax from scratch if N is small?
            # N=100, A@x is 10k mults. 100 passes = 1M ops. Trivial.
            # So we can just recompute Ax each time we flip something, or even each pass.
            
            # Let's do a pass:
            # 1. Compute gains = x * (A @ x)
            # 2. If any gain > 0, flip the one with max gain?
            #    Or flip all?
            #    Let's flip the one with max gain. It's a greedy step.
            
            max_gain = np.max(gains)
            if max_gain > 1e-6: # Tolerance
                v = np.argmax(gains)
                # Flip v
                x[v] = -x[v]
                improved = True
                # Update Ax efficiently?
                # Ax = A @ x is fast enough for N=100.
                # But we can update: Ax -= 2 * x[v] * A[:, v] ... wait x[v] is already flipped in the array?
                # Let's just recompute to avoid bugs, it's cheap.
                Ax = A @ x
            else:
                break
            
            if not improved:
                break

        # Convert back to 0/1 labels
        # x == 1 -> label 1, x == -1 -> label 0
        return (x > 0).astype(int)

    # Time budget management
    # Total limit 120s.
    # Leave 5s for overhead and final checks.
    # We will perform multiple restarts with different spectral initializations.
    
    # Get eigenvectors
    # np.linalg.eigh is for symmetric matrices. A is symmetric (unweighted graph).
    # We need the top few eigenvectors to create diverse starts.
    try:
        vals, vecs = np.linalg.eigh(A)
        # Sort by eigenvalue descending
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]
    except np.linalg.LinAlgError:
        # Fallback to random if eigh fails
        vecs = np.random.randn(n, min(10, n)).astype(np.float32)
        vals = np.zeros(min(10, n))

    # Select a few top eigenvectors for initialization
    # Usually the top 1 is best (Fiedler vector is 2nd smallest for Laplacian, but for Max Cut, 
    # the principal eigenvector of A is often used, or related to Ncut).
    # Actually, for Max Cut, the problem is NP-hard. Spectral methods often use the leading eigenvector.
    
    # Let's use the top 5 eigenvectors as seeds.
    num_seeds = min(5, n)
    seeds = vecs[:, :num_seeds]
    
    best_labels = None
    best_score = -1
    
    # Time allocation
    total_deadline = time.time() + 115.0 # 115s limit to be safe
    
    # We can afford many local search runs.
    # Each local search run is very fast (ms).
    # Let's loop through seeds, and maybe add random perturbations.
    
    # 1. Try each spectral seed
    for i in range(num_seeds):
        if time.time() > total_deadline:
            break
            
        seed_vec = seeds[:, i]
        # Threshold at 0
        init_labels = (seed_vec > 0).astype(int)
        
        # Run local search
        # Allocate time for this run: 10s? No, it's fast. 
        # The local search itself will stop when converged.
        # We just need to ensure we don't run *too many* if they are slow (they aren't).
        # But to be safe, pass a tight time limit to local_search, e.g., 10s.
        # Since N=100, it will converge in <1s usually.
        
        labels = local_search(init_labels, A, None, 10.0)
        
        # Score
        # Cut size = 0.5 * sum_{u,v} A[u,v] (1 - x[u]x[v])
        # Or just count edges where labels differ.
        # Vectorized score:
        x = np.where(labels == 0, -1.0, 1.0).astype(np.float32)
        # Ax = A @ x
        # Score = 0.5 * (sum(deg) - sum(x * Ax))
        # sum(deg) = 2 * E (total edges weight)
        # Let's compute directly for clarity and speed.
        # diff = labels[:, None] != labels[None, :]
        # score = (A * diff).sum() / 2 ? No, A is 0/1.
        # score = np.sum(A * (labels[:, None] != labels[None, :])) / 2
        # This creates a 100x100 bool array. Fine.
        
        diff = labels[:, None] != labels[None, :]
        score = 0.5 * np.sum(A * diff)
        
        if score > best_score:
            best_score = score
            best_labels = labels.copy()

    # 2. If time remains, do random restarts
    # How much time passed?
    elapsed = time.time()
    time_remaining = total_deadline - elapsed
    
    if time_remaining > 2.0: # If we have more than 2 seconds
        # Estimate how many random restarts we can do.
        # Each restart takes ~0.1s? (guess). Let's say 0.05s.
        # We can do many.
        rng = np.random.default_rng(42)
        
        # Loop until time runs out
        while time.time() < total_deadline - 1.0:
            # Random init
            init_labels = rng.integers(0, 2, size=n).astype(int)
            
            # Local search
            labels = local_search(init_labels, A, None, 5.0)
            
            # Score
            diff = labels[:, None] != labels[None, :]
            score = 0.5 * np.sum(A * diff)
            
            if score > best_score:
                best_score = score
                best_labels = labels.copy()
                
    if best_labels is None:
        # Fallback
        best_labels = np.zeros(n, dtype=int)
        
    return best_labels
# EVOLVE-BLOCK-END
