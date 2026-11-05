"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def partition(adjacency):
    """
    Partition vertices into two groups to maximize the cut size.
    Uses Simulated Annealing with vectorized gain updates and multi-start.
    """
    start_time = time.time()
    # Evaluator kills at 150s. Leave 5s margin.
    deadline = start_time + 145.0
    
    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)
    
    # Precompute neighbors and degrees for fast updates
    neighbors = [np.where(adj[i] == 1)[0] for i in range(n)]
    
    best_labels = None
    best_cut = -1
    rng = np.random.default_rng(42)
    
    # Number of restarts depends on available time, but we can run many
    # short SA runs or fewer long ones. 
    # Let's try a strategy: run SA until time is up, keeping the best solution 
    # across restarts.
    
    # We will run multiple restarts. 
    # Each restart: 
    #   - Random init
    #   - SA for a fixed number of steps or until time slice is up
    
    # Estimate time per step? Hard to predict exactly.
    # Let's just loop restarts and check time at the start of each restart 
    # and inside the SA loop.
    
    max_restarts = 1000 # Cap just in case
    restart_count = 0
    
    while time.time() < deadline:
        restart_count += 1
        if restart_count > max_restarts:
            break
            
        # Initialize x with -1, 1
        x = rng.choice([-1, 1], size=n)
        
        # Calculate initial gains: gain[v] = x[v] * sum(x[neighbors])
        # S_v = sum(x[neighbors])
        # gain = x * S
        # Using matrix multiplication: S = adj @ x
        S = adj @ x
        gains = x * S
        
        # Current cut size
        # Cut = 0.5 * (E - sum(x_i x_j))
        # sum(x_i x_j) = 0.5 * (x^T adj x) because adj is symmetric
        # Actually, x^T adj x = sum_i sum_j x_i adj_ij x_j = sum_{(i,j) in E} 2 x_i x_j
        # So sum_{(i,j) in E} x_i x_j = 0.5 * x^T adj x
        # Cut = 0.5 * (E - 0.5 * x^T adj x)
        # Let's just calculate cut directly from gains or x for simplicity at check points
        # Or update incrementally.
        
        # Let's track cut size incrementally.
        # Initial cut:
        diff = (x[None, :] != x[:, None])
        current_cut = int(np.sum(adj * diff))
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = (x + 1) // 2 # Convert -1,1 to 0,1
            
        # Simulated Annealing Parameters
        # Initial temperature
        # Max possible gain is roughly max degree. 
        # Let's set T0 high enough to accept bad moves initially.
        T = 5.0 
        T_end = 0.1
        # Cooling rate
        # We want to cool down over the duration of this restart.
        # Let's define a time slice for this restart, e.g., 10% of remaining time or fixed steps.
        # Better: run SA until a local minimum is reached for a while, or just fixed steps.
        # Given N=100, we can do many steps.
        # Let's try 5000 steps per restart.
        steps = 5000
        
        for _ in range(steps):
            if time.time() > deadline:
                break
                
            # Pick random vertex
            v = rng.integers(0, n)
            
            # Get gain
            delta = gains[v]
            
            # Acceptance criteria
            accept = False
            if delta > 0:
                accept = True
            else:
                # Metropolis criterion
                # exp(delta / T)
                # If T is small, this is ~0
                if T > 1e-6:
                    # Use log to avoid underflow/overflow issues if delta is large negative
                    # But delta is bounded by degree * 1. Max degree < 100.
                    # delta/T is safe.
                    if rng.random() < np.exp(delta / T):
                        accept = True
            
            if accept:
                # Flip v
                x[v] = -x[v]
                
                # Update cut size
                current_cut += delta
                
                # Update gains
                # 1. Gain for v becomes -gain[v]
                old_gain_v = gains[v]
                gains[v] = -old_gain_v
                
                # 2. Gain for neighbors u:
                # gain[u] = x[u] * S[u]
                # S[u] changes by -2 * x_v (old x_v)
                # Wait, x_v changed from x_v to -x_v.
                # Change in x_v is -2 * x_v_old.
                # So S[u] -= 2 * x_v_old
                # New gain[u] = x[u] * (S[u] - 2 * x_v_old) = gain[u] - 2 * x[u] * x_v_old
                
                nbrs = neighbors[v]
                if len(nbrs) > 0:
                    # Vectorized update
                    # gains[nbrs] -= 2 * x[nbrs] * x_v_old
                    gains[nbrs] -= 2 * x[nbrs] * x[v] # Note: x[v] is the NEW value?
                    # Let's check signs.
                    # Old state: x_v.
                    # New state: -x_v.
                    # Change in x_v: -2 * x_v_old.
                    # Contribution to S[u] from v is x_v.
                    # Old contribution: x_v_old.
                    # New contribution: -x_v_old.
                    # Delta S[u] = -2 * x_v_old.
                    # Delta gain[u] = x[u] * Delta S[u] = -2 * x[u] * x_v_old.
                    # In code, we just flipped x[v]. So x[v] is now -x_v_old.
                    # So x_v_old = -x[v] (current).
                    # Delta gain[u] = -2 * x[u] * (-x[v]) = 2 * x[u] * x[v].
                    
                    # Let's re-verify.
                    # gain[u] = x[u] * sum(x[nbrs_u])
                    # Let S_u = sum(x[nbrs_u]).
                    # gain[u] = x[u] S_u.
                    # Flip v. x_v -> -x_v.
                    # For u in N(v):
                    # S_u' = S_u - x_v - (-x_v) = S_u - 2x_v. (Here x_v is old value)
                    # gain[u]' = x[u] (S_u - 2x_v) = gain[u] - 2 x[u] x_v.
                    # In code:
                    # old_x_v = x[v] before flip.
                    # We do x[v] = -x[v].
                    # So current x[v] = -old_x_v.
                    # We need to subtract 2 * x[u] * old_x_v.
                    # old_x_v = -x[v] (current).
                    # So subtract 2 * x[u] * (-x[v]) = add 2 * x[u] * x[v].
                    
                    gains[nbrs] += 2 * x[nbrs] * x[v]
                
                # Update best if improved
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = (x + 1) // 2
        
        # Cool down
        # Simple exponential cooling based on steps
        T *= 0.8
        if T < T_end:
            T = T_end

    return best_labels
# EVOLVE-BLOCK-END
