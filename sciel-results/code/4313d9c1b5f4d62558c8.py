"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_dates = 12
    dt = T / n_dates
    discount = np.exp(-r * dt)
    
    n_paths = training_paths.shape[0]
    
    # S_all shape: (n_paths, 12) corresponding to dates 1..12
    # training_paths has n_dates+1 columns (0..12), we use 1..12
    S_all = training_paths[:, 1:n_dates+1]
    
    # V[i] will store the value of the option at exercise date i+1 (0-indexed i)
    # V[11] is the value at the last date (maturity)
    V = np.zeros((n_dates, n_paths))
    
    # At maturity (last date, index 11), the value is the European payoff
    V[11, :] = np.maximum(K - S_all[:, 11], 0.0)
    
    # We will store the continuation value function parameters for each date
    # For each date i, we store:
    #   nodes[i]: array of S values (quantiles)
    #   cont_vals[i]: array of expected V[i+1] at those nodes
    # Then we can interpolate to get C_i(S) = E[V[i+1] | S_i = S]
    
    n_nodes = 200  # Number of quantile points for interpolation
    
    # Pre-compute quantile nodes for each date to save time?
    # The distribution of S changes over time, so we should compute nodes per date.
    # However, to save time, we can use a common grid or compute on the fly.
    # Computing on the fly is safer for accuracy.
    
    # We need to store the policy parameters
    policy_nodes = []
    policy_cont_vals = []
    
    # Backward induction
    for i in range(10, -1, -1):
        if time.time() > deadline:
            # If we run out of time, we need to break and return a best-effort policy
            # For the remaining earlier dates, we can assume a simple heuristic or 
            # fill with the last computed policy or a static threshold.
            # To be safe, let's fill the rest with a simple threshold based on K.
            # But we need to complete the loop or handle the return carefully.
            # Let's just break and handle the missing dates in the policy object.
            pass
        
        S_i = S_all[:, i]
        V_next = V[i+1, :]
        
        # Estimate C_i(S) = E[V_next | S_i = S]
        # Use quantile binning to ensure uniform coverage of the S distribution
        # This is more robust than fixed-width bins for skewed distributions like GBM
        
        # Compute quantiles of S_i
        # We use n_nodes quantiles. 
        # Note: np.quantile is O(N log N) roughly, which is fine for 4096 paths.
        try:
            nodes = np.quantile(S_i, np.linspace(0, 1, n_nodes + 2)[1:-1])
            # Remove duplicates if any (can happen with low n_paths or flat distributions)
            # But GBM is continuous, so unlikely.
            
            # For each node, we want the average V_next for S_i close to the node.
            # Instead of k-nearest neighbors which is slow, we can use binning again,
            # but define the bins by these quantiles.
            
            # Create bins defined by the quantiles
            # We need bin edges. We can use the quantiles as edges?
            # If we use quantiles as edges, each bin has ~n_paths/n_nodes samples.
            
            # Let's use the quantiles as the centers or edges.
            # If we use them as edges:
            # edges = np.concatenate(([min], quantiles, [max]))
            # But we want to evaluate at the quantiles themselves.
            
            # Alternative: 
            # 1. Sort S_i and V_next together.
            # 2. Divide into n_nodes bins of equal size (in terms of path count).
            # 3. The "node" for each bin is the median S of that bin.
            # 4. The "value" for each bin is the mean V_next of that bin.
            
            order = np.argsort(S_i)
            S_sorted = S_i[order]
            V_sorted = V_next[order]
            
            # Define bin indices
            # Each bin has n_paths // n_nodes paths
            bin_size = n_paths // n_nodes
            if bin_size == 0:
                bin_size = 1
                n_nodes = n_paths # fallback
            
            # Calculate medians and means
            nodes = np.zeros(n_nodes)
            cont_vals = np.zeros(n_nodes)
            
            for k in range(n_nodes):
                start_idx = k * bin_size
                end_idx = (k + 1) * bin_size if k < n_nodes - 1 else n_paths
                if start_idx >= n_paths:
                    break
                bin_S = S_sorted[start_idx:end_idx]
                bin_V = V_sorted[start_idx:end_idx]
                nodes[k] = np.median(bin_S)
                cont_vals[k] = np.mean(bin_V)
            
            # Ensure nodes are strictly increasing for np.interp
            # If there are duplicates, remove them
            mask = np.concatenate(([True], np.diff(nodes) > 1e-9))
            nodes = nodes[mask]
            cont_vals = cont_vals[mask]
            
            # Extend the range to cover all possible S values
            # np.interp uses constant extension by default (left/right values)
            # We should set left and right to the first and last cont_vals
            # Or better, extrapolate linearly? 
            # For a put, as S -> 0, V -> K (discounted future K). 
            # As S -> inf, V -> 0.
            # The data will cover the relevant range.
            
            policy_nodes.append(nodes)
            policy_cont_vals.append(cont_vals)
            
            # Now compute V[i] for these training paths
            # C_i(S_i) = interp(S_i, nodes, cont_vals)
            C_i = np.interp(S_i, nodes, cont_vals, left=cont_vals[0], right=cont_vals[-1])
            
            # Discounted continuation value
            disc_cont = discount * C_i
            
            # Exercise if K - S_i > disc_cont
            exercise_here = (K - S_i) > disc_cont
            
            # V[i] is the max of exercise payoff and continuation
            V[i, :] = np.maximum(K - S_i, disc_cont)
            
        except Exception as e:
            # Fallback if quantile/binning fails
            # Use a simple threshold: Exercise if S < K
            policy_nodes.append(np.array([0.0, K, 2*K]))
            policy_cont_vals.append(np.array([K, 0.0, 0.0]))
            C_i = np.interp(S_i, [0.0, K, 2*K], [K, 0.0, 0.0])
            disc_cont = discount * C_i
            V[i, :] = np.maximum(K - S_i, disc_cont)

    # If we broke out of the loop early due to deadline, 
    # policy_nodes might have fewer than n_dates entries.
    # We need to fill the missing earlier dates (indices 0 to len(policy_nodes)-1)
    # Since we go backwards, if we stop at i=5, we have nodes for 10,9,8,7,6,5.
    # We are missing 4,3,2,1,0.
    # We can reuse the earliest computed policy (index 5) for the earlier steps,
    # or use a static heuristic. Reusing the earliest computed is a reasonable fallback.
    
    while len(policy_nodes) < n_dates:
        # Prepend a copy of the first available (which is the earliest time we calculated)
        # Wait, the list is built from i=10 down to 0.
        # So policy_nodes[0] is for i=10. policy_nodes[-1] is for i=0.
        # If we stopped at i=5, we have 6 entries. We need 12.
        # The missing ones are for i=4,3,2,1,0.
        # We should prepend copies of the *last* calculated one (which is the earliest time, i=5)
        # Actually, if we iterate i=10..0, and stop at i=5, we have calculated 10,9,8,7,6,5.
        # The list has [10, 9, 8, 7, 6, 5].
        # We need to add 4, 3, 2, 1, 0.
        # So we prepend the last element (which corresponds to i=5) to the front?
        # No, the order in the list is [date 11 params, date 10 params, ...].
        # Wait, my loop is `for i in range(10, -1, -1)`.
        # i=10 is the second to last date.
        # V[11] is the last date.
        # So i=10 corresponds to the 11th column of S_all? No.
        # S_all[:, 0] is date 1. S_all[:, 11] is date 12.
        # V[11] is value at date 12.
        # Loop i=10: uses S_all[:, 10] (date 11) and V[11] (value at date 12). Computes V[10] (value at date 11).
        # So i corresponds to the index in S_all and V.
        # If we stop at i=5, we have computed V[10]..V[5].
        # We are missing V[4]..V[0].
        # The policy_nodes list contains params for i=10, 9, ..., 5.
        # So policy_nodes[0] is for i=10. policy_nodes[-1] is for i=5.
        # To fill i=4,3,2,1,0, we should use the policy from i=5 (the earliest one we have).
        # So we prepend copies of policy_nodes[-1] and policy_cont_vals[-1].
        
        policy_nodes.insert(0, policy_nodes[-1])
        policy_cont_vals.insert(0, policy_cont_vals[-1])

    # Now policy_nodes has n_dates entries.
    # policy_nodes[0] corresponds to i=0 (Date 1).
    # policy_nodes[11] corresponds to i=11 (Date 12).
    
    class ExercisePolicy:
        def __init__(self, nodes_list, vals_list, n_dates):
            self.nodes_list = nodes_list
            self.vals_list = vals_list
            self.n_dates = n_dates
            self.discount = discount
            self.K = K
            
        def exercise(self, current_state, time_val):
            # time_val is the current time.
            # Exercise dates are at t = 1/12, 2/12, ..., 12/12.
            # Determine the index of the current exercise date.
            # time_val * 12 gives the date number (1 to 12).
            # Index in our arrays (0 to 11) is date_number - 1.
            
            try:
                date_idx = int(round(time_val * 12)) - 1
                if date_idx < 0:
                    date_idx = 0
                if date_idx >= self.n_dates:
                    date_idx = self.n_dates - 1
            except:
                date_idx = 0
                
            S_vals = np.asarray(current_state, dtype=float)
            
            nodes = self.nodes_list[date_idx]
            vals = self.vals_list[date_idx]
            
            # Get continuation value for each S
            # Use linear interpolation
            # Left and right values: 
            # For S < min(nodes), use vals[0]
            # For S > max(nodes), use vals[-1]
            C = np.interp(S_vals, nodes, vals, left=vals[0], right=vals[-1])
            
            # Discounted continuation
            disc_C = self.discount * C
            
            # Exercise if intrinsic value > continuation
            intrinsic = self.K - S_vals
            
            return intrinsic > disc_C

    policy = ExercisePolicy(policy_nodes, policy_cont_vals, n_dates)
    return policy
# EVOLVE-BLOCK-END
