import random
import time
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    start_time = time.time()
    deadline = start_time + 290  # Safety margin for 300s limit

    m = len(clauses)
    
    # Precompute clause literals as 0-indexed variable indices and signs
    # clause_vars[i] = list of var indices
    # clause_signs[i] = list of 1 for positive, -1 for negative
    clause_vars = []
    clause_signs = []
    
    # Incidence lists: var_pos_clauses[v] = list of clause indices containing +x_v
    # var_neg_clauses[v] = list of clause indices containing -x_v
    var_pos_clauses = [[] for _ in range(n)]
    var_neg_clauses = [[] for _ in range(n)]
    
    for i, clause in enumerate(clauses):
        vars_list = []
        signs_list = []
        for lit in clause:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            vars_list.append(v)
            signs_list.append(s)
            if s == 1:
                var_pos_clauses[v].append(i)
            else:
                var_neg_clauses[v].append(i)
        clause_vars.append(vars_list)
        clause_signs.append(signs_list)

    def weighted_walksat(A, max_steps):
        """
        Weighted WalkSAT / Max-SAT heuristic.
        A: numpy int8 array of size n (0 or 1)
        """
        # Weights for each clause
        weights = np.ones(m, dtype=np.float32)
        # Satisfaction counts: number of literals satisfied in each clause
        # sat_count[i] = sum(1 for lit in clause[i] if A matches lit)
        sat_count = np.zeros(m, dtype=np.int8)
        
        # Initialize sat_count
        for i in range(m):
            c_vars = clause_vars[i]
            c_signs = clause_signs[i]
            count = 0
            for v, s in zip(c_vars, c_signs):
                if (A[v] == 1 and s == 1) or (A[v] == 0 and s == -1):
                    count += 1
            sat_count[i] = count
            
        violated_mask = (sat_count == 0)
        num_violated = np.sum(violated_mask)
        
        if num_violated == 0:
            return True

        # Weight decay factor
        decay = 0.99
        # Weight increase factor for violated clauses
        increase = 1.5
        # Probability to pick greedily (highest weighted violated clause) vs random violated
        p_greedy = 0.7
        
        for step in range(max_steps):
            if time.time() > deadline:
                return False
            
            if num_violated == 0:
                return True

            # Select a violated clause
            if rng.random() < p_greedy:
                # Find the violated clause with the maximum weight
                # We only care about violated clauses
                # Create a masked view of weights where violated_mask is True
                # To avoid full array max, we can just pick randomly from violated if m is small, 
                # but weighted selection is the point.
                # Since m is likely small (planted 3-SAT usually m ~ 4-5*n), 
                # finding max over ~250 elements is fast.
                # However, doing this every step might be slow in Python.
                # Optimization: Keep a list of violated clause indices.
                # For now, simple approach: pick random violated, but bias by weight?
                # Actually, let's just pick the max weight among violated.
                
                # Efficient way: 
                # weights_masked = np.where(violated_mask, weights, 0)
                # best_ci = np.argmax(weights_masked)
                # But we need to ensure it's actually violated (argmax could be 0 if all weights are 0, but they start at 1 and decay, so they are > 0)
                
                # Alternative: Sample from violated clauses with probability proportional to weight.
                # This is harder to implement efficiently without a heap or similar.
                # Let's stick to: Pick random violated clause, but if we have time, use max weight.
                # Given n=50, m is small. Let's just iterate violated clauses to find max weight?
                # No, let's use the numpy argmax trick.
                
                # To make it faster, we can just pick a random violated clause index
                # and rely on the weight update to make "hard" clauses more likely to be picked
                # in subsequent steps if they remain violated? No, that's not how it works.
                
                # Let's implement a simple weighted choice.
                # Get indices of violated clauses
                violated_indices = np.where(violated_mask)[0]
                if len(violated_indices) == 0:
                    return True
                    
                # Weights of these violated clauses
                w = weights[violated_indices]
                # Normalize for sampling? Or just use rng.choice with weights?
                # rng.choice doesn't take weights directly.
                # Use np.random.choice
                # But we need to use our rng for reproducibility? 
                # The task doesn't strictly require reproducibility of the random seed across runs, 
                # but using a fixed seed is good practice. 
                # np.random.choice uses global numpy state. Let's use our rng.
                
                # Manual weighted choice
                total_w = np.sum(w)
                if total_w == 0:
                    ci = rng.choice(violated_indices.tolist())
                else:
                    # Cumulative sum
                    cum_w = np.cumsum(w)
                    # Pick random threshold
                    r = rng.random() * total_w
                    # Find index
                    ci_idx = np.searchsorted(cum_w, r)
                    if ci_idx >= len(violated_indices):
                        ci_idx = len(violated_indices) - 1
                    ci = violated_indices[ci_idx]
            else:
                # Random violated clause
                violated_indices = np.where(violated_mask)[0]
                ci = rng.choice(violated_indices.tolist())

            # Get the clause details
            c_vars = clause_vars[ci]
            c_signs = clause_signs[ci]
            
            # Calculate gain for each variable in the clause
            # Gain = (# clauses that become satisfied) - (# clauses that become violated)
            # We only need to consider clauses connected to these variables.
            
            best_gain = -100
            best_var = -1
            best_sign = 0 # The sign we are flipping to
            
            # We will store gains for random selection if needed, but greedy is primary
            gains = []
            
            for v, s in zip(c_vars, c_signs):
                # Current value of v
                cur_val = A[v]
                new_val = 1 - cur_val
                
                # Calculate delta
                # If new_val == 1:
                #   Clauses with +v increase sat_count by 1
                #   Clauses with -v decrease sat_count by 1
                # If new_val == 0:
                #   Clauses with -v increase sat_count by 1
                #   Clauses with +v decrease sat_count by 1
                
                # We need to count how many clauses change from 0 to >0 (satisfied)
                # and how many change from >0 to 0 (violated).
                
                become_sat = 0
                become_violated = 0
                
                if new_val == 1:
                    # +v becomes true, -v becomes false
                    for c_idx in var_pos_clauses[v]:
                        if sat_count[c_idx] == 0:
                            become_sat += 1
                    for c_idx in var_neg_clauses[v]:
                        if sat_count[c_idx] == 1:
                            become_violated += 1
                else:
                    # -v becomes true, +v becomes false
                    for c_idx in var_neg_clauses[v]:
                        if sat_count[c_idx] == 0:
                            become_sat += 1
                    for c_idx in var_pos_clauses[v]:
                        if sat_count[c_idx] == 1:
                            become_violated += 1
                
                gain = become_sat - become_violated
                gains.append((gain, v, new_val))
            
            # Select variable
            if rng.random() < p_greedy:
                # Max gain
                gains.sort(key=lambda x: -x[0])
                best_gain, best_var, new_val = gains[0]
            else:
                # Random from clause
                best_gain, best_var, new_val = rng.choice(gains)
            
            # Apply flip
            A[best_var] = new_val
            
            # Update sat_count
            if new_val == 1:
                for c_idx in var_pos_clauses[best_var]:
                    sat_count[c_idx] += 1
                for c_idx in var_neg_clauses[best_var]:
                    sat_count[c_idx] -= 1
            else:
                for c_idx in var_neg_clauses[best_var]:
                    sat_count[c_idx] += 1
                for c_idx in var_pos_clauses[best_var]:
                    sat_count[c_idx] -= 1
            
            # Update weights
            # Decay all weights
            weights *= decay
            # Increase weight of violated clauses
            violated_mask = (sat_count == 0)
            weights[violated_mask] *= increase
            
            num_violated = np.sum(violated_mask)
            
        return False

    best_assignment = None
    best_violated = m + 1
    
    max_restarts = 1000
    base_steps = 10000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Random initial assignment
        A = (np.random.rand(n) < 0.5).astype(np.int8)
        
        # Try Weighted WalkSAT
        # Adaptive steps: start small, increase if not found?
        # Or just fixed.
        steps = base_steps + rng.randint(0, 2000)
        
        if weighted_walksat(A, steps):
            return A.astype(bool).tolist()
            
        # Track best
        # Count violated
        v_count = 0
        for i in range(m):
            c_vars = clause_vars[i]
            c_signs = clause_signs[i]
            sat = 0
            for v, s in zip(c_vars, c_signs):
                if (A[v] == 1 and s == 1) or (A[v] == 0 and s == -1):
                    sat += 1
            if sat == 0:
                v_count += 1
                
        if v_count < best_violated:
            best_violated = v_count
            best_assignment = A.copy()
            
        # If we get close to solution, we might want to increase steps for next restart
        # But simple is better.

    if best_assignment is not None:
        # Final check
        # Re-verify best_assignment
        v_count = 0
        for i in range(m):
            c_vars = clause_vars[i]
            c_signs = clause_signs[i]
            sat = 0
            for v, s in zip(c_vars, c_signs):
                if (best_assignment[v] == 1 and s == 1) or (best_assignment[v] == 0 and s == -1):
                    sat += 1
            if sat == 0:
                v_count += 1
        if v_count == 0:
            return best_assignment.astype(bool).tolist()
            
        return best_assignment.astype(bool).tolist()
        
    return [rng.random() < 0.5 for _ in range(n)]
