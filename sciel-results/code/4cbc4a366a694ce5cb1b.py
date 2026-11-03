import numpy as np
import random
import itertools
import time

def construct():
    D = 11
    supports = list(itertools.combinations(range(D), 4))
    n_sup = len(supports)
    
    # Build adjacency matrix (conflict graph)
    # Conflict if intersection size is 3
    # A[i, j] = 1 if supports[i] and supports[j] intersect in 3 elements
    # Note: supports are tuples of sorted indices.
    
    # Optimization: Map support tuple to index
    sup_to_idx = {sup: i for i, sup in enumerate(supports)}
    
    # Precompute intersection sizes?
    # 330 nodes. 330*330 = 108,900 pairs.
    # Intersection check is fast.
    
    A = np.zeros((n_sup, n_sup), dtype=np.int8)
    for i in range(n_sup):
        s_i = set(supports[i])
        for j in range(i + 1, n_sup):
            s_j = set(supports[j])
            if len(s_i.intersection(s_j)) == 3:
                A[i, j] = 1
                A[j, i] = 1
                
    # QUBO Parameters
    # Energy E = -sum(x) + lambda * sum(A_ij * x_i * x_j)
    # We want to minimize E.
    # If no conflicts, E = -k. Min E -> Max k.
    # If conflicts, penalty added.
    
    lambda_pen = 2.0  # Penalty weight. Must be > 1 to discourage conflicts strongly.
    
    # Initialize x
    # Greedy init: pick nodes with min degree?
    # Or random.
    
    # Let's try a few restarts or a long run.
    # Time limit 120s.
    
    best_k = 0
    best_x = np.zeros(n_sup, dtype=np.int8)
    
    # Helper to get valid k from x (if we allow violations during SA, we need to clean up)
    # But if lambda is high, final x should be valid.
    # To be safe, we will check validity at the end.
    
    def get_k_valid(x):
        # Count bits
        k = np.sum(x)
        # Check for conflicts
        # If any conflict, we must remove vertices to make it valid.
        # Greedy repair: while conflict exists, remove one.
        # This is expensive to do in the loop, so we do it at the end.
        # For "score" estimation during SA, we use the QUBO energy.
        return k
        
    # Simulated Annealing
    # State: x (array of 0/1)
    # Move: flip x[i]
    
    # Precompute row sums for faster delta E?
    # Delta E for flipping x[i] from 0 to 1:
    # E_new - E_old = -1 + lambda * (sum_{j != i} A[i,j] x[j])
    # Let h_i = sum_{j != i} A[i,j] x[j]
    # Delta E = -1 + lambda * h_i
    
    # If x[i] is 1 and we flip to 0:
    # Delta E = 1 - lambda * h_i
    
    # We can maintain h array dynamically.
    
    def run_sa(initial_x, T_start, T_end, steps):
        x = initial_x.copy()
        h = np.dot(A, x)  # h[i] = sum_j A[i,j] x[j]. Note A[i,i]=0 so this is sum_{j!=i}
        
        current_E = -np.sum(x) + lambda_pen * 0.5 * (x @ (A @ x))
        # Note: sum_{i<j} A_ij x_i x_j = 0.5 * x^T A x
        
        best_local_k = 0
        best_local_x = x.copy()
        
        T = T_start
        cooling = (T_end / T_start) ** (1.0 / steps)
        
        for step in range(steps):
            # Pick random i
            i = random.randint(0, n_sup - 1)
            
            if x[i] == 0:
                # Flip to 1
                # Delta E = -1 + lambda * h[i]
                dE = -1.0 + lambda_pen * h[i]
                if dE < 0 or (T > 0 and random.random() < np.exp(-dE / T)):
                    x[i] = 1
                    # Update h
                    # h[j] += A[j,i] for all j
                    h += A[:, i]
                    current_E += dE
            else:
                # Flip to 0
                # Delta E = 1 - lambda * h[i]
                dE = 1.0 - lambda_pen * h[i]
                if dE < 0 or (T > 0 and random.random() < np.exp(-dE / T)):
                    x[i] = 0
                    # Update h
                    h -= A[:, i]
                    current_E += dE
            
            T *= cooling
            
            # Track best "valid" or "low energy" state?
            # We track the state with minimum energy.
            # At the end, we will repair if necessary.
            if current_E < (best_local_k * -1): # Rough heuristic, better to track min E
                 pass # This logic is flawed. Let's just track min E.
                 
        # Actually, let's track the state with the minimum energy encountered.
        # But we need to return the x that gives the highest VALID k.
        # So we should periodically check if x is valid and update best_valid.
        
        return x

    # Refined SA Loop with periodic validity check
    def run_sa_optimized():
        # Init: Greedy IS
        # Pick node with min degree
        x = np.zeros(n_sup, dtype=np.int8)
        remaining = set(range(n_sup))
        degrees = [np.sum(A[i]) for i in range(n_sup)]
        
        while remaining:
            # Find min degree in remaining
            min_deg = float('inf')
            best_v = None
            for v in remaining:
                if degrees[v] < min_deg:
                    min_deg = degrees[v]
                    best_v = v
                    if min_deg == 0: break
            x[best_v] = 1
            # Update degrees for neighbors?
            # Simpler: just remove best_v and its neighbors from remaining
            to_remove = {best_v}
            neighbors = np.where(A[best_v] == 1)[0]
            for n in neighbors:
                if n in remaining:
                    to_remove.add(n)
            remaining.difference_update(to_remove)
            
        # Now run SA
        T = 5.0
        steps = 200000
        cooling = 0.9999 # Very slow cooling
        
        h = np.dot(A, x)
        current_E = -np.sum(x) + lambda_pen * 0.5 * (x @ (A @ x))
        
        best_valid_k = np.sum(x)
        best_valid_x = x.copy()
        
        # Check if initial is valid
        # Since it was built as IS, it should be.
        
        for step in range(steps):
            i = random.randint(0, n_sup - 1)
            
            if x[i] == 0:
                dE = -1.0 + lambda_pen * h[i]
                if dE < 0 or (T > 0 and random.random() < np.exp(-dE / T)):
                    x[i] = 1
                    h += A[:, i]
                    current_E += dE
            else:
                dE = 1.0 - lambda_pen * h[i]
                if dE < 0 or (T > 0 and random.random() < np.exp(-dE / T)):
                    x[i] = 0
                    h -= A[:, i]
                    current_E += dE
                    
            T *= cooling
            
            # Periodic check for valid solution
            if step % 1000 == 0:
                # Check for conflicts
                # x is valid if sum(A_ij x_i x_j) == 0
                # conflicts = 0.5 * (x @ (A @ x))
                # But we have h?
                # Sum of conflicts = 0.5 * sum_i x_i h_i
                conflict_count = 0.5 * np.dot(x, h)
                
                if conflict_count == 0:
                    k = np.sum(x)
                    if k > best_valid_k:
                        best_valid_k = k
                        best_valid_x = x.copy()
                else:
                    # If not valid, we can try to repair locally?
                    # Or just ignore for "best_valid" tracking.
                    # The SA will hopefully move towards lower E (fewer conflicts).
                    pass
                    
        return best_valid_x

    x_best = run_sa_optimized()
    
    # Final Repair: If x_best still has conflicts (unlikely if lambda high and we tracked valid),
    # remove one vertex from each conflict.
    # But we tracked best_valid_x, so it should be valid.
    # Just to be safe, verify.
    # If it's not valid, we must fix it.
    # Let's do a simple greedy repair on x_best if needed.
    
    # Actually, the code above returns best_valid_x which is guaranteed valid by the check.
    # However, if the SA never found a valid state better than the greedy init, it returns greedy init.
    # Greedy init is valid.
    
    # Construct vectors
    vectors = []
    for idx in range(n_sup):
        if x_best[idx] == 1:
            sup = supports[idx]
            for signs in itertools.product([1, -1], repeat=4):
                v = [0] * D
                for k, s in enumerate(sup):
                    v[s] = signs[k]
                vectors.append(v)
                
    return np.array(vectors, dtype=np.int64)
