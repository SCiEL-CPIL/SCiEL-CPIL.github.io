"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit
    
    # Generate all weight-4 vectors with ±1 entries
    # C(11,4) = 330 positions, 2^4 = 16 sign combos -> 5280 vectors
    candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.float64)
    n_candidates = len(candidates)
    
    # Precompute Gram matrix: G[i,j] = candidates[i] · candidates[j]
    # Norm squared = 4 for all
    # Constraint: |u-v|^2 >= 4  <=>  8 - 2*u·v >= 4  <=>  u·v <= 2
    G = np.dot(candidates, candidates.T)
    
    # Build adjacency: A[i,j] = 1 if candidates[i] and candidates[j] are incompatible (dot > 2)
    A = (G > 2 + 1e-9).astype(np.int8)
    np.fill_diagonal(A, 0)
    
    # Degree of each node
    degrees = A.sum(axis=1)
    
    def greedy_select(order, time_limit):
        """Greedy selection following the given order."""
        selected = []
        selected_set = set()
        for idx in order:
            if time.time() > time_limit:
                break
            # Check compatibility with all selected
            compatible = True
            for s in selected:
                if A[idx, s]:
                    compatible = False
                    break
            if compatible:
                selected.append(idx)
                selected_set.add(idx)
        return selected
    
    def simulated_annealing(time_limit, seed=42):
        """
        Simulated annealing on a continuous relaxation.
        
        State: x in [0,1]^n, where x[i] is the "inclusion probability" of candidate i.
        Objective: maximize sum(x) - penalty * sum_{i,j incompatible} f(x[i], x[j])
        where f is a smooth penalty that grows when both x[i] and x[j] are large.
        
        We use f(x_i, x_j) = (x_i * x_j)^2 which is smooth and penalizes co-inclusion.
        Temperature T controls the smoothness: we use a soft penalty
        penalty(T) = sum_{i,j: A[i,j]} (x_i * x_j)^2 / T  (as T -> 0, hard constraint)
        
        Actually, let's use a different approach: 
        Energy E(x) = -sum(x) + beta * sum_{i<j, A[i,j]} max(0, x_i*x_j - threshold)^2
        
        Simpler: use E(x) = -sum(x) + beta * sum_{i<j, A[i,j]} (x_i * x_j)^2
        
        Gradient descent with simulated annealing:
        - At each step, pick a random coordinate i, update x[i] by a small step
        - Accept with probability exp(-deltaE / T) if deltaE > 0
        """
        rng = np.random.RandomState(seed)
        n = n_candidates
        
        # Initialize with a greedy solution as starting point
        order = list(range(n))
        # Sort by degree (low degree first) for better initial solution
        order.sort(key=lambda i: degrees[i])
        init_sel = greedy_select(order, time.time() + 5.0)
        
        x = np.zeros(n)
        for idx in init_sel:
            x[idx] = 1.0
        
        # Parameters
        T_initial = 1.0
        T_final = 0.01
        cooling_rate = 0.9995
        step_size = 0.1
        beta = 5.0  # penalty weight
        
        T = T_initial
        best_x = x.copy()
        best_obj = _objective(x, beta)
        
        n_incompat = A.sum() // 2  # number of incompatible pairs
        
        while time.time() < time_limit:
            # Pick random coordinate
            i = rng.randint(n)
            # Propose new value: x[i] + delta
            delta = step_size * rng.choice([-1, 1])
            new_xi = np.clip(x[i] + delta, 0.0, 1.0)
            if new_xi == x[i]:
                continue
            
            # Compute change in objective
            # E = -sum(x) + beta * sum_{j: A[i,j]} (x[i]*x[j])^2
            # deltaE = -(new_xi - x[i]) + beta * sum_{j: A[i,j]} ((new_xi*x[j])^2 - (x[i]*x[j])^2)
            neighbors = np.where(A[i] == 1)[0]
            if len(neighbors) == 0:
                deltaE = -(new_xi - x[i])
            else:
                xj = x[neighbors]
                deltaE = -(new_xi - x[i]) + beta * np.sum((new_xi * xj)**2 - (x[i] * xj)**2)
            
            if deltaE <= 0 or rng.random() < np.exp(-deltaE / max(T, 1e-10)):
                x[i] = new_xi
                obj = _objective(x, beta)
                if obj < best_obj:
                    best_obj = obj
                    best_x = x.copy()
            
            # Cool down
            T *= cooling_rate
            if T < T_final:
                T = T_final
        
        # Round best_x to get integer solution
        # Take all i where x[i] > 0.5, then fix conflicts greedily
        threshold = 0.5
        candidate_indices = np.where(best_x > threshold)[0]
        
        # Greedy fix: sort by best_x value (higher first), add if compatible
        candidate_indices = candidate_indices[np.argsort(-best_x[candidate_indices])]
        final_sel = []
        for idx in candidate_indices:
            compatible = True
            for s in final_sel:
                if A[idx, s]:
                    compatible = False
                    break
            if compatible:
                final_sel.append(idx)
        
        return final_sel
    
    def _objective(x, beta):
        """Compute objective: -sum(x) + beta * sum_{i<j, A[i,j]} (x_i*x_j)^2"""
        neg_sum = -np.sum(x)
        # Efficient computation: for each i, sum over neighbors j
        # (x_i * x_j)^2 = x_i^2 * x_j^2
        # sum_{i<j, A[i,j]} x_i^2 * x_j^2
        x2 = x ** 2
        # Use the adjacency matrix
        # sum_i x_i^2 * sum_{j: A[i,j]} x_j^2  / 2  (since each pair counted twice)
        neighbor_sums = A.dot(x2)
        penalty = 0.5 * np.sum(x2 * neighbor_sums)
        return neg_sum + beta * penalty
    
    # Try multiple strategies
    best_selected = []
    np.random.seed(42)
    
    # Strategy 1: Original fixed order (greedy by index)
    fixed_order = list(range(n_candidates))
    sel = greedy_select(fixed_order, deadline - 60.0)
    if len(sel) > len(best_selected):
        best_selected = sel
    
    # Strategy 2: Greedy by degree (low degree first)
    degree_order = list(range(n_candidates))
    degree_order.sort(key=lambda i: degrees[i])
    sel = greedy_select(degree_order, deadline - 60.0)
    if len(sel) > len(best_selected):
        best_selected = sel
    
    # Strategy 3: Multiple random orderings with simple greedy
    time_remaining = deadline - time.time() - 20.0
    n_trials = min(30, max(5, int(time_remaining / 2.0)))
    for trial in range(n_trials):
        if time.time() > deadline - 20.0:
            break
        perm = np.random.permutation(n_candidates)
        sel = greedy_select(perm, deadline - 20.0)
        if len(sel) > len(best_selected):
            best_selected = sel
    
    # Strategy 4: Simulated annealing
    if time.time() < deadline - 10.0:
        sa_time = deadline - 10.0 - time.time()
        sa_trials = max(1, int(sa_time / 15.0))
        for trial in range(sa_trials):
            if time.time() > deadline - 10.0:
                break
            sa_seed = 42 + trial * 1000
            sel = simulated_annealing(deadline - 10.0 - (trial + 1) * (sa_time / sa_trials), seed=sa_seed)
            if len(sel) > len(best_selected):
                best_selected = sel
    
    if not best_selected:
        # Fallback: return a minimal valid set
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)
    
    return candidates[best_selected].astype(np.int64)
# EVOLVE-BLOCK-END
