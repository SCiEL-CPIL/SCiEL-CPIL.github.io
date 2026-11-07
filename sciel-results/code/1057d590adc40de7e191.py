"""Multi-objective optimizer for RE37: MOEA/D with Tchebycheff decomposition."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """MOEA/D multi-objective optimizer with deadline safety."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        # Number of subproblems. 20-30 is good for 200 budget.
        self.n_sub = 20 

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        d = self.dim
        n_obj = self.n_obj
        n_sub = self.n_sub

        rng = np.random.default_rng(0)

        # 1. Generate Weight Vectors on Simplex
        # For 3 objectives, we need points on a triangle.
        # Use a grid with H subdivisions.
        H = 5
        weights = []
        for i in range(H + 1):
            for j in range(H + 1 - i):
                k = H - i - j
                if i + j + k == H:
                    weights.append([i / H, j / H, k / H])
        
        # We generated (H+1)(H+2)/2 points. For H=5, 21 points.
        # We want n_sub points. If len > n_sub, sample uniformly.
        if len(weights) > n_sub:
            indices = rng.choice(len(weights), n_sub, replace=False)
            weights = [weights[i] for i in indices]
        elif len(weights) < n_sub:
            # Pad with random weights if grid is too small
            while len(weights) < n_sub:
                w = rng.random(n_obj)
                w /= np.sum(w)
                weights.append(w)
        
        weights = np.array(weights)
        n_sub = len(weights)

        # 2. Initialize Population
        X = lo + (hi - lo) * rng.random((n_sub, d))
        F = np.array([problem(x) for x in X])
        used = n_sub

        # Record all evaluated points
        all_F = [F.copy()]

        # 3. Determine Neighborhoods
        # Neighborhood size
        T = 5 
        neighbors = []
        for i in range(n_sub):
            # Sort other weights by distance to weight i
            dists = np.linalg.norm(weights - weights[i], axis=1)
            # Exclude self
            dists[i] = np.inf
            idx = np.argsort(dists)[:T]
            neighbors.append(idx)

        # 4. Tchebycheff Fitness
        def tchebycheff(f, w):
            # w is 1D array of shape (n_obj,)
            # f is 1D array of shape (n_obj,)
            # Maximize min? No, minimize.
            # G = max_k ( w_k * | f_k - z_k^* | )
            # z^* is ideal point. We estimate it dynamically.
            return f # Placeholder, need z_star

        # We need ideal point z_star.
        # Estimate z_star as min of F so far.
        
        def calc_tcheb(F_mat, W_mat, z_star):
            # F_mat: (n, n_obj)
            # W_mat: (n, n_obj)
            # z_star: (n_obj,)
            # G[i, j] = max_k ( W[i, k] * | F[i, k] - z_star[k] | )
            # We want to minimize G for each subproblem j using solution i?
            # Usually, we compute G(x, lambda^k) for solution x and weight k.
            # Here we compute for all pairs to update neighbors.
            # Shape: (n_sub, n_sub)
            # G[i, j] = max_k ( weights[j, k] * abs(F[i, k] - z_star[k]) )
            
            # F[i, k] - z_star[k]
            diff = np.abs(F_mat[:, np.newaxis, :] - z_star[np.newaxis, np.newaxis, :])
            # weights[j, k]
            w_exp = weights[np.newaxis, :, :]
            # G[i, j, k] = w_exp[j, k] * diff[i, k]
            G = np.max(w_exp * diff, axis=2)
            return G

        # 5. Main Loop
        while used + n_sub <= self.budget:
            if time.time() >= deadline:
                break

            # Update ideal point
            z_star = np.min(F, axis=0)
            
            # Calculate Tchebycheff matrix for current population
            # G[i, j] is fitness of solution i for subproblem j
            G = calc_tcheb(F, weights, z_star)
            
            # Generate Offspring
            Q_X = np.zeros((n_sub, d))
            Q_F = np.zeros((n_sub, n_obj))
            
            for k in range(n_sub):
                # Pick parents from neighborhood of k
                nbs = neighbors[k]
                # Select best 2 from neighbors based on G[:, k]
                # Actually, standard MOEA/D picks random from neighbors or best.
                # Let's pick the 2 best from neighbors for subproblem k.
                nb_indices = nbs
                # Sort neighbors by fitness for subproblem k
                nb_fitness = G[nb_indices, k]
                sorted_nb = nb_indices[np.argsort(nb_fitness)]
                
                p1_idx = sorted_nb[0]
                p2_idx = sorted_nb[1] if len(sorted_nb) > 1 else p1_idx
                
                x1 = X[p1_idx]
                x2 = X[p2_idx]
                
                # SBX Crossover
                child = np.zeros(d)
                for j in range(d):
                    if rng.random() < 0.5:
                        # Beta distribution approximation
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / 20.0)
                        else:
                            beta = (2 * (1 - u)) ** (1 / 20.0)
                        child[j] = 0.5 * ((1 + beta) * x1[j] + (1 - beta) * x2[j])
                    else:
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / 20.0)
                        else:
                            beta = (2 * (1 - u)) ** (1 / 20.0)
                        child[j] = 0.5 * ((1 + beta) * x2[j] + (1 - beta) * x1[j])
                
                # Polynomial Mutation
                for j in range(d):
                    if rng.random() < 1.0 / d:
                        u = rng.random()
                        if u < 0.5:
                            delta = (2 * u) ** (1 / 20.0) - 1
                        else:
                            delta = 1 - (2 * (1 - u)) ** (1 / 20.0)
                        child[j] = child[j] + delta * (hi[j] - lo[j])
                
                child = np.clip(child, lo, hi)
                Q_X[k] = child
                Q_F[k] = problem(child)
                used += 1

            # Record offspring
            all_F.append(Q_F.copy())

            # Update Population
            # For each offspring k, check its neighbors
            # Calculate Tchebycheff for offspring
            G_off = calc_tcheb(Q_F, weights, z_star)
            
            for k in range(n_sub):
                nbs = neighbors[k]
                for j in nbs:
                    # If offspring k is better than current X[j] for subproblem j
                    if G_off[k, j] < G[j, j]:
                        X[j] = Q_X[k]
                        F[j] = Q_F[k]
                        G[j, j] = G_off[k, j] # Update self-fitness

            # Note: G matrix needs to be fully updated for next iteration?
            # We only updated diagonal G[j,j]. The off-diagonals G[i,j] for i != j 
            # are not strictly necessary if we recompute G at the start of the next loop.
            # We do recompute G at the start of the while loop.

        # Use remaining budget for local search or random
        while used < self.budget:
            if time.time() >= deadline:
                break
            # Pick random point from population and mutate
            idx = rng.integers(n_sub)
            child = X[idx] + 0.05 * (hi - lo) * rng.standard_normal(d)
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1
            all_F.append(fc.reshape(1, -1))
            
            # Check if it improves any neighbor
            z_star = np.min(F, axis=0)
            G_child = calc_tcheb(fc.reshape(1, -1), weights, z_star)
            for j in neighbors[idx]:
                if G_child[0, j] < G[j, j]:
                    X[j] = child
                    F[j] = fc
                    G[j, j] = G_child[0, j]

        return X

# EVOLVE-BLOCK-END
