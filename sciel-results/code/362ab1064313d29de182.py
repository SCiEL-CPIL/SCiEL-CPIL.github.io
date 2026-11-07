"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-start CMA-ES for black-box continuous optimization.
    
    Strategy:
    1. Initialize with a diverse set of random points.
    2. Run multiple independent CMA-ES instances in a round-robin fashion.
    3. Each CMA-ES instance adapts its step size and covariance matrix to the local landscape.
    4. This approach handles the rotated and bimodal nature of Lunacek bi-Rastrigin better 
       than single-trajectory methods or static GP surrogates.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Deadline guard
        start_time = time.time()
        max_time = 280.0  # Safety margin

        rng = np.random.default_rng(123)
        
        # CMA-ES parameters
        # For dim=5, a population size lambda of 10-20 is standard.
        # We use 4 parallel CMA-ES instances to balance exploration and exploitation.
        n_agents = 4
        lambda_ = 10  # Population size per agent
        
        # Ensure we have enough budget for at least one full generation per agent
        if budget < n_agents * lambda_:
             # Fallback to simple random search if budget is extremely small (unlikely given 500)
             X = rng.uniform(-5, 5, size=(budget, dim))
             Y = np.array([func(x) for x in X])
             return X[np.argmin(Y)], Y.min()

        # Initialize agents
        # Start with diverse centers
        centers = rng.uniform(-4, 4, size=(n_agents, dim))
        
        # State for each agent
        # CMA-ES state: mean, sigma, C, p_sigma, p_c
        agents = []
        for i in range(n_agents):
            agents.append({
                'mean': centers[i].copy(),
                'sigma': 1.5,  # Initial step size
                'C': np.eye(dim),
                'p_sigma': np.zeros(dim),
                'p_c': np.zeros(dim),
                'history': [] # Store (x, y) for this agent to select parents
            })
            
        # Global best
        best_y = np.inf
        best_x = None
        
        # Pre-compute CMA-ES hyperparameters
        # Standard CMA-ES formulas
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = np.min(1 - c_1, 2 * (weights[0] - 0.5) / ((dim + 2)**2 + mu_eff))
        
        # Evaluation loop
        calls = 0
        gen = 0
        
        while calls < budget:
            if time.time() - start_time > max_time:
                break
                
            # Round-robin: pick an agent
            agent_idx = gen % n_agents
            agent = agents[agent_idx]
            
            # Generate offspring for this agent
            # We generate lambda_ points
            # To save time and ensure we don't exceed budget mid-generation, 
            # we generate exactly lambda_ points if we have enough budget left.
            # If not, we generate the remaining budget.
            n_to_gen = min(lambda_, budget - calls)
            
            if n_to_gen == 0:
                break
                
            # Sample from multivariate normal
            # We need the Cholesky factor of C for sampling
            try:
                L = np.linalg.cholesky(agent['C'])
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset C
                agent['C'] = np.eye(dim)
                L = np.eye(dim)
                
            # Generate random vectors
            z = rng.standard_normal((n_to_gen, dim))
            # x = mean + sigma * L @ z.T
            offspring = agent['mean'] + agent['sigma'] * (z @ L.T)
            
            # Clip to bounds [-5, 5]
            # Note: CMA-ES usually works better without hard clipping if the domain is large enough,
            # but the problem requires clipping. We clip for evaluation.
            offspring_clipped = np.clip(offspring, -5, 5)
            
            # Evaluate offspring
            new_y = np.array([func(x) for x in offspring_clipped])
            calls += n_to_gen
            
            # Update global best
            if np.min(new_y) < best_y:
                best_y = np.min(new_y)
                best_idx = np.argmin(new_y)
                best_x = offspring_clipped[best_idx].copy()
            
            # If we didn't generate a full generation, we can't properly update CMA-ES state
            # without biasing it. We only update if we have a full generation.
            if n_to_gen == lambda_:
                # Sort offspring by fitness
                idx_sorted = np.argsort(new_y)
                parents = offspring_clipped[idx_sorted[:mu]]
                parents_y = new_y[idx_sorted[:mu]]
                
                # Update mean
                new_mean = np.sum(weights[:, None] * parents, axis=0)
                
                # Update p_sigma
                agent['p_sigma'] = (1 - c_sigma) * agent['p_sigma'] + \
                                    np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * \
                                    agent['sigma'] / np.linalg.norm(L) * \
                                    (np.linalg.solve(L, (new_mean - agent['mean']) / agent['sigma']).T)
                
                # Heuristic for sigma update (CSA - Cumulative Step Adaptation)
                # ||p_sigma|| / E||N(0,I)||
                h_sigma = (np.linalg.norm(agent['p_sigma']) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1)))) / \
                          np.sqrt(dim) * (1 + c_sigma)
                h_sigma = min(h_sigma, 1.4 + 2 / (dim + 1))
                
                # Update p_c
                agent['p_c'] = (1 - c_c) * agent['p_c'] + \
                                h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * \
                                (new_mean - agent['mean']) / agent['sigma']
                
                # Update C
                # delta_h = (1 + (1 - h_sigma) * c_c * (2 - c_c)) * (np.linalg.norm(agent['p_c'])**2 / dim - 1)
                # This term is often omitted or simplified in basic implementations for speed/stability
                # Let's use the standard rank-1 + rank-mu update
                delta_h = 0 # Simplification for stability and speed
                
                C_old = agent['C']
                agent['C'] = (1 + c_1 + c_mu - c_1) * C_old + \
                              c_1 * (np.outer(agent['p_c'], agent['p_c']) + delta_h * np.eye(dim)) + \
                              c_mu * np.sum(weights[:, None, None] * np.outer(parents - new_mean, (parents - new_mean).T), axis=0)
                
                # Update sigma
                agent['sigma'] = agent['sigma'] * np.exp(c_sigma / d_sigma * (np.linalg.norm(agent['p_sigma']) / np.sqrt(dim) - 1))
                
                # Clip sigma to reasonable bounds
                agent['sigma'] = np.clip(agent['sigma'], 0.01, 10.0)
                
                # Update mean
                agent['mean'] = new_mean
                
                # Ensure C is symmetric and positive semi-definite
                agent['C'] = (agent['C'] + agent['C'].T) / 2
                # Add small identity to ensure PD
                agent['C'] += 1e-6 * np.eye(dim)
                
                # Reset if sigma is too small or mean is out of bounds
                if agent['sigma'] < 0.01:
                    agent['sigma'] = 1.0
                    agent['C'] = np.eye(dim)
                    agent['p_sigma'] = np.zeros(dim)
                    agent['p_c'] = np.zeros(dim)
                    agent['mean'] = rng.uniform(-4, 4, size=dim)
            
            gen += 1
            
        return best_x, best_y

# EVOLVE-BLOCK-END
