"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Parallel Population of (1+1)-ES with adaptive step-size and restarting."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # 10s safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_f = np.inf
        best_x = None

        def call_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            return float(func(x))

        def check_deadline():
            return (time.time() - start_time) > deadline

        # --- Phase 1: Initialization ---
        # Create a diverse population of agents
        # We use a mix of deterministic and random starts
        n_agents = 50
        agents = []
        
        # Agent 0: Origin
        agents.append(np.zeros(dim))
        
        # Agent 1-4: Deterministic shifts
        agents.append(np.full(dim, 1.0))
        agents.append(np.full(dim, -1.0))
        agents.append(np.full(dim, 2.0))
        agents.append(np.full(dim, -2.0))
        
        # Agent 5+: Random starts
        rng_init = np.random.default_rng(42)
        for _ in range(n_agents - 5):
            x = rng_init.uniform(-4.5, 4.5, dim)
            agents.append(x)

        # State for each agent
        # x: current position
        # fx: current fitness
        # sigma: step size
        # stagnation: count of steps without improvement
        agent_states = []
        
        for x0 in agents:
            x0 = np.clip(x0, -5, 5)
            if check_deadline() or calls_made >= budget:
                break
            fx = call_func(x0)
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()
            
            agent_states.append({
                'x': x0.copy(),
                'fx': fx,
                'sigma': 1.0,
                'stagnation': 0
            })

        if not agent_states:
            return best_x, best_f

        # --- Phase 2: Parallel (1+1)-ES Loop ---
        # Run until budget or time limit
        # Each iteration: for each agent, take a step
        # If improved: move, increase sigma slightly, reset stagnation
        # If not improved: stay, decrease sigma, increment stagnation
        # If stagnation > threshold: restart agent from a perturbed best_x or random point
        
        max_stagnation = 30  # Restart if no improvement for 30 steps
        sigma_increase = 1.05
        sigma_decrease = 0.9
        sigma_min = 1e-10
        sigma_max = 10.0
        
        while calls_made < budget and not check_deadline():
            # Iterate over a random subset of agents to ensure diversity in updates
            # Or just iterate all. With 50 agents and 500 budget, we get ~10 generations.
            # To make it more effective, we can process agents in rounds.
            
            # Shuffle agent indices to avoid bias
            idxs = np.arange(len(agent_states))
            np.random.shuffle(idxs)
            
            for i in idxs:
                if calls_made >= budget or check_deadline():
                    break
                
                agent = agent_states[i]
                x = agent['x']
                fx = agent['fx']
                sigma = agent['sigma']
                
                # Generate candidate
                rng_step = np.random.default_rng(hash((i, calls_made)) & 0xFFFFFFFF)
                y = x + sigma * rng_step.standard_normal(dim)
                y = np.clip(y, -5, 5)
                
                fy = call_func(y)
                
                if fy < fx:
                    # Improvement
                    agent['x'] = y
                    agent['fx'] = fy
                    agent['sigma'] = min(sigma * sigma_increase, sigma_max)
                    agent['stagnation'] = 0
                    
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                else:
                    # No improvement
                    agent['sigma'] = max(sigma * sigma_decrease, sigma_min)
                    agent['stagnation'] += 1
                    
                    # Check for restart
                    if agent['stagnation'] > max_stagnation:
                        # Restart logic:
                        # 50% chance to restart near current best
                        # 50% chance to restart randomly
                        if np.random.rand() < 0.5 and best_x is not None:
                            x_new = best_x + 0.5 * rng_step.standard_normal(dim)
                        else:
                            x_new = rng_step.uniform(-4.5, 4.5, dim)
                        
                        x_new = np.clip(x_new, -5, 5)
                        fx_new = call_func(x_new)
                        
                        agent['x'] = x_new
                        agent['fx'] = fx_new
                        agent['sigma'] = 1.0
                        agent['stagnation'] = 0
                        
                        if fx_new < best_f:
                            best_f = fx_new
                            best_x = x_new.copy()

        # --- Phase 3: Final Polish ---
        # Use the best_x found. If budget remains, do a very fine-grained (1+1)-ES
        # or Nelder-Mead on the best point.
        if best_x is not None and calls_made < budget and not check_deadline():
            # Simple local refinement with very small steps
            x_refine = best_x.copy()
            f_refine = best_f
            sigma_refine = 0.1
            steps = min(50, budget - calls_made)
            
            rng_refine = np.random.default_rng(999)
            for _ in range(steps):
                if calls_made >= budget or check_deadline():
                    break
                y = x_refine + sigma_refine * rng_refine.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = call_func(y)
                if fy < f_refine:
                    x_refine = y
                    f_refine = fy
                    sigma_refine *= 1.1
                else:
                    sigma_refine *= 0.8
                    if sigma_refine < 1e-8:
                        break
            
            if f_refine < best_f:
                best_f = f_refine
                best_x = x_refine.copy()

        return best_x, best_f

# EVOLVE-BLOCK-END
