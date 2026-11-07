import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0 # 5s safety margin
        
        dim = self.dim
        budget = self.budget
        
        # Multi-Start Strategy
        # We maintain a list of "agents". Each agent is a (1+lambda)-ES instance.
        # If an agent stagnates, we replace it with a new one.
        
        # Parameters
        num_agents = 5
        lam = 10 # Population size per agent
        stagnation_threshold = 20 # Generations without improvement before killing an agent
        
        # Global state
        best_x_global = None
        best_fx_global = np.inf
        calls = 0
        
        # Agent structure:
        # x_mean: current center
        # sigma: current step size
        # best_fx_local: best found by this agent
        # gen: current generation
        # stagnation_count: generations since last improvement for this agent
        # active: boolean
        
        agents = []
        
        # Initialize agents with diverse starting points
        # Use a grid-like or random spread to ensure coverage
        # For dim=5, random is fine, but let's ensure they are distinct
        for i in range(num_agents):
            x0 = np.random.uniform(-4, 4, dim) # Slightly inside the box
            # Evaluate initial point to set baseline
            if calls < budget:
                fx0 = func(x0)
                calls += 1
                if fx0 < best_fx_global:
                    best_fx_global = fx0
                    best_x_global = x0.copy()
                agents.append({
                    'x_mean': x0.copy(),
                    'sigma': 2.0, # Start with a reasonably large step
                    'best_fx_local': fx0,
                    'gen': 0,
                    'stagnation': 0,
                    'active': True
                })
            else:
                break
        
        if not agents:
            # Fallback if we ran out of budget immediately (unlikely)
            x0 = np.zeros(dim)
            fx0 = func(x0)
            calls += 1
            return x0, fx0

        # Main Loop
        # We process agents in a round-robin fashion.
        # Each "step" involves one agent generating lam offspring, evaluating them,
        # and updating its state.
        # To balance the budget, we cycle through active agents.
        
        agent_idx = 0
        
        while calls < budget and time.time() < deadline:
            # Find next active agent
            active_agents = [a for a in agents if a['active']]
            if not active_agents:
                break
            
            # Select agent round-robin
            # To avoid expensive list filtering every time, we can just pick the next one
            # and check if it's active. If not, skip.
            # Given small num_agents, filtering is fine.
            
            # We want to spread the budget. 
            # Let's just iterate through the original list of agents and skip inactive ones.
            # Actually, simpler: just pick the agent at agent_idx % num_agents.
            # If it's inactive, we need to replace it or skip.
            
            # Let's manage a queue of active indices.
            # Simpler approach: Just cycle 0..num_agents-1.
            # If agent is inactive, we revive it or create new?
            # Let's just create a new agent if it's dead and we have budget.
            
            idx = agent_idx % num_agents
            agent_idx += 1
            
            if not agents[idx]['active']:
                # Revive this agent slot with a new random start or near global best
                # Strategy: 50% chance random, 50% chance near global best
                if np.random.rand() < 0.5 and best_x_global is not None:
                    # Perturb global best
                    x0 = best_x_global + np.random.randn(dim) * 1.0
                else:
                    x0 = np.random.uniform(-4, 4, dim)
                
                # Clip to box
                x0 = np.clip(x0, -5, 5)
                
                # Evaluate new start
                if calls < budget:
                    fx0 = func(x0)
                    calls += 1
                    if fx0 < best_fx_global:
                        best_fx_global = fx0
                        best_x_global = x0.copy()
                    
                    agents[idx] = {
                        'x_mean': x0.copy(),
                        'sigma': 2.0,
                        'best_fx_local': fx0,
                        'gen': 0,
                        'stagnation': 0,
                        'active': True
                    }
                else:
                    break
            
            agent = agents[idx]
            
            # Generate population for this agent
            # (1+lambda)-ES
            Z = np.random.randn(lam, dim)
            Y = agent['x_mean'][None, :] + agent['sigma'] * Z
            Y = np.clip(Y, -5, 5)
            
            # Evaluate offspring
            F = np.full(lam, np.inf)
            for i in range(lam):
                if calls >= budget or time.time() > deadline:
                    break
                F[i] = func(Y[i])
                calls += 1
                if F[i] < best_fx_global:
                    best_fx_global = F[i]
                    best_x_global = Y[i].copy()
            
            if calls >= budget or time.time() > deadline:
                break
            
            # Update Agent
            # Find best offspring
            best_idx = np.argmin(F)
            best_fx_offspring = F[best_idx]
            best_x_offspring = Y[best_idx]
            
            if best_fx_offspring < agent['best_fx_local']:
                # Improvement
                agent['best_fx_local'] = best_fx_offspring
                agent['x_mean'] = best_x_offspring.copy()
                agent['stagnation'] = 0
                # Slightly increase sigma? Or keep? 
                # In (1+1)-ES, we usually increase if improved, decrease if not.
                # Here, if improved, we can try to expand slightly to explore more.
                agent['sigma'] *= 1.1
            else:
                # No improvement
                agent['stagnation'] += 1
                # Shrink sigma
                agent['sigma'] *= 0.95
                # If stagnation is high, kill the agent
                if agent['stagnation'] > stagnation_threshold:
                    agent['active'] = False
                    # It will be revived in the next cycle when we hit this index
            
            # Safety check on sigma
            agent['sigma'] = np.clip(agent['sigma'], 1e-6, 10.0)

        return best_x_global, best_fx_global
