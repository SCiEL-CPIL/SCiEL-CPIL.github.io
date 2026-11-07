import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        
        dim = self.dim
        budget = self.budget
        
        # Initial global best
        self.best_x = np.random.uniform(-5, 5, dim)
        self.best_f = func(self.best_x)
        self.calls += 1
        
        if self.calls >= budget or time.time() > deadline:
            return self.best_x, self.best_f
        
        # Multi-start (1+1)-ES with momentum for handling multimodal Lunacek function
        # This approach uses random restarts to escape local minima while momentum
        # helps maintain directional progress within each run
        
        # Parameters for (1+1)-ES with momentum
        alpha = 0.8  # momentum coefficient
        sigma0 = 2.0  # initial step size
        stagnation_window = 20  # number of steps without improvement before considering restart
        max_stagnation = 40  # maximum consecutive steps before forced restart
        
        # State variables
        x = self.best_x.copy()
        best_x = self.best_x.copy()
        best_f = self.best_f
        sigma = sigma0
        momentum = np.zeros(dim)
        no_improve_count = 0
        total_steps = 0
        
        while self.calls < budget and time.time() < deadline:
            # Generate trial step
            z = np.random.randn(dim)
            step = sigma * (z + alpha * momentum)
            
            # Update momentum
            momentum = alpha * momentum + (1 - alpha) * z
            
            # Propose new point
            x_trial = x + step
            x_trial = np.clip(x_trial, -5, 5)
            
            # Evaluate
            f_trial = func(x_trial)
            self.calls += 1
            total_steps += 1
            
            # Check if improved
            if f_trial < best_f:
                best_f = f_trial
                best_x = x_trial.copy()
                no_improve_count = 0
                # Keep momentum for successful step
            else:
                no_improve_count += 1
                # Decay momentum slightly on failure
                momentum *= 0.95
            
            # Decide whether to accept the step
            if f_trial < best_f or (f_trial <= self.best_f and np.random.rand() < 0.5):
                x = x_trial
                # Adjust sigma: increase on success, decrease on failure
                if f_trial < self.best_f:
                    sigma *= 1.1
                else:
                    sigma *= 0.9
            else:
                # Reject: keep x, decrease sigma
                sigma *= 0.9
            
            # Clamp sigma
            sigma = np.clip(sigma, 1e-6, 5.0)
            
            # Restart if stagnant
            if no_improve_count >= stagnation_window:
                # Random restart to escape local minimum
                if no_improve_count >= max_stagnation or np.random.rand() < 0.3:
                    # Full random restart
                    x = np.random.uniform(-5, 5, dim)
                else:
                    # Local perturbation around best
                    x = best_x + np.random.uniform(-1.5, 1.5, dim)
                    x = np.clip(x, -5, 5)
                
                # Reset parameters
                sigma = sigma0
                momentum = np.zeros(dim)
                no_improve_count = 0
            
            # Safety check
            if time.time() > deadline - 5:
                break
        
        # Update best
        self.best_x = best_x
        self.best_f = best_f
        
        return self.best_x, self.best_f
