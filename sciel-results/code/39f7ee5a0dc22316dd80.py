import numpy as np

def heuristics(demands, capacity):
        n = len(demands)
        demands = np.array(demands)
        
        # Compute all pairwise sums
        # sums[i, j] = demands[i] + demands[j]
        sums = demands[:, None] + demands[None, :]
        
        # Compute waste
        waste = capacity - sums
        
        # Initialize heuristic matrix
        eta = np.zeros((n, n))
        
        # Valid pairs are those where waste >= 0
        valid_mask = waste >= 0
        
        # For valid pairs, eta = 1 / (waste + 1)
        # We add 1 to avoid division by zero and to scale
        # If waste is 0, eta = 1. If waste is 149, eta = 1/150.
        
        # We only calculate where valid
        # To avoid division by zero or negative, we use np.where
        # But we want 0 for invalid.
        
        eta[valid_mask] = 1.0 / (waste[valid_mask] + 1.0)
        
        # The task says: "Heuristic entries that are not positive become 1e-9"
        # My eta is either 0 (invalid) or positive (valid).
        # 0 is not positive, so it will become 1e-9 by the evaluator.
        # This is correct.
        
        return eta
