import numpy as np


def heuristics(prizes, distance_matrix, max_len):
        """
        Computes a heuristic matrix for ACO using a Scaled Inverse Cost approach.
        This approach uses an inverse relationship between the incremental path cost
        and the heuristic value, weighted by the prize at the destination node.
        This is distinct from Gaussian decay, offering a different balance between
        exploration (via prize) and exploitation (via short distance).
        """
        N = len(prizes)
        d_to_start = distance_matrix[:, 0]
        
        # Incremental cost: distance i->j plus distance j->0
        # This represents the additional length if we go i->j and then must return to 0
        # (or just the cost of the step, but the constraint is total length <= max_len)
        # The constraint is: current_len + d[i,j] + d[j,0] <= max_len.
        # So the "cost" of moving to j is effectively d[i,j] + d[j,0] in terms of feasibility.
        
        effective_cost = distance_matrix + d_to_start[None, :]
        
        # Normalize prizes to [0, 1] range (they already are, but ensure max is 1 for stability)
        # If all prizes are 0, avoid division issues.
        max_prize = np.max(prizes)
        if max_prize == 0:
            normalized_prizes = np.ones_like(prizes)
        else:
            normalized_prizes = prizes / max_prize
            
        # Inverse cost heuristic
        # We add a small epsilon to the denominator to avoid division by zero
        # and to ensure that even very short distances don't create infinite values.
        # The scale 'scale' determines how quickly the heuristic drops off with distance.
        # A smaller scale makes the heuristic more sensitive to distance.
        scale = 0.2 
        epsilon = 1e-6
        
        # Heuristic = Prize * (1 / (Cost + Scale))
        # This creates a matrix where high prizes and low costs are favored.
        
        heuristic = normalized_prizes[None, :] / (effective_cost + scale)
        
        # Ensure finite and positive
        heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
        
        return heuristic
