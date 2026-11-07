import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Compute a heuristic matrix for ant colony optimization of orienteering.
    
    Strategy: Model the heuristic as a dynamic "regret" metric where the attractiveness
    of moving to node j from i is inversely proportional to the maximum prize value of
    any other unvisited node k that is also reachable from i within the remaining length budget.
    This biases ants away from choices that lead to lower opportunity costs compared to immediate alternatives.
    
    Since the heuristic is static (computed once), we approximate the "dynamic" regret by
    considering the global set of nodes and their reachability constraints.
    """
    prizes = np.asarray(prizes, dtype=float)
    d = np.asarray(distance_matrix, dtype=float)
    n = len(prizes)
    eps = 1e-9
    
    # Pre-compute return costs: d[j, 0]
    return_cost = d[:, 0]
    
    # For each pair (i, j), the total cost to go from i to j and back to 0 is d[i,j] + d[j,0].
    # However, the constraint is: length_so_far + d[i,j] + d[j,0] <= max_len.
    # Since length_so_far varies, a static heuristic cannot perfectly capture this.
    # But we can approximate the "feasibility" or "costliness" of the move.
    
    # Let's implement the specific directive:
    # "attractiveness ... is inversely proportional to the maximum prize value of any other 
    #  unvisited node k that is also reachable from i within the remaining length budget"
    
    # Static Approximation:
    # 1. For each node j, compute its "regret score" based on how much prize we "lose" 
    #    by not picking the best alternative.
    # 2. The best alternative for any node j is the max prize among all other nodes.
    # 3. If we pick j, we give up max(prizes[k] for k != j).
    # 4. Regret_j = max(prizes[k] for k != j) - prizes[j].
    # 5. We want to MINIMIZE regret, so attractiveness should be inversely related to regret?
    #    Or, attractiveness is proportional to (prizes[j] / (max_other_prize[j] + eps)).
    
    # Let's compute max_other_prize for each j.
    max_prize = np.max(prizes)
    second_max_prize = np.partition(prizes, -2)[-2] if n > 1 else 0.0
    
    max_other_prizes = np.zeros(n)
    for j in range(n):
        if prizes[j] == max_prize:
            # If j is the max, the max other is the second max
            # (assuming unique max for simplicity, or handle ties)
            # More robust:
            masked = prizes.copy()
            masked[j] = 0.0
            max_other_prizes[j] = np.max(masked)
        else:
            max_other_prizes[j] = max_prize
            
    # Regret Ratio R_j = prizes[j] / (max_other_prizes[j] + eps)
    # High R_j means j is a good choice relative to alternatives.
    R = prizes / (max_other_prizes + eps)
    
    # Now, incorporate distance.
    # The directive says: "inversely proportional to the maximum prize value of any other 
    #  unvisited node k that is also reachable from i within the remaining length budget"
    
    # This implies that if there is a very high prize node k reachable from i, 
    # then moving to j (which is not k) is less attractive because we "could have" gone to k.
    # So, attractiveness of j from i should be penalized by the existence of high-prize 
    # reachable nodes.
    
    # Let's define a "penalty" for move i->j based on the max prize of nodes reachable from i.
    # Reachable from i within max_len: d[i,k] + d[k,0] <= max_len.
    # Since we don't know length_so_far, we assume the worst case (length_so_far=0) for reachability?
    # Or the best case? 
    # If we assume length_so_far=0, then any node k with d[i,k]+d[k,0] <= max_len is reachable.
    # This is a static approximation of "reachable from i".
    
    # Let's compute for each i, the max prize of nodes k such that d[i,k] + d[k,0] <= max_len.
    # Let's call this MaxReachablePrize[i].
    
    # Note: Node 0 has prize 0.
    # For each i, mask nodes k where d[i,k] + d[k,0] > max_len.
    # Then take max prize of remaining nodes.
    
    MaxReachablePrize = np.zeros(n)
    for i in range(n):
        # Compute costs from i to all k and back to 0
        costs = d[i, :] + return_cost
        # Mask infeasible nodes
        feasible = costs <= max_len
        # Set infeasible prizes to 0
        masked_prizes = np.where(feasible, prizes, 0.0)
        MaxReachablePrize[i] = np.max(masked_prizes)
        
    # Now, the attractiveness of moving from i to j should be:
    # H[i, j] = (Prize[j] / (MaxReachablePrize[i] + eps)) * (1 / (d[i,j] + d[j,0] + eps))
    
    # Why?
    # 1. Prize[j] / MaxReachablePrize[i]: If there is a very high prize node reachable from i,
    #    and j is not that node, the ratio is low. If j IS the high prize node, the ratio is 1.
    #    This implements the "regret" idea: you are penalized for not picking the best reachable node.
    # 2. 1 / (d[i,j] + d[j,0]): Standard distance penalty. Shorter moves are better.
    
    # Let's refine:
    # The directive says: "inversely proportional to the maximum prize value of any other unvisited node k..."
    # So, H[i,j] ~ 1 / MaxOtherPrizeReachableFromI?
    # But that would make ALL moves less attractive if there is a high prize node.
    # That doesn't help distinguish between moving to the high prize node vs a low prize node.
    # The key is that the high prize node SHOULD be attractive.
    # So, H[i,j] should be HIGH for the high prize node.
    # Therefore, H[i,j] ~ Prize[j] / MaxReachablePrize[i] makes sense.
    
    # Let's try this combination.
    
    # Vectorized computation:
    # total_cost[i, j] = d[i, j] + d[j, 0]
    total_cost = d + return_cost[None, :]
    
    # Ratio[i, j] = prizes[j] / MaxReachablePrize[i]
    # Note: MaxReachablePrize[i] is the max prize reachable from i.
    # If i cannot reach any high prize node, MaxReachablePrize[i] is low, so ratio is high?
    # Wait, if MaxReachablePrize[i] is low, it means no high prize nodes are reachable.
    # Then any move is "good" relative to the alternatives (which are also low).
    # So ratio = Prize[j] / LowValue = High. This is correct.
    
    # If MaxReachablePrize[i] is high (e.g., 0.9), and Prize[j] is low (e.g., 0.1),
    # ratio = 0.1 / 0.9 = 0.11. Low attractiveness.
    # If Prize[j] is high (e.g., 0.9), ratio = 0.9 / 0.9 = 1.0. High attractiveness.
    
    # This seems to align with the directive.
    
    ratio = prizes[None, :] / (MaxReachablePrize[:, None] + eps)
    
    # Distance penalty
    dist_penalty = 1.0 / (total_cost + eps)
    
    # Combine
    heuristic = ratio * dist_penalty
    
    # Ensure positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
