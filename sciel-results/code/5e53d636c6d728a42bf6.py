"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _run_aco(prizes, distance_matrix, max_len, heuristic, n_ants=20, n_iters=30,
             rho=0.9, deposit_div=10.0, seed=0, deadline=None):
    """Run ACO with given heuristic and return best total prize."""
    n = len(prizes)
    rng = np.random.default_rng(seed)

    # Pheromone matrix, initialized to 1
    pheromone = np.ones((n, n))

    best_total = 0.0

    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break

        iter_best = 0.0

        for _ in range(n_ants):
            if deadline is not None and time.time() > deadline:
                break

            visited = [0]
            current = 0
            length = 0.0
            total_prize = prizes[0]

            while True:
                # Find candidate nodes
                candidates = []
                cand_indices = []
                for j in range(n):
                    if j in visited:
                        continue
                    new_len = length + distance_matrix[current, j] + distance_matrix[j, 0]
                    if new_len <= max_len:
                        candidates.append(pheromone[current, j] * heuristic[current, j])
                        cand_indices.append(j)

                if not candidates:
                    break

                candidates = np.array(candidates)
                total = candidates.sum()
                if total <= 0:
                    break
                probs = candidates / total
                j = rng.choice(cand_indices, p=probs)

                length = length + distance_matrix[current, j]
                total_prize += prizes[j]
                visited.append(j)
                current = j

            if total_prize > iter_best:
                iter_best = total_prize

        if iter_best > best_total:
            best_total = iter_best

        # Update pheromone
        pheromone = pheromone * rho + iter_best / deposit_div

    return best_total


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an instance-specific opportunity-cost heuristic matrix.
    
    The heuristic entry (i, j) represents the attractiveness of moving from i to j.
    We model this as: prize[j] * (remaining_budget_fraction_after_visit)^alpha
    where remaining_budget_fraction accounts for the distance cost of going i->j->0
    relative to the total budget.
    
    We use a static approximation: for each node j, compute a "value density" that
    balances prize against the round-trip distance cost from the origin (node 0),
    since node 0 is the start and end point.
    
    Specifically, for entry (i, j):
      - The cost of visiting j from i and returning to 0 is d[i,j] + d[j,0]
      - The "slack" is max_len - (current_length + d[i,j] + d[j,0]), but since we don't
        know current_length statically, we use a proxy: the budget consumed by the
        leg i->j->0 relative to max_len.
      - We define cost_fraction = (d[i,j] + d[j,0]) / max_len
      - The heuristic is prize[j] * (1 - cost_fraction)^beta, clipped to be positive.
    
    This penalizes nodes that consume too much of the budget relative to their prize,
    encouraging the ACO to prioritize high-value nodes that leave room for more visits.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)
    
    # Ensure prize at node 0 is 0 (as per task description)
    prizes = prizes.copy()
    prizes[0] = 0.0
    
    # Compute for each pair (i, j) the cost fraction
    # cost[i, j] = (d[i,j] + d[j,0]) / max_len
    # d[j,0] is the same for all i, so we can vectorize
    d_j_0 = distance_matrix[:, 0]  # shape (n,)
    d_i_j = distance_matrix  # shape (n, n)
    
    cost = (d_i_j + d_j_0[None, :]) / max_len  # shape (n, n)
    
    # Clamp cost to [0, 1] to avoid negative (1 - cost)
    cost = np.clip(cost, 0.0, 1.0)
    
    # Slack factor: (1 - cost)^beta, with beta=2 to strongly penalize high-cost moves
    beta = 2.0
    slack = (1.0 - cost) ** beta
    
    # Heuristic: prize[j] * slack[i, j]
    # prize[j] is broadcast across rows
    heuristic = prizes[None, :] * slack
    
    # Ensure all entries are positive (as required by task: non-positive become 1e-9)
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    # Ensure finite values
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
