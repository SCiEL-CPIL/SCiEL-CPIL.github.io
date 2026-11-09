"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 120.0
_SOFT_DEADLINE = 115.0

# State for the bandit heuristic
# We assume one instance is evaluated at a time.
# If the problem size changes, we reset the state.
_state = {
    'n': 0,
    'regret': None,
    'visited_count': 0,
    'instance_id': None
}

def _reset_state(n):
    _state['n'] = n
    _state['regret'] = np.zeros(n)
    _state['visited_count'] = 0
    _state['instance_id'] = None

def _get_instance_id(distance_matrix):
    # A simple hash to detect if we are starting a new instance
    # using the first few elements and shape
    if distance_matrix.shape[0] != _state['n']:
        return "new"
    # Check if the matrix is different (just in case, though unlikely to swap matrices of same size without resetting)
    # For speed, we assume the evaluator resets or uses distinct instances.
    # A robust ID would be hash of the matrix, but that's expensive.
    # Given the contract, we assume sequential instances.
    # We can use the sum of the first row as a cheap proxy for change.
    proxy = np.sum(distance_matrix[0, :10])
    if _state['instance_id'] is None:
        _state['instance_id'] = proxy
        return "init"
    if abs(_state['instance_id'] - proxy) > 1e-9:
        return "new"
    return "same"

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit using a regret-minimization bandit strategy.
    """
    # Deadline check
    if time.time() > _SOFT_DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return current_node
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
        
    n = distance_matrix.shape[0]
    
    # Detect new instance
    inst_status = _get_instance_id(distance_matrix)
    if inst_status in ["new", "init"] or _state['regret'] is None:
        _reset_state(n)
        _state['instance_id'] = np.sum(distance_matrix[0, :10])
        _state['regret'] = np.zeros(n)
        _state['visited_count'] = 0

    # Compute immediate costs
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    dist_to_destination = distance_matrix[destination_node, unvisited_nodes]
    
    # Immediate regret: cost of taking this node now
    # We want to minimize the edge (current->next) and eventually (next->...->dest)
    # A simple proxy for "future cost" is the distance from the candidate to the destination.
    # So immediate regret = dist(current, candidate) + dist(candidate, destination)
    immediate_regret = dist_from_current + dist_to_destination
    
    # Update the EWMA of regret for the chosen node? 
    # Actually, we update the regret for the node we *pick* after we pick it?
    # Or do we use the current regret to *inform* the pick?
    # Standard bandit: Pick based on regret (minimize estimated regret), then update regret for that arm.
    # Here, the "arm" is the next node.
    # We want to avoid nodes that have high historical regret.
    
    # The "regret" stored in _state['regret'] is the historical average of the immediate cost
    # incurred when that node was chosen (or considered?).
    # If we only update when chosen, the regret for unchosen nodes remains 0 (or initial).
    # This biases towards unvisited nodes.
    
    # Better: Update the regret for ALL unvisited nodes? No, that's expensive and not standard.
    # Standard UCB/Regret Minimization:
    # Score = ImmediateCost + HistoricalRegret[arm]
    # We pick the arm with the minimum Score.
    # Then we update HistoricalRegret[arm] with the observed ImmediateCost.
    
    # However, in TSP, the "cost" of a node is context-dependent.
    # The immediate cost here is `dist_from_current + dist_to_destination`.
    # This is a good proxy for the "quality" of the extension.
    
    # Combine immediate and historical
    # We need to map unvisited_nodes to their historical regret
    hist_regret = _state['regret'][unvisited_nodes]
    
    # We want to minimize the total score
    # To make it a "bandit", we should explore.
    # But for TSP, exploitation (greedy) is usually good.
    # Let's use a "Softmin" or "Softplus" approach?
    # Or simply: Score = Immediate + Alpha * Hist
    # If Hist is 0 for unvisited, it acts like greedy.
    # As we visit nodes, their Hist increases.
    # But we only visit each node ONCE. So Hist is only updated once per node.
    # This means the "learning" is minimal because each arm is pulled only once.
    
    # CRITICAL INSIGHT: In a constructive TSP heuristic, each node is visited exactly once.
    # Therefore, a standard bandit that learns from *past pulls of the same arm* is useless
    # because there are no past pulls!
    
    # We must redefine "learning".
    # Learning must be from the *structure* of the problem revealed by previous choices.
    # Or, we use a different model:
    # Instead of learning about the *node*, learn about the *strategy* or *context*.
    
    # Alternative Structural Change: **Lookahead / 2-opt style local repair within the heuristic**
    # Or **Savings Heuristic (Clarke-Wright)** adaptation?
    # The current code is NN with a destination bias.
    # A materially different family is **Savings-Based** or **Edge-Selection**.
    
    # Let's try a **Savings Heuristic** style selection.
    # In Savings, we start with individual tours and merge them.
    # But we are building a single tour sequentially.
    
    # How to adapt Savings to sequential construction?
    # At each step, we have a current path: Start -> ... -> Current.
    # We want to add a node Next.
    # The "savings" of adding Next between Current and... where?
    # We don't know the end yet.
    
    # Let's try **Nearest Neighbor with 2-Opt Lookahead**.
    # When picking the next node, we don't just look at `dist(current, next)`.
    # We look at `dist(current, next) + dist(next, best_neighbor_of_next)`.
    # This is a 2-step lookahead.
    
    # Let's implement a **2-Step Lookahead Greedy**.
    # For each candidate `u` in unvisited:
    #   cost(u) = dist(current, u)
    #   if there are other unvisited nodes v != u:
    #      best_next_v = argmin(dist(u, v))
    #      cost(u) += dist(u, best_next_v)
    # Pick u with min cost.
    
    # This is O(N^2) per step, so O(N^3) total. For N=50, 125,000 ops. Very fast.
    # This is a different design family: **Lookahead Greedy** vs **Phase-Aware Greedy**.
    
    # Let's refine the lookahead.
    # The immediate cost is `dist(current, u)`.
    # The future cost is approximated by `dist(u, argmin_{v in unvisited\{u}} dist(u, v))`.
    # This avoids picking a node that is a "dead end" (isolated from the rest).
    
    # Implementation:
    # 1. Get dist from current to all unvisited.
    # 2. For each unvisited node u, find the closest other unvisited node v.
    # 3. Score = dist(current, u) + dist(u, v).
    # 4. Pick min score.
    
    # Optimization:
    # We can vectorize the "closest other unvisited" part?
    # dist_matrix[unvisited, unvisited] gives a matrix of distances between all unvisited.
    # We need the min of each row, excluding the diagonal (self).
    
    if n_unvisited > 2:
        # Extract submatrix of distances between unvisited nodes
        sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        
        # Set diagonal to infinity to exclude self
        np.fill_diagonal(sub_dist, np.inf)
        
        # Find min distance for each node to any other unvisited node
        min_dist_to_other = np.min(sub_dist, axis=1)
        
        # Total score: edge from current + edge to next best neighbor
        scores = dist_from_current + min_dist_to_other
        
        # Tie-breaking: if scores are close, prefer the one closer to current?
        # Or just argmin.
        idx = np.argmin(scores)
        
    else:
        # Fallback to simple NN if lookahead is not applicable
        idx = np.argmin(dist_from_current)
        
    return int(unvisited_nodes[idx])

# EVOLVE-BLOCK-END
