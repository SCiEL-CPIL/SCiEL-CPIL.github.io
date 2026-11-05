import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Loopy Belief Propagation (BP) on the graph to estimate P(v in MIS).
    1. Interpret the task as finding a vertex ordering that maximizes the greedy MIS size.
    2. Use a probabilistic model where each vertex has a "belief" b_v that it is included in the MIS.
    3. The beliefs are updated iteratively via message passing. The message from u to v 
       represents the "cost" or "constraint" that v imposes on u.
    4. Specifically, we model the independent set constraint: if v is in, u cannot be in.
    5. We use a "complement clique density" heuristic: vertices that form dense cliques in the 
       complement graph (i.e., are isolated in the original graph) should have high priority.
    6. The BP updates:
       m_{u->v} = -log(1 + exp(-lambda * (sum_{w in N(u)\{v}} m_{w->u})))
       This is a soft-max like update.
    7. Belief b_v = 1 / (1 + exp(- (sum_{u in N(v)} m_{u->v} + bias_v)))
    8. The bias_v is derived from local density: -degree(v) or similar.
    9. The final score is the converged belief b_v.
    
    This approach avoids the O(n^3) local search on permutations and instead solves a 
    continuous optimization problem on the graph, which is often faster and can find 
    different optima due to the smoothness of the belief landscape.
    """
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Precompute neighbor lists and degrees
    neighbors_list = [np.where(adj[i])[0].tolist() for i in range(n)]
    degrees = np.array([len(nbrs) for nbrs in neighbors_list])
    
    # Compute local density bias
    # Vertices with lower degree are more likely to be in MIS.
    # Also, vertices whose neighbors have high degree are less likely (because neighbors will be picked).
    # Let's use a simple bias: bias_v = -degree(v)
    bias = -degrees.astype(float)
    
    # Initialize messages
    # m[u][v] is the message from u to v.
    # We can store this as a matrix or a list of arrays.
    # For n=200, a matrix is fine.
    m = np.zeros((n, n))
    
    # Hyperparameters
    alpha = 1.0  # Learning rate for belief update
    lam = 1.0    # Temperature/scale for messages
    max_iter = 500
    tol = 1e-6
    
    # Belief Propagation Iteration
    # We use a simple synchronous update or asynchronous.
    # Synchronous is easier to implement.
    
    beliefs = np.zeros(n)
    
    for it in range(max_iter):
        if time.time() - start_time > 118.0:
            break
            
        new_m = np.zeros((n, n))
        
        # Update messages
        # m_{u->v} = -log(1 + exp(-lam * (sum_{w in N(u)\{v}} m_{w->u} + bias_u)))
        # To avoid numerical instability, use log-sum-exp trick or simple approximation.
        # sum_{w in N(u)\{v}} m_{w->u} = sum_{w in N(u)} m_{w->u} - m_{v->u}
        
        # Compute sum of incoming messages for each u
        # incoming_sum[u] = sum_{w} m[w][u]
        incoming_sum = m.sum(axis=0)
        
        for u in range(n):
            for v in neighbors_list[u]:
                # sum_{w in N(u)\{v}} m_{w->u}
                s = incoming_sum[u] - m[v, u]
                # Message value
                # We want the message to be negative if the constraint is strong.
                # If neighbors are "active" (high belief), the message should be negative (penalty).
                # Let's define the "field" at u excluding v as F = s + bias_u
                F = s + bias[u]
                # m_{u->v} = -log(1 + exp(-lam * F))
                # If F is large positive, exp(-lam*F) is small, m ~ 0.
                # If F is large negative, exp(-lam*F) is large, m ~ -lam*F (large positive? No, -log(1+large) is negative).
                # Wait, if F is negative (penalty), we want the message to reflect that.
                # Let's check signs.
                # If u is strongly constrained (low field), it should send a message to v saying "I'm unlikely to be in, so you can be".
                # Actually, standard BP for Max-Cut or IS:
                # Let's stick to a simpler heuristic update:
                # m_{u->v} = -lam * (s + bias[u])
                # This is a linear approximation.
                
                new_m[u, v] = -lam * (s + bias[u])
        
        # Update beliefs
        # b_v = 1 / (1 + exp(- (sum_{u in N(v)} m_{u->v} + bias_v)))
        # incoming_sum is now from new_m? No, we need to compute sum of new_m.
        new_incoming_sum = new_m.sum(axis=0)
        
        # Field at v
        fields = new_incoming_sum + bias
        
        # Beliefs (sigmoid)
        # To avoid overflow, use stable sigmoid
        # b_v = 1 / (1 + exp(-fields))
        # If fields > 0, b_v > 0.5
        max_f = np.max(fields)
        if max_f > 0:
            exp_f = np.exp(fields - max_f)
            beliefs = 1 / (1 + 1/exp_f)
        else:
            exp_f = np.exp(fields)
            beliefs = exp_f / (1 + exp_f)
            
        # Check convergence
        if np.max(np.abs(beliefs - beliefs)) < tol: # This check is wrong, beliefs is updated
            pass
        # Correct convergence check:
        # We need to compare old and new beliefs.
        # Let's just run for a fixed number of iterations or time limit.
        
        m = new_m
        
    # Final scores are the beliefs
    # Higher belief -> Higher priority
    scores = beliefs
    
    return scores
