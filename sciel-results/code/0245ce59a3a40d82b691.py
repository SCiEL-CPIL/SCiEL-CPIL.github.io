"""
Strategy: WalkSAT-style Max-2-SAT for Maximum Cut.
Formulation:
- Variables: x_i in {0, 1}
- Clauses: For each edge (i, j), the clause (x_i != x_j) is satisfied if the edge is cut.
- Objective: Maximize number of satisfied clauses (cut edges).
- Search: WalkSAT.
  1. Pick a random unsatisfied clause (uncut edge).
  2. Flip one of its variables.
     - With probability p (walk probability), flip the variable that increases the number of satisfied clauses the most (greedy step).
     - With probability 1-p, flip the variable that minimizes the decrease in satisfied clauses (or maximizes increase, but usually just random among the two to escape local minima). 
     - Actually, standard WalkSAT for Max-SAT:
       - Let C be an unsatisfied clause.
       - Let v1, v2 be its literals.
       - If we flip v1, delta1 = change in total satisfied clauses.
       - If we flip v2, delta2 = change in total satisfied clauses.
       - With prob p, flip the one with larger delta (greedy).
       - With prob 1-p, flip the one with smaller delta (random walk).
  3. This allows escaping local optima where no single flip improves the cut.

We will use a multi-start strategy with WalkSAT, using spectral initialization for the first start and random for others.
"""

import numpy as np
import time

def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = labels[None, :] != labels[:, None]
    return int((adjacency & diff).sum() // 2)

def _spectral_init(adjacency):
    """Initialize using the sign of the Fiedler vector."""
    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1)
    L = np.diag(degrees) - adjacency.astype(float)
    
    try:
        vals, vecs = np.linalg.eigh(L)
        fiedler = vecs[:, 1]
        labels = (fiedler >= 0).astype(np.int8)
        return labels
    except Exception:
        return np.random.randint(0, 2, size=n, dtype=np.int8)

def _walksat_maxcut(adjacency, labels, max_iters, deadline, walk_prob=0.5):
    """
    WalkSAT for Maximum Cut.
    labels: 0/1 array of size n.
    """
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    best_cut = current_cut
    best_labels = labels.copy()
    
    # Precompute edge list for faster unsatisfied clause selection
    # adjacency is bool matrix. Get indices of True.
    edges = np.argwhere(adjacency)
    # Only keep i < j to avoid duplicates
    edges = edges[edges[:, 0] < edges[:, 1]]
    m = len(edges)
    
    if m == 0:
        return best_labels, best_cut

    # Precompute adjacency matrix as float for delta calculations?
    # Actually, for a single flip, we can compute delta efficiently.
    # Let's maintain a "satisfied" count or just recompute cut size?
    # Recomputing cut size is O(N^2). For N=100, it's 10k ops.
    # If we do 100k iterations, that's 1e9 ops. Might be slow in Python loop.
    # We need to update the cut size incrementally.
    
    # Let's maintain the cut size incrementally.
    # When we flip vertex v:
    # Delta = sum_{u in N(v)} (1 if x_u == x_v else -1) * A_{uv}
    # Wait.
    # If x_u == x_v (edge not cut), flipping v makes x_u != x_v (edge cut). Delta +1.
    # If x_u != x_v (edge cut), flipping v makes x_u == x_v (edge not cut). Delta -1.
    # So Delta = (# neighbors u with x_u == x_v) - (# neighbors u with x_u != x_v)
    # Let deg_same = number of neighbors with same label.
    # Let deg_diff = number of neighbors with different label.
    # Delta = deg_same - deg_diff = 2 * deg_same - degree(v).
    
    # We need to track deg_same for each vertex? Or compute on the fly?
    # Computing on the fly for a specific v:
    # neighbors = np.where(adjacency[v])[0]
    # same = (labels[neighbors] == labels[v]).sum()
    # diff = len(neighbors) - same
    # delta = same - diff
    
    # This is O(degree(v)). Sum of degrees is 2M.
    # In each iteration, we pick an unsatisfied edge (u, v).
    # We need to evaluate delta for flipping u and delta for flipping v.
    # This requires O(degree(u) + degree(v)) work.
    
    # To pick an unsatisfied edge, we need to know which edges are unsatisfied.
    # An edge (u, v) is unsatisfied if labels[u] == labels[v].
    # We can maintain a list of unsatisfied edges? Or just pick a random edge and check?
    # Picking a random edge and checking if it's unsatisfied is O(1).
    # If it is satisfied, we reject and pick another.
    # As the cut grows, the fraction of unsatisfied edges drops.
    # Initially ~50% are unsatisfied (random init).
    # Near optimal, very few are unsatisfied.
    # This rejection sampling might become inefficient.
    
    # Alternative: Maintain a set of unsatisfied edges.
    # When we flip v, the status of all edges incident to v flips.
    # So we can update the set.
    # But updating a set of edges in Python is slow.
    
    # Let's stick to the simple approach first:
    # 1. Pick random edge from edges list.
    # 2. Check if unsatisfied (labels[u] == labels[v]).
    # 3. If satisfied, retry (max 10 times, then pick random vertex).
    # 4. If unsatisfied, calculate delta for u and v.
    # 5. Flip based on walk_prob.
    # 6. Update cut size.
    
    start_time = time.time()
    
    # Precompute neighbors for each vertex for faster delta calculation
    neighbors = [np.where(adjacency[i])[0] for i in range(n)]
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
        
        # Find an unsatisfied edge
        # Try up to 20 random edges
        unsat_edge = None
        for _ in range(20):
            e_idx = np.random.randint(m)
            u, v = edges[e_idx]
            if labels[u] == labels[v]:
                unsat_edge = (u, v)
                break
        
        if unsat_edge is None:
            # All edges are satisfied? Or we failed to find one.
            # If all satisfied, we are at a perfect cut (or local opt with no uncut edges? No, uncut edge means same label).
            # If we can't find an unsatisfied edge, the cut is perfect or very close.
            # Let's just pick a random vertex to flip to escape potential local optima if we are stuck.
            # Or break if we are confident.
            # For now, pick a random vertex.
            u = np.random.randint(n)
            v = -1 # dummy
            # Calculate delta for flipping u
            neigh_u = neighbors[u]
            same_u = (labels[neigh_u] == labels[u]).sum()
            diff_u = len(neigh_u) - same_u
            delta_u = same_u - diff_u
            # No v to compare, so just flip if delta > 0 or randomly?
            # Let's just flip randomly to explore.
            if np.random.rand() < 0.5 or delta_u > 0:
                labels[u] = 1 - labels[u]
                current_cut += delta_u
        else:
            u, v = unsat_edge
            
            # Calculate delta for flipping u
            neigh_u = neighbors[u]
            same_u = (labels[neigh_u] == labels[u]).sum()
            diff_u = len(neigh_u) - same_u
            delta_u = same_u - diff_u
            
            # Calculate delta for flipping v
            neigh_v = neighbors[v]
            same_v = (labels[neigh_v] == labels[v]).sum()
            diff_v = len(neigh_v) - same_v
            delta_v = same_v - diff_v
            
            # WalkSAT logic
            # delta is the change in the number of satisfied clauses (cut edges).
            # We want to maximize this.
            # With prob walk_prob, pick the one with larger delta (greedy).
            # With prob 1-walk_prob, pick the one with smaller delta (random walk).
            
            if np.random.rand() < walk_prob:
                if delta_u >= delta_v:
                    flip_node = u
                    delta = delta_u
                else:
                    flip_node = v
                    delta = delta_v
            else:
                if delta_u <= delta_v:
                    flip_node = u
                    delta = delta_u
                else:
                    flip_node = v
                    delta = delta_v
            
            # Perform flip
            labels[flip_node] = 1 - labels[flip_node]
            current_cut += delta
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
                
    return best_labels, best_cut

def partition(adjacency):
    """
    Main partition function using WalkSAT.
    """
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_labels = None
    best_cut = -1
    
    # Phase 1: Spectral Init + WalkSAT
    labels = _spectral_init(adjacency)
    # Run WalkSAT for a significant portion of the time
    # Estimate: 100k iterations per second?
    # Let's run until deadline - 20s for the first run.
    labels, cut = _walksat_maxcut(adjacency, labels, 200000, deadline - 20.0, walk_prob=0.5)
    if cut > best_cut:
        best_cut = cut
        best_labels = labels.copy()
    
    # Phase 2: Random Restarts + WalkSAT
    remaining = deadline - time.time() - 5.0
    if remaining > 2.0:
        # Determine number of restarts based on time
        # Each restart takes some time. Let's assume ~1s per restart if we limit it.
        # Or just run a fixed number of shorter runs.
        num_restarts = 10
        time_per_restart = remaining / num_restarts
        
        for r in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
            
            # Random init
            labels = np.random.randint(0, 2, size=n, dtype=np.int8)
            
            # Run WalkSAT for this time slice
            sub_deadline = time.time() + time_per_restart - 0.1
            # Scale iterations roughly with time
            # If 1s allows ~50k iterations, then time_per_restart * 50000
            est_iters = int(time_per_restart * 50000)
            labels, cut = _walksat_maxcut(adjacency, labels, est_iters, sub_deadline, walk_prob=0.5)
            
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
    
    if best_labels is None:
        best_labels = _spectral_init(adjacency)
        
    return best_labels
