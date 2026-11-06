import time
import random
from itertools import combinations
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# Precompute popcounts for all 16-bit masks
N = 16
ALL_MASK = (1 << N) - 1
POP_COUNTS = [0] * (1 << N)
for i in range(1 << N):
    POP_COUNTS[i] = bin(i).count('1')

# Precompute all possible triples as bitmasks
ALL_TRIPLES = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            mask = (1 << a) | (1 << b) | (1 << c)
            ALL_TRIPLES.append(mask)

def is_linear(existing_edges, new_edge):
    """Check if new_edge shares at most 1 point with all existing_edges."""
    for e in existing_edges:
        if (e & new_edge) != 0:
            if POP_COUNTS[e & new_edge] > 1:
                return False
    return True

def calc_alpha(edges_masks):
    """Calculate independence number of the hypergraph defined by edges_masks."""
    if not edges_masks:
        return N
    
    bad = bytearray(1 << N)
    
    for e in edges_masks:
        comp = ALL_MASK ^ e
        sub = comp
        while True:
            bad[e | sub] = 1
            if sub == 0:
                break
            sub = (sub - 1) & comp
            
    max_alpha = 0
    for i in range(1 << N):
        if not bad[i]:
            pc = POP_COUNTS[i]
            if pc > max_alpha:
                max_alpha = pc
                if max_alpha == N:
                    break
    return max_alpha

def calc_alpha_ilp(edges_masks):
    """Calculate independence number using ILP (minimum hitting set).
    
    We want to find the minimum number of vertices to remove so that every
    triple is hit. This is the minimum vertex cover in the 3-uniform hypergraph.
    alpha = N - tau, where tau is the minimum hitting set size.
    """
    if not edges_masks:
        return N
    
    n_edges = len(edges_masks)
    
    # Build constraint matrix: each row is a triple, columns are vertices 0..15
    # Constraint: for each triple, sum of selected vertices >= 1
    A = np.zeros((n_edges, N), dtype=int)
    for i, mask in enumerate(edges_masks):
        for v in range(N):
            if mask & (1 << v):
                A[i, v] = 1
    
    # Objective: minimize sum of x_v (number of vertices in hitting set)
    c = np.ones(N, dtype=float)
    
    # Constraints: A @ x >= 1 for each edge
    # Convert to standard form: -A @ x <= -1
    lower = np.ones(n_edges)
    upper = np.full(n_edges, np.inf)
    
    constraints = LinearConstraint(-A, -1, 0)
    bounds = Bounds(0, 1)
    
    result = milp(c=c, constraints=constraints, bounds=bounds, integrality=np.ones(N))
    
    if result.status == 0:  # success
        tau = int(round(result.fun))
        return N - tau
    else:
        # Fallback to brute force
        return calc_alpha(edges_masks)

def random_greedy_packing():
    """Generate a random maximal linear packing."""
    edges = []
    available = ALL_TRIPLES[:]
    random.shuffle(available)
    
    for t in available:
        if is_linear(edges, t):
            edges.append(t)
            
    return edges

def local_search(edges, time_limit):
    """Improve the independence number via local search."""
    current_alpha = calc_alpha(edges)
    best_alpha = current_alpha
    best_edges = edges[:]
    
    while time.time() < time_limit:
        improved = False
        
        # First try adding edges (cheaper: just check linearity, alpha only decreases)
        for t in ALL_TRIPLES:
            if time.time() >= time_limit:
                break
            if is_linear(edges, t):
                new_edges = edges + [t]
                new_alpha = calc_alpha(new_edges)
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_edges = new_edges[:]
                    edges = new_edges[:]
                    current_alpha = new_alpha
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try swapping one edge
        for i in range(len(edges)):
            if time.time() >= time_limit:
                break
                
            e_to_remove = edges[i]
            remaining_edges = edges[:i] + edges[i+1:]
            
            for new_e in ALL_TRIPLES:
                if new_e == e_to_remove:
                    continue
                if is_linear(remaining_edges, new_e):
                    candidate_edges = remaining_edges + [new_e]
                    new_alpha = calc_alpha(candidate_edges)
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        best_edges = candidate_edges
                        edges = candidate_edges[:]
                        current_alpha = new_alpha
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            break  # Local optimum reached
        
    return best_edges, best_alpha

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_edges = None
    best_alpha = N + 1
    
    # Do several quick random restarts first
    num_quick_restarts = 5
    for _ in range(num_quick_restarts):
        if time.time() >= deadline:
            break
        candidate_edges = random_greedy_packing()
        candidate_alpha = calc_alpha(candidate_edges)
        if candidate_alpha < best_alpha:
            best_alpha = candidate_alpha
            best_edges = candidate_edges[:]
    
    # If we haven't found anything good, use a simple fallback
    if best_edges is None:
        seed_triples = [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
        best_edges = []
        for t in seed_triples:
            mask = 0
            for v in t:
                mask |= (1 << v)
            best_edges.append(mask)
        best_alpha = calc_alpha(best_edges)
    
    # Now do local search on the best found so far
    if time.time() < deadline - 10:
        improved_edges, improved_alpha = local_search(best_edges, deadline - 5)
        if improved_alpha < best_alpha:
            best_alpha = improved_alpha
            best_edges = improved_edges[:]
    
    # If we still have time, try more random restarts
    while time.time() < deadline - 15:
        candidate_edges = random_greedy_packing()
        candidate_alpha = calc_alpha(candidate_edges)
        
        if candidate_alpha < best_alpha:
            best_alpha = candidate_alpha
            best_edges = candidate_edges[:]
            
            # Local search on the new best
            if time.time() < deadline - 10:
                improved_edges, improved_alpha = local_search(candidate_edges, deadline - 5)
                if improved_alpha < best_alpha:
                    best_alpha = improved_alpha
                    best_edges = improved_edges[:]
        
        # Early termination if we've found a very good solution
        if best_alpha <= 3:
            break
    
    # Final polish: use ILP to verify and potentially improve alpha estimate
    if best_edges and time.time() < deadline - 2:
        ilp_alpha = calc_alpha_ilp(best_edges)
        if ilp_alpha < best_alpha:
            best_alpha = ilp_alpha
    
    # Convert best_masks back to list of triples [a, b, c]
    result = []
    for mask in best_edges:
        triple = []
        for i in range(N):
            if mask & (1 << i):
                triple.append(i)
        result.append(triple)
        
    return result
