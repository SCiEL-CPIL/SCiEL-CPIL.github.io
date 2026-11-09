"""Direct bootstrap for R(5,5): extend Paley(37) with a time-boxed heuristic search."""
import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START

DEADLINE = time.time() + 270  # safety margin: 330s hard, aim to finish by 270s


def _build_paley37():
    """Return adjacency bitset list for Paley graph on 37 vertices (vertices 0..36)."""
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj


def _find_k4_sets(adj, n):
    """Find all K4 cliques in graph with n vertices. Return list of 4-tuples."""
    k4_list = []
    for i in range(n):
        neigh_i = [j for j in range(n) if j > i and (adj[i] & (1 << j))]
        for j in range(n):
            if j <= i or not (adj[i] & (1 << j)):
                continue
            neigh_ij = [k for k in range(n) if k > j and (adj[i] & (1 << k)) and (adj[j] & (1 << k))]
            for k1 in range(len(neigh_ij)):
                a = neigh_ij[k1]
                for k2 in range(k1 + 1, len(neigh_ij)):
                    b = neigh_ij[k2]
                    if adj[a] & (1 << b):
                        k4_list.append((i, j, a, b))
    return k4_list


def _find_ind4_sets(adj, n):
    """Find all independent 4-sets in graph with n vertices. Return list of 4-tuples."""
    ind4_list = []
    for i in range(n):
        non_neigh_i = [j for j in range(n) if j > i and not (adj[i] & (1 << j))]
        for j in non_neigh_i:
            non_neigh_ij = [k for k in range(n) if k > j and not (adj[i] & (1 << k)) and not (adj[j] & (1 << k))]
            for k1 in range(len(non_neigh_ij)):
                a = non_neigh_ij[k1]
                for k2 in range(k1 + 1, len(non_neigh_ij)):
                    b = non_neigh_ij[k2]
                    if not (adj[a] & (1 << b)):
                        ind4_list.append((i, j, a, b))
    return ind4_list


def _try_add_vertex_sa(adj, n, k4_list, ind4_list, time_limit=10.0):
    """
    Try to add vertex n (new index = n) by choosing edges to 0..n-1.
    Uses Optimized Simulated Annealing.
    Returns new_adj or None if failed.
    """
    new_n = n + 1
    
    # Precompute constraints per vertex for fast delta cost calculation
    # k4_by_vertex[v] = list of indices of K4s containing v
    # ind4_by_vertex[v] = list of indices of ind4s containing v
    
    k4_by_vertex = [[] for _ in range(n)]
    for idx, k4 in enumerate(k4_list):
        for v in k4:
            k4_by_vertex[v].append(idx)
            
    ind4_by_vertex = [[] for _ in range(n)]
    for idx, ind4 in enumerate(ind4_list):
        for v in ind4:
            ind4_by_vertex[v].append(idx)
            
    # Precompute masks for K4s and ind4s
    k4_masks = []
    for k4 in k4_list:
        mask = 0
        for v in k4:
            mask |= (1 << v)
        k4_masks.append(mask)
        
    ind4_masks = []
    for ind4 in ind4_list:
        mask = 0
        for v in ind4:
            mask |= (1 << v)
        ind4_masks.append(mask)

    def get_initial_cost(state_mask):
        cost = 0
        for mask in k4_masks:
            if (state_mask & mask) == mask:
                cost += 1
        for mask in ind4_masks:
            if (state_mask & mask) == 0:
                cost += 1
        return cost

    def calc_delta_cost(state_mask, v):
        """
        Calculate change in cost if we flip bit v.
        state_mask is the state BEFORE flipping.
        """
        delta = 0
        # Check K4s containing v
        # If v is 0 in state_mask, it becomes 1.
        # K4 violation if all 4 bits are 1.
        # Before: v=0, so K4 was not fully 1 (unless... wait, if v=0, it can't be fully 1).
        # After: v=1. K4 becomes fully 1 if other 3 bits are 1.
        # If v is 1 in state_mask, it becomes 0.
        # Before: v=1. K4 was fully 1 if other 3 bits are 1.
        # After: v=0. K4 is not fully 1.
        
        if (state_mask & (1 << v)) == 0:
            # v goes 0 -> 1
            # K4s: gain violation if other 3 are 1
            for idx in k4_by_vertex[v]:
                mask = k4_masks[idx]
                others = mask & ~(1 << v)
                if (state_mask & others) == others:
                    delta += 1
            # ind4s: lose violation if other 3 are 0 (because before v=0, if others 0, it was a violation; after v=1, it's not)
            for idx in ind4_by_vertex[v]:
                mask = ind4_masks[idx]
                others = mask & ~(1 << v)
                if (state_mask & others) == 0:
                    delta -= 1
        else:
            # v goes 1 -> 0
            # K4s: lose violation if other 3 are 1
            for idx in k4_by_vertex[v]:
                mask = k4_masks[idx]
                others = mask & ~(1 << v)
                if (state_mask & others) == others:
                    delta -= 1
            # ind4s: gain violation if other 3 are 0
            for idx in ind4_by_vertex[v]:
                mask = ind4_masks[idx]
                others = mask & ~(1 << v)
                if (state_mask & others) == 0:
                    delta += 1
        return delta

    best_state = None
    best_cost = float('inf')
    
    start_time = time.time()
    max_restarts = 1000
    
    while time.time() - start_time < time_limit:
        # Generate initial state
        state = 0
        for i in range(n):
            if random.random() < 0.5:
                state |= (1 << i)
        
        current_cost = get_initial_cost(state)
        
        if current_cost == 0:
            best_state = state
            break
            
        if current_cost < best_cost:
            best_cost = current_cost
            best_state = state
            
        # Simulated Annealing
        temp = 1.0
        cooling = 0.995
        min_temp = 0.01
        
        while temp > min_temp:
            v = random.randint(0, n - 1)
            delta = calc_delta_cost(state, v)
            new_cost = current_cost + delta
            
            if new_cost < 0: # Should not happen, but safety
                new_cost = 0
                
            if delta < 0 or random.random() < math.exp(-delta / temp):
                state ^= (1 << v)
                current_cost = new_cost
                if current_cost == 0:
                    best_state = state
                    break
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_state = state
            temp *= cooling
            
        if best_cost == 0:
            break
            
    if best_cost == 0:
        new_adj = adj[:] + [0]
        for v in range(n):
            if best_state & (1 << v):
                new_adj[new_n] |= (1 << v)
                new_adj[v] |= (1 << new_n)
        return new_adj
        
    return None


def construct():
    adj = _build_paley37()
    n = 37
    
    best_n = 37
    best_adj = adj[:]
    
    # Precompute K4 and independent 4-sets for the base graph
    k4_list = _find_k4_sets(adj, n)
    ind4_list = _find_ind4_sets(adj, n)
    
    # Try to extend the graph by adding vertices one at a time
    for target_n in range(38, 43):
        if time.time() > DEADLINE:
            break
        
        # Recompute k4 and ind4 for current graph
        k4_list = _find_k4_sets(adj, n)
        ind4_list = _find_ind4_sets(adj, n)
        
        # Allocate time for this step
        time_remaining = DEADLINE - time.time()
        if time_remaining < 15:
            break
            
        # Try SA with a time limit
        # We can try a few different time limits if we have budget
        # First try 10s
        new_adj = _try_add_vertex_sa(adj, n, k4_list, ind4_list, time_limit=10.0)
        
        if new_adj is None and time.time() < DEADLINE - 20:
            # Retry with longer time
            new_adj = _try_add_vertex_sa(adj, n, k4_list, ind4_list, time_limit=20.0)
            
        if new_adj is not None:
            adj = new_adj
            n = target_n
            best_n = n
            best_adj = adj[:]
            print(f"Extended to n={n}")
        else:
            break
    
    # Build edge list from best_adj
    n_best = best_n
    edges = []
    for i in range(n_best):
        for j in range(i + 1, n_best):
            if best_adj[i] & (1 << j):
                edges.append((i, j))
    
    return n_best, edges

# EVOLVE-BLOCK-END
