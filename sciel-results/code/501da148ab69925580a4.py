"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from collections import defaultdict

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    
    Strategy:
    1. Type 1 vectors (norm^2=2, two +/-1s): All 220 are mutually compatible.
       Score = 220.
       
    2. Type 3 vectors (norm^2=4, four +/-1s):
       Max norm = 2. Min dist^2 >= 4.
       Dist^2 = 8 - 2*dot. Need dot <= 2.
       Conflicts if dot > 2.
       Find large independent set in conflict graph.
       
    Optimizations:
    - Bitset-based greedy for speed
    - Local search (2-opt) to improve greedy solutions
    - Multiple restarts with time-aware budget
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    # --- Generate Type 1 Vectors (Baseline) ---
    type1_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                type1_vectors.append(v)
    type1_array = np.array(type1_vectors, dtype=np.int64)
    best_result = type1_array.copy()
    best_size = len(best_result)
    
    # --- Generate Type 3 Vectors ---
    type3_vectors = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            type3_vectors.append(v)
    type3_array = np.array(type3_vectors, dtype=np.int64)
    n3 = len(type3_array)
    
    if n3 == 0:
        return best_result
        
    # Compute dot products for Type 3
    dots3 = type3_array @ type3_array.T
    
    # Conflict if dot > 2
    conflict3 = (dots3 > 2)
    np.fill_diagonal(conflict3, False)
    
    # Build adjacency lists for local search
    adj_lists = [np.where(conflict3[i])[0].tolist() for i in range(n3)]
    degrees3 = np.array([len(adj) for adj in adj_lists])
    
    # Precompute neighbor sets as frozensets for fast membership
    neighbor_sets = [set(adj) for adj in adj_lists]
    
    def greedy_independent_set_fast(conflict, degrees, n, rng, adj_lists, neighbor_sets):
        """Fast greedy using bitset operations."""
        # Use a bitset representation for remaining nodes
        # For n=5280, we can use Python integers as bitsets
        remaining_bits = (1 << n) - 1
        selected = []
        
        # Work with a list of remaining nodes for efficiency
        remaining_list = list(range(n))
        remaining_set = set(remaining_list)
        deg = degrees.copy()
        
        while remaining_set:
            # Find node with minimum degree among remaining
            min_deg = min(deg[i] for i in remaining_set)
            # Collect all nodes with min degree
            candidates = [i for i in remaining_set if deg[i] == min_deg]
            chosen = rng.choice(candidates)
            
            selected.append(chosen)
            
            # Remove chosen and its neighbors from remaining
            to_remove = neighbor_sets[chosen] | {chosen}
            to_remove_in_remaining = to_remove & remaining_set
            
            for node in to_remove_in_remaining:
                remaining_set.discard(node)
                # Update degrees of other remaining nodes
                for neighbor in neighbor_sets[node]:
                    if neighbor in remaining_set:
                        deg[neighbor] -= 1
        
        return selected

    def local_search_2opt(selected, conflict, n, adj_lists, neighbor_sets, rng, max_iterations=1000):
        """Improve independent set via 2-opt swaps."""
        selected_set = set(selected)
        current_size = len(selected)
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Try to add any node not in selected that doesn't conflict
            for i in range(n):
                if i not in selected_set:
                    # Check if i conflicts with any selected node
                    conflicts_with_selected = neighbor_sets[i] & selected_set
                    if not conflicts_with_selected:
                        selected_set.add(i)
                        current_size += 1
                        improved = True
            
            # Try 2-opt: remove one selected, add two non-selected
            if not improved:
                # For each selected node, try removing it and adding up to 2 non-selected
                for remove_candidate in list(selected_set):
                    # Nodes that would be freed by removing remove_candidate
                    freed_nodes = [i for i in range(n) 
                                   if i not in selected_set 
                                   and remove_candidate in neighbor_sets[i]
                                   and not (neighbor_sets[i] & (selected_set - {remove_candidate}))]
                    
                    if len(freed_nodes) >= 1:
                        # Try adding all freed nodes
                        new_size = current_size - 1 + len(freed_nodes)
                        if new_size > current_size:
                            selected_set.discard(remove_candidate)
                            for node in freed_nodes:
                                selected_set.add(node)
                            current_size = new_size
                            improved = True
                            break
        
        return list(selected_set)

    best_t3_selected = []
    best_t3_size = 0
    
    # Run multiple restarts with local search
    attempts = 0
    consecutive_no_improvement = 0
    
    while time.time() < deadline:
        attempts += 1
        rng = np.random.RandomState(attempts * 7919 + 42)
        
        # Greedy pass
        selected = greedy_independent_set_fast(conflict3, degrees3, n3, rng, adj_lists, neighbor_sets)
        
        # Local search improvement
        selected = local_search_2opt(selected, conflict3, n3, adj_lists, neighbor_sets, rng, max_iterations=500)
        
        if len(selected) > best_t3_size:
            best_t3_size = len(selected)
            best_t3_selected = selected.copy()
            consecutive_no_improvement = 0
        else:
            consecutive_no_improvement += 1
        
        # Early stop if we've been stuck for a while and have a good solution
        if consecutive_no_improvement > 50 and best_t3_size > best_size:
            break
        
        # Check time more frequently in later stages
        if attempts % 10 == 0 and time.time() > deadline - 15:
            break

    if best_t3_size > best_size:
        best_result = type3_array[best_t3_selected]
        best_size = best_t3_size
        
    return best_result
# EVOLVE-BLOCK-END
