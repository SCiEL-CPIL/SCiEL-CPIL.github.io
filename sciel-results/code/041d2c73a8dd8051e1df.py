import numpy as np
import time
import itertools
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s limit
    
    # 1. Generate Type A vectors (one non-zero entry +/- 2)
    type_a_vectors = []
    for i in range(D):
        for sign in (2, -2):
            v = np.zeros(D, dtype=np.int64)
            v[i] = sign
            type_a_vectors.append(v)
    type_a_vectors = np.array(type_a_vectors)
    
    # 2. Generate Type B vectors (four non-zero entries +/- 1)
    type_b_vectors = []
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, sign in zip(support, signs):
                v[idx] = sign
            type_b_vectors.append(v)
    type_b_vectors = np.array(type_b_vectors)
    
    n_b = len(type_b_vectors)
    
    # 3. Compute incompatibility graph for Type B
    # Two Type B vectors are incompatible if |dot product| >= 3
    # (since both have norm 2, distance^2 = 8 - 2*dot, need distance >= 2 => dot <= 2)
    # Actually: distance^2 = ||v1-v2||^2 = ||v1||^2 + ||v2||^2 - 2*v1.v2 = 8 - 2*dot
    # Need distance >= 2 (norm), so distance^2 >= 4 => 8 - 2*dot >= 4 => dot <= 2
    # So incompatible if dot > 2, i.e., dot >= 3
    
    dot_products = type_b_vectors @ type_b_vectors.T
    adjacency = (dot_products >= 3)
    np.fill_diagonal(adjacency, False)
    
    # Also check Type A vs Type B compatibility
    # Type A has norm 2, Type B has norm 2
    # Distance^2 = 8 - 2*dot, need >= 4 => dot <= 2
    # Type A dot Type B: Type A has one entry ±2, Type B has four ±1 entries
    # If the non-zero entry of Type A is in the support of Type B, dot = ±2
    # Otherwise dot = 0
    # So always compatible! Good.
    
    # Type A vs Type A: same index, opposite sign: dot = -4, dist^2 = 8-2*(-4)=16>=4 OK
    # same index, same sign: dot = 4, dist^2 = 8-8=0 < 4 NOT OK
    # different indices: dot = 0, dist^2 = 8 >= 4 OK
    # So Type A vectors with same index and same sign are incompatible with each other
    # But we only have one of each sign per index, so all Type A are mutually compatible
    
    # 4. Use spectral method + local search to find large independent set in Type B graph
    
    best_selected_b = np.array([], dtype=int)
    best_size = 0
    
    def evaluate_solution(selected_indices):
        """Check if selected indices form an independent set and return size"""
        if len(selected_indices) == 0:
            return 0
        if len(selected_indices) == 1:
            return 1
        # Check all pairs
        for i in range(len(selected_indices)):
            for j in range(i+1, len(selected_indices)):
                if adjacency[selected_indices[i], selected_indices[j]]:
                    return -1  # invalid
        return len(selected_indices)
    
    def local_search(selected_set, max_iterations=1000):
        """Improve independent set using swap operations"""
        selected_set = set(selected_set)
        available = set(range(n_b)) - selected_set
        current_size = len(selected_set)
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            # Try adding one vertex
            for v in available:
                if not any(adjacency[v, u] for u in selected_set):
                    selected_set.add(v)
                    available.discard(v)
                    improved = True
                    break
            
            if not improved:
                # Try swap: remove one, add one or more
                for u in list(selected_set):
                    # If we remove u, which new vertices can we add?
                    neighbors_of_u = set(np.where(adjacency[u])[0])
                    potential_new = available - neighbors_of_u
                    # Check which of potential_new are compatible with selected_set - {u}
                    remaining = selected_set - {u}
                    compatible_new = []
                    for v in potential_new:
                        if not any(adjacency[v, w] for w in remaining):
                            compatible_new.append(v)
                    
                    if len(compatible_new) > 1:
                        # Remove u, add all compatible_new
                        selected_set.discard(u)
                        for v in compatible_new:
                            selected_set.add(v)
                            available.discard(v)
                        improved = True
                        break
                    elif len(compatible_new) == 1:
                        # Try single swap
                        v = compatible_new[0]
                        selected_set.discard(u)
                        selected_set.add(v)
                        available.discard(v)
                        improved = True
                        break
            
            if not improved:
                break
        
        return list(selected_set)
    
    # Spectral initialization
    def spectral_init(seed=42):
        """Use eigenvector of Laplacian to get initial solution"""
        np.random.seed(seed)
        
        # Build Laplacian
        degree = adjacency.sum(axis=1)
        L = np.diag(degree) - adjacency.astype(float)
        
        # Get leading eigenvector (smallest non-zero eigenvalue)
        try:
            # Use power iteration on (I - L/n) or similar
            # Or just use random + greedy
            # For simplicity, use random restarts with greedy
            return None
        except:
            return None
    
    # Try multiple random restarts with greedy + local search
    rng = np.random.RandomState(42)
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Random ordering
        order = rng.permutation(n_b)
        
        # Greedy: pick vertices in random order if compatible
        selected = []
        selected_set = set()
        
        for v in order:
            if not any(adjacency[v, u] for u in selected_set):
                selected.append(v)
                selected_set.add(v)
        
        # Local search to improve
        improved = local_search(selected_set, max_iterations=500)
        improved_size = len(improved)
        
        if improved_size > best_size:
            best_size = improved_size
            best_selected_b = np.array(improved, dtype=int)
    
    # Also try min-degree greedy as baseline
    degrees = adjacency.sum(axis=1)
    available = np.ones(n_b, dtype=bool)
    selected_min_deg = []
    
    while np.any(available):
        if time.time() > deadline:
            break
        avail_indices = np.where(available)[0]
        if len(avail_indices) == 0:
            break
        avail_degrees = degrees[avail_indices]
        min_deg_idx_in_avail = np.argmin(avail_degrees)
        chosen_global_idx = avail_indices[min_deg_idx_in_avail]
        selected_min_deg.append(chosen_global_idx)
        available[chosen_global_idx] = False
        neighbors = np.where(adjacency[chosen_global_idx])[0]
        available[neighbors] = False
    
    if len(selected_min_deg) > best_size:
        best_size = len(selected_min_deg)
        best_selected_b = np.array(selected_min_deg, dtype=int)
    
    # Convert selected indices to vectors
    selected_b_vectors = type_b_vectors[best_selected_b]
    
    # Combine with Type A
    all_vectors = np.vstack([type_a_vectors, selected_b_vectors])
    
    return all_vectors

