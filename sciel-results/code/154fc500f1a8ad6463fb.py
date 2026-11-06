"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Assign a priority score to each vertex for a greedy Maximum Independent Set decoder.
    
    Strategy: Spectral initialization + Simulated Annealing on Permutations with insertion moves.
    1. Initialize the vertex order using a spectral heuristic (Fiedler vector of the Laplacian).
    2. Treat the ordering as a discrete state space.
    3. Use Simulated Annealing to swap or insert vertices in the ordering, accepting worse solutions
       with a decaying probability to escape local optima of the greedy decoder.
    4. Return the scores corresponding to the best ordering found (inverse rank).
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    adj = np.asarray(adjacency)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    adj_bool = adj.astype(bool)
    degree = adj_bool.sum(axis=1).astype(float)
    
    # Fast Greedy Simulator
    # Returns the size of the independent set produced by the greedy decoder 
    # given a specific ordering of vertices.
    def simulate_greedy(order):
        selected = 0
        available = np.ones(n, dtype=bool)
        
        for v in order:
            if not available[v]:
                continue
            selected += 1
            available[v] = False
            # Mark neighbors as unavailable
            available[adj_bool[v]] = False
            
            # Early exit if no vertices available
            if not available.any():
                break
                
        return selected

    # Initial Heuristic using Spectral Information
    def get_initial_order():
        # Compute Laplacian L = D - A
        D = np.diag(degree)
        L = D - adj_bool.astype(float)
        
        # Power iteration to find the Fiedler vector (eigenvector corresponding to second smallest eigenvalue)
        # We want to find the eigenvector with the second smallest eigenvalue.
        # Since L is positive semi-definite, the smallest eigenvalue is 0.
        # We can use power iteration on (L - mu*I) or just use a simple approach.
        # For n=200, we can just use np.linalg.eigh for accuracy and speed is acceptable for initialization.
        
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
            # Sort by eigenvalue
            idx = np.argsort(eigvals)
            # The first eigenvector is the constant vector (eigenvalue 0).
            # The second is the Fiedler vector.
            fiedler = eigvecs[:, idx[1]]
        except:
            fiedler = np.random.normal(0, 1, n)
        
        # Combine spectral info with degree
        # Lower degree and lower Fiedler value (or higher, depending on sign) is better.
        # We want to select vertices that are "peripheral" and low degree.
        # The Fiedler vector often separates the graph. 
        # Let's use a combination: degree + alpha * fiedler_value
        # We need to decide the sign. Let's normalize fiedler to have mean 0.
        fiedler = (fiedler - fiedler.mean()) / (fiedler.std() + 1e-8)
        
        # We want low degree. What about fiedler?
        # In a path graph, the ends have extreme fiedler values. 
        # Usually, we want to pick vertices that are less "central".
        # Let's try sorting by (degree + 0.5 * fiedler).
        # Actually, the sign of the fiedler vector is arbitrary.
        # Let's just use the absolute value or a random sign?
        # Better: use the raw value. If the graph is bipartite, one side is positive, one negative.
        # Independent set in bipartite graph is one of the partitions.
        # So picking all vertices with positive fiedler value (if it aligns with one partition) is good.
        # But we need an ordering.
        
        # Let's use a score that favors low degree and "extreme" fiedler values?
        # Or just low degree + noise?
        # The previous best used degree + 0.1 * neighbor_degree_sum.
        # Let's try: degree + 0.2 * fiedler
        
        sort_key = degree + 0.2 * fiedler + np.random.normal(0, 0.1, n)
        return np.argsort(sort_key, kind='stable')

    # Main Search Loop: Simulated Annealing on the Permutation
    
    # Initial state
    current_order = get_initial_order()
    current_size = simulate_greedy(current_order)
    
    best_order = current_order.copy()
    best_size = current_size
    
    # Annealing Parameters
    T = 1.0  # Initial temperature
    cooling = 0.9999  # Slower cooling to allow more exploration
    
    # Calibrate speed
    t_start = time.time()
    test_order = current_order.copy()
    simulate_greedy(test_order)
    t_end = time.time()
    avg_time_per_iter = (t_end - t_start) / 1.0
    
    max_iters = 500000  # High cap, will break on time
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
            
        # Generate Neighbor: Swap or Insert
        if np.random.random() < 0.5:
            # Swap
            idx1 = np.random.randint(n)
            idx2 = np.random.randint(n)
            if idx1 == idx2:
                continue
            new_order = current_order.copy()
            new_order[idx1], new_order[idx2] = new_order[idx2], new_order[idx1]
        else:
            # Insert: Remove a random vertex and insert it at a random position
            idx_remove = np.random.randint(n)
            v = current_order[idx_remove]
            new_order = np.delete(current_order, idx_remove)
            idx_insert = np.random.randint(len(new_order) + 1)
            new_order = np.insert(new_order, idx_insert, v)
            
        new_size = simulate_greedy(new_order)
        
        delta = new_size - current_size
        
        # Acceptance Criteria
        if delta > 0:
            # Always accept improvement
            current_order = new_order
            current_size = new_size
            
            if current_size > best_size:
                best_size = current_size
                best_order = current_order.copy()
        else:
            # Accept worse move with probability exp(delta / T)
            if np.random.random() < np.exp(delta / T):
                current_order = new_order
                current_size = new_size
        
        # Cool down
        T *= cooling
        
        # Restart mechanism
        if T < 0.01:
            # Probability to restart from best found or a new random heuristic
            if np.random.random() < 0.1:
                # Restart from best
                current_order = best_order.copy()
                current_size = best_size
                T = 0.5 # Reheat slightly
            elif np.random.random() < 0.1:
                # Restart from new heuristic
                current_order = get_initial_order()
                current_size = simulate_greedy(current_order)
                T = 0.5

    # Convert best_order to scores
    scores = np.zeros(n)
    # The greedy decoder sorts by descending score.
    # Vertex at position 0 in best_order should have the highest score.
    scores[best_order] = n - np.arange(n)
    
    return scores
# EVOLVE-BLOCK-END
