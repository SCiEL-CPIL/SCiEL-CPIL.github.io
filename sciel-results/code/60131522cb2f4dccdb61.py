import numpy as np
import itertools
import random
import time

D = 11

def construct():
    # Use weight 4 vectors
    w = 4
    indices = list(range(D))
    
    # Generate all weight-w vectors
    vectors = []
    for support in itertools.combinations(indices, w):
        for signs in itertools.product([1, -1], repeat=w):
            v = [0] * D
            for i, s in zip(support, signs):
                v[i] = s
            vectors.append(v)
            
    N = len(vectors)
    V = np.array(vectors, dtype=np.int32)
    
    # Calculate dot products
    # 5280 x 5280 matrix
    Dots = V @ V.T
    
    limit = w // 2
    # Conflicts where dot > limit
    # We want to select a subset with no conflicts
    # Build adjacency list
    adj = [set() for _ in range(N)]
    
    # Efficiently find conflicts
    # Using np.where on the whole matrix might be memory heavy if done poorly
    # But 5280x5280 is ~28M booleans, which is ~28MB. Manageable.
    mask = Dots > limit
    np.fill_diagonal(mask, False)
    
    # Get indices of conflicts
    # Iterating row by row is safer for memory
    for i in range(N):
        # Get indices where mask[i] is True
        conflicts = np.nonzero(mask[i])[0]
        if len(conflicts) > 0:
            adj[i] = set(conflicts.tolist())
            
    # Greedy Maximum Independent Set
    best_set = []
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    # Try a few restarts with different random orderings
    # Since the graph is sparse, greedy is decent.
    # We can also try a simple local search if time permits, but greedy is fast.
    
    for restart in range(200):
        if time.time() - start_time > deadline:
            break
            
        # Order by degree, with random perturbation
        degrees = [len(adj[i]) for i in range(N)]
        # Perturb to break ties and explore different solutions
        keys = [degrees[i] + random.random() for i in range(N)]
        order = sorted(range(N), key=lambda i: keys[i])
        
        selected = []
        available = set(range(N))
        
        for i in order:
            if i not in available:
                continue
            # Select i
            selected.append(i)
            available.remove(i)
            # Remove neighbors
            for n in adj[i]:
                if n in available:
                    available.remove(n)
                    
        if len(selected) > len(best_set):
            best_set = selected
            
    # Return the best set found
    if not best_set:
        # Fallback to seed-like solution if something went wrong
        # But weight 4 should always find something
        return np.array(vectors[:10], dtype=np.int64)
        
    return np.array([vectors[i] for i in best_set], dtype=np.int64)
