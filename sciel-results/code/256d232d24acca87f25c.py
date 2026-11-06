import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill
    
    n = 16
    # Generate all C(16,3) = 560 possible triples
    all_triples = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                all_triples.append((a, b, c))
    
    n_triples = len(all_triples)
    
    # Map each triple to its index
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute conflict graph: for each triple, which other triples conflict with it
    # Two triples conflict if they are part of the same K(2,2,2) configuration
    # A K(2,2,2) is formed by 3 disjoint pairs {x,x'}, {y,y'}, {z,z'}
    # The 8 transversals are all combinations of picking one from each pair
    
    # For each triple (x,y,z), find all K(2,2,2) configurations it belongs to
    # and add the other 7 transversals as conflicts
    
    conflict_graph = [[] for _ in range(n_triples)]
    
    # For each triple, enumerate all possible K(2,2,2) configs it could be part of
    for idx, (x, y, z) in enumerate(all_triples):
        # x, y, z are the three elements of this triple
        # For a K(2,2,2), we need three disjoint pairs
        # This triple picks one element from each pair
        # So we need to find x', y', z' such that {x,x'}, {y,y'}, {z,z'} are disjoint pairs
        
        used = {x, y, z}
        for x_prime in range(n):
            if x_prime in used:
                continue
            for y_prime in range(n):
                if y_prime in used or y_prime == x_prime:
                    continue
                for z_prime in range(n):
                    if z_prime in used or z_prime == x_prime or z_prime == y_prime:
                        continue
                    
                    # Check if all 8 transversals exist in all_triples
                    # The 8 transversals are:
                    # (x,y,z), (x,y,z'), (x,y',z), (x,y',z'), (x',y,z), (x',y,z'), (x',y',z), (x',y',z')
                    transversals = [
                        (x, y, z),
                        (x, y, z_prime),
                        (x, y_prime, z),
                        (x, y_prime, z_prime),
                        (x_prime, y, z),
                        (x_prime, y, z_prime),
                        (x_prime, y_prime, z),
                        (x_prime, y_prime, z_prime)
                    ]
                    
                    # Sort each tuple and check if they're all valid triples
                    valid_config = True
                    other_indices = []
                    for t in transversals:
                        t_sorted = tuple(sorted(t))
                        if t_sorted not in triple_to_idx:
                            valid_config = False
                            break
                        if t_sorted != (x, y, z):
                            other_indices.append(triple_to_idx[t_sorted])
                    
                    if valid_config:
                        # Add conflicts (avoid duplicates)
                        for oidx in other_indices:
                            if oidx not in conflict_graph[idx]:
                                conflict_graph[idx].append(oidx)
    
    # Now perform multiple random restarts with "Largest Degree First" greedy
    best_result = None
    best_size = 0
    
    # Number of restarts - we'll do as many as we can before deadline
    max_restarts = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        # Random ordering with bias toward high-degree vertices
        # Create a list of (random_priority, degree, index)
        candidates = []
        for i in range(n_triples):
            degree = len(conflict_graph[i])
            # Mix random and degree-based priority
            priority = random.random() * 0.5 + degree * 0.5
            candidates.append((priority, degree, i))
        
        # Sort by priority (higher first)
        candidates.sort(key=lambda x: (-x[0], -x[1]))
        
        # Greedy selection
        selected = set()
        removed = set()
        
        for _, _, idx in candidates:
            if idx in removed:
                continue
            
            # Check if this triple conflicts with any already selected triple
            can_add = True
            for conflict_idx in conflict_graph[idx]:
                if conflict_idx in selected:
                    can_add = False
                    break
            
            if can_add:
                selected.add(idx)
                # Mark all conflicts as removed
                for conflict_idx in conflict_graph[idx]:
                    removed.add(conflict_idx)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_result = [all_triples[i] for i in sorted(selected)]
    
    if best_result is None:
        # Fallback: just return a small valid set
        return [[0, 1, 2], [0, 1, 3], [0, 1, 4]]
    
    return [list(t) for t in best_result]

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
