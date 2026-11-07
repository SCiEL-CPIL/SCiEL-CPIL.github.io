import time
import random
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 20
    trees = instance_trees(n, 611)
    
    # Pre-compute edges for each tree
    tree_edges = trees
    
    # Helper to generate a random injective mapping for a tree of size k
    # Returns a list of k distinct integers from 0..19
    def random_mapping(k):
        hosts = list(range(n))
        random.shuffle(hosts)
        return hosts[:k]
    
    # Calculate the set of edges used by a given mapping for a given tree
    def get_edges(tree_idx, mapping):
        edges = set()
        for u, v in tree_edges[tree_idx]:
            hu = mapping[u]
            hv = mapping[v]
            if hu > hv:
                hu, hv = hv, hu
            edges.add((hu, hv))
        return edges
    
    # Calculate total conflicts (number of edges used by >1 tree)
    # Returns the number of "extra" uses, i.e., sum(count(e) - 1 for e if count(e) > 1)
    # Or simply, total edges used - unique edges used.
    # If we want 0 conflicts, unique edges should be sum of tree edges = 190.
    # Total edges is always 190.
    # So conflicts = 190 - len(union_of_all_edges)
    def calculate_conflicts(mappings):
        all_edges = set()
        for i in range(n):
            all_edges.update(get_edges(i, mappings[i]))
        return 190 - len(all_edges)
    
    # Initial State: Greedy or Random?
    # Let's start with Random, then Anneal.
    # Or better, Greedy Init:
    # Sort trees by size descending?
    # Let's just use Random Init for simplicity, SA will fix it.
    
    mappings = [random_mapping(i+1) for i in range(n)]
    current_conflicts = calculate_conflicts(mappings)
    
    best_mappings = mappings[:]
    best_conflicts = current_conflicts
    
    if best_conflicts == 0:
        return mappings
    
    # Simulated Annealing
    T = 5.0
    cooling = 0.999
    rng = Random(12345)
    
    while T > 0.01 and time.time() < deadline:
        # Pick a random tree to modify
        idx = rng.randint(0, n - 1)
        k = idx + 1
        
        # Generate a new mapping
        # Option 1: Full random
        # Option 2: Local change (swap or move)
        # Full random is good for escaping local minima.
        new_mapping = random_mapping(k)
        
        # To speed up conflict calculation, we can update incrementally
        # But 20 trees is small. Recalculation is fine.
        
        old_mapping = mappings[idx]
        mappings[idx] = new_mapping
        
        new_conflicts = calculate_conflicts(mappings)
        delta = new_conflicts - current_conflicts
        
        if delta < 0 or rng.random() < pow(2.71828, -delta / T):
            current_conflicts = new_conflicts
            if current_conflicts < best_conflicts:
                best_conflicts = current_conflicts
                best_mappings = mappings[:]
                if best_conflicts == 0:
                    break
        else:
            mappings[idx] = old_mapping
            
        T *= cooling
        
    # Return the best mapping found
    # Convert to the required format: list of lists
    # Entry k-1 is a list of k distinct int host vertices
    result = []
    for i in range(n):
        result.append(best_mappings[i])
        
    return result
