import random
import time

def construct():
    d = 7
    n_vertices = 1 << d
    edges = []
    edge_map = {}
    
    # Generate all edges
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, v))
                edge_map[(u, v)] = len(edges) - 1
    
    n_edges = len(edges)
    
    # Precompute conflicts for each edge
    # conflict[e] is a list of 6 tuples of 3 edges
    # If we add edge e, we check if any of these triples are fully in selected
    conflicts = [[] for _ in range(n_edges)]
    
    for idx, (u, v) in enumerate(edges):
        # Find the bit flipped
        diff = u ^ v
        i = diff.bit_length() - 1
        
        for j in range(d):
            if j == i:
                continue
            # The other two vertices of the C4
            u2 = u ^ (1 << j)
            v2 = v ^ (1 << j)
            
            # The 3 other edges
            e1 = (min(u, u2), max(u, u2))
            e2 = (min(v, v2), max(v, v2))
            e3 = (min(u2, v2), max(u2, v2))
            
            idx1 = edge_map[e1]
            idx2 = edge_map[e2]
            idx3 = edge_map[e3]
            
            conflicts[idx].append((idx1, idx2, idx3))
    
    def is_valid_add(edge_idx, selected):
        # selected is a set of edge indices
        for t in conflicts[edge_idx]:
            if all(x in selected for x in t):
                return False
        return True
    
    def greedy_fill(selected, available_indices):
        # available_indices is a list of edge indices to try
        # We try to add them in the given order
        for idx in available_indices:
            if idx not in selected:
                if is_valid_add(idx, selected):
                    selected.add(idx)
        return selected

    start_time = time.time()
    time_limit = 110.0
    
    # Initial greedy
    random.seed(42) # Deterministic start for reproducibility in initial phase? 
    # Actually, random is better for search.
    all_indices = list(range(n_edges))
    random.shuffle(all_indices)
    current_set = set()
    greedy_fill(current_set, all_indices)
    
    best_set = set(current_set)
    best_size = len(current_set)
    
    while time.time() - start_time < time_limit:
        # Kick: remove 1 random edge
        if not current_set:
            break
            
        # Remove 1 edge
        remove_idx = random.choice(list(current_set))
        new_set = set(current_set)
        new_set.remove(remove_idx)
        
        # Greedy fill with remaining edges in random order
        # To be fast, we can just shuffle the list of all edges not in new_set
        # But creating a new list every time is O(N).
        # N=448, so it's fine.
        
        remaining = [i for i in all_indices if i not in new_set]
        random.shuffle(remaining)
        
        greedy_fill(new_set, remaining)
        
        new_size = len(new_set)
        
        if new_size > best_size:
            best_size = new_size
            best_set = new_set
            current_set = new_set
        elif new_size == len(current_set):
            # Random walk
            current_set = new_set
        else:
            # Accept worse solution with some probability (annealing)
            # Simple heuristic: accept if it's not much worse
            # Or just always accept to explore?
            # If we always accept, we might drift to smaller sets.
            # Let's accept with probability 0.1
            if random.random() < 0.1:
                current_set = new_set
        
    # Convert best_set to list of [u, v]
    result = []
    for idx in best_set:
        result.append(list(edges[idx]))
        
    return result
