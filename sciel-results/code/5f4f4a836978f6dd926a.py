import random
import time
import itertools

def construct():
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    num_triples = len(all_triples)
    
    # Map triple to index
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute hyperedges (K(2,2,2) configurations)
    hyperedges = []
    triple_to_hyperedges = [[] for _ in range(num_triples)]
    
    # Iterate over all combinations of 6 points
    for p6 in itertools.combinations(points, 6):
        partitions = [
            [(0,1),(2,3),(4,5)],
            [(0,1),(2,4),(3,5)],
            [(0,1),(2,5),(3,4)],
            [(0,2),(1,3),(4,5)],
            [(0,2),(1,4),(3,5)],
            [(0,2),(1,5),(3,4)],
            [(0,3),(1,2),(4,5)],
            [(0,3),(1,4),(2,5)],
            [(0,3),(1,5),(2,4)],
            [(0,4),(1,2),(3,5)],
            [(0,4),(1,3),(2,5)],
            [(0,4),(1,5),(2,3)],
            [(0,5),(1,2),(3,4)],
            [(0,5),(1,3),(2,4)],
            [(0,5),(1,4),(2,3)]
        ]
        
        for part in partitions:
            pairs = []
            for r1, r2 in part:
                pairs.append((p6[r1], p6[r2]))
            
            h_triples = []
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        a = pairs[0][0] if i == 0 else pairs[0][1]
                        b = pairs[1][0] if j == 0 else pairs[1][1]
                        c = pairs[2][0] if k == 0 else pairs[2][1]
                        t = tuple(sorted((a, b, c)))
                        h_triples.append(t)
            
            h_indices = [triple_to_idx[t] for t in h_triples]
            hyperedges.append(h_indices)
            
            h_idx = len(hyperedges) - 1
            for t_idx in h_indices:
                triple_to_hyperedges[t_idx].append(h_idx)
                
    num_hyperedges = len(hyperedges)
    
    # Precompute for each hyperedge, the list of triples it contains
    hyperedge_triples = [set(h) for h in hyperedges]
    
    start_time = time.time()
    time_limit = 115
    best_size = 0
    best_set = set()
    
    def count_conflicts(t_idx, counts):
        """Count how many hyperedges would be violated if we add t_idx."""
        conflicts = 0
        for h_idx in triple_to_hyperedges[t_idx]:
            if counts[h_idx] == 7:
                conflicts += 1
        return conflicts
    
    def is_valid_add(t_idx, counts):
        return count_conflicts(t_idx, counts) == 0
    
    def greedy_construct(seed_val):
        """
        Priority-based greedy: at each step, pick the valid triple with 
        minimum conflict potential (number of hyperedges at capacity 6).
        """
        rng = random.Random(seed_val)
        current_set = set()
        counts = [0] * num_hyperedges
        
        # Build a list of all triples
        remaining = list(range(num_triples))
        rng.shuffle(remaining)
        
        while remaining:
            # Find all valid triples in remaining
            valid_triples = []
            for t_idx in remaining:
                if is_valid_add(t_idx, counts):
                    # Score: number of hyperedges this triple shares that are at count 6
                    # (adding would make them 7, blocking 1 more triple per such hyperedge)
                    score = 0
                    for h_idx in triple_to_hyperedges[t_idx]:
                        if counts[h_idx] == 6:
                            score += 1
                    valid_triples.append((score, rng.random(), t_idx))
            
            if not valid_triples:
                break
            
            # Pick the one with lowest score, tie-break randomly
            valid_triples.sort(key=lambda x: (x[0], x[1]))
            _, _, chosen = valid_triples[0]
            
            current_set.add(chosen)
            remaining.remove(chosen)
            
            for h_idx in triple_to_hyperedges[chosen]:
                counts[h_idx] += 1
        
        return current_set, counts
    
    def local_search_swap(current_set, counts, max_iterations=200):
        """
        Local search: try swapping one triple out and another in to improve size.
        """
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Try to find a beneficial swap
            # Remove a random triple, then try to add a better one
            if not current_set:
                break
                
            removed = random.choice(list(current_set))
            new_set = set(current_set)
            new_counts = list(counts)
            
            # Remove the triple
            new_set.discard(removed)
            for h_idx in triple_to_hyperedges[removed]:
                new_counts[h_idx] -= 1
            
            # Now try to add up to 2 triples to compensate
            added_count = 0
            for t_idx in range(num_triples):
                if t_idx in new_set:
                    continue
                if is_valid_add(t_idx, new_counts):
                    new_set.add(t_idx)
                    for h_idx in triple_to_hyperedges[t_idx]:
                        new_counts[h_idx] += 1
                    added_count += 1
                    if added_count >= 2:
                        break
            
            if len(new_set) > len(current_set):
                current_set = new_set
                counts = new_counts
                improved = True
        
        return current_set, counts
    
    def greedy_fill_only(current_set, counts):
        """Add as many valid triples as possible without removing any."""
        improved = True
        while improved:
            improved = False
            for t_idx in range(num_triples):
                if t_idx in current_set:
                    continue
                if is_valid_add(t_idx, counts):
                    current_set.add(t_idx)
                    for h_idx in triple_to_hyperedges[t_idx]:
                        counts[h_idx] += 1
                    improved = True
        return current_set, counts
    
    # Multi-start: try several different seeds and keep the best
    num_starts = 30
    seeds = list(range(num_starts))
    random.shuffle(seeds)
    
    for seed in seeds:
        if time.time() - start_time > time_limit - 5:
            break
            
        # Greedy construction
        current_set, current_counts = greedy_construct(seed)
        current_set, current_counts = greedy_fill_only(current_set, current_counts)
        
        # Local search with swaps
        current_set, current_counts = local_search_swap(current_set, current_counts)
        current_set, current_counts = greedy_fill_only(current_set, current_counts)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
        
        # If we have time, do more local search iterations on the best
        if len(current_set) == best_size and best_size > 0:
            # Try additional swap-based improvements
            for _ in range(50):
                if time.time() - start_time > time_limit - 5:
                    break
                if not current_set:
                    break
                removed = random.choice(list(current_set))
                new_set = set(current_set)
                new_counts = list(current_counts)
                new_set.discard(removed)
                for h_idx in triple_to_hyperedges[removed]:
                    new_counts[h_idx] -= 1
                
                added = 0
                for t_idx in range(num_triples):
                    if t_idx in new_set:
                        continue
                    if is_valid_add(t_idx, new_counts):
                        new_set.add(t_idx)
                        for h_idx in triple_to_hyperedges[t_idx]:
                            new_counts[h_idx] += 1
                        added += 1
                        if added >= 2:
                            break
                
                if len(new_set) > len(current_set):
                    current_set = new_set
                    current_counts = new_counts
                    current_set, current_counts = greedy_fill_only(current_set, current_counts)
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = set(current_set)
    
    return [list(all_triples[i]) for i in best_set]
