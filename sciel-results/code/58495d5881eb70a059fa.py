"""Linear triple systems on 16 points - minimize independence number."""
import itertools
import random
import time
import numpy as np

def construct():
    deadline = time.time() + 145  # Safety margin before 150s kill
    n = 16
    all_points = list(range(n))
    
    # Precompute all possible triples
    all_triples = list(itertools.combinations(range(n), 3))
    triple_set = set(all_triples)
    
    # For each pair, store which triples contain that pair
    pair_to_triples = {}
    for t in all_triples:
        a, b, c = t
        for pair in [(a, b), (a, c), (b, c)]:
            pair = (min(pair), max(pair))
            if pair not in pair_to_triples:
                pair_to_triples[pair] = []
            pair_to_triples[pair].append(t)
    
    def get_pairs(triple):
        a, b, c = triple
        return frozenset({(min(a,b), max(a,b)), (min(a,c), max(a,c)), (min(b,c), max(b,c))})
    
    def is_linear(triples):
        """Check if any two triples share at most one point."""
        pair_counts = {}
        for t in triples:
            for p in get_pairs(t):
                pair_counts[p] = pair_counts.get(p, 0) + 1
                if pair_counts[p] > 1:
                    return False
        return True
    
    def independence_number(triples):
        """Find the largest subset of points containing no triple from the system."""
        if not triples:
            return n
        
        triple_set_local = set(triples)
        
        # Try to find largest independent set
        # Check from size n down to 1
        best = 0
        for size in range(n, best, -1):
            found = False
            for subset in itertools.combinations(range(n), size):
                subset_set = set(subset)
                has_triple = False
                for t in triple_set_local:
                    if t[0] in subset_set and t[1] in subset_set and t[2] in subset_set:
                        has_triple = True
                        break
                if not has_triple:
                    found = True
                    break
            if found:
                best = size
                break
        return best
    
    def greedy_construct():
        """Build a linear triple system greedily, trying to minimize independence number."""
        # Start with a random ordering of triples
        random.shuffle(all_triples)
        triples = []
        used_pairs = set()
        
        for t in all_triples:
            pairs = get_pairs(t)
            if not any(p in used_pairs for p in pairs):
                triples.append(t)
                used_pairs.update(pairs)
        
        return triples
    
    def local_search(triples, max_iters=1000):
        """Improve the triple system via local search."""
        if not is_linear(triples):
            return triples
        
        current_alpha = independence_number(triples)
        triples_list = list(triples)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Try to swap a triple with another to reduce independence number
            improved = False
            for i in range(len(triples_list)):
                if time.time() > deadline - 5:
                    break
                
                old_triple = triples_list[i]
                old_pairs = get_pairs(old_triple)
                
                # Find a replacement triple
                for new_triple in all_triples:
                    if new_triple == old_triple:
                        continue
                    
                    new_pairs = get_pairs(new_triple)
                    
                    # Check if new_triple can replace old_triple
                    # The remaining triples (excluding i) must not conflict with new_triple
                    can_replace = True
                    for j in range(len(triples_list)):
                        if j == i:
                            continue
                        other_pairs = get_pairs(triples_list[j])
                        if any(p in new_pairs for p in other_pairs):
                            can_replace = False
                            break
                    
                    if not can_replace:
                        continue
                    
                    # Try the swap
                    triples_list[i] = new_triple
                    new_alpha = independence_number(triples_list)
                    
                    if new_alpha < current_alpha:
                        current_alpha = new_alpha
                        improved = True
                        break
                    else:
                        triples_list[i] = old_triple
                
                if improved:
                    break
            
            if not improved:
                break
        
        return triples_list, current_alpha
    
    # Strategy: use multiple random restarts with greedy + local search
    best_triples = None
    best_alpha = n
    
    # First, try a few greedy constructions
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        
        triples = greedy_construct()
        if not triples:
            continue
        
        triples_list, alpha = local_search(triples, max_iters=200)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples_list
    
    # Try more intensive local search on the best found so far
    if best_triples is not None and time.time() < deadline - 15:
        best_triples, best_alpha = local_search(best_triples, max_iters=5000)
    
    # Additional refinement: try to add more triples if possible
    if best_triples is not None and time.time() < deadline - 10:
        current_triples = list(best_triples)
        used_pairs = set()
        for t in current_triples:
            used_pairs.update(get_pairs(t))
        
        # Try to add more triples
        for t in all_triples:
            if time.time() > deadline - 5:
                break
            pairs = get_pairs(t)
            if not any(p in used_pairs for p in pairs):
                # Check if adding this triple increases independence number
                test_triples = current_triples + [t]
                new_alpha = independence_number(test_triples)
                if new_alpha <= best_alpha:
                    current_triples.append(t)
                    used_pairs.update(pairs)
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
        
        best_triples = current_triples
    
    if best_triples is None:
        # Fallback: return the initial seed
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_triples
