import time
import itertools
import random

N = 16
ALL_TRIPLES = list(itertools.combinations(range(N), 3))
TRIPLE_PAIRS = [set(itertools.combinations(t, 2)) for t in ALL_TRIPLES]
TRIPLE_MASKS = [sum(1 << x for x in t) for t in ALL_TRIPLES]
UNIVERSE_MASK = (1 << N) - 1

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_alpha = N + 1
    best_triples = []
    
    def calc_alpha_from_masks(masks):
        """Fast independence number calculation using bitmask marking."""
        if not masks:
            return N
        
        independent = [True] * (1 << N)
        
        for m in masks:
            y = (~m) & UNIVERSE_MASK
            z = y
            while True:
                independent[m | z] = False
                if z == 0:
                    break
                z = (z - 1) & y
        
        max_alpha = 0
        for i in range(1 << N):
            if independent[i]:
                c = bin(i).count("1")
                if c > max_alpha:
                    max_alpha = c
        return max_alpha
    
    def build_linear_graph_backtrack(timeout=5.0):
        """Backtracking CSP solver with forward checking to build dense linear 3-graph."""
        t_start = time.time()
        triples = []
        used_pairs = set()
        masks = []
        
        # Precompute candidate triples sorted by some heuristic
        candidates = list(range(len(ALL_TRIPLES)))
        
        def backtrack(remaining_candidates, depth=0):
            nonlocal triples, used_pairs, masks
            if time.time() - t_start > timeout:
                return False
            
            if not remaining_candidates:
                return True
            
            # Try each candidate
            for i in range(len(remaining_candidates)):
                idx = remaining_candidates[i]
                pairs = TRIPLE_PAIRS[idx]
                if pairs.isdisjoint(used_pairs):
                    # Add this triple
                    triples.append(ALL_TRIPLES[idx])
                    used_pairs |= pairs
                    masks.append(TRIPLE_MASKS[idx])
                    
                    # Recurse
                    new_remaining = remaining_candidates[:i] + remaining_candidates[i+1:]
                    if backtrack(new_remaining, depth + 1):
                        return True
                    
                    # Backtrack
                    triples.pop()
                    used_pairs -= pairs
                    masks.pop()
            
            return False
        
        backtrack(candidates)
        return triples, masks
    
    def local_search_reduce_alpha(triples, masks, max_iterations=5000):
        """Local search to reduce independence number via swap/drop/add moves."""
        t_start = time.time()
        current_alpha = calc_alpha_from_masks(masks)
        used_pairs = set()
        for t in triples:
            used_pairs |= TRIPLE_PAIRS[ALL_TRIPLES.index(t)]
        
        best_alpha = current_alpha
        best_triples = list(triples)
        best_masks = list(masks)
        best_used_pairs = set(used_pairs)
        
        for iteration in range(max_iterations):
            if time.time() - t_start > 10.0:
                break
            
            # Try random swap: remove one triple, add another
            if len(triples) > 0:
                remove_idx = random.randint(0, len(triples) - 1)
                removed_triple = triples[remove_idx]
                removed_pairs = TRIPLE_PAIRS[ALL_TRIPLES.index(removed_triple)]
                removed_mask = TRIPLE_MASKS[ALL_TRIPLES.index(removed_triple)]
                
                # Find a replacement that doesn't conflict
                possible_replacements = []
                for idx in range(len(ALL_TRIPLES)):
                    if idx == ALL_TRIPLES.index(removed_triple):
                        continue
                    pairs = TRIPLE_PAIRS[idx]
                    # Check if this triple can be added after removing the old one
                    new_used = (best_used_pairs - removed_pairs)
                    if pairs.isdisjoint(new_used):
                        possible_replacements.append(idx)
                
                if possible_replacements:
                    add_idx = random.choice(possible_replacements)
                    new_triples = best_triples[:remove_idx] + best_triples[remove_idx+1:] + [ALL_TRIPLES[add_idx]]
                    new_masks = best_masks[:remove_idx] + best_masks[remove_idx+1:] + [TRIPLE_MASKS[add_idx]]
                    new_alpha = calc_alpha_from_masks(new_masks)
                    
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        best_triples = new_triples
                        best_masks = new_masks
                        best_used_pairs = set()
                        for t in best_triples:
                            best_used_pairs |= TRIPLE_PAIRS[ALL_TRIPLES.index(t)]
            
            # Try adding a triple if possible
            if random.random() < 0.3:
                possible_adds = []
                for idx in range(len(ALL_TRIPLES)):
                    pairs = TRIPLE_PAIRS[idx]
                    if pairs.isdisjoint(best_used_pairs):
                        possible_adds.append(idx)
                
                if possible_adds:
                    add_idx = random.choice(possible_adds)
                    new_triples = best_triples + [ALL_TRIPLES[add_idx]]
                    new_masks = best_masks + [TRIPLE_MASKS[add_idx]]
                    new_alpha = calc_alpha_from_masks(new_masks)
                    
                    if new_alpha <= best_alpha:
                        best_alpha = new_alpha
                        best_triples = new_triples
                        best_masks = new_masks
                        best_used_pairs = set()
                        for t in best_triples:
                            best_used_pairs |= TRIPLE_PAIRS[ALL_TRIPLES.index(t)]
            
            # Try dropping a triple
            if random.random() < 0.2 and len(best_triples) > 3:
                drop_idx = random.randint(0, len(best_triples) - 1)
                new_triples = best_triples[:drop_idx] + best_triples[drop_idx+1:]
                new_masks = best_masks[:drop_idx] + best_masks[drop_idx+1:]
                new_alpha = calc_alpha_from_masks(new_masks)
                
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_triples = new_triples
                    best_masks = new_masks
                    best_used_pairs = set()
                    for t in best_triples:
                        best_used_pairs |= TRIPLE_PAIRS[ALL_TRIPLES.index(t)]
        
        return best_alpha, best_triples, best_masks
    
    # Strategy: Run multiple backtracking builds with different randomizations, then local search
    num_restarts = 0
    while time.time() < deadline:
        seed = num_restarts
        random.seed(seed)
        
        # Build initial linear graph via backtracking
        triples, masks = build_linear_graph_backtrack(timeout=3.0)
        
        if not triples:
            num_restarts += 1
            continue
        
        # Apply local search to reduce alpha
        alpha, improved_triples, improved_masks = local_search_reduce_alpha(triples, masks, max_iterations=2000)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = improved_triples
            if best_alpha <= 4:
                break
        
        num_restarts += 1
    
    return best_triples
