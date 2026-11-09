"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import itertools
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n = 16
    all_points = list(range(n))
    
    # Precompute all possible triples
    all_triples = list(itertools.combinations(range(n), 3))
    
    # We want to find a set of triples where any two share at most one point
    # and minimize the independence number (largest subset with no triple)
    
    # Strategy: Use a greedy + local search approach
    # 1. Start with a random valid system
    # 2. Try to improve by swapping/replacing triples
    
    def is_valid_system(triples):
        """Check if any two triples share at most one point."""
        for i in range(len(triples)):
            for j in range(i + 1, len(triples)):
                shared = set(triples[i]) & set(triples[j])
                if len(shared) > 1:
                    return False
        return True
    
    def independence_number(triples):
        """Compute the independence number (largest subset with no triple)."""
        triple_set = set(frozenset(t) for t in triples)
        
        # Check all subsets from largest to smallest
        # For n=16, this is expensive but we can use a smarter approach
        # Use the fact that if a set S contains a triple, it's not independent
        
        # Binary search on size? No, just check from n down
        # Actually, for small n=16, we can be smarter
        
        # A set is independent if it contains no triple from our system
        # We want the largest such set
        
        # Use a simple approach: try all subsets of size k from n down
        # But 2^16 = 65536, and for each we check if any triple is contained
        
        # Better: use the complement - find smallest hitting set
        # Or just check subsets directly
        
        # For efficiency, we'll check subsets of size k from n down to 1
        # A subset S is independent iff no triple is fully contained in S
        
        for k in range(n, 0, -1):
            for subset in itertools.combinations(range(n), k):
                subset_set = set(subset)
                # Check if any triple is fully contained
                independent = True
                for t in triples:
                    if set(t).issubset(subset_set):
                        independent = False
                        break
                if independent:
                    return k
        return 0
    
    def faster_independence_number(triples):
        """Faster independence number using bit masks."""
        triple_masks = []
        for t in triples:
            mask = 0
            for p in t:
                mask |= (1 << p)
            triple_masks.append(mask)
        
        n = 16
        # Check all subsets from largest to smallest
        for k in range(n, 0, -1):
            for subset in itertools.combinations(range(n), k):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                # Check if any triple is fully contained in this subset
                independent = True
                for tm in triple_masks:
                    if (tm & mask) == tm:
                        independent = False
                        break
                if independent:
                    return k
        return 0
    
    def random_valid_system():
        """Generate a random valid system of triples."""
        triples = []
        used_pairs = set()
        
        # Shuffle triples and greedily add
        candidate_triples = all_triples[:]
        random.shuffle(candidate_triples)
        
        for t in candidate_triples:
            # Check if this triple conflicts with existing ones
            ok = True
            t_set = set(t)
            for existing in triples:
                shared = t_set & set(existing)
                if len(shared) > 1:
                    ok = False
                    break
            if ok:
                triples.append(list(t))
            
            # Limit number of triples to keep search space manageable
            # With too many triples, independence number might be too small
            # But we need enough to cover the space
            if len(triples) >= 20:
                break
        
        return triples
    
    def local_search(initial_triples, max_time=30):
        """Improve the system via local search."""
        best_triples = list(initial_triples)
        best_score = faster_independence_number(best_triples)
        start_ls = time.time()
        
        while time.time() - start_ls < max_time and time.time() < deadline:
            improved = False
            current_score = best_score
            
            # Try replacing one triple with another
            for i in range(len(best_triples)):
                if time.time() >= deadline:
                    return best_triples, best_score
                
                # Try removing triple i and adding a new one
                remaining = [t for j, t in enumerate(best_triples) if j != i]
                
                # Find a good replacement
                candidate_triples = all_triples[:]
                random.shuffle(candidate_triples)
                
                for new_t in candidate_triples:
                    new_t_set = set(new_t)
                    # Check compatibility with remaining
                    ok = True
                    for existing in remaining:
                        if len(new_t_set & set(existing)) > 1:
                            ok = False
                            break
                    if not ok:
                        continue
                    
                    # Check if new_t is already in remaining
                    if any(set(t) == new_t_set for t in remaining):
                        continue
                    
                    candidate_system = remaining + [list(new_t)]
                    score = faster_independence_number(candidate_system)
                    
                    if score < best_score:
                        best_triples = candidate_system
                        best_score = score
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                # Try adding a new triple
                candidate_triples = all_triples[:]
                random.shuffle(candidate_triples)
                
                for new_t in candidate_triples:
                    new_t_set = set(new_t)
                    ok = True
                    for existing in best_triples:
                        if len(new_t_set & set(existing)) > 1:
                            ok = False
                            break
                    if not ok:
                        continue
                    if any(set(t) == new_t_set for t in best_triples):
                        continue
                    
                    candidate_system = best_triples + [list(new_t)]
                    score = faster_independence_number(candidate_system)
                    
                    if score < best_score:
                        best_triples = candidate_system
                        best_score = score
                        improved = True
                        break
            
            if not improved:
                # Try removing a triple
                if len(best_triples) > 5:
                    for i in range(len(best_triples)):
                        candidate_system = [t for j, t in enumerate(best_triples) if j != i]
                        score = faster_independence_number(candidate_system)
                        if score < best_score:
                            best_triples = candidate_system
                            best_score = score
                            improved = True
                            break
            
            if not improved:
                # Restart with a new random system
                new_initial = random_valid_system()
                new_score = faster_independence_number(new_initial)
                if new_score < best_score:
                    best_triples = new_initial
                    best_score = new_score
                else:
                    break
        
        return best_triples, best_score
    
    # Main search loop
    best_triples = None
    best_score = n + 1  # Start with worst case
    
    # First, try a few random initializations
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        initial = random_valid_system()
        
        if not initial:
            continue
        
        triples, score = local_search(initial, max_time=20)
        
        if score < best_score:
            best_score = score
            best_triples = triples
        
        # If we've found a good score, we can stop early
        if best_score <= 4:
            break
        
        # Limit total restarts
        if num_restarts >= 10:
            break
    
    if best_triples is None:
        # Fallback: return a simple valid system
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_triples
# EVOLVE-BLOCK-END
