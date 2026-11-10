"""Linear triple systems on 16 points with small independence number."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    best_triples = None
    best_alpha = 17  # infinity sentinel

    # Helper: check if adding triple keeps linear property
    def can_add(triples, new_triple):
        a, b, c = new_triple
        new_set = frozenset((a, b, c))
        for t in triples:
            inter = len(new_set & frozenset(t))
            if inter > 1:
                return False
        return True

    # Helper: compute independence number (max subset with no triple)
    def independence_number(triples):
        if not triples:
            return 16
        # Check all subsets of size k from high to low
        # A subset S is independent if no triple is fully contained in S
        triple_sets = [frozenset(t) for t in triples]
        points = list(range(16))
        
        # Try from size 16 down
        for size in range(16, 0, -1):
            for subset in combinations(points, size):
                subset_f = frozenset(subset)
                covered = False
                for ts in triple_sets:
                    if ts <= subset_f:
                        covered = True
                        break
                if not covered:
                    return size
        return 0

    # Helper: greedy construction with randomization
    def greedy_construct(random_seed, strategy):
        random.seed(random_seed)
        triples = []
        
        # Generate all possible triples
        all_triples = list(combinations(range(16), 3))
        random.shuffle(all_triples)
        
        for t in all_triples:
            if can_add(triples, t):
                triples.append(list(t))
                # Optional: stop early if we have enough triples
                # More triples generally helps reduce independence number
                if len(triples) >= 40:  # cap to avoid too dense
                    break
        
        return triples

    # Phase 1: Try structured constructions (Steiner-like)
    # Known: Steiner system S(2,3,16) doesn't exist, but we can try near-linear systems
    
    # Phase 2: Randomized greedy with many restarts
    attempt = 0
    while time.time() < deadline - 5.0:
        seed = attempt * 7919 + 42
        triples = greedy_construct(seed, "basic")
        
        if len(triples) >= 10:
            alpha = independence_number(triples)
            if alpha < best_alpha:
                best_alpha = alpha
                best_triples = triples
                if best_alpha <= 6:  # very good, stop early
                    break
        attempt += 1
        
        # Every 50 attempts, try a different strategy: prioritize covering
        if attempt % 50 == 0:
            triples2 = greedy_covering(seed)
            if triples2:
                alpha2 = independence_number(triples2)
                if alpha2 < best_alpha:
                    best_alpha = alpha2
                    best_triples = triples2
                    if best_alpha <= 6:
                        break

    # Phase 3: Local search / hill climbing on the best solution
    if best_triples is not None and time.time() < deadline - 10.0:
        best_triples = local_search(best_triples, best_alpha, deadline)
        best_alpha = independence_number(best_triples)

    if best_triples is None:
        # Fallback: return a valid linear system
        best_triples = [[i, i + 1, i + 2] for i in range(0, 16, 3)]
        # Ensure validity
        valid = []
        for t in best_triples:
            if can_add(valid, t):
                valid.append(t)
        best_triples = valid

    return best_triples


def greedy_covering(random_seed):
    """Greedy that prioritizes triples covering under-represented points."""
    import random
    random.seed(random_seed)
    
    triples = []
    point_count = [0] * 16
    
    all_triples = list(combinations(range(16), 3))
    random.shuffle(all_triples)
    
    for t in all_triples:
        if can_add_triples(triples, t):
            triples.append(list(t))
            for p in t:
                point_count[p] += 1
            if len(triples) >= 50:
                break
    
    return triples


def can_add_triples(triples, new_triple):
    a, b, c = new_triple
    new_set = frozenset((a, b, c))
    for t in triples:
        if len(new_set & frozenset(t)) > 1:
            return False
    return True


def local_search(triples, current_alpha, deadline):
    """Try to improve by swapping triples."""
    import time
    import random
    import copy
    
    best = copy.deepcopy(triples)
    best_alpha = current_alpha
    attempts = 0
    
    while time.time() < deadline - 5.0 and attempts < 5000:
        attempts += 1
        candidate = copy.deepcopy(best)
        
        # Randomly remove a triple and try to add a different one
        if candidate:
            idx = random.randint(0, len(candidate) - 1)
            removed = candidate.pop(idx)
            
            # Try to add a random valid triple
            all_triples = list(combinations(range(16), 3))
            random.shuffle(all_triples)
            
            added = False
            for t in all_triples:
                if can_add_triples(candidate, t):
                    candidate.append(list(t))
                    added = True
                    break
            
            if added:
                alpha = independence_number(candidate)
                if alpha < best_alpha:
                    best_alpha = alpha
                    best = candidate
                    if best_alpha <= 6:
                        break
                elif alpha == best_alpha and random.random() < 0.3:
                    # Accept with some probability for diversity
                    best = candidate
    
    return best
# EVOLVE-BLOCK-END
