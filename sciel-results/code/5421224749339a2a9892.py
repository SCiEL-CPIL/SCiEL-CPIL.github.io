import time
import random
import itertools
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # Precompute all possible triples on points 0..15
    all_triples = list(itertools.combinations(range(16), 3))
    
    # For each triple, precompute which other triples it conflicts with (shares >= 2 points)
    # Two triples conflict if they share at least 2 points
    triple_set = set(all_triples)
    conflict = {}
    for t in all_triples:
        conflict[t] = set()
    
    # Build conflict graph
    for i, t1 in enumerate(all_triples):
        for j in range(i + 1, len(all_triples)):
            t2 = all_triples[j]
            # Check if they share at least 2 points
            common = set(t1) & set(t2)
            if len(common) >= 2:
                conflict[t1].add(t2)
                conflict[t2].add(t1)

    # Convert to bitmask representation for speed
    # Each triple is a 16-bit mask
    triple_masks = {}
    for t in all_triples:
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks[t] = mask

    # For fast independence number calculation using bitmask
    # A system is a set of triples. The independence number is the largest set of points
    # that contains no triple entirely within it.
    
    def compute_alpha(triples_list):
        """Compute independence number of the hypergraph defined by triples_list."""
        # For each point, compute which triples contain it
        point_to_triples = [[] for _ in range(16)]
        for idx, t in enumerate(triples_list):
            for p in t:
                point_to_triples[p].append(idx)
        
        # Use bitmask for the set of triples that are "hit" by a given point set
        # A point set S is independent if no triple is entirely contained in S
        # i.e., for every triple t, t is NOT a subset of S
        # Equivalently, S is independent if for every triple t, there exists p in t with p not in S
        
        # Better approach: enumerate subsets by size from largest to smallest
        # For a subset S (as bitmask), check if any triple is contained in S
        # A triple t = (a,b,c) is contained in S iff (S has bits a, b, c)
        
        # Precompute for each triple its mask
        triple_masks_list = []
        for t in triples_list:
            m = 0
            for p in t:
                m |= (1 << p)
            triple_masks_list.append(m)
        
        n = len(triples_list)
        
        # Check subsets from size 16 down to 1
        # But we can be smarter: start from high sizes
        # For size k, check all C(16,k) subsets
        # A subset S is independent iff no triple mask is a subset of S
        # i.e., for all i, triple_masks_list[i] & S != triple_masks_list[i]
        # i.e., for all i, (triple_masks_list[i] & ~S) != 0
        
        full_mask = (1 << 16) - 1
        
        # Try sizes from 16 down to 1
        for size in range(16, 0, -1):
            # Generate all subsets of this size
            # Use itertools.combinations
            found = False
            for subset in itertools.combinations(range(16), size):
                S = 0
                for p in subset:
                    S |= (1 << p)
                # Check if any triple is contained in S
                is_independent = True
                for tm in triple_masks_list:
                    if (tm & S) == tm:  # triple is subset of S
                        is_independent = False
                        break
                if is_independent:
                    return size
        return 0

    # Local search: start with a random valid system, then try to improve
    # A valid system is a set of triples where any two share at most one point
    
    def is_valid_system(triples_list):
        """Check if any two triples share at most one point."""
        for i in range(len(triples_list)):
            for j in range(i + 1, len(triples_list)):
                common = set(triples_list[i]) & set(triples_list[j])
                if len(common) >= 2:
                    return False
        return True

    def random_valid_system(num_triples=None):
        """Generate a random valid system."""
        if num_triples is None:
            num_triples = random.randint(10, 25)
        
        triples = []
        used_pairs = set()  # Track pairs of points that are used together in a triple
        
        # Shuffle all triples
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        
        for t in shuffled:
            if len(triples) >= num_triples:
                break
            # Check if this triple conflicts with any existing triple
            conflict_flag = False
            for existing in triples:
                common = set(t) & set(existing)
                if len(common) >= 2:
                    conflict_flag = True
                    break
            if not conflict_flag:
                triples.append(t)
        
        return triples

    def local_search(initial_system, time_limit):
        """Perform local search to minimize independence number."""
        current = initial_system[:]
        current_alpha = compute_alpha(current)
        best = current[:]
        best_alpha = current_alpha
        
        while time.time() < time_limit:
            # Try moves: add a triple, remove a triple, swap a triple
            improved = False
            
            # Try adding a triple
            shuffled_add = all_triples[:]
            random.shuffle(shuffled_add)
            for new_t in shuffled_add:
                candidate = current[:] + [new_t]
                if is_valid_system(candidate):
                    alpha = compute_alpha(candidate)
                    if alpha < current_alpha:
                        current = candidate
                        current_alpha = alpha
                        if current_alpha < best_alpha:
                            best = current[:]
                            best_alpha = current_alpha
                        improved = True
                        break
                    elif alpha == current_alpha and random.random() < 0.1:
                        # Occasionally accept equal alpha to escape local optima
                        current = candidate
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try removing a triple
            if len(current) > 1:
                shuffled_remove = list(range(len(current)))
                random.shuffle(shuffled_remove)
                for idx in shuffled_remove:
                    candidate = current[:idx] + current[idx+1:]
                    alpha = compute_alpha(candidate)
                    if alpha < current_alpha:
                        current = candidate
                        current_alpha = alpha
                        if current_alpha < best_alpha:
                            best = current[:]
                            best_alpha = current_alpha
                        improved = True
                        break
                    elif alpha == current_alpha and random.random() < 0.05:
                        current = candidate
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try swapping: remove one, add another
            if len(current) > 0:
                shuffled_swap = list(range(len(current)))
                random.shuffle(shuffled_swap)
                for idx in shuffled_swap:
                    shuffled_add = all_triples[:]
                    random.shuffle(shuffled_add)
                    for new_t in shuffled_add:
                        if new_t == current[idx]:
                            continue
                        candidate = current[:idx] + [new_t] + current[idx+1:]
                        if is_valid_system(candidate):
                            alpha = compute_alpha(candidate)
                            if alpha < current_alpha:
                                current = candidate
                                current_alpha = alpha
                                if current_alpha < best_alpha:
                                    best = current[:]
                                    best_alpha = current_alpha
                                improved = True
                                break
                            elif alpha == current_alpha and random.random() < 0.05:
                                current = candidate
                                improved = True
                                break
                    if improved:
                        break
            
            if not improved:
                # No improvement found, restart with a new random system
                if best_alpha <= 3:
                    break
                current = random_valid_system()
                current_alpha = compute_alpha(current)
                if current_alpha < best_alpha:
                    best = current[:]
                    best_alpha = current_alpha
        
        return best, best_alpha

    # Run multiple random restarts
    best_system = None
    best_alpha = 17  # worst case
    
    while time.time() < deadline:
        # Generate a random initial system
        initial = random_valid_system()
        if not initial:
            continue
        
        # Run local search with a time slice
        time_slice = time.time() + min(30.0, deadline - time.time())
        if time_slice <= time.time():
            break
        
        system, alpha = local_search(initial, time_slice)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = system
        
        if best_alpha <= 3:
            break
    
    if best_system is None:
        # Fallback: return the seed program's result
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_system
