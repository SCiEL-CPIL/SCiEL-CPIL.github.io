import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    N = 16
    all_points = list(range(N))
    
    def calc_independence(triples):
        # triples is a list of (a,b,c)
        # Precompute masks for triples
        triple_masks = []
        for t in triples:
            m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            triple_masks.append(m)
        
        max_alpha = 0
        # Iterate all subsets
        # Optimization: We can iterate and check if subset is independent
        # A subset S is independent if it does not contain any triple from triple_masks.
        # Instead of checking all triples for each subset, we can mark subsets that ARE independent.
        # But 2^16 is small.
        
        # A faster way: 
        # For each subset mask, check if it contains any triple.
        # We can precompute for each triple mask, which supersets are "covered".
        # Actually, just iterating 65536 subsets and checking against ~30 triples is 2M ops. Very fast.
        
        for mask in range(1 << N):
            # Check if mask contains any triple
            is_independent = True
            for tm in triple_masks:
                if (mask & tm) == tm:
                    is_independent = False
                    break
            if is_independent:
                size = bin(mask).count('1')
                if size > max_alpha:
                    max_alpha = size
                    # Early exit if max_alpha is 16? No, we want min.
                    # If we find a large one, we keep looking for larger to know the max.
                    # We want the MAX size of an independent set.
        return max_alpha

    def is_linear(triples, new_triple):
        a, b, c = new_triple
        for t in triples:
            inter = len(set(t) & set(new_triple))
            if inter > 1:
                return False
        return True

    def random_valid_triple(triples):
        # Try to find a random triple not conflicting with existing ones
        # For efficiency, we can just pick random points and check
        # But as triples grow, this becomes slower.
        # Simple approach:
        for _ in range(100):
            t = tuple(random.sample(all_points, 3))
            if is_linear(triples, t):
                return t
        return None

    # Initialize with a greedy construction
    current_triples = []
    # Add random triples until no more can be added easily
    while time.time() < start_time + 5.0:
        t = random_valid_triple(current_triples)
        if t is None:
            break
        current_triples.append(t)
        
    # Ensure we have a decent number of triples
    # If we have too few, restart or add more
    if len(current_triples) < 10:
        current_triples = []
        while time.time() < start_time + 5.0:
            t = random_valid_triple(current_triples)
            if t is None:
                break
            current_triples.append(t)

    best_triples = current_triples[:]
    best_alpha = calc_independence(best_triples)
    
    # Local Search: Simulated Annealing / Hill Climbing
    # Moves: Add a triple, Remove a triple, Swap a triple
    # Objective: Minimize alpha
    
    current_alpha = best_alpha
    current_set = set(current_triples)
    
    while time.time() < deadline:
        # Generate a move
        move_type = random.random()
        if move_type < 0.4:
            # Add a triple
            new_t = random_valid_triple(list(current_set))
            if new_t is not None:
                new_set = current_set | {new_t}
                # Check if this improves or maintains alpha
                # To save time, we might only calculate alpha if we think it's promising
                # But alpha calculation is fast.
                new_alpha = calc_independence(list(new_set))
                if new_alpha < current_alpha:
                    current_set = new_set
                    current_alpha = new_alpha
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_triples = list(current_set)
                elif new_alpha == current_alpha and random.random() < 0.1:
                    # Accept sideways move sometimes
                    current_set = new_set
        elif move_type < 0.7:
            # Remove a triple
            if current_set:
                t_to_remove = random.choice(list(current_set))
                new_set = current_set - {t_to_remove}
                new_alpha = calc_independence(list(new_set))
                if new_alpha < current_alpha:
                    current_set = new_set
                    current_alpha = new_alpha
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_triples = list(current_set)
                elif new_alpha == current_alpha and random.random() < 0.05:
                    current_set = new_set
        else:
            # Swap: Remove one, add one
            if current_set:
                t_to_remove = random.choice(list(current_set))
                temp_set = current_set - {t_to_remove}
                new_t = random_valid_triple(list(temp_set))
                if new_t is not None:
                    new_set = temp_set | {new_t}
                    new_alpha = calc_independence(list(new_set))
                    if new_alpha < current_alpha:
                        current_set = new_set
                        current_alpha = new_alpha
                        if current_alpha < best_alpha:
                            best_alpha = current_alpha
                            best_triples = list(current_set)
                    elif new_alpha == current_alpha and random.random() < 0.1:
                        current_set = new_set

    # Convert back to list of lists
    return [list(t) for t in best_triples]
