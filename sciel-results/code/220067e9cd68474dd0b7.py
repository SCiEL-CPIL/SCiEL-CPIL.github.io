import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n_points = 16
    points = list(range(n_points))
    
    def get_alpha(triples):
        """
        Calculate the independence number (largest subset with no triple).
        Uses a branch and bound approach for speed on 16 points.
        """
        if not triples:
            return n_points
        
        # Precompute adjacency or just check membership
        # For 16 points, checking all subsets of size k is feasible if k is small,
        # but alpha might be large.
        # Better: Use a greedy/branching algorithm to find max independent set.
        
        # Convert triples to a set of frozensets for fast lookup
        triple_set = set(frozenset(t) for t in triples)
        
        # We want to find the max size S such that no t in triple_set is a subset of S.
        # This is equivalent to finding the max size of a set S that is "triple-free".
        
        # Since n=16 is small, we can iterate over subsets in decreasing size?
        # 2^16 = 65536 subsets. Checking each subset against ~20 triples is fast.
        # Let's just iterate all subsets from size 16 down to 1 and return the first valid one.
        # This is O(2^n * m) where m is num triples. 65536 * 20 is ~1.3M ops. Very fast.
        
        # Optimization: Iterate by size
        max_size = n_points
        for size in range(max_size, 0, -1):
            # Generate all subsets of given size
            found = False
            for subset in itertools.combinations(points, size):
                subset_set = frozenset(subset)
                # Check if any triple is contained in subset_set
                is_independent = True
                for t in triple_set:
                    if t.issubset(subset_set):
                        is_independent = False
                        break
                if is_independent:
                    return size
            # If no subset of this size is independent, continue to smaller sizes
        return 0

    def is_linear(triples):
        """Check if any two triples share more than 1 point."""
        for i in range(len(triples)):
            for j in range(i+1, len(triples)):
                if len(set(triples[i]).intersection(triples[j])) > 1:
                    return False
        return True

    def generate_initial_system():
        """Generate a random maximal linear triple system."""
        triples = []
        available_pairs = set(itertools.combinations(points, 2))
        random.shuffle(list(available_pairs))
        
        # We want to be dense to lower alpha.
        # Greedy: pick random pairs, form triples if possible.
        # Actually, a better start is a random collection of disjoint triples + filling.
        
        # Let's try a greedy approach that adds triples as long as linearity holds.
        # To maximize density, we should prefer triples that cover "uncovered" points or pairs.
        
        # Simple random greedy:
        all_pairs = list(itertools.combinations(points, 2))
        random.shuffle(all_pairs)
        
        # We need to ensure linearity.
        # State: count of pairs used? No, linearity is about triples sharing points.
        # Two triples {a,b,c} and {a,b,d} share {a,b} -> Invalid.
        # So each pair can appear in at most one triple.
        # This is a packing of pairs into triples.
        
        used_pairs = set()
        
        # Randomly try to form triples
        # Shuffle points to break symmetry
        shuffled_points = points[:]
        random.shuffle(shuffled_points)
        
        # We can't just pick random triples because we might get stuck with low density.
        # Let's use a randomized greedy:
        # 1. Start with empty list.
        # 2. While possible, pick a random unused pair (p1, p2).
        # 3. Try to find a third point p3 such that (p1, p3) and (p2, p3) are also unused.
        # 4. If found, add {p1, p2, p3} and mark pairs used.
        
        triples = []
        # To improve speed, keep a list of available pairs?
        # With 16 points, 120 pairs. 120^2 is small.
        
        # Let's just iterate through shuffled pairs and try to complete them.
        # This might not be maximal.
        
        # Better: Randomized backtracking or just simple greedy with restarts?
        # Given time constraints, simple greedy is okay for initialization.
        
        for p1, p2 in all_pairs:
            if (p1, p2) in used_pairs or (p2, p1) in used_pairs:
                continue
            # Find candidates for p3
            candidates = []
            for p3 in points:
                if p3 == p1 or p3 == p2:
                    continue
                # Check if pairs (p1, p3) and (p2, p3) are free
                if (p1, p3) not in used_pairs and (p3, p1) not in used_pairs and \
                   (p2, p3) not in used_pairs and (p3, p2) not in used_pairs:
                    candidates.append(p3)
            
            if candidates:
                p3 = random.choice(candidates)
                triples.append([p1, p2, p3])
                used_pairs.add((p1, p2))
                used_pairs.add((p1, p3))
                used_pairs.add((p2, p3))
                
                # Optional: Stop if we have enough? No, maximize density.
                
        return triples

    # Initialize with a few restarts to find a good starting point
    best_system = None
    best_alpha = n_points
    
    for _ in range(5):
        sys_init = generate_initial_system()
        if is_linear(sys_init):
            a = get_alpha(sys_init)
            if a < best_alpha:
                best_alpha = a
                best_system = sys_init
                
    if best_system is None:
        # Fallback
        return [[3*i, 3*i+1, 3*i+2] for i in range(5)]
        
    current_system = best_system
    current_alpha = best_alpha
    
    # Simulated Annealing
    T = 100.0
    T_min = 1.0
    cooling_rate = 0.999
    
    # Moves:
    # 1. Remove a random triple.
    # 2. Add a random valid triple (if space allows).
    # 3. Swap: Remove one, try to add two? Or just Add/Remove.
    # Let's stick to Add/Remove to maintain simplicity and linearity checks.
    
    # Precompute all possible triples? 16C3 = 560.
    all_possible_triples = list(itertools.combinations(points, 3))
    
    while time.time() < deadline and T > T_min:
        # Generate neighbor
        new_system = current_system[:]
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to add a triple
            # Pick a random triple from all_possible_triples
            # Check if it can be added (linearity with existing)
            t = random.choice(all_possible_triples)
            t_set = set(t)
            valid = True
            for existing in new_system:
                if len(t_set.intersection(existing)) > 1:
                    valid = False
                    break
            if valid:
                new_system.append(t)
            else:
                continue # Skip this step if invalid
        else:
            # Try to remove a triple
            if not new_system:
                continue
            idx = random.randint(0, len(new_system) - 1)
            new_system.pop(idx)
            
        # Evaluate new system
        new_alpha = get_alpha(new_system)
        
        # Acceptance criterion
        if new_alpha < current_alpha:
            current_system = new_system
            current_alpha = new_alpha
            if current_alpha == 4: # Theoretical lower bound for 16 points? 
                # Actually, Steiner Triple System on 13 points has alpha 5?
                # For 16, if we can get 4 or 5, that's great.
                # Let's stop if we hit a very low value.
                # But we want to minimize.
                pass
        else:
            # Delta E is (new_alpha - current_alpha)
            delta_e = new_alpha - current_alpha
            if delta_e > 0:
                prob = pow(2.71828, -delta_e / T)
                if random.random() < prob:
                    current_system = new_system
                    current_alpha = new_alpha
            # If delta_e == 0, accept (to allow exploration)
            else:
                current_system = new_system
                # current_alpha stays same
                
        T *= cooling_rate
        
    return current_system

# Ensure the function is defined
if __name__ == "__main__":
    # Simple test
    res = construct()
    print(f"Result: {len(res)} triples")
    # print(res)
