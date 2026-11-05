import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # Bitmask utilities
    # Points 0-15, use 16-bit integers
    
    def subset_size(mask):
        return bin(mask).count('1')
    
    def get_bits(mask):
        bits = []
        for i in range(16):
            if mask & (1 << i):
                bits.append(i)
        return bits
    
    def triple_mask(a, b, c):
        return (1 << a) | (1 << b) | (1 << c)
    
    # Precompute all pairs for linearity check
    # For a triple (a,b,c), the pairs are (a,b), (a,c), (b,c)
    # Encode pair as a 16-bit mask with two bits set
    
    def pair_mask(a, b):
        return (1 << a) | (1 << b)
    
    # Check if a set of triples is linear (any two share at most one point)
    def is_linear(triples):
        pair_counts = {}
        for t in triples:
            a, b, c = t
            pairs = [(a, b), (a, c), (b, c)]
            for p in pairs:
                key = (min(p), max(p))
                if key in pair_counts:
                    pair_counts[key] += 1
                    if pair_counts[key] > 1:
                        return False
                else:
                    pair_counts[key] = 1
        return True
    
    # Compute independence number by checking all subsets up to size k
    # Returns the largest k such that there exists an independent set of size k
    def independence_number(triples, max_check=16):
        if not triples:
            return max_check
        # Precompute for each point, which triples contain it
        point_to_triples = [[] for _ in range(16)]
        for i, t in enumerate(triples):
            for p in t:
                point_to_triples[p].append(i)
        
        # Check subsets of decreasing size
        for k in range(max_check, 0, -1):
            # Check all subsets of size k
            # For efficiency, use recursion with pruning
            found = False
            def check_subset(points, size, mask):
                nonlocal found
                if found:
                    return
                if size == 0:
                    # Check if this subset is independent
                    # A subset is independent if no triple is fully contained in it
                    for t in triples:
                        tm = triple_mask(*t)
                        if tm & mask == tm:
                            return
                    found = True
                    return
                if len(points) < size:
                    return
                for i in range(len(points) - size + 1):
                    p = points[i]
                    new_mask = mask | (1 << p)
                    remaining = points[i+1:]
                    check_subset(remaining, size - 1, new_mask)
                    if found:
                        return
            
            all_points = list(range(16))
            check_subset(all_points, k, 0)
            if found:
                return k
        return 0
    
    # Faster independence check using precomputed triple masks
    def independence_number_fast(triple_masks, max_check=16):
        """triple_masks: list of 16-bit masks for each triple"""
        if not triple_masks:
            return max_check
        
        # For each subset, check if any triple is contained
        # Use a different approach: for each subset, check against all triples
        for k in range(max_check, 0, -1):
            # Generate all k-subsets
            def gen_subsets(n, k, start=0, current=None):
                if current is None:
                    current = []
                if len(current) == k:
                    mask = 0
                    for p in current:
                        mask |= (1 << p)
                    yield mask
                    return
                for i in range(start, n - k + len(current) + 1):
                    current.append(i)
                    yield from gen_subsets(n, k, i + 1, current)
                    current.pop()
            
            for mask in gen_subsets(16, k):
                # Check if any triple is contained in this subset
                independent = True
                for tm in triple_masks:
                    if tm & mask == tm:
                        independent = False
                        break
                if independent:
                    return k
        return 0
    
    # Even faster: precompute for each mask which triples it contains
    # But 2^16 = 65536 masks, and we can precompute containment
    
    def precompute_containment(triple_masks):
        """For each of the 2^16 masks, store a bitmask of which triples are contained"""
        # This is too expensive to store as full bitmask if many triples
        # Instead, for each mask, just store a boolean: is it independent?
        # A mask is independent if no triple is fully contained in it
        n_masks = 1 << 16
        independent = [True] * n_masks
        
        # For each triple, mark all masks that contain it as non-independent
        for tm in triple_masks:
            # All masks that contain tm: tm | extra, where extra is any subset of ~tm
            # Iterate over all supersets of tm
            remaining = ~tm & 0xFFFF
            extra = 0
            while True:
                mask = tm | extra
                independent[mask] = False
                if extra == remaining:
                    break
                extra = (extra + 1) & remaining
                if extra == 0 and (tm | extra) != tm:  # Avoid infinite loop
                    break
        
        return independent
    
    # The above precomputation is O(3^16) which is too slow.
    # Let's use a different approach: for each mask, check against triples directly.
    # But with 16 points, the number of k-subsets is manageable for small k.
    
    # Let's use a smarter approach: 
    # For independence number, we can use the following:
    # alpha >= k iff there exists a k-subset with no triple contained.
    # We can check this by iterating over k-subsets.
    
    # For 16 points, C(16,8) = 12870, C(16,7) = 11440, etc.
    # This is feasible.
    
    def compute_alpha(triple_masks, target=7):
        """Check if there's an independent set of size >= target.
        Returns the actual independence number (capped at target for efficiency)."""
        if not triple_masks:
            return 16
        
        # Check from high to low
        for k in range(16, target - 1, -1):
            # Check all k-subsets
            from itertools import combinations
            for subset in combinations(range(16), k):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                # Check if any triple is contained
                has_triple = False
                for tm in triple_masks:
                    if tm & mask == tm:
                        has_triple = True
                        break
                if not has_triple:
                    return k  # Found independent set of size k
        return target - 1  # No independent set of size >= target
    
    # Now let's design the search
    # State: a list of triples (each is a 3-tuple of points 0-15)
    # Constraint: linear (any two triples share at most one point)
    # Objective: minimize independence number
    
    # Move operators:
    # 1. Add a random triple (if linear)
    # 2. Remove a random triple
    # 3. Replace a triple with another (if linear)
    
    # Simulated annealing
    def simulated_annealing(max_time=130):
        # Initial solution: start with a few random triples
        triples = []
        
        # Try to build an initial solution
        random.seed(42)
        
        # Start with a structured initial solution
        # Use a greedy approach to add triples
        used_pairs = set()
        
        # Greedily add triples
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            a, b, c = random.sample(range(16), 3)
            pairs = [(min(a,b), max(a,b)), (min(a,c), max(a,c)), (min(b,c), max(b,c))]
            if all(p not in used_pairs for p in pairs):
                triples.append(tuple(sorted((a, b, c))))
                for p in pairs:
                    used_pairs.add(p)
        
        # Convert to masks
        triple_masks = [triple_mask(*t) for t in triples]
        
        # Compute initial alpha
        current_alpha = compute_alpha(triple_masks, target=8)
        
        best_triples = triples[:]
        best_alpha = current_alpha
        
        # Simulated annealing
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        
        while time.time() < deadline - 5 and T > T_min:
            # Generate a move
            move_type = random.random()
            
            if move_type < 0.3:
                # Add a triple
                a, b, c = random.sample(range(16), 3)
                new_triple = tuple(sorted((a, b, c)))
                # Check linearity
                valid = True
                for t in triples:
                    overlap = len(set(new_triple) & set(t))
                    if overlap > 1:
                        valid = False
                        break
                if valid:
                    new_triples = triples + [new_triple]
                    new_masks = triple_masks + [triple_mask(*new_triple)]
                    new_alpha = compute_alpha(new_masks, target=8)
                    delta = new_alpha - current_alpha
                    if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                        triples = new_triples
                        triple_masks = new_masks
                        current_alpha = new_alpha
                        if current_alpha < best_alpha:
                            best_alpha = current_alpha
                            best_triples = triples[:]
            elif move_type < 0.6:
                # Remove a triple
                if triples:
                    idx = random.randint(0, len(triples) - 1)
                    new_triples = triples[:idx] + triples[idx+1:]
                    new_masks = triple_masks[:idx] + triple_masks[idx+1:]
                    new_alpha = compute_alpha(new_masks, target=8)
                    delta = new_alpha - current_alpha
                    if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                        triples = new_triples
                        triple_masks = new_masks
                        current_alpha = new_alpha
                        if current_alpha < best_alpha:
                            best_alpha = current_alpha
                            best_triples = triples[:]
            else:
                # Replace a triple
                if triples:
                    idx = random.randint(0, len(triples) - 1)
                    a, b, c = random.sample(range(16), 3)
                    new_triple = tuple(sorted((a, b, c)))
                    # Check linearity with all other triples
                    valid = True
                    for j, t in enumerate(triples):
                        if j == idx:
                            continue
                        overlap = len(set(new_triple) & set(t))
                        if overlap > 1:
                            valid = False
                            break
                    if valid:
                        new_triples = triples[:idx] + [new_triple] + triples[idx+1:]
                        new_masks = triple_masks[:idx] + [triple_mask(*new_triple)] + triple_masks[idx+1:]
                        new_alpha = compute_alpha(new_masks, target=8)
                        delta = new_alpha - current_alpha
                        if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                            triples = new_triples
                            triple_masks = new_masks
                            current_alpha = new_alpha
                            if current_alpha < best_alpha:
                                best_alpha = current_alpha
                                best_triples = triples[:]
            
            T *= cooling
            
            # Early exit if we found alpha <= 7
            if best_alpha <= 7:
                break
        
        return best_triples, best_alpha
    
    # Run simulated annealing
    best_triples, best_alpha = simulated_annealing()
    
    # Verify linearity
    if not is_linear(best_triples):
        # Fallback: return a simple valid system
        # Just return a few non-overlapping triples
        best_triples = [(0,1,2), (3,4,5), (6,7,8), (9,10,11), (12,13,14), (0,3,6)]
    
    # Convert to list of lists
    return [list(t) for t in best_triples]
