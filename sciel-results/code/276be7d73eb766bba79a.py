import itertools
import random
import time

def construct():
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    
    # Precompute pair indices for each triple
    # Pair index: i * n + j for i < j
    triple_pairs = []
    for t in all_triples:
        a, b, c = t
        pairs = [(a, b), (a, c), (b, c)]
        # Sort pairs to ensure consistent representation if needed, 
        # but for a set of pairs, order of elements in pair doesn't matter 
        # as long as we are consistent. 
        # Let's use a frozenset of the two points, or just a tuple (min, max).
        p1 = (min(a, b), max(a, b))
        p2 = (min(a, c), max(a, c))
        p3 = (min(b, c), max(b, c))
        triple_pairs.append(frozenset([p1, p2, p3]))
        
    # Also precompute bitmask for each triple for alpha calculation
    triple_masks = []
    for t in all_triples:
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks.append(mask)

    def calc_alpha(system_indices):
        """
        system_indices: list of indices into all_triples
        Returns the independence number.
        """
        if not system_indices:
            return n
            
        # Mark all supersets of each triple in the system as "not independent"
        # ind_mask[i] will be True if mask i is independent
        # We use a list of booleans of size 2^16
        ind = [True] * (1 << n)
        
        for idx in system_indices:
            t_mask = triple_masks[idx]
            # Iterate all supersets of t_mask
            # Supersets are t_mask | submask_of_complement
            complement = ((1 << n) - 1) ^ t_mask
            # Iterate all submasks of complement
            sub = complement
            while True:
                sup = t_mask | sub
                ind[sup] = False
                if sub == 0:
                    break
                sub = (sub - 1) & complement
            
        # Find the largest mask with ind[mask] == True
        max_size = 0
        for mask in range(1 << n):
            if ind[mask]:
                size = bin(mask).count('1')
                if size > max_size:
                    max_size = size
                    if max_size == n: # Early exit if full set is independent (shouldn't happen for large systems)
                        break
        return max_size

    best_alpha = n
    best_system = []
    
    start_time = time.time()
    deadline = start_time + 115 # Leave 5s margin
    
    rng = random.Random(42)
    
    # We will run random restarts
    while time.time() < deadline:
        # Shuffle triples
        shuffled_indices = list(range(len(all_triples)))
        rng.shuffle(shuffled_indices)
        
        used_pairs = set()
        system_indices = []
        
        for idx in shuffled_indices:
            pairs = triple_pairs[idx]
            # Check if any pair is already used
            if not (used_pairs & pairs):
                system_indices.append(idx)
                used_pairs |= pairs
                
        # Evaluate alpha
        alpha = calc_alpha(system_indices)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = system_indices
            
            # If we hit a theoretical lower bound, we can stop.
            # For n=16, k=3, alpha=4 is a strong candidate for minimum.
            # Let's see if we can go lower. 
            # If alpha=3, every 4 points contain a triple.
            # Number of 4-subsets is 1820. Each triple covers 4 such subsets (completing the 4th point).
            # 37 triples * 4 = 148 < 1820. So alpha=3 is impossible?
            # Wait, a triple $\{a,b,c\}$ is contained in 4 sets of size 4: $\{a,b,c,d\}$ for $d \in \{0..15\} \setminus \{a,b,c\}$.
            # If $\alpha=3$, then EVERY set of size 4 must contain a triple.
            # This means the union of the "4-sets containing each triple" must cover all 4-sets.
            # Total 4-sets: $\binom{16}{4} = 1820$.
            # Coverage by one triple: 4.
            # Max coverage by 37 triples: $37 \times 4 = 148$.
            # Since $148 < 1820$, it is impossible to cover all 4-sets.
            # Therefore, there exists a 4-set that contains no triple.
            # So $\alpha \ge 4$.
            
            if best_alpha == 4:
                # We found the theoretical minimum. We can break early to save time,
                # but we might want to verify it's valid or just return it.
                # The task asks to minimize. 4 is the min possible.
                # We can break immediately.
                break

    # Convert indices back to list of triples
    result = [list(all_triples[i]) for i in best_system]
    return result
