import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n_points = 16
    all_pairs = list(combinations(range(n_points), 2))
    total_pairs = len(all_pairs)
    
    # Pre-compute triples and their pair indices
    # triple -> (pair_idx_1, pair_idx_2, pair_idx_3)
    triple_to_pairs = {}
    all_triples = []
    
    # Map pair to index for fast lookup
    pair_to_idx = {p: i for i, p in enumerate(all_pairs)}
    
    for t in combinations(range(n_points), 3):
        a, b, c = t
        p1 = (a, b)
        p2 = (a, c)
        p3 = (b, c)
        
        # Ensure consistent ordering for pair lookup (a<b, etc. already handled by combinations)
        idx1 = pair_to_idx[p1]
        idx2 = pair_to_idx[p2]
        idx3 = pair_to_idx[p3]
        
        triple_to_pairs[t] = (idx1, idx2, idx3)
        all_triples.append(t)
        
    # Shuffle triples to explore different branches
    random.seed(42)
    random.shuffle(all_triples)
    
    # State for B&B
    used_pairs = [False] * total_pairs
    current_triples = []
    best_system = None
    best_alpha = 16 # We want to minimize alpha
    
    # Helper: Check if adding a triple is valid
    def can_add(triple):
        p1, p2, p3 = triple_to_pairs[triple]
        return not used_pairs[p1] and not used_pairs[p2] and not used_pairs[p3]
        
    def add_triple(triple):
        p1, p2, p3 = triple_to_pairs[triple]
        used_pairs[p1] = True
        used_pairs[p2] = True
        used_pairs[p3] = True
        current_triples.append(triple)
        
    def remove_triple():
        triple = current_triples.pop()
        p1, p2, p3 = triple_to_pairs[triple]
        used_pairs[p1] = False
        used_pairs[p2] = False
        used_pairs[p3] = False

    # Helper: Calculate independence number (exact)
    def calc_alpha(system):
        if not system:
            return n_points
            
        # Create a set of triples for fast checking
        # Actually, for n=16, we can just iterate subsets or use a smarter check.
        # Let's use the bitmask approach from the seed, it's very fast.
        triple_masks = []
        for a, b, c in system:
            mask = (1 << a) | (1 << b) | (1 << c)
            triple_masks.append(mask)
            
        max_size = 0
        # Optimization: If we have many triples, alpha is likely small.
        # We can check from size 16 down.
        # But iterating all 2^16 masks is ~65k ops. 
        # 1000 calls to this is 65M ops, which might be too slow in Python if done naively.
        # Let's optimize: 
        # A set S is independent if no triple is a subset of S.
        # Equivalent: For each triple, if it is in S, S is not independent.
        
        # Faster check:
        # Precompute for each point, which triples contain it? No.
        # Let's stick to the bitmask but optimize the loop.
        
        # Actually, we can prune the search. 
        # If current_alpha_lower_bound > best_alpha, we might stop? No, we are calculating exact.
        
        # Let's try a different approach for alpha:
        # Start with all points. Remove points that form triples?
        # No, that's for vertex cover.
        
        # Let's just use the bitmask loop but break early if possible.
        # We want max size.
        # Iterate masks from (1<<16)-1 down to 0? No, size isn't monotonic with value.
        # Iterate by size k from 16 down to 0.
        # If we find an independent set of size k, return k.
        
        # Generating combinations of size k is expensive to generate repeatedly.
        # Let's just iterate all masks once.
        # To speed up, we can use numpy if available, but standard lib is safer.
        
        # Let's try a heuristic check first:
        # If the number of triples is high, alpha is low.
        
        # For the B&B, we don't want to call this on every node.
        # We should only call it when we reach a "maximal" packing or periodically.
        # Or use a lower bound on alpha for pruning.
        # Lower bound on alpha: 
        # If we pick k points randomly, prob of containing a triple is ~ T * (k/16)^3.
        # This is weak.
        
        # Let's just call it at leaves (maximal packings) or every N steps.
        # Given the time limit, we might not reach many leaves if we go deep.
        # A maximal packing usually has ~16-20 triples.
        # Depth ~20. Branching factor ~ (remaining valid triples).
        # This tree is huge.
        
        # Alternative Strategy: 
        # Instead of pure B&B on triples, use a **Greedy + Local Search** but with a **different objective function** or **representation**.
        # The current program does Random Greedy.
        # Let's try **Simulated Annealing on the set of triples**.
        # Move: Swap a triple in the system with a new triple, or add/remove a triple.
        # Constraint: Linearity (no pair repeated).
        # Objective: Minimize Alpha.
        
        # This is structurally different from "build from scratch". It optimizes a valid system.
        
        pass # The B&B above is too complex to implement efficiently in 150s with Python overhead for the evaluation step.
        
    # Let's pivot to the Simulated Annealing strategy immediately as it's more robust for this time limit.
    # The prompt asks for a DIFFERENT strategy. 
    # Current: Random Greedy Packing.
    # New: Simulated Annealing on Linear Triple Systems.
    
    def calc_alpha_fast(system):
        if not system:
            return n_points
        triple_masks = []
        for a, b, c in system:
            mask = (1 << a) | (1 << b) | (1 << c)
            triple_masks.append(mask)
            
        max_size = 0
        # Iterate all subsets
        # Optimization: If a mask contains a triple, it's not independent.
        # We can precompute a "hit" table? 
        # For each point mask, does it contain any triple?
        # 2^16 = 65536.
        # Let's just do the loop. It's fast enough for ~100-200 evaluations.
        
        for mask in range(1 << 16):
            is_indep = True
            for tm in triple_masks:
                if (mask & tm) == tm:
                    is_indep = False
                    break
            if is_indep:
                size = bin(mask).count('1')
                if size > max_size:
                    max_size = size
                    # If we hit 16, we can break? No, we want max.
                    # If we hit a size close to 16, we can break?
                    # Actually, if we find a set of size 16, alpha=16.
                    # We want to minimize.
                    pass
        return max_size

    # SA Implementation
    # State: List of triples
    # Moves:
    # 1. Add a random valid triple.
    # 2. Remove a random triple.
    # 3. Swap: Remove a triple, add a different valid triple.
    
    # Initialize with a greedy packing
    def initial_greedy():
        random.seed(1)
        used = set()
        triples = []
        all_t = list(combinations(range(16), 3))
        random.shuffle(all_t)
        for t in all_t:
            a, b, c = t
            p1 = (a,b) if a<b else (b,a)
            p2 = (a,c) if a<c else (c,a)
            p3 = (b,c) if b<c else (c,b)
            if p1 not in used and p2 not in used and p3 not in used:
                triples.append(t)
                used.add(p1); used.add(p2); used.add(p3)
        return triples

    current_system = initial_greedy()
    current_alpha = calc_alpha_fast(current_system)
    best_system = current_system[:]
    best_alpha = current_alpha
    
    T = 1.0
    T_min = 0.01
    cooling = 0.99
    
    # Pre-compute all triples for SA moves
    all_triples_sa = list(combinations(range(16), 3))
    
    # To make SA efficient, we need to check validity quickly.
    # Maintain a set of used pairs.
    used_pairs_set = set()
    for a, b, c in current_system:
        used_pairs_set.add((a,b) if a<b else (b,a))
        used_pairs_set.add((a,c) if a<c else (c,a))
        used_pairs_set.add((b,c) if b<c else (c,b))
        
    def is_valid_triple(t):
        a, b, c = t
        p1 = (a,b) if a<b else (b,a)
        p2 = (a,c) if a<c else (c,a)
        p3 = (b,c) if b<c else (c,b)
        return p1 not in used_pairs_set and p2 not in used_pairs_set and p3 not in used_pairs_set
        
    def add_to_used(t):
        a, b, c = t
        used_pairs_set.add((a,b) if a<b else (b,a))
        used_pairs_set.add((a,c) if a<c else (c,a))
        used_pairs_set.add((b,c) if b<c else (c,b))
        
    def remove_from_used(t):
        a, b, c = t
        used_pairs_set.discard((a,b) if a<b else (b,a))
        used_pairs_set.discard((a,c) if a<c else (c,a))
        used_pairs_set.discard((b,c) if b<c else (c,b))

    # SA Loop
    # We need to be careful with time. 
    # calc_alpha_fast is the bottleneck.
    # We can cache alpha for systems? No, systems are unique.
    # We can approximate alpha? 
    # For now, let's limit the number of SA steps.
    
    steps = 0
    max_steps = 5000 # Adjust based on time
    
    while time.time() < deadline - 5.0:
        steps += 1
        if steps > max_steps:
            break
            
        # Generate a move
        move_type = random.random()
        
        # 1. Add a triple
        if move_type < 0.3:
            # Find a random valid triple
            candidate = random.choice(all_triples_sa)
            if is_valid_triple(candidate):
                current_system.append(candidate)
                add_to_used(candidate)
                new_alpha = calc_alpha_fast(current_system)
                delta = new_alpha - current_alpha
                if delta < 0 or random.random() < pow(2.718, -delta / T):
                    current_alpha = new_alpha
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        best_system = current_system[:]
                else:
                    # Revert
                    current_system.pop()
                    remove_from_used(candidate)
                    
        # 2. Remove a triple
        elif move_type < 0.6:
            if current_system:
                idx = random.randint(0, len(current_system)-1)
                t = current_system[idx]
                current_system.pop(idx)
                remove_from_used(t)
                new_alpha = calc_alpha_fast(current_system)
                delta = new_alpha - current_alpha
                if delta < 0 or random.random() < pow(2.718, -delta / T):
                    current_alpha = new_alpha
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        best_system = current_system[:]
                else:
                    # Revert
                    current_system.insert(idx, t)
                    add_to_used(t)
                    
        # 3. Swap
        else:
            if current_system:
                idx = random.randint(0, len(current_system)-1)
                old_t = current_system[idx]
                # Remove old
                current_system.pop(idx)
                remove_from_used(old_t)
                
                # Find a valid new triple
                # Try a few random ones
                new_t = None
                for _ in range(10):
                    candidate = random.choice(all_triples_sa)
                    if is_valid_triple(candidate):
                        new_t = candidate
                        break
                if new_t:
                    current_system.insert(idx, new_t)
                    add_to_used(new_t)
                    new_alpha = calc_alpha_fast(current_system)
                    delta = new_alpha - current_alpha
                    if delta < 0 or random.random() < pow(2.718, -delta / T):
                        current_alpha = new_alpha
                        if new_alpha < best_alpha:
                            best_alpha = new_alpha
                            best_system = current_system[:]
                    else:
                        # Revert
                        current_system[idx] = old_t
                        remove_from_used(new_t)
                        add_to_used(old_t)
                else:
                    # Revert
                    current_system.insert(idx, old_t)
                    add_to_used(old_t)

        # Cool down
        T *= cooling
        
        # If T is very low, we might get stuck. 
        # If we find alpha=5, we can stop?
        if best_alpha <= 5:
            break

    if best_system is None:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
        
    return best_system
