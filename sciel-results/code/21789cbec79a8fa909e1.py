"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a // gcd(a, b) * b

def _lcm_list(lst):
    return reduce(_lcm, lst)

def construct():
    """
    Return a nonempty subset of [1,22] such that ALL reciprocal subset sums,
    including the empty sum, are distinct. Maximize cardinality.
    
    Strategy: Formulate as finding the largest dissociated subset of integer
    weights w_i = LCM(1..22)/i. Use randomized local search with simulated
    annealing, tracking both cardinality and minimum subset-sum gap to guide
    exploration of the non-convex landscape.
    """
    N = 22
    universe = list(range(1, N + 1))
    
    # Precompute LCM of 1..22
    L = _lcm_list(universe)
    
    # Integer weights: w_i = L // i
    weights = [0] * (N + 1)
    for i in universe:
        weights[i] = L // i
    
    def compute_subset_sums(subset):
        """Compute all subset sums for a given subset of indices."""
        sums = {0}
        for i in subset:
            w = weights[i]
            new_sums = set()
            for s in sums:
                ns = s + w
                if ns not in sums:
                    new_sums.add(ns)
                else:
                    return None  # Collision detected
            sums.update(new_sums)
        return sums
    
    def is_dissociated(subset):
        """Check if subset has all distinct subset sums. Returns True/False."""
        return compute_subset_sums(subset) is not None
    
    def min_gap(subset):
        """Compute minimum difference between any two distinct subset sums."""
        sums = compute_subset_sums(subset)
        if sums is None:
            return 0
        if len(sums) <= 1:
            return float('inf')
        sorted_sums = sorted(sums)
        min_diff = float('inf')
        for i in range(1, len(sorted_sums)):
            diff = sorted_sums[i] - sorted_sums[i-1]
            if diff < min_diff:
                min_diff = diff
        return min_diff
    
    # Greedy construction with multiple orderings
    def greedy(order):
        S = []
        for i in order:
            candidate = S + [i]
            if is_dissociated(candidate):
                S = candidate
        return S
    
    # Try many greedy orderings to find a good initial set
    best_set = []
    best_size = 0
    best_gap = 0
    
    # Deterministic orderings first
    orderings = [
        list(range(1, N+1)),
        list(range(N, 0, -1)),
        sorted(universe, key=lambda x: weights[x]),
        sorted(universe, key=lambda x: -weights[x]),
    ]
    
    # Add random orderings
    for _ in range(100):
        orderings.append(universe[:])
    
    for order in orderings:
        S = greedy(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            best_gap = min_gap(S) if len(S) > 1 else float('inf')
        elif len(S) == best_size and len(S) > 1:
            g = min_gap(S)
            if g > best_gap:
                best_gap = g
                best_set = S[:]
    
    # Local search with simulated annealing
    current_set = best_set[:]
    current_size = len(current_set)
    current_gap = min_gap(current_set) if current_size > 1 else float('inf')
    
    global_best_set = current_set[:]
    global_best_size = current_size
    global_best_gap = current_gap
    
    start_time = time.time()
    timeout = 50.0  # Leave 10s buffer
    
    T = 50.0  # Initial temperature (in terms of gap units)
    cooling = 0.999
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() - start_time < timeout:
        if no_improve_count > max_no_improve:
            break
            
        # Generate a random move
        move_type = random.random()
        
        if move_type < 0.35:
            # Try to add an element
            candidates_to_add = [i for i in universe if i not in current_set]
            if not candidates_to_add:
                continue
            i = random.choice(candidates_to_add)
            new_set = current_set + [i]
            if is_dissociated(new_set):
                new_size = len(new_set)
                new_gap = min_gap(new_set) if new_size > 1 else float('inf')
                # Accept if size increases or gap improves
                if new_size > current_size or (new_size == current_size and new_gap > current_gap):
                    current_set = new_set
                    current_size = new_size
                    current_gap = new_gap
                    no_improve_count = 0
                    if current_size > global_best_size or (current_size == global_best_size and current_gap > global_best_gap):
                        global_best_set = current_set[:]
                        global_best_size = current_size
                        global_best_gap = current_gap
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
                
        elif move_type < 0.65:
            # Try to remove an element (to enable future additions)
            if len(current_set) <= 2:
                continue
            i = random.choice(current_set)
            new_set = [x for x in current_set if x != i]
            # Always valid
            # Accept with probability based on how much it hurts
            # We're trading size for potential future gains
            if random.random() < 0.15:
                current_set = new_set
                current_size = len(current_set)
                current_gap = min_gap(current_set) if current_size > 1 else float('inf')
                no_improve_count += 1
            else:
                no_improve_count += 1
        else:
            # Swap: remove one, add another
            if len(current_set) == 0:
                continue
            i_remove = random.choice(current_set)
            candidates_to_add = [i for i in universe if i not in current_set]
            if not candidates_to_add:
                continue
            i_add = random.choice(candidates_to_add)
            new_set = [x for x in current_set if x != i_remove] + [i_add]
            if is_dissociated(new_set):
                new_size = len(new_set)
                new_gap = min_gap(new_set) if new_size > 1 else float('inf')
                # Accept if same size but better gap, or we use SA to sometimes accept worse
                if new_size == current_size and new_gap > current_gap:
                    current_set = new_set
                    current_gap = new_gap
                    no_improve_count = 0
                    if current_size > global_best_size or (current_size == global_best_size and current_gap > global_best_gap):
                        global_best_set = current_set[:]
                        global_best_size = current_size
                        global_best_gap = current_gap
                else:
                    # Simulated annealing acceptance for same-size moves
                    delta_gap = new_gap - current_gap
                    if delta_gap < 0 and random.random() < min(1.0, -delta_gap / max(T, 1.0)):
                        current_set = new_set
                        current_gap = new_gap
                    no_improve_count += 1
            else:
                no_improve_count += 1
        
        T *= cooling
    
    # Final improvement pass: try to add any remaining elements
    for i in universe:
        if i not in global_best_set:
            candidate = global_best_set + [i]
            if is_dissociated(candidate):
                global_best_set = candidate
    
    if not global_best_set:
        global_best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return global_best_set
# EVOLVE-BLOCK-END
