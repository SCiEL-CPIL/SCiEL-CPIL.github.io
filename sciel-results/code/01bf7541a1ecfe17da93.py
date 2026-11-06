import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    # Pre-compute all 4-subsets of [16] as bitmasks
    all_masks = []
    mask_to_combo = {}
    for combo in combinations(range(16), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_masks.append(mask)
        mask_to_combo[mask] = combo

    n = len(all_masks)  # 1820

    # Pre-compute pairwise intersections for all pairs of masks
    # This is O(n^2) = ~3.3M operations, which is fast in Python if optimized
    # Actually, let's just compute on the fly but use a faster check.
    
    # Strategy: Maximum Independent Set in a 3-uniform hypergraph.
    # Vertices: all 1820 masks.
    # Hyperedges: triples (i, j, k) that form a sunflower.
    # We want to find the largest subset of vertices with no hyperedge entirely contained in it.
    
    # Direct enumeration of all sunflower triples is expensive (1820^3 is too big).
    # Instead, we use a local search heuristic on the binary vector x of length n.
    # Objective: Maximize sum(x) subject to no sunflower triple is fully active.
    
    # Efficient Sunflower Check for a new set S against existing family F:
    # A sunflower {A, B, C} exists if A&B == B&C == C&A.
    # When adding S to F, we check if there exist A, B in F such that {A, B, S} is a sunflower.
    # This requires A&B == A&S == B&S.
    # Let I = A&S. Then we need B&S == I and A&B == I.
    # So for a fixed S, we group F by their intersection with S.
    # For each intersection value I, let G_I = {A in F | A&S == I}.
    # If there exist A, B in G_I such that A&B == I, then {A, B, S} is a sunflower.
    
    # We will implement a simulated annealing / hill climbing hybrid.
    # State: A set of active masks.
    # Move: Flip a random mask (add or remove).
    # Accept: If size increases, always accept. If size decreases, accept with probability exp(-delta/T).
    # But we want to maximize size, so we can just do "Best Improvement" or "First Improvement" hill climbing.
    # Given the time limit, a robust random restart + greedy + local search is good.
    
    # Let's use a faster representation:
    # active: set of indices
    # To check if adding index i is valid:
    #   S = all_masks[i]
    #   groups = {}
    #   for j in active:
    #       A = all_masks[j]
    #       inter = A & S
    #       if inter not in groups:
    #           groups[inter] = []
    #       groups[inter].append(j)
    #   for inter, indices in groups.items():
    #       # Check if any pair in indices has intersection == inter
    #       for p in range(len(indices)):
    #           A_idx = indices[p]
    #           A = all_masks[A_idx]
    #           for q in range(p+1, len(indices)):
    #               B_idx = indices[q]
    #               B = all_masks[B_idx]
    #               if (A & B) == inter:
    #                   return False
    #   return True
    
    # This is O(|F| * |G|^2). If |F| is ~200, this is manageable.
    
    best_family = set()
    best_size = 0
    
    # Helper to convert set of indices to list of lists
    def indices_to_result(indices):
        return [list(mask_to_combo[all_masks[i]]) for i in indices]

    # Greedy construction
    def greedy_build(order):
        fam = set()
        for i in order:
            if time.time() > deadline:
                break
            if can_add(i, fam):
                fam.add(i)
        return fam

    def can_add(new_idx, fam):
        S = all_masks[new_idx]
        groups = {}
        for j in fam:
            A = all_masks[j]
            inter = A & S
            if inter not in groups:
                groups[inter] = []
            groups[inter].append(j)
        
        for inter, indices in groups.items():
            k = len(indices)
            if k < 2:
                continue
            # Check pairs
            for p in range(k):
                A_idx = indices[p]
                A = all_masks[A_idx]
                for q in range(p+1, k):
                    B_idx = indices[q]
                    B = all_masks[B_idx]
                    if (A & B) == inter:
                        return False
        return True

    # Main loop: Random restarts with greedy, then local search
    # Local search: Try to add any unused set. If none, try swaps (remove 1, add 2).
    
    while time.time() < deadline - 10.0:
        order = list(range(n))
        random.shuffle(order)
        fam = greedy_build(order)
        
        if len(fam) > best_size:
            best_size = len(fam)
            best_family = fam.copy()
        
        # Local search on current fam
        # Try to add sets
        added_any = True
        while added_any and time.time() < deadline - 10.0:
            added_any = False
            unused = [i for i in range(n) if i not in fam]
            random.shuffle(unused)
            for i in unused:
                if time.time() > deadline - 10.0:
                    break
                if can_add(i, fam):
                    fam.add(i)
                    added_any = True
                    if len(fam) > best_size:
                        best_size = len(fam)
                        best_family = fam.copy()
                    break # Restart the loop to re-evaluate unused? Or just continue?
                    # Breaking allows us to shuffle again and potentially find another.
        
        if len(fam) > best_size:
            best_size = len(fam)
            best_family = fam.copy()
            
        # If we hit a high number, we can stop
        if best_size >= 250:
            break

    # Final local search on best_family to squeeze out more
    if time.time() < deadline:
        fam = best_family.copy()
        improved = True
        while improved and time.time() < deadline:
            improved = False
            unused = [i for i in range(n) if i not in fam]
            random.shuffle(unused)
            for i in unused:
                if time.time() > deadline:
                    break
                if can_add(i, fam):
                    fam.add(i)
                    improved = True
                    if len(fam) > best_size:
                        best_size = len(fam)
                        best_family = fam.copy()
                    break

    return indices_to_result(best_family)
