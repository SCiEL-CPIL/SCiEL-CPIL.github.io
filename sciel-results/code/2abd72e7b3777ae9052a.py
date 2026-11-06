import time
import random
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def is_linear(triples):
        n = len(triples)
        sets = [set(t) for t in triples]
        for i in range(n):
            for j in range(i + 1, n):
                if len(sets[i] & sets[j]) > 1:
                    return False
        return True

    def calc_alpha(triples):
        n = 16
        # Check from size 8 down to 1
        # Optimization: If we find an independent set of size k, return k.
        # We want the MAXIMUM size.
        for k in range(8, 0, -1):
            for subset in itertools.combinations(range(n), k):
                s = set(subset)
                has_triple = False
                for t in triples:
                    if t[0] in s and t[1] in s and t[2] in s:
                        has_triple = True
                        break
                if not has_triple:
                    return k
        return 0

    # 1. Construct a high-quality linear system on 15 points (0..14)
    # Goal: Maximize number of triples (density) to minimize alpha.
    # Strategy: Randomized greedy with backtracking/restarts.
    
    best_triples_15 = []
    best_count = 0
    
    # Try multiple random seeds to find a dense linear system
    # A linear system on 15 points can have at most 35 triples (STS(15) bound).
    # We aim for close to 35.
    
    for seed in range(50):
        if time_left() < 20:
            break
        random.seed(seed)
        triples_15 = []
        used_pairs = set()
        
        # Pre-shuffle all pairs
        all_pairs = list(itertools.combinations(range(15), 2))
        random.shuffle(all_pairs)
        
        # Greedy addition
        # To improve density, we can try to add triples that cover "hard" pairs.
        # Simple greedy: iterate pairs, if available, pick random 3rd point.
        # To make it better, we can look ahead or use a different order.
        # Let's stick to a robust greedy but with multiple restarts.
        
        # Optimization: Sort candidates for 3rd point by degree (number of available pairs)
        # to keep options open? Or random? Random is usually fine for high density.
        
        for p1, p2 in all_pairs:
            if (p1, p2) in used_pairs:
                continue
            
            candidates = [p for p in range(15) if p != p1 and p != p2]
            random.shuffle(candidates)
            
            added = False
            for p3 in candidates:
                new_triple = (min(p1, p2, p3), min(max(p1, p2, p3), max(max(p1, p2, p3), min(p1, p2, p3))), max(p1, p2, p3))
                new_pairs = [(min(p1, p3), max(p1, p3)), (min(p2, p3), max(p2, p3)), (p1, p2)]
                
                if any(p in used_pairs for p in new_pairs):
                    continue
                
                # Check linearity with existing triples
                ok = True
                for t in triples_15:
                    if len(set(new_triple) & set(t)) > 1:
                        ok = False
                        break
                
                if ok:
                    triples_15.append(new_triple)
                    for p in new_pairs:
                        used_pairs.add(p)
                    added = True
                    break
        
        if len(triples_15) > best_count:
            best_count = len(triples_15)
            best_triples_15 = triples_15[:]
            
            # If we reached the theoretical max for STS(15), we can stop early?
            # STS(15) has 35 blocks.
            if best_count == 35:
                break

    # 2. Add point 15
    # We need to add triples [15, u, v].
    # Constraint: The pairs {u, v} must be disjoint from each other (so max 7 triples).
    # Constraint: The pair {u, v} must NOT be an edge in the hypergraph of best_triples_15.
    # Why? If {u, v} is in a triple T = [u, v, w] in best_triples_15,
    # then [15, u, v] and [u, v, w] share {u, v}, violating linearity.
    
    used_pairs_15 = set()
    for t in best_triples_15:
        used_pairs_15.add((min(t[0], t[1]), max(t[0], t[1])))
        used_pairs_15.add((min(t[0], t[2]), max(t[0], t[2])))
        used_pairs_15.add((min(t[1], t[2]), max(t[1], t[2])))
        
    free_pairs = []
    for p1, p2 in itertools.combinations(range(15), 2):
        if (min(p1, p2), max(p1, p2)) not in used_pairs_15:
            free_pairs.append((p1, p2))
            
    # We need to select a matching (set of disjoint pairs) from free_pairs.
    # We want to select up to 7 pairs to minimize the independence number of the final system.
    
    # Heuristic: Select pairs that "hit" the largest independent sets of the current system (without 15).
    # Or, simpler: Random search over matchings.
    
    # Let's use ILP or a smart greedy/random local search to select the pairs for 15.
    # Since the number of free pairs is small (105 - 3*len(best_triples_15)),
    # and we need a matching of size 7, we can use a randomized local search.
    
    # First, find a valid matching of size 7 in free_pairs.
    # If free_pairs is a random graph, it likely has a large matching.
    
    def find_matching(pairs, k):
        # Greedy random matching
        pairs = pairs[:]
        random.shuffle(pairs)
        selected = []
        used_nodes = set()
        for p1, p2 in pairs:
            if p1 not in used_nodes and p2 not in used_nodes:
                selected.append((p1, p2))
                used_nodes.add(p1)
                used_nodes.add(p2)
                if len(selected) == k:
                    break
        return selected

    # Try to find a matching of size 7
    initial_matching = find_matching(free_pairs, 7)
    
    if len(initial_matching) < 7:
        # If we can't find 7, use whatever we found
        pass

    best_alpha = 16
    best_all_triples = best_triples_15 + [[15, p1, p2] for p1, p2 in initial_matching]
    
    # Local Search on the matching for point 15
    # Objective: Minimize alpha of the full system.
    
    current_matching = initial_matching[:]
    current_triples = best_triples_15 + [[15, p1, p2] for p1, p2 in current_matching]
    current_alpha = calc_alpha(current_triples)
    best_alpha = current_alpha
    best_all_triples = current_triples[:]
    
    # Iteration: Try to swap one pair in the matching with another free pair
    # that maintains the matching property (disjointness) and linearity (free pair).
    
    # Pre-compute neighbors for fast swap?
    # For each pair i in current_matching, the other points are fixed.
    # A new pair (u, v) can replace (p1, p2) if u, v are not in the set of points used by other pairs in the matching.
    
    other_points = set()
    for p1, p2 in current_matching:
        other_points.add(p1)
        other_points.add(p2)
        
    # Remove points of the pair being swapped
    # We'll do this inside the loop.
    
    # Optimization: Pre-compute alpha for the base system (15 points) to speed up?
    # No, alpha of full system depends on the triples with 15.
    
    # We can speed up calc_alpha by checking only subsets of size >= current_best_alpha.
    # But calc_alpha is already optimized to check from 8 down.
    
    # Let's run local search for the remaining time.
    
    while time_left() > 2:
        improved = False
        # Shuffle the order of pairs to try
        indices = list(range(len(current_matching)))
        random.shuffle(indices)
        
        for i in indices:
            if time_left() < 2:
                break
                
            p1_old, p2_old = current_matching[i]
            
            # Points that are "locked" by other pairs in the matching
            locked_points = set()
            for j, (p1, p2) in enumerate(current_matching):
                if j != i:
                    locked_points.add(p1)
                    locked_points.add(p2)
                    
            # Candidate pairs: (u, v) from free_pairs such that u, v not in locked_points
            # and u, v != p1_old, p2_old (actually they can be, but we are replacing, so effectively new points)
            # Wait, if u is in locked_points, we can't use it.
            # So we need u, v not in locked_points.
            
            candidates = []
            for p1, p2 in free_pairs:
                if p1 not in locked_points and p2 not in locked_points:
                    # Also ensure we don't pick the same pair we just removed? 
                    # It doesn't matter, it's a valid candidate.
                    candidates.append((p1, p2))
            
            if not candidates:
                continue
                
            random.shuffle(candidates)
            
            # Try a few random candidates
            for p1_new, p2_new in candidates[:20]: # Limit tries per swap to save time
                # Construct new matching
                new_matching = current_matching[:]
                new_matching[i] = (p1_new, p2_new)
                
                # Check if it's a valid matching (disjointness)
                # We already ensured p1_new, p2_new not in locked_points.
                # But do p1_new and p2_new conflict with each other? No, they are distinct in a pair.
                # Do they conflict with the pair being replaced? No, we are replacing it.
                # So it is a valid matching.
                
                new_triples = best_triples_15 + [[15, p1, p2] for p1, p2 in new_matching]
                
                # Verify linearity (should be true by construction)
                # if not is_linear(new_triples): continue
                
                new_alpha = calc_alpha(new_triples)
                
                if new_alpha < current_alpha:
                    current_alpha = new_alpha
                    current_matching = new_matching
                    current_triples = new_triples
                    improved = True
                    break # Restart search with new state
            if improved:
                break
                
        if not improved:
            # If no improvement in a full pass, we might be stuck in a local minimum.
            # We could try a random restart of the matching, but let's just continue
            # to be safe on time, or break if we've been stuck for a while.
            # For now, just break if we don't improve in a full pass?
            # Or continue? The space is small. Let's break to avoid wasting time.
            break

    return best_all_triples

# Test locally if run directly
if __name__ == "__main__":
    t = construct()
    print("Triples:", len(t))
    print("Linear:", is_linear(t))
    print("Alpha:", calc_alpha(t))
