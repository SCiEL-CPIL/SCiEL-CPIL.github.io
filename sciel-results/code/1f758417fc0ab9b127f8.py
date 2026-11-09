<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    n = 16
    points = list(range(n))
    
    # Generate all C(16,3) = 560 triples
    all_triples = list(combinations(range(n), 3))
    
    # For each triple, find all K(2,2,2)s it belongs to
    # A K(2,2,2) is defined by 3 disjoint pairs: (a1,a2), (b1,b2), (c1,c2)
    # The 8 transversal triples are {ai, bj, ck} for all i,j,k in {0,1}
    
    # Precompute: for each triple, list of (k222_id, position) 
    # where position indicates which of the 8 slots it fills
    # Better: for each triple, store the list of K(2,2,2) IDs it belongs to
    
    # First, enumerate all K(2,2,2)s
    # A K(2,2,2) is determined by choosing 3 disjoint pairs from 16 points
    # Number of ways: choose 6 points, then partition into 3 pairs
    # C(16,6) * 15 (number of perfect matchings on 6 points)
    
    # For each K(2,2,2), store the 8 triples that form it
    k222_triples = {}  # k222_id -> list of 8 triples
    triple_to_k222s = {t: [] for t in all_triples}  # triple -> list of k222_ids
    
    k222_id = 0
    for six_points in combinations(range(n), 6):
        # Generate all perfect matchings of these 6 points
        # There are 15 perfect matchings on 6 labeled points
        a, b, c, d, e, f = six_points
        matchings = [
            ((a,b),(c,d),(e,f)),
            ((a,b),(c,e),(d,f)),
            ((a,b),(c,f),(d,e)),
            ((a,c),(b,d),(e,f)),
            ((a,c),(b,e),(d,f)),
            ((a,c),(b,f),(d,e)),
            ((a,d),(b,c),(e,f)),
            ((a,d),(b,e),(c,f)),
            ((a,d),(b,f),(c,e)),
            ((a,e),(b,c),(d,f)),
            ((a,e),(b,d),(c,f)),
            ((a,e),(b,f),(c,d)),
            ((a,f),(b,c),(d,e)),
            ((a,f),(b,d),(c,e)),
            ((a,f),(b,e),(c,d)),
        ]
        for m in matchings:
            (p1, p2), (q1, q2), (r1, r2) = m
            # 8 transversal triples
            triples_in_k222 = [
                (p1, q1, r1),
                (p1, q1, r2),
                (p1, q2, r1),
                (p1, q2, r2),
                (p2, q1, r1),
                (p2, q1, r2),
                (p2, q2, r1),
                (p2, q2, r2),
            ]
            # Sort each triple for consistency
            triples_in_k222 = [tuple(sorted(t)) for t in triples_in_k222]
            
            # Check if all 8 are distinct (they should be since pairs are disjoint)
            if len(set(triples_in_k222)) == 8:
                k222_triples[k222_id] = triples_in_k222
                for t in triples_in_k222:
                    if t in triple_to_k222s:
                        triple_to_k222s[t].append(k222_id)
                k222_id += 1
    
    # Greedy: add triples in an order that minimizes K(2,2,2) conflicts
    # Use a randomized greedy with multiple restarts
    
    best_result = None
    best_count = 0
    
    # Track for each K(2,2,2) how many of its 8 triples are in the current set
    # When adding a triple, check if any K(2,2,2) it belongs to would reach 8
    
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        random.seed(restart * 7919)
        
        # Shuffle triples
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        
        # Count how many of each K(2,2,2)'s triples are present
        k222_count = [0] * k222_id
        
        selected = set()
        
        for t in shuffled:
            if time.time() > deadline - 5:
                break
            
            # Check if adding t would complete any K(2,2,2)
            can_add = True
            for kid in triple_to_k222s[t]:
                if k222_count[kid] >= 7:
                    can_add = False
                    break
            
            if can_add:
                selected.add(t)
                for kid in triple_to_k222s[t]:
                    k222_count[kid] += 1
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_result = list(selected)
    
    if best_result is None:
        # Fallback: star at point 0
        best_result = [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
    
    return [[a, b, c] for a, b, c in best_result]
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)  # 560
    
    # Enumerate all K(2,2,2)s and build incidence structure
    triple_to_k222s = {t: [] for t in all_triples}
    k222_to_triples = []  # list of lists of 8 triple indices
    
    triple_index = {t: i for i, t in enumerate(all_triples)}
    
    for six_points in combinations(range(n), 6):
        a, b, c, d, e, f = six_points
        matchings = [
            ((a,b),(c,d),(e,f)),
            ((a,b),(c,e),(d,f)),
            ((a,b),(c,f),(d,e)),
            ((a,c),(b,d),(e,f)),
            ((a,c),(b,e),(d,f)),
            ((a,c),(b,f),(d,e)),
            ((a,d),(b,c),(e,f)),
            ((a,d),(b,e),(c,f)),
            ((a,d),(b,f),(c,e)),
            ((a,e),(b,c),(d,f)),
            ((a,e),(b,d),(c,f)),
            ((a,e),(b,f),(c,d)),
            ((a,f),(b,c),(d,e)),
            ((a,f),(b,d),(c,e)),
            ((a,f),(b,e),(c,d)),
        ]
        for m in matchings:
            (p1, p2), (q1, q2), (r1, r2) = m
            triples_in_k222 = [
                tuple(sorted((p1, q1, r1))),
                tuple(sorted((p1, q1, r2))),
                tuple(sorted((p1, q2, r1))),
                tuple(sorted((p1, q2, r2))),
                tuple(sorted((p2, q1, r1))),
                tuple(sorted((p2, q1, r2))),
                tuple(sorted((p2, q2, r1))),
                tuple(sorted((p2, q2, r2))),
            ]
            if len(set(triples_in_k222)) == 8:
                k222_idx = len(k222_to_triples)
                triple_indices = [triple_index[t] for t in triples_in_k222]
                k222_to_triples.append(triple_indices)
                for ti in triple_indices:
                    triple_to_k222s[all_triples[ti]].append(k222_idx)
    
    num_k222s = len(k222_to_triples)
    
    # Build conflict graph: two triples conflict if they appear in the same K(2,2,2)
    # and including both would force a K(2,2,2) completion
    # Actually, the constraint is: no K(2,2,2) can have all 8 triples present.
    # So we need max subset of triples with no K(2,2,2) fully contained.
    
    # Greedy initial solution
    best_count = 0
    best_selected = set()
    
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        random.seed(restart * 7919)
        shuffled = list(range(num_triples))
        random.shuffle(shuffled)
        
        k222_count = [0] * num_k222s
        selected = set()
        
        for ti in shuffled:
            can_add = True
            for kid in triple_to_k222s[all_triples[ti]]:
                if k222_count[kid] >= 7:
                    can_add = False
                    break
            if can_add:
                selected.add(ti)
                for kid in triple_to_k222s[all_triples[ti]]:
                    k222_count[kid] += 1
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_selected = set(selected)
    
    # Local search: try to add triples that were excluded, or swap
    # For each unselected triple, check if we can add it by removing some selected triples
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        unselected = [i for i in range(num_triples) if i not in best_selected]
        random.shuffle(unselected)
        
        for ti in unselected:
            if time.time() > deadline - 15:
                break
            
            # Check which K(2,2,2)s would be violated
            violating_k222s = []
            for kid in triple_to_k222s[all_triples[ti]]:
                if k222_count_cache.get(kid, 0) >= 7:
                    violating_k222s.append(kid)
            
            if not violating_k222s:
                best_selected.add(ti)
                best_count += 1
                improved = True
            else:
                # Try to remove one triple from each violating K(2,2,2)
                # This is complex; skip for now and rely on B&B
                pass
    
    # Branch and bound for maximum independent set
    # State: (remaining_candidates, current_selected, current_count)
    # Upper bound: current_count + len(remaining_candidates)
    # Better upper bound: use the fact that for each K(2,2,2), at most 7 of its 8 triples can be selected
    
    # Precompute for each triple, the list of K(2,2,2) indices it belongs to
    triple_k222_list = [triple_to_k222s[all_triples[i]] for i in range(num_triples)]
    
    # B&B solver
    best_bnb = best_count
    best_bnb_set = set(best_selected)
    
    # Order triples by degree (number of K(2,2,2)s they belong to) - process high-degree first
    degree_order = sorted(range(num_triples), key=lambda i: len(triple_k222_list[i]), reverse=True)
    
    # Use a stack-based DFS
    # State: (pos_in_order, selected_set, k222_counts)
    # We'll use a simpler approach: recursive with memoization on small subproblems
    
    # Actually, let's use a different B&B strategy:
    # Start with all 560 triples as candidates
    # At each step, pick a triple, branch on include/exclude
    # Upper bound: current_size + max_additional (computed via greedy on remaining)
    
    def greedy_upper_bound(candidates, k222_counts):
        """Quick greedy to estimate max we can still add."""
        count = 0
        local_counts = k222_counts[:]
        for ti in candidates:
            can_add = True
            for kid in triple_k222_list[ti]:
                if local_counts[kid] >= 7:
                    can_add = False
                    break
            if can_add:
                count += 1
                for kid in triple_k222_list[ti]:
                    local_counts[kid] += 1
        return count
    
    # B&B with time limit
    bnb_deadline = deadline - 5
    stack = [(list(range(num_triples)), set(), [0] * num_k222s, 0)]
    nodes_explored = 0
    
    while stack and time.time() < bnb_deadline:
        if time.time() > bnb_deadline:
            break
        
        candidates, selected, k222_counts, count = stack.pop()
        nodes_explored += 1
        
        if not candidates:
            if count > best_bnb:
                best_bnb = count
                best_bnb_set = set(selected)
            continue
        
        # Upper bound: count + len(candidates) is trivial
        # Better: count + greedy_upper_bound(candidates, k222_counts)
        ub = count + len(candidates)
        if ub <= best_bnb:
            continue
        
        # Pick a triple to branch on - choose one with high degree among remaining
        # to maximize pruning
        branch_idx = -1
        max_deg = -1
        for ci, ti in enumerate(candidates):
            deg = len(triple_k222_list[ti])
            if deg > max_deg:
                max_deg = deg
                branch_idx = ci
        
        if branch_idx == -1:
            continue
        
        ti = candidates[branch_idx]
        remaining_after = candidates[:branch_idx] + candidates[branch_idx+1:]
        
        # Branch 1: exclude ti
        stack.append((remaining_after, selected, k222_counts, count))
        
        # Branch 2: include ti (if valid)
        can_include = True
        for kid in triple_k222_list[ti]:
            if k222_counts[kid] >= 7:
                can_include = False
                break
        
        if can_include:
            new_counts = k222_counts[:]
            for kid in triple_k222_list[ti]:
                new_counts[kid] += 1
            new_selected = set(selected)
            new_selected.add(ti)
            stack.append((remaining_after, new_selected, new_counts, count + 1))
    
    # Use the best result found
    final_selected = best_bnb_set if best_bnb > best_count else best_selected
    final_count = max(best_bnb, best_count)
    
    if not final_selected:
        final_selected = set(range(num_triples))  # fallback shouldn't happen
    
    result = [list(all_triples[i]) for i in sorted(final_selected)]
    return result
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I made an error - I referenced `k222_count_cache` which doesn't exist. Let me fix this with a cleaner implementation.

