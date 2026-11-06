# EVOLVE-BLOCK-START
import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 16
    # 1. Generate all C(16,3) = 560 possible triples
    all_triples = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                all_triples.append((a, b, c))
    
    N = len(all_triples)
    
    # 2. Precompute conflict graph
    # Two triples conflict if they are two of the 8 transversals of the same K(2,2,2) configuration.
    # i.e., there exists a set of 3 disjoint pairs such that both triples are transversals.
    # Efficient check:
    # Let t1 = {a,b,c}, t2 = {d,e,f}.
    # They conflict iff there is a permutation of t1 and t2 such that they can be embedded in pairs.
    # Specifically, t1 and t2 are transversals of the same pair-triple if and only if:
    # There exist x,y,z such that pairs are ({x1, y1}, {x2, y2}, {x3, y3}) where 
    # {x1, x2, x3} = t1 and {y1, y2, y3} = t2? No, that's only if they are disjoint.
    # 
    # General condition:
    # t1 and t2 are transversals of the same 3 disjoint pairs if and only if 
    # the union of t1 and t2 can be partitioned into 3 pairs, each containing one element from t1 and one from t2? 
    # Not exactly. The pairs are (p1, p2, p3). t1 picks one from each. t2 picks one from each.
    # So for each pair pi, t1 has one element, t2 has one element.
    # This implies that the symmetric difference of t1 and t2 is partitioned into pairs?
    # Let's use the explicit check derived earlier:
    # t1={a,b,c}, t2={d,e,f}.
    # They conflict if there is a bijection between the 3 elements of t1 and the 3 pairs, and t2 similarly,
    # such that the "partners" in the pairs are consistent.
    # Actually, simpler:
    # t1 and t2 conflict if and only if there exists a permutation of t2, say d',e',f' such that
    # the set of pairs {(a, x), (b, y), (c, z)} can be formed where {x,y,z} is a permutation of t2?
    # No. The pairs are disjoint.
    # Let's stick to the robust check:
    # t1 and t2 conflict iff there exist 3 disjoint pairs P1, P2, P3 such that 
    # |t1 ∩ Pi| = 1 and |t2 ∩ Pi| = 1 for all i.
    
    # Optimized check for two triples t1, t2:
    # If t1 == t2, no conflict (same triple).
    # If t1 and t2 are disjoint (intersection 0):
    #   They conflict if we can pair up elements of t1 with elements of t2 to form 3 pairs.
    #   This is always possible? No, the pairs must be disjoint. Since t1 and t2 are disjoint, any pairing works.
    #   Wait, if t1={1,2,3} and t2={4,5,6}, pairs (1,4),(2,5),(3,6) are disjoint. Yes.
    #   So ANY two disjoint triples conflict?
    #   Let's verify. Pairs: (1,4), (2,5), (3,6). Transversals include {1,2,3} and {4,5,6}. Yes.
    #   So all disjoint triples conflict? That would mean the independence number is small.
    #   But wait, the problem says "no three disjoint pairs have ALL EIGHT transversal triples present".
    #   If we have {1,2,3} and {4,5,6}, do we have a conflict?
    #   A conflict is defined as "all 8 transversals present".
    #   Having just 2 transversals is NOT a conflict.
    #   The conflict graph for Independent Set usually represents "cannot be in the same independent set".
    #   Here, "cannot be in the same set" means "their presence together completes a K(2,2,2) violation"?
    #   No, the violation is global: "no three disjoint pairs have all eight...".
    #   This is a hypergraph independent set problem (3-uniform? No, 8-uniform hyperedges).
    #   Standard MIS on a graph is not directly applicable because the constraint is on sets of 8.
    #   
    #   However, the current seed program checks "if adding this triple completes a K(2,2,2)".
    #   This implies a greedy approach that maintains a valid set.
    #   To use Branch and Bound, we need a binary constraint.
    #   Is it possible to reduce this to a graph?
    #   If we select a set S, it is valid if for every K(2,2,2) configuration C, |S ∩ C| < 8.
    #   This is a "partial" constraint. It's not a simple graph independent set.
    #   
    #   BUT, we can formulate it as a 0-1 Integer Program or use a specific B&B for this hypergraph.
    #   Given the small size (560 vars), we can use a recursive B&B that branches on including/excluding triples,
    #   and uses a pruning rule: if the current count of selected triples in any K(2,2,2) is 7,
    #   we cannot add the 8th.
    #   
    #   Upper Bound for B&B:
    #   We need an upper bound on how many MORE triples we can add.
    #   A simple bound: remaining candidates.
    #   A tighter bound: For each remaining candidate t, if adding it would violate any K(2,2,2) (i.e., if it is the 8th for some K), it is forbidden.
    #   If we just take the count of "currently allowed" triples, that's a valid upper bound (since we can't add forbidden ones).
    #   Even tighter: Greedy coloring of the "conflict graph" of remaining allowed nodes?
    #   Actually, since the constraint is "at most 7 per K", it's a packing problem.
    #   
    #   Let's refine the B&B strategy:
    #   State: (current_set, candidate_list, counts_for_each_K)
    #   This state space is huge.
    #   
    #   Alternative: Since the seed program gets 2.36 (approx 315-320 triples?), and the max is 560,
    #   and the constraint is "sparse" (each triple is in ~10k K's, but only a few are "full"),
    #   maybe we can use a different relaxation.
    #   
    #   Let's look at the "Min-Overlap" insight.
    #   The seed program is greedy.
    #   The directive asks for a " MATERIALLY DIFFERENT strategy".
    #   The prompt suggests: "Formulate ... as maximum weight independent set ... branch-and-bound".
    #   But as analyzed, it's not a standard MIS.
    #   
    #   However, we can approximate it.
    #   If we assume that most conflicts are local, we can use a heuristic B&B.
    #   
    #   Let's try a different angle: **Local Search with Simulated Annealing** on the validity score.
    #   Objective: Maximize number of triples.
    #   Penalty: Number of violated K(2,2,2)s.
    #   Start with a random set, maybe slightly larger than valid, or use the greedy result as start.
    #   Moves: Add a random triple, Remove a random triple, Swap (add one, remove one).
    #   Energy function: E = - (num_triples) + Lambda * (num_violations).
    #   If we can keep violations at 0, we maximize triples.
    #   
    #   This is a standard approach for constraint satisfaction/optimization.
    #   It allows "jumping" over local optima of the greedy approach.
    #   
    #   Implementation details:
    #   1. Efficiently maintain violation counts.
    #      There are 120,120 K(2,2,2) configurations.
    #      Each triple belongs to 10,296 K's.
    #      Storing counts for all K's is feasible (120k integers).
    #      Updating counts when adding/removing a triple: O(10296).
    #      This is fast enough for thousands of iterations.
    #      
    #   2. Initial solution: Use the greedy result from the seed (or a faster version of it) to get a valid starting point.
    #      Then run SA to try to add more or swap for better density?
    #      Actually, if we start valid, we can only add triples that don't create violations.
    #      SA can explore the space of valid sets.
    #      
    #   3. Better: Start with a set that might have violations, and anneal to remove them while keeping size high.
    #      Or start with the greedy valid set, and try to "swap" triples to free up capacity in dense K's.
    #      
    #   Let's implement Simulated Annealing.
    
    # Precompute K-memberships for each triple
    # k_memberships[t_index] = list of K_indices
    
    # First, enumerate all K(2,2,2) configurations (pair-triples)
    # A K is defined by 6 distinct points partitioned into 3 pairs.
    # Total Ks = 120,120.
    
    # We need a mapping from triple to list of Ks it belongs to.
    # And for each K, the list of 8 transversals (triple indices).
    
    # This precomputation might be heavy but is one-time.
    
    # Let's generate all Ks and their transversals.
    # To save memory/time, we can generate them on the fly or store compactly.
    # 120,120 Ks * 8 transversals = ~1M entries. Fine.
    
    # Step 2a: Map triple tuple to index
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Step 2b: Generate all Ks
    # Iterate over all 6-subsets of 16 points.
    # For each 6-subset, iterate over all 15 pairings.
    # For each pairing, determine the 8 transversals.
    
    # This is C(16,6) * 15 = 8008 * 15 = 120,120 iterations.
    # Inside, we do constant work.
    # This should take < 1 second in Python? Maybe 5-10s.
    # Let's optimize if possible.
    
    # Actually, we can just store for each triple the list of K-indices.
    # triple_k_list = [[] for _ in range(N)]
    
    # And for each K, we don't really need to store the transversals explicitly if we just update counts.
    # When we add triple t, we increment count[k] for all k in triple_k_list[t].
    # A violation occurs if count[k] == 8.
    # We need to track the number of violations (Ks with count 8).
    
    # Precomputation:
    # 1. Generate all Ks.
    # 2. For each K, find its 8 transversals.
    # 3. Add K index to triple_k_list for each of the 8 transversals.
    
    # Let's do this.
    
    # To speed up, we can use numpy? No, standard library is fine for 120k items.
    
    # We need to iterate over 6-subsets.
    # from itertools import combinations
    # for pts in combinations(range(16), 6):
    #    for pairing in get_pairings(pts):
    #       ...
    
    # How to get 15 pairings of 6 items?
    # Fix item 0. Pair it with 1, 2, 3, 4, or 5.
    # If paired with 1, remaining are 2,3,4,5. Pair 2 with 3,4,5.
    # 3 * 3 * 1 = 9? No.
    # Number of perfect matchings in K6 is 15.
    # Algorithm:
    # def get_pairings(points):
    #    # points is a tuple of 6
    #    p0 = points[0]
    #    rest = points[1:]
    #    for i in range(5):
    #       p1 = rest[i]
    #       pair1 = (p0, p1)
    #       remaining = [x for j, x in enumerate(rest) if j != i]
    #       # Now pair up remaining 4 points.
    #       # 3 ways.
    #       r0, r1, r2, r3 = remaining
    #       # (r0,r1),(r2,r3) ; (r0,r2),(r1,r3) ; (r0,r3),(r1,r2)
    #       yield (pair1, (r0,r1), (r2,r3))
    #       yield (pair1, (r0,r2), (r1,r3))
    #       yield (pair1, (r0,r3), (r1,r2))
    
    # This yields 15 pairings.
    
    # Now, for a given pairing (p1, p2, p3), the 8 transversals are:
    # (a,b,c) where a in p1, b in p2, c in p3.
    # Sort each triple to match the canonical form in all_triples.
    
    # Let's implement this precomputation.
    
    triple_k_list = [[] for _ in range(N)]
    
    # We don't actually need to store the Ks themselves, just the counts.
    # But we need to know which Ks exist to initialize counts to 0.
    # num_ks = 0
    
    # Optimization: We can just build the triple_k_list.
    # We don't need to store the Ks in a list, just increment num_ks.
    
    # However, for SA, we need to update counts.
    # counts = [0] * num_ks
    
    # Let's build it.
    
    # Note: This precomputation might take a few seconds.
    # If it's too slow, we can skip and use a slower check on the fly, but that defeats the purpose of SA speed.
    # Let's try to make it fast.
    
    # Actually, we can avoid generating all Ks if we use a different conflict check for SA?
    # No, SA needs fast updates.
    
    # Let's run the precomputation.
    
    # Time check
    if time.time() > deadline - 60:
        # Fallback to greedy if precomputation is too slow?
        # Unlikely, 120k iterations is fast.
        pass
        
    for pts in combinations(range(16), 6):
        # Get 15 pairings
        p0 = pts[0]
        rest = pts[1:]
        for i in range(5):
            p1 = rest[i]
            pair1 = (p0, p1)
            remaining = [x for j, x in enumerate(rest) if j != i]
            r0, r1, r2, r3 = remaining
            pairings = [
                (pair1, (r0, r1), (r2, r3)),
                (pair1, (r0, r2), (r1, r3)),
                (pair1, (r0, r3), (r1, r2))
            ]
            for pairing in pairings:
                # pairing is 3 pairs
                # Generate 8 transversals
                # p1=(a,a'), p2=(b,b'), p3=(c,c')
                # Transversals: (a,b,c), (a,b,c'), (a,b',c), (a,b',c'), (a',b,c), (a',b,c'), (a',b',c), (a',b',c')
                a, a2 = pairing[0]
                b, b2 = pairing[1]
                c, c2 = pairing[2]
                
                transversals = [
                    (a, b, c), (a, b, c2), (a, b2, c), (a, b2, c2),
                    (a2, b, c), (a2, b, c2), (a2, b2, c), (a2, b2, c2)
                ]
                
                # Sort each to match canonical form
                # And get index
                for t in transversals:
                    t_canon = tuple(sorted(t))
                    idx = triple_to_idx[t_canon]
                    # We need a unique ID for this K to update counts.
                    # We can just use a counter.
                    # But we need to associate the K ID with the triple.
                    # So we need to know which K ID this is.
                    # We can just increment a global K counter for each pairing.
                    pass
    
    # Wait, the above loop doesn't assign K IDs correctly to the triple_k_list.
    # I need a global counter for Ks.
    
    triple_k_list = [[] for _ in range(N)]
    k_counter = 0
    
    for pts in combinations(range(16), 6):
        p0 = pts[0]
        rest = pts[1:]
        for i in range(5):
            p1 = rest[i]
            pair1 = (p0, p1)
            remaining = [x for j, x in enumerate(rest) if j != i]
            r0, r1, r2, r3 = remaining
            pairings = [
                (pair1, (r0, r1), (r2, r3)),
                (pair1, (r0, r2), (r1, r3)),
                (pair1, (r0, r3), (r1, r2))
            ]
            for pairing in pairings:
                k_id = k_counter
                k_counter += 1
                
                a, a2 = pairing[0]
                b, b2 = pairing[1]
                c, c2 = pairing[2]
                
                transversals = [
                    (a, b, c), (a, b, c2), (a, b2, c), (a, b2, c2),
                    (a2, b, c), (a2, b, c2), (a2, b2, c), (a2, b2, c2)
                ]
                
                for t in transversals:
                    t_canon = tuple(sorted(t))
                    idx = triple_to_idx[t_canon]
                    triple_k_list[idx].append(k_id)
    
    num_ks = k_counter
    
    # Now we have triple_k_list.
    # We can run Simulated Annealing.
    
    # Initial solution: Greedy
    # We'll run a fast greedy to get a starting valid set.
    # Then SA tries to improve.
    
    def run_greedy(seed_val):
        random.seed(seed_val)
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        selected = []
        selected_set = set()
        # Use the fast conflict check from seed?
        # Or just use the counts?
        # For greedy, we can use the counts to check if adding t makes any count reach 8.
        # But we need to initialize counts to 0.
        
        counts = [0] * num_ks
        num_violations = 0
        
        for t in shuffled:
            t_idx = triple_to_idx[t]
            # Check if adding t creates a violation
            # It creates a violation if any k in triple_k_list[t_idx] has count[k] == 7
            # (because it would become 8)
            # Actually, if count[k] is already 8, it's already a violation.
            # We want to maintain 0 violations.
            # So we can only add t if for all k in triple_k_list[t_idx], count[k] < 7.
            
            can_add = True
            for k in triple_k_list[t_idx]:
                if counts[k] >= 7:
                    can_add = False
                    break
            
            if can_add:
                selected.append(t)
                selected_set.add(t)
                for k in triple_k_list[t_idx]:
                    if counts[k] == 7:
                        num_violations += 1 # Not really, it becomes 8, which is a violation
                    counts[k] += 1
        
        return selected, selected_set, counts, num_violations

    # Run a few greedy restarts to find a good start
    best_start = []
    best_start_set = set()
    best_start_counts = None
    best_start_violations = 0
    
    for i in range(20):
        if time.time() > deadline - 20:
            break
        sel, sset, cnts, viols = run_greedy(i)
        if len(sel) > len(best_start):
            best_start = sel
            best_start_set = sset
            best_start_counts = cnts
            best_start_violations = viols
            
    # Now SA
    # State: selected_set, counts, num_violations
    # We want to minimize violations and maximize size.
    # Actually, we want to maximize size subject to violations == 0.
    # If we start with a valid set (violations=0), we can only add triples that don't create violations.
    # If we get stuck, we can remove a triple to allow adding others?
    # Or we can allow violations temporarily and anneal them out.
    
    # Let's use an energy function: E = -size + L * violations
    # If L is large, it will try to remove violations first.
    # But we want to KEEP size high.
    # Maybe: E = -size + L * violations^2 ?
    
    # Better: 
    # If violations > 0, we MUST remove them.
    # If violations == 0, we try to add more.
    
    # Strategy:
    # 1. Start with valid set.
    # 2. Try to add random triples. If it creates a violation, reject (or accept with prob based on T).
    # 3. If we can't add any, try to swap: remove one, add one.
    
    # Let's implement a simple SA that allows temporary violations.
    
    current_sel = best_start[:]
    current_set = set(best_start)
    current_counts = best_start_counts[:]
    current_violations = 0 # Should be 0
    
    # Recalculate violations to be sure
    for k in range(num_ks):
        if current_counts[k] >= 8:
            current_violations += 1
            
    best_size = len(current_sel)
    best_solution = current_sel[:]
    
    T = 10.0
    cooling = 0.999
    
    # We need to be able to add/remove triples quickly.
    
    def add_triple(t_idx):
        nonlocal current_counts, current_violations
        for k in triple_k_list[t_idx]:
            if current_counts[k] == 7:
                current_violations += 1
            current_counts[k] += 1
        current_set.add(all_triples[t_idx])
        current_sel.append(all_triples[t_idx])
        
    def remove_triple(t_idx):
        nonlocal current_counts, current_violations
        for k in triple_k_list[t_idx]:
            if current_counts[k] == 8:
                current_violations -= 1
            current_counts[k] -= 1
        t = all_triples[t_idx]
        current_set.discard(t)
        # Remove from list? List removal is O(N).
        # For SA, we can just keep a set for membership and a list for iteration?
        # Or just use the set for membership and calculate size from set.
        # Let's use a set for current_sel as well to make size check O(1).
        # But we need to return a list at the end.
        pass
        
    # Refactor to use sets for current state to make add/remove O(1) for membership.
    # But we need to update counts.
    
    # Let's restart SA state management.
    
    cur_set = set(best_start)
    cur_counts = best_start_counts[:]
    cur_viols = 0
    for k in range(num_ks):
        if cur_counts[k] >= 8:
            cur_viols += 1
            
    best_size = len(cur_set)
    best_sol = list(cur_set)
    
    # Precompute list of all triple indices for random selection
    all_indices = list(range(N))
    
    # SA Loop
    while time.time() < deadline - 5:
        # Pick a random move
        # 1. Add a random triple not in set
        # 2. Remove a random triple from set
        # 3. Swap
        
        move_type = random.random()
        
        if move_type < 0.5:
            # Add
            # Pick random t not in cur_set
            # To avoid O(N) check, we can just pick random index and check membership
            t_idx = random.choice(all_indices)
            if t_idx in [triple_to_idx[t] for t in cur_set]: # Slow
                continue
            # Faster: check if all_triples[t_idx] in cur_set
            if all_triples[t_idx] in cur_set:
                continue
                
            # Calculate delta
            delta_viols = 0
            for k in triple_k_list[t_idx]:
                if cur_counts[k] == 7:
                    delta_viols += 1
            # Delta size = 1
            # Delta E = -1 + L * delta_viols
            # If delta_viols > 0, E increases (bad).
            # Accept if delta_E < 0 or rand < exp(-delta_E/T)
            
            L = 100.0
            delta_E = -1.0 + L * delta_viols
            
            if delta_E < 0 or random.random() < pow(2.71828, -delta_E / T):
                # Accept
                cur_set.add(all_triples[t_idx])
                for k in triple_k_list[t_idx]:
                    if cur_counts[k] == 7:
                        cur_viols += 1
                    cur_counts[k] += 1
                
                if cur_viols == 0 and len(cur_set) > best_size:
                    best_size = len(cur_set)
                    best_sol = list(cur_set)
                    
        elif move_type < 0.8:
            # Remove
            if not cur_set:
                continue
            t = random.choice(list(cur_set))
            t_idx = triple_to_idx[t]
            
            delta_viols = 0
            for k in triple_k_list[t_idx]:
                if cur_counts[k] == 8:
                    delta_viols -= 1 # Violation goes away
            
            # Delta size = -1
            # Delta E = 1 + L * delta_viols (delta_viols is negative)
            
            L = 100.0
            delta_E = 1.0 + L * delta_viols
            
            if delta_E < 0 or random.random() < pow(2.71828, -delta_E / T):
                # Accept
                cur_set.discard(t)
                for k in triple_k_list[t_idx]:
                    if cur_counts[k] == 8:
                        cur_viols -= 1
                    cur_counts[k] -= 1
                    
        else:
            # Swap
            if not cur_set:
                continue
            t_old = random.choice(list(cur_set))
            t_old_idx = triple_to_idx[t_old]
            
            # Pick t_new not in set
            t_new_idx = random.choice(all_indices)
            if all_triples[t_new_idx] in cur_set:
                continue
            t_new = all_triples[t_new_idx]
            
            # Calculate delta
            # Remove t_old, Add t_new
            delta_viols = 0
            # Effect of removing t_old
            for k in triple_k_list[t_old_idx]:
                if cur_counts[k] == 8:
                    delta_viols -= 1
                # Note: if count was 8, it becomes 7.
                # If count was 7, it becomes 6.
                
            # Effect of adding t_new
            # We must consider the counts AFTER removal?
            # Yes, sequential.
            # But for delta calculation, we can do it in one go if we track the net change.
            # Let's simulate the counts change.
            
            # Actually, simpler:
            # Calculate delta_viols for removal
            dv_rem = 0
            for k in triple_k_list[t_old_idx]:
                if cur_counts[k] == 8:
                    dv_rem -= 1
            
            # Calculate delta_viols for addition (on the modified counts)
            # We need to know if any k in t_new has count == 7 AFTER removal.
            # This is complex.
            # Let's just accept swap if it doesn't increase violations, or with prob.
            
            # Simplified: Only allow swap if it keeps violations at 0 or reduces them.
            # If cur_viols == 0, we only swap if delta_viols == 0.
            # If cur_viols > 0, we swap if delta_viols < 0.
            
            # For now, skip swap if it's complex, or just do it if it looks good.
            # Let's just try to add t_new to a temporary copy? Too slow.
            
            # Let's approximate:
            # If t_old and t_new share no Ks, delta is independent.
            # If they share Ks, it's coupled.
            
            # Given time constraints, let's stick to Add/Remove for now.
            pass
            
        T *= cooling
        
    # Return best valid solution
    # Ensure best_sol is valid
    # It should be, as we only updated best_sol when cur_viols == 0.
    
    return [list(t) for t in best_sol]

# EVOLVE-BLOCK-END
