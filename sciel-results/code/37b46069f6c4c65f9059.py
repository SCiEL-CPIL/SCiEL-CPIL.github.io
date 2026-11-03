"""
Strategy: Model K_3(6,1) as a Maximum Independent Set (MIS) problem on the distance-2 graph.
Nodes are words in {0,1,2}^6. Edges connect words at Hamming distance 1 or 2.
A valid covering code of radius 1 corresponds to a maximal independent set in this graph.
We use a greedy construction to get an initial MIS, then apply local search (1-swap, 2-swap)
to minimize the size of the set.
"""
import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: direct sum construction
        # For q=3, n=4, r=1, size 9 (tetracode)
        # For general n, extend by appending arbitrary symbols
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            # Pad to length n
            result = []
            for head in tetracode:
                for tail in itertools.product(range(3), repeat=n - 4):
                    result.append(head + list(tail))
            return result
        
        # Generic fallback: Greedy set cover
        start_time = time.time()
        deadline = 115.0
        total = q ** n
        # Map word to int
        def vec_to_num(vec):
            num = 0
            for i in range(len(vec)):
                num += vec[i] * (q ** i)
            return num
        
        # Precompute cover sets
        cover_sets = []
        for i in range(total):
            v = [0] * n
            num = i
            for j in range(n):
                v[j] = num % q
                num //= q
            
            s = [i]
            for pos in range(n):
                orig = v[pos]
                for new_val in range(q):
                    if new_val != orig:
                        v[pos] = new_val
                        s.append(vec_to_num(v))
                v[pos] = orig
            cover_sets.append(s)
            
        selected = set()
        uncovered = np.ones(total, dtype=bool)
        
        while np.any(uncovered) and time.time() < deadline - 5:
            best_c = -1
            max_gain = 0
            for c in range(total):
                if c in selected:
                    continue
                gain = np.sum(uncovered[cover_sets[c]])
                if gain > max_gain:
                    max_gain = gain
                    best_c = c
            if max_gain == 0:
                break
            selected.add(best_c)
            uncovered[cover_sets[best_c]] = False
            
        return [[(c // (q ** i)) % q for i in range(n)] for c in selected]

    # Specific case: q=3, n=6, radius=1
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    # Precompute adjacency for distance <= 2
    # Nodes: 0..728
    # adj[u] is a set of v such that dist(u, v) <= 2
    adj = [set() for _ in range(729)]
    
    def num_to_vec(num):
        return [(num // (3 ** i)) % 3 for i in range(6)]
    
    def vec_to_num(vec):
        return sum(v * (3 ** i) for i, v in enumerate(vec))
        
    # Build graph
    # Optimized: generate neighbors directly
    for u in range(729):
        vec_u = num_to_vec(u)
        # Distance 1
        for pos in range(6):
            orig = vec_u[pos]
            for new_val in range(3):
                if new_val != orig:
                    vec_u[pos] = new_val
                    v = vec_to_num(vec_u)
                    adj[u].add(v)
            vec_u[pos] = orig
            
        # Distance 2
        for p1 in range(6):
            for p2 in range(p1 + 1, 6):
                orig1 = vec_u[p1]
                orig2 = vec_u[p2]
                for v1 in range(3):
                    if v1 == orig1: continue
                    vec_u[p1] = v1
                    for v2 in range(3):
                        if v2 == orig2: continue
                        vec_u[p2] = v2
                        v = vec_to_num(vec_u)
                        adj[u].add(v)
                vec_u[p1] = orig1
            vec_u[p2] = orig2 # restore p2? No, p2 was not changed in outer loop of p1, but inside v2 loop it was.
            # Wait, the logic above modifies vec_u. Need to be careful.
            # Let's just re-derive or reset.
            # Actually, the loop structure:
            # for p1:
            #   for p2:
            #      save orig1, orig2
            #      modify p1, p2
            #      restore p1, p2 after inner loops?
            # My code above restores p1 after the v1 loop. But p2 is modified in the v2 loop.
            # After the v2 loop, p2 is still modified.
            # So `vec_u[p2] = orig2` is needed after the v2 loop finishes?
            # The code `vec_u[p2] = orig2` is outside the `for v2` loop but inside `for v1`? No.
            # Let's rewrite this part safely.
            
    # Re-do adjacency building safely
    adj = [set() for _ in range(729)]
    for u in range(729):
        vec_u = num_to_vec(u)
        # Dist 1
        for pos in range(6):
            orig = vec_u[pos]
            for new_val in range(3):
                if new_val != orig:
                    vec_u[pos] = new_val
                    v = vec_to_num(vec_u)
                    adj[u].add(v)
            vec_u[pos] = orig
            
        # Dist 2
        for p1 in range(6):
            orig1 = vec_u[p1]
            for p2 in range(p1 + 1, 6):
                orig2 = vec_u[p2]
                for v1 in range(3):
                    if v1 == orig1: continue
                    vec_u[p1] = v1
                    for v2 in range(3):
                        if v2 == orig2: continue
                        vec_u[p2] = v2
                        v = vec_to_num(vec_u)
                        adj[u].add(v)
                    vec_u[p2] = orig2
                vec_u[p1] = orig1
                
    # Convert to list of sorted neighbors for faster iteration?
    # Sets are fine for membership, lists for iteration.
    # We will use sets for the independent set check.
    
    best_solution = None
    best_size = 729 # Upper bound
    
    # Greedy MIS Construction
    # Randomize order to escape local minima
    for restart in range(20):
        if time.time() > deadline - 10:
            break
            
        candidates = list(range(729))
        random.shuffle(candidates)
        
        current_set = set()
        # For fast "is independent" check, we maintain the set.
        # When adding x, check if any neighbor of x is in current_set.
        
        for u in candidates:
            # Check if u is adjacent to any in current_set
            # Since current_set grows, we can just check intersection
            if adj[u].isdisjoint(current_set):
                current_set.add(u)
                
        # Local Search
        # State: current_set
        # Move: 1-swap (remove u, add v) if valid and size decreases? 
        # Wait, MIS size is fixed for a given maximal independent set? No.
        # We want to MINIMIZE the size of the independent set.
        # This is the "Minimum Maximal Independent Set" problem?
        # No, a covering code corresponds to a *maximal* independent set.
        # We want the *smallest* maximal independent set.
        
        # So, we start with a maximal IS.
        # We try to remove a node u. If we remove u, the set is no longer maximal.
        # We must add a node v to restore maximality.
        # We want to find a swap (u, v) such that the new set is a valid MIS and has smaller size?
        # Removing u and adding v keeps size same.
        # To reduce size, we need to remove u and NOT add v? But then it's not maximal.
        # Is it possible to have a smaller MIS? Yes.
        # How to find it?
        # Heuristic: If removing u leaves some nodes uncovered (not dominated by current_set \ {u}),
        # we must add one of them. If we can find a v that covers all those, size is same.
        # To reduce size, we need to remove u and find a v such that the new set is MIS and...
        # Actually, the size of a MIS is not directly optimizable by simple swaps if we strictly require maximality at each step.
        # However, the "independent set" part is the constraint (distance >= 3).
        # The "covering" part is maximality (every node is in set or adjacent to set).
        
        # Let's stick to the property:
        # 1. Independent: no two codewords within dist 2.
        # 2. Maximal: no other word can be added.
        
        # If we remove u, the set is still independent.
        # It might not be maximal.
        # If it IS maximal, we reduced size by 1!
        # So, first check: can we remove any u such that the remaining set is still maximal?
        # If yes, remove it.
        
        # Check maximality of S:
        # For every x not in S, is x adjacent to some y in S?
        # Equivalently: Does S dominate the graph?
        
        def is_maximal(S):
            # For each x not in S, check if adj[x] intersects S
            # Optimization: check if any node is NOT dominated
            # Dominated(x) = x in S OR (adj[x] & S) is not empty
            # We only care about x not in S.
            for x in range(729):
                if x in S:
                    continue
                if adj[x].isdisjoint(S):
                    return False
            return True
            
        # Local Search Loop
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try to remove a node
            # Iterate over nodes in S
            # Check if S \ {u} is maximal
            # This is expensive: O(|S| * 729 * |adj|)
            # |S| ~ 73, 729 nodes. 73 * 729 * ~100 = 5 million ops. Feasible.
            
            S_list = list(current_set)
            random.shuffle(S_list)
            
            for u in S_list:
                # Check if current_set - {u} is maximal
                new_S = current_set - {u}
                if is_maximal(new_S):
                    current_set = new_S
                    improved = True
                    if len(current_set) < best_size:
                        best_size = len(current_set)
                        best_solution = list(current_set)
                    break # Restart loop with new S
            
        # If stuck, try 1-swap to escape local minimum?
        # Or just accept this MIS.
        # Given time, simple deletion search is powerful.
        
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_solution = list(current_set)
            
    # Fallback if something went wrong
    if best_solution is None:
        # Use the direct sum construction (size 81)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        best_solution = [vec_to_num(h) for h in tetracode]
        # Wait, vec_to_num returns int. We need list of lists at the end.
        # Let's convert best_solution (list of ints) to list of lists.
        
    # Convert result to list of lists
    result = [num_to_vec(c) for c in best_solution]
    
    # Final verification (optional but good for safety)
    # Check independence
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            dist = sum(1 for k in range(6) if result[i][k] != result[j][k])
            if dist <= 2:
                # Invalid! Fallback to 81
                tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
                
    # Check coverage
    covered = [False] * 729
    for c in best_solution:
        # c is int
        vec_c = num_to_vec(c)
        # Mark c and neighbors
        covered[c] = True
        for pos in range(6):
            orig = vec_c[pos]
            for new_val in range(3):
                if new_val != orig:
                    vec_c[pos] = new_val
                    v = vec_to_num(vec_c)
                    covered[v] = True
            vec_c[pos] = orig
            
    if not all(covered):
        # Invalid! Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
        
    return result

# EVOLVE-BLOCK-END
