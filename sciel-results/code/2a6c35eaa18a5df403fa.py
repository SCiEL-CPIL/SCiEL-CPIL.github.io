import time
import random
from collections import defaultdict
import math

def construct():
    start_time = time.time()
    deadline = start_time + 82  # safety margin before 90s kill

    # Precompute all forbidden triples (a,b,c) with a<b<c, 1/a = 1/b + 1/c
    # Equivalent to: bc = a(b+c) => c = ab/(b-a)
    triples = []
    triple_map = defaultdict(set)  # element -> set of triples containing it

    for a in range(1, 181):
        for b in range(a + 1, 181):
            den = b - a
            if den == 0:
                continue
            num = a * b
            if num % den == 0:
                c = num // den
                if c > b and c <= 180:
                    triple = (a, b, c)
                    triples.append(triple)
                    triple_map[a].add(triple)
                    triple_map[b].add(triple)
                    triple_map[c].add(triple)

    # For branch-and-bound: precompute conflict structure
    # conflicts[x] = set of elements that conflict with x (i.e., there exists a triple containing both)
    conflicts = defaultdict(set)
    for (a, b, c) in triples:
        conflicts[a].add(b)
        conflicts[a].add(c)
        conflicts[b].add(a)
        conflicts[b].add(c)
        conflicts[c].add(a)
        conflicts[c].add(b)

    # For each element, precompute the pairs that would conflict
    # conflict_pairs[x] = list of (y, z) such that {x,y,z} is a forbidden triple
    conflict_pairs = defaultdict(list)
    for (a, b, c) in triples:
        conflict_pairs[a].append((b, c))
        conflict_pairs[b].append((a, c))
        conflict_pairs[c].append((a, b))

    # Precompute triple lookup: for each pair (x,y), which z values complete a triple
    pair_to_third = {}
    for (a, b, c) in triples:
        pair_to_third[(a, b)] = c
        pair_to_third[(a, c)] = b
        pair_to_third[(b, c)] = a

    def check_conflict(x, S):
        """Check if adding x to set S creates any forbidden triple."""
        for y, z in conflict_pairs[x]:
            if y in S and z in S:
                return True
        return False

    def can_add(x, S):
        """Check if x can be added to S without creating a forbidden triple."""
        return not check_conflict(x, S)

    def greedy_build(order, S_init=None):
        """Build a set greedily following the given order."""
        S = set(S_init) if S_init is not None else set()
        for x in order:
            if x in S:
                continue
            if not check_conflict(x, S):
                S.add(x)
        return S

    best_S = set()
    best_size = 0

    # Strategy 1: Start with large numbers, add small ones
    S1 = set(range(91, 181))
    candidates = list(range(1, 91))
    random.shuffle(candidates)
    for x in candidates:
        if not check_conflict(x, S1):
            S1.add(x)
    if len(S1) > best_size:
        best_size = len(S1)
        best_S = set(S1)

    # Strategy 1b: Start with large numbers, add small ones in descending order (deterministic)
    S1b = set(range(91, 181))
    for x in range(90, 0, -1):
        if not check_conflict(x, S1b):
            S1b.add(x)
    if len(S1b) > best_size:
        best_size = len(S1b)
        best_S = set(S1b)

    # Strategy 2: Pure greedy from all numbers in random order
    all_nums = list(range(1, 181))
    random.shuffle(all_nums)
    S2 = greedy_build(all_nums)
    if len(S2) > best_size:
        best_size = len(S2)
        best_S = set(S2)

    # Strategy 3: Greedy with small numbers first
    S3 = greedy_build(list(range(1, 181)))
    if len(S3) > best_size:
        best_size = len(S3)
        best_S = set(S3)

    # Strategy 4: Greedy with large numbers first
    S4 = greedy_build(list(range(180, 0, -1)))
    if len(S4) > best_size:
        best_size = len(S4)
        best_S = set(S4)

    # Strategy 5: Multiple random restarts
    for _ in range(8):
        if time.time() > deadline - 35:
            break
        nums = list(range(1, 181))
        random.shuffle(nums)
        S5 = greedy_build(nums)
        if len(S5) > best_size:
            best_size = len(S5)
            best_S = set(S5)

    # Strategy 6: Weighted greedy - prioritize elements with fewer conflicts
    if time.time() < deadline - 30:
        # Count how many triples each element participates in
        degree = {x: len(conflict_pairs[x]) for x in range(1, 181)}
        # Sort by degree ascending (fewer conflicts first)
        order = sorted(range(1, 181), key=lambda x: degree[x])
        S6 = greedy_build(order)
        if len(S6) > best_size:
            best_size = len(S6)
            best_S = set(S6)

    # Local search on best_S
    S = set(best_S)

    # Phase 1: Try to add any remaining elements
    remaining = [x for x in range(1, 181) if x not in S]
    for x in remaining:
        if not check_conflict(x, S):
            S.add(x)
    best_size = len(S)
    best_S = set(S)

    # Phase 2: Local search - remove one, add multiple
    while time.time() < deadline - 10:
        improved = False
        S_list = list(S)
        random.shuffle(S_list)

        for remove_elem in S_list:
            if time.time() > deadline - 10:
                break
            S.remove(remove_elem)

            # Try to add elements
            added = []
            for x in remaining:
                if x in S:
                    continue
                if not check_conflict(x, S):
                    S.add(x)
                    added.append(x)

            if len(added) >= 1 and len(S) > len(best_S):
                best_size = len(S)
                best_S = set(S)
                improved = True
                remaining = [x for x in range(1, 181) if x not in S]
            else:
                # Revert
                for x in added:
                    S.discard(x)
                S.add(remove_elem)

        if not improved:
            break

    # Phase 3: Pair removal - remove two, try to add three
    S = set(best_S)
    remaining = [x for x in range(1, 181) if x not in S]

    while time.time() < deadline - 5:
        improved = False
        S_list = list(S)
        random.shuffle(S_list)

        for i in range(len(S_list)):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, len(S_list)):
                if time.time() > deadline - 5:
                    break
                r1, r2 = S_list[i], S_list[j]
                S.remove(r1)
                S.remove(r2)

                added = []
                for x in remaining:
                    if x in S:
                        continue
                    if not check_conflict(x, S):
                        S.add(x)
                        added.append(x)

                if len(added) >= 3 and len(S) > len(best_S):
                    best_size = len(S)
                    best_S = set(S)
                    improved = True
                    remaining = [x for x in range(1, 181) if x not in S]
                else:
                    for x in added:
                        S.discard(x)
                    S.add(r1)
                    S.add(r2)

        if not improved:
            break

    # Phase 4: Branch and Bound for Maximum Independent Set in 3-uniform hypergraph
    # Use the greedy solution as lower bound, try to prove optimality or find better
    if time.time() < deadline - 10:
        BnB_best_size = best_size
        BnB_best_S = set(best_S)
        nodes_visited = 0
        max_nodes = 500000  # safety limit
        
        def compute_upper_bound(candidates_set, current_size):
            """Compute an upper bound on the size of any independent set extendable from current.
            Uses a greedy coloring-like bound adapted for 3-uniform hypergraphs.
            Simple bound: current_size + |candidates_set| (trivial)
            Better: use the fact that each triple can have at most 2 elements.
            """
            # Trivial bound
            bound = current_size + len(candidates_set)
            
            # Improved bound: count how many triples are "active" (have 2 elements in current)
            # Each such triple can contribute at most 1 more element from candidates
            # But this is complex; use a simpler packing bound
            # For now, use the trivial bound but with a greedy coloring heuristic
            # Color candidates so that same-colored elements can't all be in an independent set
            # For 3-uniform hypergraph, use a simple greedy coloring based on conflicts
            
            if not candidates_set:
                return bound
            
            # Greedy coloring: assign colors such that no triple has all 3 same color
            # This gives an upper bound via the maximum color class size
            color_map = {}
            next_color = 0
            cand_list = sorted(candidates_set, key=lambda x: len(conflicts[x] & candidates_set))
            
            for x in cand_list:
                # Find a color that doesn't create a monochromatic triple
                used_colors = set()
                for y, z in conflict_pairs[x]:
                    if y in candidates_set and z in candidates_set:
                        cy = color_map.get(y)
                        cz = color_map.get(z)
                        if cy is not None and cy == cz:
                            used_colors.add(cy)
                        elif cy is not None:
                            # y has color cy, z uncolored or different
                            pass
                        elif cz is not None:
                            pass
                
                # Also check pairs with already colored elements
                for y in conflicts[x]:
                    if y in candidates_set and y in color_map:
                        cy = color_map[y]
                        # Check if there's a z in candidates_set with same color as y
                        # such that (x,y,z) is a triple
                        for z, w in conflict_pairs[x]:
                            if w in candidates_set and w in color_map and color_map[w] == cy:
                                if (x, y, w) in triples or (x, w, y) in triples or (y, x, w) in triples:
                                    used_colors.add(cy)
                
                # Assign smallest available color
                c = 0
                while c in used_colors:
                    c += 1
                color_map[x] = c
                if c >= next_color:
                    next_color = c + 1
            
            # Upper bound: current_size + sum of min(size of color class, 2) 
            # Actually for 3-uniform, each color class can have at most... it's complex
            # Use: current_size + len(candidates_set) - (number of triples that would be violated)
            # Simpler: just use current_size + len(candidates_set) for now
            # But we can use a better bound: count maximum matching in the conflict graph
            # For now, return the trivial bound
            return bound
        
        def branch_and_bound(candidates_list, current_set, current_size, depth=0):
            """Branch and bound search for maximum independent set."""
            nonlocal BnB_best_size, BnB_best_S, nodes_visited
            
            if time.time() > deadline - 5:
                return
            if nodes_visited >= max_nodes:
                return
            
            nodes_visited += 1
            
            # Update best if current is better
            if current_size > BnB_best_size:
                BnB_best_size = current_size
                BnB_best_S = set(current_set)
            
            # Upper bound check
            if not candidates_list:
                return
            
            # Quick upper bound: current_size + len(candidates_list)
            if current_size + len(candidates_list) <= BnB_best_size:
                return
            
            # Choose a pivot element
            # Heuristic: pick element with most conflicts in candidates
            pivot_idx = 0
            max_conflicts = -1
            for i, x in enumerate(candidates_list):
                conf_count = len(conflicts[x] & current_set)
                if conf_count > max_conflicts:
                    max_conflicts = conf_count
                    pivot_idx = i
            
            pivot = candidates_list[pivot_idx]
            
            # Branch 1: Include pivot
            new_set = set(current_set)
            new_set.add(pivot)
            new_candidates = []
            for x in candidates_list:
                if x == pivot:
                    continue
                if can_add(x, new_set):
                    new_candidates.append(x)
            
            # Filter new_candidates to maintain consistency
            # Remove elements that conflict with each other via pivot
            # Actually, we need to be more careful: if we add pivot, some triples may be completed
            # The remaining candidates must not form a triple with any two elements in new_set
            
            branch_and_bound(new_candidates, new_set, current_size + 1, depth + 1)
            
            # Branch 2: Exclude pivot
            # When excluding pivot, we may still be able to add other elements
            # But we need to be careful about the structure
            remaining_candidates = candidates_list[:pivot_idx] + candidates_list[pivot_idx+1:]
            branch_and_bound(remaining_candidates, current_set, current_size, depth + 1)
        
        # Run BnB from the best greedy solution
        # Start with empty set, but use the greedy solution to guide the search
        # Better: start with the greedy solution's elements as "included" and search for additions
        # Actually, let's do a full BnB but with good ordering
        
        # Use degree-based ordering: elements with fewer conflicts first
        degree = {x: len(conflicts[x]) for x in range(1, 181)}
        initial_order = sorted(range(1, 181), key=lambda x: degree[x])
        
        branch_and_bound(initial_order, set(), 0)
        
        if BnB_best_size > best_size:
            best_size = BnB_best_size
            best_S = set(BnB_best_S)

    # Final verification: ensure no forbidden triples
    def verify_set(S):
        for (a, b, c) in triples:
            if a in S and b in S and c in S:
                return False
        return True
    
    if not verify_set(best_S):
        # Fallback to a known valid set
        result = list(range(91, 181))
        if verify_set(set(result)):
            return result
        else:
            # This shouldn't happen, but be safe
            return [180]
    
    result = list(best_S)
    if not result:
        result = list(range(91, 181))

    return result
