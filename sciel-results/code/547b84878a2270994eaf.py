def construct():
    import time
    import random
    from collections import defaultdict

    deadline = time.time() + 140.0

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        if not S:
            return True
        S_list = list(S)
        S_set = set(S_list)
        for i, a in enumerate(S_list):
            for j in range(i, len(S_list)):
                b = S_list[j]
                if a + b in S_set:
                    return False
        return True

    def build_conflict_graph(vals):
        """Build a graph where edges represent sum-free conflicts.
        Two elements a, b conflict if there exists c in vals such that a+b=c, a+c=b, or b+c=a.
        Actually, for sum-free, we need: no a+b=c where a,b,c are all in the subset.
        So conflicts are triples (a,b,c) where a+b=c.
        We model this as: for each element x, find all pairs (y,z) in vals such that y+z=x.
        If we include x, we cannot include both y and z.
        This is a 3-uniform hypergraph, not a graph.
        
        Alternative: Build a graph where we connect x to y if there exists z such that x+y=z or x+z=y or y+z=x.
        But that's not quite right either.
        
        Let's use a different approach: For each element x, compute its "conflict set" - the elements that cannot coexist with x.
        Actually, x can coexist with y as long as there's no z in the subset such that x+y=z or x+z=y or y+z=x.
        
        For the heuristic, let's use a greedy approach with conflict tracking.
        """
        vals_set = set(vals)
        # For each x, find all y such that x+y is in vals_set (then x and y conflict with x+y)
        # Also x-y in vals_set means x = y + (x-y), so x conflicts with y and x-y
        conflict_map = defaultdict(set)
        for x in vals:
            for y in vals:
                if x + y in vals_set:
                    # x + y = z, so {x, y, z} is a conflict triple
                    conflict_map[x].add(y)
                    conflict_map[x].add(x + y)
                    conflict_map[y].add(x)
                    conflict_map[y].add(x + y)
                    conflict_map[x + y].add(x)
                    conflict_map[x + y].add(y)
                if x - y in vals_set and x != y:
                    # x = y + (x-y), so {x, y, x-y} is a conflict triple
                    conflict_map[x].add(y)
                    conflict_map[x].add(x - y)
                    conflict_map[y].add(x)
                    conflict_map[y].add(x - y)
                    conflict_map[x - y].add(x)
                    conflict_map[x - y].add(y)
        return conflict_map

    def solve_instance(vals, time_limit=40.0):
        n = len(vals)
        vals_set = set(vals)
        start_time = time.time()
        
        # Build conflict structure
        conflict_map = build_conflict_graph(vals)
        
        # Strategy: Start with a large initial set, then iteratively remove elements that cause the most conflicts
        # to make room for adding more elements.
        
        def greedy_add(S, x, conflict_map):
            """Check if x can be added to S without violating sum-free."""
            if 2*x in S:
                return False
            for y in S:
                if x + y in S:
                    return False
                if x - y in S:
                    return False
            return True

        def conflict_degree(x, S):
            """Count how many elements in S conflict with x."""
            count = 0
            for y in S:
                if x + y in S or x - y in S or y - x in S:
                    count += 1
            return count

        # Initial candidates using different strategies
        candidates = []
        
        # Odds (provably sum-free)
        odds = [x for x in vals if x % 2 != 0]
        candidates.append(set(odds))
        
        # Positives
        pos = [x for x in vals if x > 0]
        candidates.append(set(pos))
        
        # Negatives
        neg = [x for x in vals if x < 0]
        candidates.append(set(neg))
        
        # Top half by value
        sorted_vals = sorted(vals, reverse=True)
        candidates.append(set(sorted_vals[:n//2 + 1]))
        
        # Bottom half by value
        candidates.append(set(sorted_vals[-(n//2 + 1):]))
        
        # Top half by absolute value
        sorted_abs = sorted(vals, key=lambda x: -abs(x))
        candidates.append(set(sorted_abs[:n//2 + 1]))
        
        # Bottom half by absolute value
        candidates.append(set(sorted_abs[-(n//2 + 1):]))
        
        # Greedy constructions with different orderings
        orderings = [
            sorted(vals, key=lambda x: -abs(x)),
            sorted(vals, key=lambda x: x),
            sorted(vals, key=lambda x: -x),
            sorted(vals, key=lambda x: abs(x)),
            sorted(vals, key=lambda x: random.Random(hash(x)).random()),
            sorted(vals, key=lambda x: random.Random(42 + hash(x)).random()),
        ]
        
        for ordering in orderings:
            S = set()
            for x in ordering:
                if greedy_add(S, x, conflict_map):
                    S.add(x)
            candidates.append(S)
        
        # New strategy: Conflict-based removal
        # Start with all elements, then remove elements to make it sum-free
        def conflict_based_remove(vals, conflict_map, time_limit, start_time):
            """Start with all elements, iteratively remove the element that causes the most conflicts."""
            S = set(vals)
            
            # Find all conflict triples
            conflict_triples = []
            seen_triples = set()
            for x in vals:
                for y in vals:
                    if x + y in vals_set:
                        triple = tuple(sorted([x, y, x + y]))
                        if triple not in seen_triples:
                            seen_triples.add(triple)
                            conflict_triples.append(triple)
            
            # Count conflicts per element
            conflict_count = defaultdict(int)
            for a, b, c in conflict_triples:
                conflict_count[a] += 1
                conflict_count[b] += 1
                conflict_count[c] += 1
            
            # Iteratively remove the element with highest conflict count
            while time.time() - start_time < time_limit:
                if not conflict_triples:
                    break
                
                # Find the element with highest conflict count in current S
                max_conflict = -1
                max_elem = None
                for elem in S:
                    if conflict_count[elem] > max_conflict:
                        max_conflict = conflict_count[elem]
                        max_elem = elem
                
                if max_elem is None or max_conflict <= 0:
                    break
                
                # Remove max_elem
                S.remove(max_elem)
                
                # Update conflict triples: remove all triples containing max_elem
                new_triples = []
                for triple in conflict_triples:
                    if max_elem not in triple:
                        new_triples.append(triple)
                    else:
                        # Update conflict counts for the other two elements
                        for other in triple:
                            if other != max_elem:
                                conflict_count[other] -= 1
                conflict_triples = new_triples
            
            return S

        # Add conflict-based removal result
        cb_result = conflict_based_remove(vals, conflict_map, time_limit * 0.3, start_time)
        if is_sum_free(cb_result):
            candidates.append(cb_result)
        
        # Improve each candidate using local search
        def improve_candidate(S, conflict_map, time_limit, start_time):
            """Improve S by trying to add elements and removing conflicts."""
            S = set(S)
            local_rng = random.Random(hash(frozenset(S)))
            
            # First, greedily add any possible elements
            improved = True
            while improved and time.time() - start_time < time_limit:
                improved = False
                for x in vals_set:
                    if x in S:
                        continue
                    if greedy_add(S, x, conflict_map):
                        S.add(x)
                        improved = True
                        break
            
            # Now, try swaps: remove y, add x
            for _ in range(500):
                if time.time() - start_time > time_limit:
                    break
                
                if not S:
                    break
                
                # Try a random swap
                y = local_rng.choice(list(S))
                x_candidates = [x for x in vals_set if x not in S]
                if not x_candidates:
                    break
                x = local_rng.choice(x_candidates)
                
                # Check if removing y and adding x would improve
                S_test = S - {y}
                if greedy_add(S_test, x, conflict_map):
                    S_test.add(x)
                    # Try to add more elements
                    added_any = False
                    for z in vals_set:
                        if z in S_test:
                            continue
                        if greedy_add(S_test, z, conflict_map):
                            S_test.add(z)
                            added_any = True
                    
                    if len(S_test) >= len(S):
                        S = S_test
            
            return S

        best_S = set()
        best_size = -1
        
        # Sort candidates by size, largest first
        candidates.sort(key=len, reverse=True)
        
        for S in candidates:
            if time.time() - start_time > time_limit:
                break
            
            if not is_sum_free(S):
                continue
            
            S_improved = improve_candidate(S, conflict_map, time_limit * 0.7, start_time)
            
            if not is_sum_free(S_improved):
                continue
                
            if len(S_improved) > best_size:
                best_size = len(S_improved)
                best_S = S_improved
        
        return best_S

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        
        if size == 60:
            t_lim = 20.0
        elif size == 120:
            t_lim = 40.0
        else:
            t_lim = 60.0
            
        S = solve_instance(vals, t_lim)
        
        if not is_sum_free(S):
            S = set(x for x in vals if x % 2 != 0)
        
        results.append(sorted(S))
        
        if time.time() > deadline:
            break

    while len(results) < 3:
        results.append([])

    return results[:3]
