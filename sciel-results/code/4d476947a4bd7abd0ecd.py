"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def build_hyperedges(S):
        """Build 3-uniform hyperedges: (i,j,k) where S[i]+S[j]==S[k]."""
        S_set = set(S)
        n = len(S)
        idx_map = {x: i for i, x in enumerate(S)}
        hyperedges = set()
        
        for i in range(n):
            for j in range(i, n):  # a=b allowed
                s = S[i] + S[j]
                if s in S_set:
                    k = idx_map[s]
                    if k != i and k != j:  # distinct elements in triple
                        hyperedges.add(frozenset((i, j, k)))
                    # If k==i or k==j, that means S[i]+S[j]==S[i] or S[j],
                    # which implies S[j]==0 or S[i]==0, but 0 is excluded.
                    # So k is always distinct from i and j for nonzero values.
                    # Actually wait: if i==j, then 2*S[i]=S[k], k could equal i only if S[i]=0.
                    # Since values are nonzero, k != i and k != j always.
        return list(hyperedges)

    def can_add_fast(x, S_set, blocked_by):
        """Check if x can be added. blocked_by maps element -> set of elements that conflict with it."""
        if 2 * x in S_set:
            return False
        for a in S_set:
            if a + x in S_set:
                return False
            if x - a in S_set:
                return False
        return True

    def greedy_initial(S, order_key):
        """Greedy construction with given ordering."""
        S_sorted = sorted(S, key=order_key)
        result = []
        result_set = set()
        for x in S_sorted:
            if time.time() > DEADLINE - 5:
                break
            if can_add_fast(x, result_set, None):
                result.append(x)
                result_set.add(x)
        return result

    def local_search_efficient(S, initial_set, time_budget=15.0):
        """Efficient local search with conflict tracking."""
        current = set(initial_set)
        best = set(current)
        best_size = len(current)
        
        # Precompute all possible conflicts
        S_set = set(S)
        
        # For each element, precompute which other elements it conflicts with
        # conflict[i] = set of j such that S[i]+S[j] is in S
        # Actually, we need: if we have elements A and B, we can't have A+B.
        # So conflict is: for pair (a,b), if a+b in S, then not all three can be in set.
        
        # Better: for each element x, compute which elements y make x+y in S
        # and which elements z make z+x in S (i.e., z = (x+y)-x = y, so x+z in S means z+x in S)
        
        # Let's precompute: for each x in S, neighbors[x] = set of y in S where x+y in S or y+x in S or 2x in S
        # Actually simpler: for each x, bad_with[x] = set of y such that x+y in S_set
        # If x is in our set, then no y in bad_with[x] can also be in our set (if x+y in S)
        # AND no z such that z+x in S_set (which is same as x+z in S_set)
        
        # Let me re-think. The constraint is: for all a,b in T, a+b not in T.
        # Equivalently: for all a,b,c in T, a+b != c.
        # 
        # If I have current set T, and I want to add x:
        # - 2*x not in T
        # - For all a in T: a+x not in T and x-a not in T (since (x-a)+a = x, so if x-a in T and a in T, then (x-a)+a=x in T violates)
        #   Wait, that's not right. The constraint is no three elements a,b,c with a+b=c.
        #   If I add x, I need to check:
        #   - x+x not in T (i.e., 2x not in T)
        #   - For each a in T: a+x not in T (since a+x=x would need x in T, but we're checking a+x in T)
        #     Actually: if a in T and a+x in T, then a+(a+x) = 2a+x... no wait.
        #     The triple is (a, x, a+x): if a in T, x in T, and a+x in T, that's a violation.
        #     So: for each a in T, a+x must NOT be in T.
        #   - For each a in T: x-a must NOT be in T? No, that's not a constraint.
        #     The constraint is a+b=c. If we have a in T, x in T, and we check if x-a in T:
        #     (x-a) + a = x. So if (x-a) in T, a in T, and x in T, that's a violation.
        #     So yes: for each a in T, (x-a) must NOT be in T.
        #
        # So can_add(x, T):
        #   2*x not in T
        #   For all a in T: (a+x) not in T AND (x-a) not in T
        
        start_time = time.time()
        improved = True
        max_rounds = 50
        
        while improved and max_rounds > 0 and time.time() < start_time + time_budget:
            if time.time() > DEADLINE - 10:
                break
            improved = False
            max_rounds -= 1
            
            # Try adding each candidate
            for x in S:
                if x in current:
                    continue
                # Check if x can be added
                valid = True
                if 2 * x in current:
                    valid = False
                else:
                    for a in current:
                        if a + x in current or x - a in current:
                            valid = False
                            break
                if valid:
                    current.add(x)
                    improved = True
            
            # Try removing elements that block many additions
            for a in list(current):
                temp = set(current)
                temp.discard(a)
                # Count how many we could add
                addable = 0
                for x in S:
                    if x in temp:
                        continue
                    valid = True
                    if 2 * x in temp:
                        valid = False
                    else:
                        for b in temp:
                            if b + x in temp or x - b in temp:
                                valid = False
                                break
                    if valid:
                        addable += 1
                # If removing a lets us add more than 1, do it
                if addable > 1:
                    current = temp
                    improved = True
            
            if len(current) > best_size:
                best = set(current)
                best_size = len(current)
        
        return list(best)

    def solve_with_branch_and_bound(S, time_budget=30.0):
        """Branch and bound on the hypergraph for small instances."""
        n = len(S)
        if n > 100:
            return None
        
        hyperedges = build_hyperedges(S)
        
        # For each element, find which hyperedges contain it
        elem_to_hedges = {i: [] for i in range(n)}
        for he in hyperedges:
            for v in he:
                elem_to_hedges[v].append(he)
        
        best_solution = set()
        best_size = 0
        
        def search(selected, forbidden, remaining, depth=0):
            nonlocal best_solution, best_size
            
            if time.time() > DEADLINE - 15:
                return
            
            # Pruning: if even selecting all remaining can't beat best
            if len(selected) + len(remaining) <= best_size:
                return
            
            if not remaining:
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_solution = set(selected)
                return
            
            # Choose next element to branch on
            # Pick element with most constraints
            elem = max(remaining, key=lambda e: len(elem_to_hedges[e]))
            remaining.discard(elem)
            
            # Option 1: Include elem
            new_selected = set(selected)
            new_selected.add(elem)
            new_forbidden = set(forbidden)
            # Check if including elem creates violations with already selected
            valid = True
            for he in elem_to_hedges[elem]:
                others = [v for v in he if v != elem]
                if all(v in new_selected for v in others):
                    valid = False
                    break
                # If one other is in selected, the third must be forbidden
                for v in others:
                    if v in new_selected:
                        for w in others:
                            if w != v and w not in new_forbidden:
                                new_forbidden.add(w)
            if valid:
                search(new_selected, new_forbidden, set(remaining), depth+1)
            
            # Option 2: Exclude elem
            search(selected, forbidden, set(remaining), depth+1)
        
        search(set(), set(), set(range(n)))
        
        if best_solution:
            return [S[i] for i in best_solution]
        return None

    def solve_instance(S, time_budget=30.0):
        """Solve one instance with multiple strategies."""
        start = time.time()
        n = len(S)
        
        best = None
        best_size = 0
        
        # For small instances, try branch and bound
        if n <= 80 and time_budget > 20:
            bb_result = solve_with_branch_and_bound(S, time_budget=min(20, time_budget))
            if bb_result and len(bb_result) > best_size:
                best = bb_result
                best_size = len(bb_result)
        
        # Multiple greedy strategies
        strategies = [
            lambda x: x % 2 != 0,  # Odds
            lambda x: (-abs(x), x),  # Large |x| first
            lambda x: (abs(x), x),  # Small |x| first
            lambda x: (-x, abs(x)),  # Positive first
            lambda x: (x, abs(x)),  # Negative first
        ]
        
        for key_func in strategies:
            if time.time() > DEADLINE - 15:
                break
            
            initial = greedy_initial(S, key_func)
            
            if len(initial) > best_size:
                best = initial
                best_size = len(initial)
            
            # Local search improvement
            if time.time() < DEADLINE - 20:
                improved = local_search_efficient(S, initial, time_budget=min(10, DEADLINE - 20 - time.time()))
                if len(improved) > best_size:
                    best = improved
                    best_size = len(improved)
        
        # Random restarts
        rng = random.Random(123)
        for _ in range(3):
            if time.time() > DEADLINE - 15:
                break
            S_shuffled = S[:]
            rng.shuffle(S_shuffled)
            initial = []
            init_set = set()
            for x in S_shuffled:
                if can_add_fast(x, init_set, None):
                    initial.append(x)
                    init_set.add(x)
            
            if len(initial) > best_size:
                best = initial
                best_size = len(initial)
            
            if time.time() < DEADLINE - 20:
                improved = local_search_efficient(S, initial, time_budget=min(8, DEADLINE - 20 - time.time()))
                if len(improved) > best_size:
                    best = improved
                    best_size = len(improved)
        
        if best is None:
            best = [x for x in S if x % 2 != 0]
        
        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        
        # Allocate time based on instance size
        if size <= 100:
            budget = 25.0
        elif size <= 200:
            budget = 40.0
        else:
            budget = 55.0
        
        best = solve_instance(S, time_budget=budget)
        
        # Final verification
        if not is_sum_free(best):
            best = [x for x in S if x % 2 != 0]
        
        results.append(best)
    
    return results
# EVOLVE-BLOCK-END
