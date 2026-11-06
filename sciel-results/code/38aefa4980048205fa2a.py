"""Sum-free subset maximization via maximum independent set on conflict graph."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        """Check if set S is sum-free: no a+b=c for a,b,c in S (a=b allowed)."""
        S_set = set(S)
        for a in S_set:
            for b in S_set:
                if a + b in S_set:
                    return False
        return True

    def build_conflict_graph(elements):
        """
        Build a conflict graph where an edge between i and j means
        elements[i] and elements[j] cannot both be in a sum-free set.
        
        Two elements x, y conflict if:
        - x + y is in elements (then x+y = z for some z, and if x,y,z all in S, violation)
          Wait, that's not quite right. The condition is: no a,b,c in S with a+b=c.
          So if we have x and y in S, we need x+y NOT in S.
          Similarly, if we have x and y in S, we need y-x NOT in S (since x + (y-x) = y, 
          but that requires y-x to be in S, which is a different condition).
          
          Actually, let me think again. The conflict graph for maximum independent set
          approach: two vertices i, j are adjacent if elements[i] and elements[j] 
          CANNOT both be in a sum-free set.
          
          When can x and y NOT both be in a sum-free set?
          - If x + y is in the element set, then having x, y, and x+y would be a violation.
            But we're just deciding whether to include x and y; x+y might or might not be included.
            So x and y alone don't conflict unless x+y is forced to be included.
            
          Hmm, this is a 3-uniform hypergraph problem, not a 2-uniform graph.
          The conflict is between triples (a, b, c) where a+b=c.
          
          For the maximum independent set approach on a graph, we need to reduce this.
          One approach: create a graph where vertices are elements, and add an edge between
          x and y if including both would force a conflict. But that's not straightforward.
          
          Alternative: Use a different formulation. Let's use a greedy + local search 
          approach that's more effective for this type of problem.
        """
        n = len(elements)
        # Create a set for fast lookup
        elem_set = set(elements)
        
        # For each pair (i, j), check if elements[i] + elements[j] is in elem_set
        # If so, then we cannot have all three of elements[i], elements[j], elements[i]+elements[j]
        # This is a 3-uniform hyperedge.
        
        # For a simpler approach, let's use a greedy algorithm with smart ordering
        # and then local search.
        
        return None  # Not using graph approach directly

    def greedy_build(elements, order):
        """Greedy: add elements in given order if it keeps the set sum-free."""
        S = []
        S_set = set()
        for x in order:
            ok = True
            if 2 * x in S_set:
                ok = False
            if ok:
                for a in S:
                    if a + x in S_set:
                        ok = False
                        break
            if ok:
                S.append(x)
                S_set.add(x)
        return S

    def can_add(x, S_set):
        """Check if x can be added to S_set while keeping it sum-free."""
        if 2 * x in S_set:
            return False
        for a in S_set:
            if a + x in S_set:
                return False
        return True

    def local_search(S, elements, max_iters=50000, timeout=10.0):
        """Try to improve S by swapping out one element and in another."""
        S_set = set(S)
        S_list = list(S)
        available = [x for x in elements if x not in S_set]
        best_size = len(S_list)
        no_improve_count = 0
        t0 = time.time()
        rng = random.Random(12345)

        while max_iters > 0 and no_improve_count < 500:
            max_iters -= 1
            if time.time() - t0 > timeout:
                break
            
            # Try random swap: remove r from S, add a from available
            if not S_list or not available:
                break
                
            r_idx = rng.randrange(len(S_list))
            r = S_list[r_idx]
            a = rng.choice(available)
            if a == r:
                continue
            
            # Tentative: remove r, add a
            S_set.remove(r)
            S_list.pop(r_idx)
            
            # Check if adding a keeps sum-free
            if can_add(a, S_set):
                # Add a
                S_list.append(a)
                S_set.add(a)
                no_improve_count = 0
                # Update available
                if a in available:
                    available.remove(a)
                available.append(r)
            else:
                # Revert
                S_list.append(r)
                S_set.add(r)
                no_improve_count += 1

        return S_list

    def local_search2(S, elements, max_iters=50000, timeout=10.0):
        """More aggressive local search: try all possible single swaps."""
        S_set = set(S)
        S_list = list(S)
        available = [x for x in elements if x not in S_set]
        best_size = len(S_list)
        no_improve_count = 0
        t0 = time.time()

        # First, try to add any available element
        for a in available:
            if time.time() - t0 > timeout:
                break
            if can_add(a, S_set):
                S_list.append(a)
                S_set.add(a)
                available.remove(a)
                no_improve_count = 0

        # Then try swaps
        for _ in range(max_iters):
            if time.time() - t0 > timeout:
                break
            if no_improve_count > 200:
                break
            if not S_list or not available:
                break
                
            improved = False
            for r_idx, r in enumerate(S_list):
                for a in available:
                    if a == r:
                        continue
                    # Tentative: remove r, add a
                    S_set.remove(r)
                    if can_add(a, S_set):
                        # Add a
                        S_list.pop(r_idx)
                        S_list.append(a)
                        S_set.add(a)
                        available.remove(a)
                        available.append(r)
                        no_improve_count = 0
                        improved = True
                        break
                    else:
                        S_set.add(r)
                if improved:
                    break
            if not improved:
                no_improve_count += 1

        return S_list

    def solve_instance(elements, time_budget=30.0):
        """Solve one instance: try multiple greedy orderings, then local search."""
        n = len(elements)
        best = []
        best_size = 0
        t0 = time.time()

        # Multiple greedy orderings
        orderings = []
        # Natural order
        orderings.append(list(elements))
        # Reverse
        orderings.append(list(reversed(elements)))
        # By absolute value
        orderings.append(sorted(elements, key=abs))
        orderings.append(sorted(elements, key=abs, reverse=True))
        # By value
        orderings.append(sorted(elements))
        # Random shuffles
        rng = random.Random(42)
        for _ in range(10):
            o = list(elements)
            rng.shuffle(o)
            orderings.append(o)

        for order in orderings:
            if time.time() - t0 > time_budget - 5:
                break
            S = greedy_build(elements, order)
            if len(S) > best_size:
                best_size = len(S)
                best = list(S)

        # Local search from best greedy result
        if time.time() - t0 < time_budget - 2:
            remaining_time = time_budget - (time.time() - t0) - 2
            improved = local_search(best, elements, max_iters=50000, timeout=remaining_time)
            if len(improved) > best_size:
                best = improved
                best_size = len(improved)

        # Try local_search2 from a few different starting points
        for i in range(min(3, len(orderings))):
            if time.time() - t0 > time_budget - 2:
                break
            S = greedy_build(elements, orderings[i])
            remaining_time = time_budget - (time.time() - t0) - 2
            improved = local_search2(S, elements, max_iters=20000, timeout=remaining_time)
            if len(improved) > best_size:
                best = improved
                best_size = len(improved)

        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() - start_time > DEADLINE:
            results.append([])
            continue
        elems = instance(size, seed)
        # Allocate more time to larger instances
        budget = 40.0 if size <= 60 else (30.0 if size <= 120 else 25.0)
        sol = solve_instance(elems, time_budget=budget)
        # Verify sum-free
        if not is_sum_free(sol):
            # Fallback to odd numbers
            sol = [x for x in elems if x % 2 != 0]
        results.append(sol)

    return results
# EVOLVE-BLOCK-END
