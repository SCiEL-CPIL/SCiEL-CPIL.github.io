"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from random import Random

def construct():
    deadline = time.time() + 145.0  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        """Check if S is sum-free."""
        s = set(S)
        for a in s:
            for b in s:
                if a + b in s:
                    return False
        return True

    def build_conflict_graph(elements):
        """
        Build a conflict graph where an edge (i, j) means that elements i and j
        cannot both be in a sum-free set together with some third element.
        Actually, we need: i and j conflict if there exists k (k != i, k != j)
        such that elements[i] + elements[j] == elements[k].
        But that's not sufficient. The real constraint is: for any triple (a,b,c) 
        with a+b=c, we cannot have all three in the set.
        
        A better approach: define a "bad triple" as any (i,j,k) where e_i + e_j = e_k.
        The constraint is x_i + x_j + x_k <= 2 for each bad triple.
        
        For local search, we can define conflicts more directly:
        Two elements i and j are "in conflict" if there exists k such that
        e_i + e_j = e_k (and k is distinct from i and j).
        But actually, the issue is that if we have i and j in the set, and e_i+e_j = e_k,
        then we cannot have k in the set. So i and j "conflict with" k.
        
        Let me reframe: for local search, we want to quickly check if adding element c
        to set S creates a violation. This happens if:
        1. There exist a, b in S such that a + b = c
        2. There exists a in S such that a + c is in S (i.e., a + c = b for some b in S)
        3. There exists b in S such that c + b is in S (same as 2)
        4. 2*c is in S (case a=b=c)
        
        So for each element c, we can precompute which other elements conflict with it.
        """
        n = len(elements)
        val_to_idx = {v: i for i, v in enumerate(elements)}
        
        # For each element c (index idx), find all pairs (a, b) in elements such that a + b = c
        # and all elements d such that c + d is in elements (i.e., there exists e in elements with c+e=d)
        # and whether 2*c is in elements
        
        # conflict_with[i] = set of j such that elements[i] and elements[j] cannot both be in a sum-free set
        # This is not quite right because it depends on the third element.
        
        # Better: for each element idx, precompute:
        # - sum_partners[idx] = set of j such that elements[j] + elements[idx] is in elements
        #   (i.e., there exists k with elements[j] + elements[idx] = elements[k])
        # - For checking if idx can be added to S:
        #   Check if there exist a, b in S with a + b = elements[idx]
        #   Check if there exists a in S with a + elements[idx] in S
        #   Check if 2*elements[idx] in S
        
        # Let's precompute for each idx:
        # partners[idx] = set of j (j != idx) such that elements[j] + elements[idx] is in elements
        #   This means if both j and idx are in S, and the sum element is also in S, violation.
        #   But actually, we need: if j and idx are in S, then elements[j]+elements[idx] must NOT be in S.
        #   So the "conflict" is between the pair {j, idx} and the element k where elements[k] = elements[j]+elements[idx].
        
        # For efficient local search, let's precompute:
        # for each idx, the set of values v such that v + elements[idx] is in elements
        # and the set of values v such that elements[idx] - v is in elements (i.e., v + elements[idx] = elements[idx]... no)
        
        # Actually, let's just precompute for each idx:
        # sum_targets[idx] = set of j such that elements[idx] + elements[j] is in elements
        #   (so if idx and j are both in S, then the sum element cannot be in S)
        # diff_targets[idx] = set of j such that elements[j] - elements[idx] is in elements
        #   (so if idx and j are both in S, and elements[j] - elements[idx] is in S, then violation: 
        #    elements[j] = elements[idx] + (elements[j]-elements[idx]), so elements[idx] + (elements[j]-elements[idx]) = elements[j])
        #   Wait, that's: a = elements[idx], b = elements[j]-elements[idx], c = elements[j]. a+b=c.
        #   So if idx, (j-idx), and j are all in S, violation.
        
        # Let me simplify. For a set S, a violation exists if there are a, b, c in S with a+b=c.
        # To check if adding element x to S creates a violation:
        # 1. Is there a, b in S with a+b=x? 
        # 2. Is there a in S with a+x in S? (i.e., a+x=c for some c in S)
        # 3. Is 2*x in S?
        
        # Precompute for each element value x:
        # - For check 1: we need to know if x can be written as sum of two elements in S
        # - For check 2: we need to know if x+y is in S for some y in S
        # - For check 3: is 2x in S?
        
        # For local search with swaps, let's precompute a "conflict" structure:
        # For each pair (i, j), does elements[i] + elements[j] appear in elements?
        # If yes, then i, j, and k (where elements[k] = elements[i]+elements[j]) form a bad triple.
        
        # Let's precompute:
        # triple_conflicts[i] = set of (j, k) such that elements[i] + elements[j] = elements[k]
        # But this is O(n^3) in the worst case. For n=240, that's ~13.8M which is fine.
        
        # Actually, let's precompute for each i:
        # sum_conflicts[i] = set of j such that elements[i] + elements[j] is in elements
        #   (and j != i, and the sum element index != i and != j)
        
        # And for local search, when checking if we can add element idx to S:
        # - For each a_idx in S: check if elements[idx] - elements[a_idx] is in S (this covers a+b=idx)
        #   Actually: a + b = idx means b = idx - a. So check if (elements[idx] - elements[a_idx]) is in S.
        # - For each a_idx in S: check if elements[a_idx] + elements[idx] is in S (this covers a+idx=c)
        # - Check if 2*elements[idx] is in S
        
        # Let's precompute for each idx:
        # add_conflicts[idx] = set of j such that elements[idx] + elements[j] is in elements
        # sub_conflicts[idx] = set of j such that elements[j] - elements[idx] is in elements  
        #   (i.e., elements[j] = elements[idx] + elements[k] for some k... no)
        #   elements[j] - elements[idx] = elements[k] means elements[j] = elements[idx] + elements[k]
        #   So if idx and k are in S, and j is in S, then elements[idx] + elements[k] = elements[j], violation.
        
        # I think the cleanest precomputation is:
        # For each idx, precompute the set of values that "interact" with elements[idx]
        
        # Let's just do it directly in the local search for now, with some precomputation.
        
        # Precompute: val_to_idx
        # For each idx, precompute:
        #   plus[idx] = set of j such that elements[idx] + elements[j] is in elements (j can equal idx)
        #   minus[idx] = set of j such that elements[j] - elements[idx] is in elements (j can equal idx)
        
        plus = [set() for _ in range(n)]
        minus = [set() for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                s = elements[i] + elements[j]
                if s in val_to_idx:
                    k = val_to_idx[s]
                    plus[i].add(j)
                    plus[j].add(i)
                d = elements[j] - elements[i]
                if d in val_to_idx:
                    k = val_to_idx[d]
                    minus[i].add(j)
        
        return plus, minus, val_to_idx

    def greedy_init(elements, order_seed=0):
        """Greedy initialization with a specific ordering."""
        n = len(elements)
        rng = Random(order_seed)
        
        # Try different orderings: by abs value, random, by value
        orderings = [
            sorted(range(n), key=lambda i: -abs(elements[i])),
            sorted(range(n), key=lambda i: elements[i]),
            sorted(range(n), key=lambda i: -elements[i]),
            list(range(n)),
        ]
        rng.shuffle(orderings)
        
        best_subset = []
        best_size = 0
        
        for order in orderings:
            included = set()
            for idx in order:
                c_val = elements[idx]
                ok = True
                # Check 1: is there a, b in included with a+b = c_val?
                for a_idx in included:
                    a_val = elements[a_idx]
                    b_val = c_val - a_val
                    if b_val in {elements[i] for i in included}:
                        ok = False
                        break
                if not ok:
                    continue
                # Check 2: is there a in included with a + c_val in included?
                for a_idx in included:
                    a_val = elements[a_idx]
                    if a_val + c_val in {elements[i] for i in included}:
                        ok = False
                        break
                if not ok:
                    continue
                # Check 3: is 2*c_val in included?
                if 2 * c_val in {elements[i] for i in included}:
                    ok = False
                    continue
                if ok:
                    included.add(idx)
            
            if len(included) > best_size:
                best_size = len(included)
                best_subset = list(included)
        
        return [elements[i] for i in best_subset]

    def local_search(elements, initial_subset, time_limit=30.0):
        """Local search with swaps to improve the sum-free subset."""
        n = len(elements)
        val_to_idx = {v: i for i, v in enumerate(elements)}
        
        # Convert initial_subset to set of indices
        included = set()
        for v in initial_subset:
            if v in val_to_idx:
                included.add(val_to_idx[v])
        
        best_size = len(included)
        best_included = set(included)
        
        start_time = time.time()
        rng = random.Random(42)
        
        # Precompute for quick conflict checking
        # For each idx, precompute the set of j such that elements[idx] + elements[j] is in elements
        plus = [set() for _ in range(n)]
        for i in range(n):
            for j in range(n):
                s = elements[i] + elements[j]
                if s in val_to_idx:
                    plus[i].add(j)
        
        def can_add(idx, current_included):
            """Check if idx can be added to current_included."""
            c_val = elements[idx]
            # Check 1: a + b = c_val for a, b in current_included
            for a_idx in current_included:
                a_val = elements[a_idx]
                b_val = c_val - a_val
                if b_val in {elements[i] for i in current_included}:
                    return False
            # Check 2: a + c_val in current_included for a in current_included
            for a_idx in current_included:
                a_val = elements[a_idx]
                if a_val + c_val in {elements[i] for i in current_included}:
                    return False
            # Check 3: 2*c_val in current_included
            if 2 * c_val in {elements[i] for i in current_included}:
                return False
            return True
        
        def can_remove(idx, current_included):
            """Removing always keeps sum-free property, so always True."""
            return True
        
        # Local search: try single swaps and additions
        while time.time() - start_time < time_limit:
            improved = False
            
            # Try additions
            for idx in range(n):
                if idx not in included:
                    if can_add(idx, included):
                        included.add(idx)
                        improved = True
                        if len(included) > best_size:
                            best_size = len(included)
                            best_included = set(included)
            
            # Try swaps: remove one, add another
            if not improved:
                for out_idx in list(included):
                    temp_included = included - {out_idx}
                    for in_idx in range(n):
                        if in_idx not in temp_included:
                            if can_add(in_idx, temp_included):
                                included = temp_included | {in_idx}
                                improved = True
                                if len(included) > best_size:
                                    best_size = len(included)
                                    best_included = set(included)
                                break
                    if improved:
                        break
            
            if not improved:
                # Try random restarts or perturbations
                if rng.random() < 0.1:
                    # Remove a random element and try to re-add
                    if included:
                        out_idx = rng.choice(list(included))
                        included.discard(out_idx)
                        # Try to add back or add others
                        for idx in range(n):
                            if idx not in included:
                                if can_add(idx, included):
                                    included.add(idx)
                                    break
        
        return [elements[i] for i in best_included]

    def solve_with_ilp(elements):
        """Solve using ILP if available and fast enough."""
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            n = len(elements)
            if n == 0:
                return []
            
            val_to_idx = {v: i for i, v in enumerate(elements)}
            
            # Collect constraints
            constraint_rows = []
            for i in range(n):
                for j in range(i, n):
                    s = elements[i] + elements[j]
                    if s in val_to_idx:
                        k = val_to_idx[s]
                        constraint_rows.append((i, j, k))
            
            M = len(constraint_rows)
            
            if M == 0:
                return elements[:]
            
            # For large n, ILP might be slow. Use it only for small n.
            if n > 100:
                return None  # Skip ILP for large sets
            
            A = np.zeros((M, n), dtype=np.float64)
            for idx, (i, j, k) in enumerate(constraint_rows):
                A[idx, i] += 1.0
                A[idx, j] += 1.0
                A[idx, k] += 1.0
            
            c = -np.ones(n)
            bounds = Bounds(lb=0, ub=1)
            cons = LinearConstraint(A, lb=0, ub=2)
            integrality = np.ones(n)
            
            res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds)
            
            if res.success:
                selected_indices = [i for i in range(n) if res.x[i] > 0.5]
                return [elements[i] for i in selected_indices]
            else:
                return None
                
        except Exception:
            return None

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elems = instance(size, seed)
        
        # Try ILP first for small sets
        ilp_result = solve_with_ilp(elems)
        if ilp_result is not None:
            results.append(ilp_result)
            continue
        
        # Greedy initialization
        initial = greedy_init(elems)
        
        # Local search with time budget
        remaining_time = deadline - time.time()
        time_per_instance = max(5.0, remaining_time / 3.0)
        subset = local_search(elems, initial, time_limit=time_per_instance)
        
        results.append(subset)
        
        # Check if we're running out of time
        if time.time() > deadline - 10:
            # Return what we have
            break
    
    # Ensure we have 3 results
    while len(results) < 3:
        size, seed = ((60, 801), (120, 802), (240, 803))[len(results)]
        elems = instance(size, seed)
        # Fallback: odd numbers
        results.append([x for x in elems if x % 2])
    
    return results
# EVOLVE-BLOCK-END
