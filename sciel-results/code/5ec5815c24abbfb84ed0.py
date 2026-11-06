import time
import random
import numpy as np

def construct():
    deadline = time.time() + 145  # Safety margin

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def solve_instance(elements, time_budget=40.0):
        n = len(elements)
        if n == 0:
            return []
        
        elem_to_idx = {x: i for i, x in enumerate(elements)}
        
        # Precompute conflicts
        # conflict[i] is a set of indices j such that elements[i] and elements[j] cannot be in the same sum-free set.
        # Conditions for conflict between i and j:
        # 1. i == j and 2*elements[i] is in elements (i.e., 2x = y)
        # 2. i != j and elements[i] + elements[j] is in elements
        
        conflicts = [set() for _ in range(n)]
        
        for i in range(n):
            # Check 2*elements[i]
            double_val = 2 * elements[i]
            if double_val in elem_to_idx:
                j = elem_to_idx[double_val]
                if i != j: # If i==j, it's a self-loop, handled by not picking i? 
                           # Actually if 2x=x -> x=0, but 0 is excluded. So i!=j always.
                    conflicts[i].add(j)
                    conflicts[j].add(i)
            
            # Check sums
            # To avoid duplicates, we only add i < j here? 
            # The loop above handles 2x=y. 
            # For x+y=z, we need to ensure we don't miss any pair.
            # We can just iterate all pairs i, j.
        
        # Better: Iterate all pairs i, j
        # Reset conflicts
        conflicts = [set() for _ in range(n)]
        
        for i in range(n):
            for j in range(i, n):
                s = elements[i] + elements[j]
                if s in elem_to_idx:
                    k = elem_to_idx[s]
                    # The triple is (i, j, k). 
                    # i and j conflict if k is in the set? No.
                    # i and j can coexist ONLY IF k is NOT in the set.
                    # So, if we pick i and j, we MUST NOT pick k.
                    # This is a 3-way constraint.
                    # My previous "conflict graph" model (2-clique) is an approximation.
                    # If I use a 2-conflict graph where i~j if i+j is in V, 
                    # then an independent set S implies no i,j in S have i+j in V.
                    # This ensures sum-free property.
                    # Does it restrict valid sets? 
                    # Yes, it's a sufficient condition, not necessary.
                    # Example: {1, 2, 3}. 1+2=3. 
                    # Graph edges: (1,2), (1,3) [if 1+? no], (2,3) [2+? no].
                    # Wait, 1+2=3. So 1 and 2 conflict.
                    # 1+3=4 (not in set).
                    # 2+3=5 (not in set).
                    # Independent sets: {1,3}, {2,3}, {1}, {2}, {3}, {}.
                    # Max independent set size 2.
                    # Is {1,3} sum-free? 1+3=4 (no). 1+1=2 (no, 2 not in {1,3}). 3+3=6 (no). Yes.
                    # Is {2,3} sum-free? 2+3=5 (no). 2+2=4 (no). 3+3=6 (no). Yes.
                    # Is {1,2} sum-free? 1+2=3 (3 not in {1,2}). 1+1=2 (2 IS in {1,2}). NO.
                    # So {1,2} is NOT sum-free.
                    # The graph model correctly identifies (1,2) as an edge, so {1,2} is not independent.
                    
                    # What about 2a=c?
                    # If 2a=c, then a+a=c.
                    # In graph model: a and c conflict? 
                    # a+c? No.
                    # We need to add edge between a and c if 2a=c.
                    
                    # So, for each i, j:
                    # if elements[i] + elements[j] == elements[k]:
                    #    Add edge (i, j)
                    #    Add edge (i, k) ? No. i and k CAN coexist if j is not present.
                    #    Add edge (j, k) ? No.
                    
                    # Wait, the constraint is: NOT (i in S AND j in S AND k in S).
                    # This is NOT a simple graph independent set constraint.
                    # Independent set in G means no edge (u,v) has both u,v in S.
                    # If we add edge (i,j), we forbid {i,j}.
                    # But {i,j} is only forbidden if k is in S.
                    # If k is NOT in S, {i,j} is fine.
                    # So a simple conflict graph is too restrictive.
                    
        # Therefore, we CANNOT use a simple Max Independent Set on a 2-conflict graph.
        # We must use a heuristic that handles the 3-way constraint.
        
        # Heuristic: Greedy with local search.
        # State: S (set of indices)
        # Check if S is sum-free:
        #   For all a,b in S, a+b not in S.
        
        def is_sum_free(idx_set):
            vals = {elements[i] for i in idx_set}
            for a in vals:
                for b in vals:
                    if a + b in vals:
                        return False
            return True

        best_set = set()
        start_time = time.time()
        
        # Multiple restarts
        for restart in range(50):
            if time.time() - start_time > time_budget:
                break
            
            rng = random.Random(123 + restart)
            order = list(range(n))
            rng.shuffle(order)
            
            current_set = set()
            
            # Greedy Phase
            for idx in order:
                # Try adding idx
                # Check if adding idx violates sum-free
                # Violations:
                # 1. elements[idx] + elements[idx] in current_set
                # 2. elements[idx] + x in current_set for any x in current_set
                # 3. x + y = elements[idx] for any x,y in current_set
                
                can_add = True
                val = elements[idx]
                
                # Check 1: 2*val in current_set
                if val * 2 in {elements[i] for i in current_set}:
                    can_add = False
                
                if can_add:
                    # Check 2 and 3
                    # To make this fast, we can maintain a set of values
                    # But O(n) check per insertion is O(n^2) total per restart.
                    # n=240, n^2=57k. 50 restarts = 2.8M ops. Very fast.
                    
                    curr_vals = {elements[i] for i in current_set}
                    
                    # Check if val + curr_val in curr_vals
                    for cv in curr_vals:
                        if val + cv in curr_vals:
                            can_add = False
                            break
                        # Check if val + val in curr_vals (already done above, but cv can be val? No, val not in curr_vals yet)
                        # Wait, if cv is in curr_vals, and val+cv is in curr_vals, violation.
                        # Also need to check if val + val is in curr_vals (done).
                        # And if any two elements in curr_vals sum to val?
                        # That is: exists a,b in curr_vals such that a+b=val.
                        # This is harder to check in O(n).
                        # But if we maintain curr_vals as a set, checking "exists a,b in C s.t. a+b=val" is O(|C|).
                        # We can just do it.
                    
                    if can_add:
                        # Check if any pair in current_set sums to val
                        # We can do this efficiently if we just iterate?
                        # O(n^2) per insertion is too slow.
                        # Let's just do the O(n) check for a+b=val?
                        # Actually, if we just add val, we only need to check:
                        # 1. val+val in S
                        # 2. val+x in S for x in S
                        # 3. x+y = val for x,y in S
                        
                        # We already checked 1 and 2.
                        # For 3: 
                        # We can break early.
                        found_sum = False
                        curr_list = list(current_set)
                        for a_idx in curr_list:
                            a_val = elements[a_idx]
                            if val - a_val in curr_vals:
                                # Need to ensure we don't count a_val itself if val-a_val == a_val (i.e., 2*a_val=val)
                                # If val - a_val == a_val, then 2*a_val = val.
                                # This means a_val + a_val = val.
                                # This is a violation if a_val is in S.
                                # Since a_val is in S, this is a violation.
                                # So just checking if (val - a_val) in curr_vals is sufficient.
                                # If val - a_val == a_val, then a_val is in curr_vals, so it returns True. Correct.
                                found_sum = True
                                break
                        if found_sum:
                            can_add = False
                            
                if can_add:
                    current_set.add(idx)
            
            # Local Search Phase
            # Try to add more elements
            improved = True
            while improved and time.time() - start_time < time_budget:
                improved = False
                curr_vals = {elements[i] for i in current_set}
                
                # Try adding any element not in current_set
                for idx in range(n):
                    if idx in current_set:
                        continue
                    val = elements[idx]
                    # Check if can add
                    # 1. 2*val in curr_vals
                    if val * 2 in curr_vals:
                        continue
                    # 2. val + x in curr_vals
                    conflict = False
                    for cv in curr_vals:
                        if val + cv in curr_vals:
                            conflict = True
                            break
                    if conflict:
                        continue
                    # 3. x + y = val
                    conflict = False
                    for cv in curr_vals:
                        if val - cv in curr_vals:
                            conflict = True
                            break
                    if conflict:
                        continue
                    
                    # Can add
                    current_set.add(idx)
                    curr_vals.add(val)
                    improved = True
                    break # Restart scan? Or continue? 
                    # Restarting scan ensures we don't miss interactions, but is slower.
                    # Continuing is faster. Let's continue.
            
            if len(current_set) > len(best_set):
                best_set = set(current_set)
        
        return [elements[i] for i in best_set]

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        remaining = deadline - time.time()
        if remaining < 5:
            elems = instance(size, seed)
            # Fallback: take all positives
            results.append([x for x in elems if x > 0])
            continue

        elems = instance(size, seed)
        # Allocate time based on size
        # n=60: 10s, n=120: 20s, n=240: 40s
        time_budget = min(remaining - 2, 10 * (size / 60.0))
        res = solve_instance(elems, time_budget)
        results.append(res)

    return results
