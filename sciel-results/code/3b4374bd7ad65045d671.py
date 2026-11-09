"""Rewritten program: greedy max-involvement removal + simulated annealing polish."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def solve_one(values, time_budget):
        n = len(values)
        if n <= 2:
            return list(values)

        # Build conflict structure: for each element, track which pairs (a,b) in the set have a+b == c
        # An element c conflicts with the pair (a,b) if a+b==c.
        # For greedy removal, we want to remove elements that are "involved" in the most conflicts.
        
        # Precompute: for each c in values, count how many pairs (a,b) with a+b==c exist
        # Also track for each element x, how many times it appears as a, b, or c in a conflict triple
        val_set = set(values)
        val_idx = {v: i for i, v in enumerate(values)}
        
        # Count involvement: for each element, how many conflict triples it participates in
        involvement = [0] * n
        
        # For each pair (a, b) where a+b is in the set, increment involvement of a, b, and a+b
        # But we need to be careful: a=b is allowed, so we check all pairs
        for i in range(n):
            a = values[i]
            for j in range(i, n):
                b = values[j]
                s = a + b
                if s in val_set:
                    c_idx = val_idx[s]
                    involvement[i] += 1
                    involvement[j] += 1
                    involvement[c_idx] += 1
        
        # Greedy removal: repeatedly remove the element with highest involvement
        # until no conflicts remain
        in_set = [True] * n
        remaining = n
        
        # We'll use a priority approach: recompute involvement after each removal
        # For small n (up to 240), this is feasible
        
        # Actually, let's do iterative removal:
        # At each step, find the element with max involvement among remaining, remove it
        # Recompute involvement for remaining elements
        
        # More efficient: maintain conflict counts dynamically
        # conflict_count[i] = number of pairs (j,k) in remaining such that values[j]+values[k]==values[i]
        # Also for each i, as an addend: how many pairs where i is one of the addends
        
        # Let's use a different approach: 
        # For the current set, an element c is "bad" if there exist a,b in set with a+b=c
        # We want to remove elements to eliminate all such triples.
        
        # Greedy: while conflicts exist, remove the element that is in the most conflict triples
        # A conflict triple is (a,b,c) where a+b=c, all in set.
        
        # Let's track: for each element, the number of conflict triples it's part of
        # triple_count[i] = number of (a,b) in remaining such that a+b == values[i]
        #                   + number of (b) in remaining such that values[i]+b == values[k] for some k in remaining
        #                   + number of (a) in remaining such that a+values[i] == values[k] for some k in remaining
        
        # This is getting complex. Let's just do:
        # While there's a conflict, find the most "central" element and remove it.
        
        # Simpler greedy: 
        # 1. Find all conflict triples
        # 2. Count how many triples each element is in
        # 3. Remove the element in the most triples
        # 4. Repeat until no conflicts
        
        def get_conflicts(indices):
            """Return list of conflict triples (i,j,k) where values[i]+values[j]==values[k]"""
            idx_set = set(indices)
            conflicts = []
            for i in indices:
                for j in indices:
                    if j < i:
                        continue
                    s = values[i] + values[j]
                    if s in val_set:
                        k = val_idx[s]
                        if k in idx_set:
                            conflicts.append((i, j, k))
            return conflicts
        
        # Initial set of indices
        current_indices = list(range(n))
        
        # Iterative greedy removal
        max_iters = n  # safety
        for _ in range(max_iters):
            if time.time() > deadline - time_budget:
                break
            conflicts = get_conflicts(current_indices)
            if not conflicts:
                break
            # Count involvement
            cnt = [0] * n
            for (i, j, k) in conflicts:
                cnt[i] += 1
                cnt[j] += 1
                cnt[k] += 1
            # Find element with max count among current
            max_cnt = -1
            max_el = -1
            for idx in current_indices:
                if cnt[idx] > max_cnt:
                    max_cnt = cnt[idx]
                    max_el = idx
            if max_el >= 0:
                current_indices.remove(max_el)
        
        # Now current_indices is sum-free. Use it as starting point for simulated annealing.
        in_arr = [False] * n
        for idx in current_indices:
            in_arr[idx] = True
        current_size = len(current_indices)
        
        # Simulated annealing to add back / swap elements
        # We want to maximize size while maintaining sum-free property.
        
        # Precompute for fast conflict checking:
        # For a candidate element x to add to set S, check if there exist a,b in S with a+b=x
        # Or if x+a=b for some a,b in S, or x+b=a for some a,b in S
        # i.e., x conflicts with S if:
        #   - exists a,b in S: a+b == x
        #   - exists a in S: x+a is in S (then x, a, x+a form a triple)
        #   - exists a in S: a-x is in S (then a-x, x, a form a triple)
        
        # For O(n) check: given set S (as a boolean array or set of values), 
        # check if adding x creates a conflict:
        #   1. For each a in S: if x-a in S, conflict (a + x = x+a... wait)
        #      Actually: conflict if exists a,b in S such that a+b=x, OR a+x=b, OR x+b=a
        #      a+b=x: for each a in S, check if x-a in S
        #      a+x=b: for each a in S, check if a+x in S
        #      x+b=a: same as a+x=b by symmetry
        # So: conflict if (exists a in S: x-a in S) OR (exists a in S: a+x in S)
        
        # Let's maintain a set of current values for fast lookup
        current_val_set = set(values[i] for i in current_indices)
        
        def check_add(idx):
            """Check if adding values[idx] to current_val_set creates a conflict."""
            x = values[idx]
            for a in current_val_set:
                if (x - a) in current_val_set:
                    return False
                if (a + x) in current_val_set:
                    return False
            return True
        
        def check_remove(idx):
            """Removing always safe, but we check if it was causing conflicts (not needed for validity)."""
            return True
        
        # Simulated annealing
        rng = random.Random(42)
        T = 10.0
        T_min = 0.01
        cooling = 0.999
        best_indices = list(current_indices)
        best_size = current_size
        no_improve = 0
        max_no_improve = 5000
        
        while T > T_min and no_improve < max_no_improve:
            if time.time() > deadline - time_budget:
                break
            
            # Random move: try to add a random element, or swap (remove one, add one)
            not_in = [i for i in range(n) if not in_arr[i]]
            if not not_in:
                break
            
            move_type = rng.random()
            if move_type < 0.5 and len(not_in) > 0:
                # Try to add
                idx = rng.choice(not_in)
                if check_add(idx):
                    # Accept
                    in_arr[idx] = True
                    current_val_set.add(values[idx])
                    current_size += 1
                    no_improve = 0
                    if current_size > best_size:
                        best_size = current_size
                        best_indices = [i for i in range(n) if in_arr[i]]
                else:
                    no_improve += 1
            elif move_type < 0.8 and len(not_in) > 0 and current_size > 0:
                # Swap: remove a random in-set element, try to add a random out-set element
                idx_out = rng.choice(not_in)
                idx_in = rng.choice(current_indices)
                # Temporarily remove idx_in
                old_val = values[idx_in]
                in_arr[idx_in] = False
                current_val_set.remove(old_val)
                current_size -= 1
                # Try to add idx_out
                if check_add(idx_out):
                    in_arr[idx_out] = True
                    current_val_set.add(values[idx_out])
                    current_size += 1
                    no_improve = 0
                    if current_size > best_size:
                        best_size = current_size
                        best_indices = [i for i in range(n) if in_arr[i]]
                else:
                    # Revert
                    in_arr[idx_in] = True
                    current_val_set.add(old_val)
                    current_size += 1
                    no_improve += 1
            else:
                # Just try to add a random element (already covered)
                if not_in:
                    idx = rng.choice(not_in)
                    if check_add(idx):
                        in_arr[idx] = True
                        current_val_set.add(values[idx])
                        current_size += 1
                        no_improve = 0
                        if current_size > best_size:
                            best_size = current_size
                            best_indices = [i for i in range(n) if in_arr[i]]
                    else:
                        no_improve += 1
            
            T *= cooling
        
        # Return the best found
        return [values[i] for i in best_indices]

    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    # Allocate time proportionally, with more time for larger instances
    total_budget = 130  # seconds
    weights = [1, 2, 4]
    total_w = sum(weights)
    
    for (size, seed), w in zip(instances, weights):
        time_budget = (w / total_w) * total_budget
        vals = instance(size, seed)
        subset = solve_one(vals, time_budget)
        results.append(subset)
    
    return results
# EVOLVE-BLOCK-END
