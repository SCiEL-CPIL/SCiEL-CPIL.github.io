"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def can_add(S, S_set, x):
        """Check if adding x to S keeps it sum-free."""
        if 2 * x in S_set:
            return False
        for a in S:
            if a + x in S_set:
                return False
        return True

    def greedy_augment(S, all_elements):
        """Try to add elements from all_elements not in S, in a given order."""
        S_set = set(S)
        S_list = list(S)
        for x in all_elements:
            if x not in S_set:
                if can_add(S_list, S_set, x):
                    S_set.add(x)
                    S_list.append(x)
        return S_list

    def solve_mis_bitset(vals, time_budget=30.0):
        """
        Solve Maximum Independent Set on the sum-free conflict graph using
        branch-and-bound with Tomita-style bitset ordering.
        Conflict: a and b conflict if a+b is in vals OR 2a in vals (for a=b case handled separately).
        Actually, conflict graph: edge between i,j if vals[i]+vals[j] is in vals_set.
        Also self-loop if 2*vals[i] in vals_set (meaning i cannot be in set with itself, 
        but since we pick distinct indices, self-loop means i is excluded entirely? 
        No: sum-free condition is no a,b,c in S such that a+b=c. a=b allowed.
        So if 2*a = c and c in S, that's bad. But if we pick a, we just can't pick c if 2a=c.
        The conflict is between a and c if 2a=c. Also between a and b if a+b=c for some c in S.
        This is not a simple pairwise conflict graph in the standard sense because the "sum" 
        depends on a third element. 
        Wait: S is sum-free if for all a,b in S, a+b not in S.
        This is equivalent to: for all a,b,c in S, a+b != c.
        This defines a 3-uniform hypergraph of conflicts. MIS in a hypergraph is harder.
        However, we can model it as a graph if we fix the "sum" element.
        Let's stick to the standard graph approach: 
        Two elements x, y are incompatible if there exists z in vals such that x+y=z.
        But this is not transitive.
        Actually, the standard reduction is:
        Create a graph G where vertices are elements.
        Edge (u,v) exists if u+v is in vals.
        Then a set S is sum-free iff S is an independent set in G?
        No. If u+v = w, and w is in S, then {u,v,w} is not sum-free.
        But if w is NOT in S, then {u,v} is fine.
        So the conflict depends on the third element.
        This is a 3-SAT like structure.
        
        Let's use a different approach: 
        The condition "S is sum-free" is equivalent to:
        For all a in S, (a + S) ∩ S = ∅.
        Or: S ∩ (S+S) = ∅.
        
        Since n is small (60, 120, 240), we can use a branch-and-bound on the elements.
        Order elements. Decide include/exclude.
        Use bitsets for speed.
        
        Let's implement a standard B&B for "Max Sum-Free Subset".
        State: (idx, current_set_bits, current_size, best_size)
        Upper bound: current_size + (n - idx)
        Pruning: if upper bound <= best_size, return.
        
        To make it faster, we can use a greedy upper bound:
        From the remaining elements, greedily add as many as possible to current_set.
        This gives a tighter upper bound.
        
        We will use bitsets (Python integers) for the set of available elements and current set.
        """
        n = len(vals)
        if n == 0:
            return []
        
        vals_set = set(vals)
        
        # Precompute for each i, the set of j such that vals[i] + vals[j] is in vals_set
        # This is the "conflict" if both i and j are in S and the sum is also in S.
        # But the sum element k is also in S.
        # So if we include i and j, we must exclude k where vals[i]+vals[j]=vals[k].
        # This is a constraint propagation problem.
        
        # Let's use a simpler B&B with bitsets.
        # bits: n-bit integer, bit i set if vals[i] is in current set.
        # avail: n-bit integer, bit i set if vals[i] is still available to be added.
        
        # Precompute conflict masks:
        # conflict_mask[i]: bitmask of all j such that vals[i] + vals[j] is in vals_set.
        # Note: if vals[i] + vals[j] = vals[k], then k is in conflict_mask[i] and conflict_mask[j].
        # Also, if 2*vals[i] = vals[k], then k is in conflict_mask[i] (and i in conflict_mask[k] if we consider symmetric, but 2i=k is specific).
        # Actually, for a pair (i,j) to be added to S, we must ensure that no k in S has vals[k] = vals[i]+vals[j].
        # And no k in S has vals[k] = 2*vals[i] if i=j (but we add distinct indices, so i!=j in the pair addition step, unless we add i twice which we don't).
        # Wait, the condition is: when we add a new element x to S, we check if 2x in S or if there exists a in S such that a+x in S.
        # This is a forward check.
        
        # Let's implement B&B with forward checking.
        # State: (idx, S_bits, S_list, best_size, best_list)
        # We iterate idx from 0 to n-1.
        # At each step, try to include vals[idx] if possible, then exclude.
        
        # To speed up, we sort vals by some heuristic (e.g., absolute value descending) to prune earlier.
        # But the conflict depends on the values, not just index.
        
        # Let's try a different structural change: 
        # Use a SAT solver? No, standard library only.
        # Use a heuristic search with strong pruning:
        # 1. Sort elements by a key that makes conflicts likely (e.g., large absolute values).
        # 2. Use a priority queue for best-first search? Or depth-first with good pruning.
        
        # Let's implement a DFS B&B.
        # We will use a global best.
        # Time limit check.
        
        best_size = 0
        best_set = set()
        
        # Precompute for each i, a bitmask of j such that vals[i] + vals[j] is in vals_set.
        # Let's call this sum_mask[i].
        # If we have current set S, and we want to add x, we check:
        # 1. 2*x not in S
        # 2. For all a in S, a+x not in S
        # This is equivalent to: (S & sum_mask_for_x) == 0, where sum_mask_for_x is the set of a such that a+x is in vals_set.
        # Wait, sum_mask[i] was defined as j such that vals[i]+vals[j] in vals_set.
        # So if we add x (index i), we need to check if any a in S is such that vals[a]+vals[i] in vals_set.
        # That is exactly: S & sum_mask[i] == 0.
        # Also need to check 2*vals[i] in S. That is: if there exists k in S such that vals[k] = 2*vals[i].
        # Let's precompute double_mask[i]: bitmask of k such that vals[k] = 2*vals[i].
        # Then condition is: (S & sum_mask[i]) == 0 AND (S & double_mask[i]) == 0.
        # Note: sum_mask[i] might include i itself if 2*vals[i] in vals_set. 
        # If 2*vals[i] = vals[i], then vals[i]=0, but 0 is excluded.
        # If 2*vals[i] = vals[j] with j!=i, then j is in sum_mask[i] and i is in sum_mask[j].
        # So the double case is covered if we define sum_mask[i] to include j where vals[i]+vals[j] in vals_set.
        # Let's verify: if we add i, and k is in S with vals[k]=2*vals[i], then vals[i]+vals[i]=vals[k].
        # So i+i=k. This is not captured by "a in S such that a+i in S" unless a=i.
        # But a must be in S, and we are adding i, so a cannot be i yet.
        # So we need the explicit double check.
        
        # Precomputation
        index_of_val = {v: i for i, v in enumerate(vals)}
        sum_mask = [0] * n
        double_mask = [0] * n
        
        for i in range(n):
            vi = vals[i]
            # For each j, check if vi + vals[j] is in vals_set
            # This is O(n^2)
            for j in range(n):
                s = vi + vals[j]
                if s in index_of_val:
                    k = index_of_val[s]
                    sum_mask[i] |= (1 << k)
            # Check double
            dv = 2 * vi
            if dv in index_of_val:
                k = index_of_val[dv]
                double_mask[i] |= (1 << k)
        
        # We will use a DFS.
        # To improve pruning, we can use a greedy upper bound.
        # Given current S and available elements (from idx to n-1), 
        # greedily add elements to S to estimate max possible size.
        
        def greedy_upper_bound(S_bits, start_idx):
            # S_bits is the current set.
            # We try to add elements from start_idx to n-1 in order.
            # This is a lower bound on the max possible additions, so it's an upper bound on the final size.
            count = 0
            S_temp = S_bits
            for i in range(start_idx, n):
                bit = 1 << i
                if S_temp & bit:
                    continue
                # Check if i can be added to S_temp
                if (S_temp & sum_mask[i]) == 0 and (S_temp & double_mask[i]) == 0:
                    S_temp |= bit
                    count += 1
            return count

        # DFS
        # We'll use a recursive function with a counter to limit depth/time.
        # Since n can be 240, recursion depth is fine, but branching factor is 2.
        # We need strong pruning.
        
        # Let's sort the elements to improve pruning.
        # Elements that are "more conflicting" should be decided earlier.
        # A simple heuristic: sort by the number of conflicts (popcount of sum_mask[i] | double_mask[i]).
        # But the indices in masks refer to the original order.
        # If we reorder vals, we must reorder masks.
        
        # Let's try reordering.
        # Calculate conflict count for each i.
        conflict_counts = []
        for i in range(n):
            mask = sum_mask[i] | double_mask[i]
            # Count bits
            c = 0
            m = mask
            while m:
                c += 1
                m &= m - 1
            conflict_counts.append(c)
        
        # Create order: indices sorted by conflict count descending
        order = sorted(range(n), key=lambda i: conflict_counts[i], reverse=True)
        
        # Reorder vals and masks
        new_vals = [vals[i] for i in order]
        # We need to remap the masks.
        # old_index -> new_index
        old_to_new = {old_i: new_i for new_i, old_i in enumerate(order)}
        
        new_sum_mask = [0] * n
        new_double_mask = [0] * n
        for new_i, old_i in enumerate(order):
            old_mask = sum_mask[old_i]
            new_m = 0
            # Remap bits in old_mask to new indices
            for old_j in range(n):
                if old_mask & (1 << old_j):
                    new_j = old_to_new[old_j]
                    new_m |= (1 << new_j)
            new_sum_mask[new_i] = new_m
            
            old_dmask = double_mask[old_i]
            new_dm = 0
            for old_j in range(n):
                if old_dmask & (1 << old_j):
                    new_j = old_to_new[old_j]
                    new_dm |= (1 << new_j)
            new_double_mask[new_i] = new_dm
        
        # Now solve on new_vals with new masks.
        # We want to return the indices in the NEW order, then map back to original.
        
        best_size = 0
        best_bits = 0
        
        # Use a list to store the current path (bits)
        # We'll use a recursive function that takes (idx, current_bits)
        # And updates global best.
        
        # To avoid deep recursion overhead, we can use a stack.
        # But recursion is cleaner. Let's try recursion with a time check.
        
        import sys
        sys.setrecursionlimit(10000)
        
        def dfs(idx, current_bits):
            nonlocal best_size, best_bits
            
            # Time check
            if time.time() - start_time > DEADLINE - 5:
                return
                
            # Upper bound
            if idx == n:
                # Count bits in current_bits
                c = 0
                m = current_bits
                while m:
                    c += 1
                    m &= m - 1
                if c > best_size:
                    best_size = c
                    best_bits = current_bits
                return
            
            # Prune: even if we add all remaining, can we beat best?
            # Simple bound: current_size + (n - idx)
            c = 0
            m = current_bits
            while m:
                c += 1
                m &= m - 1
            if c + (n - idx) <= best_size:
                return
                
            # Greedy upper bound (tighter)
            # This is expensive. Maybe only do it every few levels or if simple bound is close.
            # Let's skip greedy UB for now to save time, rely on simple bound + ordering.
            
            # Option 1: Include idx
            bit = 1 << idx
            if (current_bits & new_sum_mask[idx]) == 0 and (current_bits & new_double_mask[idx]) == 0:
                dfs(idx + 1, current_bits | bit)
                
            # Option 2: Exclude idx
            dfs(idx + 1, current_bits)
            
        dfs(0, 0)
        
        # Map best_bits back to original values
        result_indices = []
        for new_i in range(n):
            if best_bits & (1 << new_i):
                result_indices.append(order[new_i])
                
        result_vals = [vals[i] for i in result_indices]
        return result_vals

    def solve_one(vals, time_budget=35.0):
        """Find a large sum-free subset of vals using multiple strategies."""
        n = len(vals)
        start = time.time()
        best = []
        best_size = 0
        
        # Strategy 1: Odd numbers (always sum-free) + augment
        odd = [x for x in vals if x % 2 != 0]
        cand = greedy_augment(odd, vals)
        if len(cand) > best_size:
            best = cand
            best_size = len(cand)
        
        # Strategy 2: Greedy from empty, sorted by various orderings
        orderings = [
            sorted(vals, key=lambda x: abs(x), reverse=True),  # largest abs first
            sorted(vals, key=lambda x: abs(x)),                  # smallest abs first
            sorted(vals, reverse=True),                          # largest value first
            sorted(vals),                                        # smallest value first
        ]
        
        for ordering in orderings:
            if time.time() - start > time_budget * 0.3:
                break
            cand = greedy_augment([], ordering)
            if len(cand) > best_size:
                best = cand
                best_size = len(cand)
        
        # Strategy 3: MIS B&B
        remaining = time_budget - (time.time() - start)
        if remaining > 5.0:
            try:
                cand = solve_mis_bitset(vals, time_budget=remaining - 1.0)
                if len(cand) > best_size:
                    best = cand
                    best_size = len(cand)
            except Exception:
                pass # Fallback to previous best
        
        # Final safety check
        if not is_sum_free(best):
            best = odd
        
        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2 != 0])
            continue
        
        vals = instance(size, seed)
        
        # Allocate time based on instance size
        if size <= 60:
            budget = 30.0
        elif size <= 120:
            budget = 40.0
        else:
            budget = 50.0
        
        # Ensure we don't exceed deadline
        remaining = DEADLINE - elapsed - 5
        budget = min(budget, remaining)
        
        result = solve_one(vals, time_budget=budget)
        results.append(result)

    return results
# EVOLVE-BLOCK-END
