import time
import random
import sys

N = 256

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 25.0  # Safety margin before 30s limit
    
    # Bitmask representation for the set of integers [1, 256]
    # We use Python integers as bitsets.
    # Bit i represents integer (i+1) if we use 0-indexed bits for 1..256
    # Let's map x in [1, 256] to bit index x-1.
    
    # Precompute sum constraints
    # A Sidon set S is valid if all sums a+b (a<=b) are distinct.
    # This is equivalent to: for any distinct pairs (a,b) and (c,d) with a<=b, c<=d,
    # if a+b == c+d, then {a,b} == {c,d}.
    
    # To make backtracking efficient, we track:
    # 1. The set of selected elements (bitmask)
    # 2. The set of used sums (bitmask or set, but sums range 2..512, so 511 bits)
    # 3. The set of "blocked" elements. An element x is blocked if adding it 
    #    violates Sidon property.
    
    # Actually, a simpler CSP formulation:
    # We want to select a subset S such that:
    # 1. S is Sidon.
    # 2. S is Maximal: For every x in [1, N] \ S, S U {x} is NOT Sidon.
    
    # "S U {x} is NOT Sidon" means there exist a, b in S U {x} such that a+b is a duplicate sum.
    # Since S is Sidon, the duplicate must involve x.
    # So, x is blocked if:
    # (2*x is in Sums(S)) OR (exists a in S such that (x+a) is in Sums(S) and x+a != 2*x)
    # Wait, if x+a = b+c with a,b,c in S, then S wouldn't be Sidon.
    # So if S is Sidon, x is blocked iff:
    # - 2x is a sum of two elements in S (including same element)
    # - OR there exists a in S such that x+a is a sum of two elements in S.
    
    # Let's use a backtracking search that builds S.
    # To ensure maximality, we must ensure that at the end, no x can be added.
    # This is hard to enforce incrementally unless we decide on maximality early.
    
    # Alternative Strategy: Randomized Local Search on the "Blocked" property.
    # Start with a random Sidon set.
    # Try to remove elements. If removal keeps it Sidon AND keeps it Maximal, accept.
    # If removal breaks Maximality (i.e., the removed element becomes addable), we need to "fix" it.
    
    # Better Strategy: "Minimal Maximal" via Hitting Set of Sum Collisions.
    # Every sum s in [2, 512] can be formed by pairs.
    # A Sidon set picks at most one pair per sum (actually, it picks elements such that pairs are unique).
    
    # Let's stick to the most robust approach for small N: 
    # **Backtracking with Pruning based on Maximal Potential**.
    
    # We will search for a set S.
    # Heuristic: Pick elements that "cover" (block) many other elements.
    
    # Data Structures:
    # selected_mask: bitmask of selected integers
    # used_sums_mask: bitmask of sums already used
    # blocked_mask: bitmask of integers that CANNOT be added to current S.
    
    # Initialization:
    # selected_mask = 0
    # used_sums_mask = 0
    # blocked_mask = 0 (initially nothing is blocked, but we haven't selected anything)
    
    # When we add x:
    # 1. Check if x is blocked. If yes, fail.
    # 2. Add x to selected.
    # 3. Update used_sums: add 2x, and x+y for all y in selected.
    # 4. Update blocked: 
    #    - x is now selected, so it's "handled".
    #    - Any y such that 2y is in new_used_sums becomes blocked.
    #    - Any y such that y+z is in new_used_sums for some z in new_selected becomes blocked.
    
    # To make this fast, we precompute for each possible sum s, the list of pairs (a,b) with a+b=s.
    # And for each x, the list of sums it creates with other elements.
    
    # However, dynamic updates are complex.
    
    # Let's try a different structural approach: **Simulated Annealing on the "Cost" of Maximality**.
    # Define Cost(S) = 0 if S is Sidon and Maximal.
    # If S is not Sidon, Cost = infinity.
    # If S is Sidon but not Maximal, Cost = number of addable elements.
    # We want to reach Cost = 0.
    
    # State: Bitmask of selected elements.
    # Moves: Add/Remove/Swap an element.
    
    def get_sums_mask(selected_mask):
        """Compute bitmask of sums for a given selected mask."""
        # This is O(K^2) where K is size.
        # We need this to be fast.
        # Since N=256, max sums is 512.
        # We can precompute sum masks for single elements? No, sums depend on pairs.
        
        # Let's just compute it on the fly but efficiently.
        sums_mask = 0
        # Extract elements
        elems = []
        m = selected_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            elems.append(idx + 1)
            m ^= lsb
        
        for i in range(len(elems)):
            a = elems[i]
            for j in range(i, len(elems)):
                b = elems[j]
                sums_mask |= (1 << (a + b))
        return sums_mask

    def is_sidon(selected_mask):
        """Check if selected_mask forms a Sidon set."""
        # We can track this during construction, but for annealing we need to check from scratch or incrementally.
        # For SA, incremental is hard. Let's use a fast check.
        # Sum of squares trick? No.
        # Just check for duplicate sums.
        sums = set()
        elems = []
        m = selected_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            elems.append(idx + 1)
            m ^= lsb
        
        for i in range(len(elems)):
            a = elems[i]
            for j in range(i, len(elems)):
                b = elems[j]
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        return True

    def count_addable(selected_mask):
        """Count how many elements in [1, N] can be added to selected_mask while keeping it Sidon."""
        if not is_sidon(selected_mask):
            return -1 # Invalid state
            
        sums_set = set()
        elems = []
        m = selected_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            elems.append(idx + 1)
            m ^= lsb
            
        for i in range(len(elems)):
            a = elems[i]
            for j in range(i, len(elems)):
                sums_set.add(elems[i] + elems[j])
                
        selected_set = set(elems)
        addable_count = 0
        
        for x in range(1, N + 1):
            if x in selected_set:
                continue
            # Check if x can be added
            # 2x must not be in sums_set
            # x + a must not be in sums_set for any a in elems
            can_add = True
            if 2 * x in sums_set:
                can_add = False
            else:
                for a in elems:
                    if x + a in sums_set:
                        can_add = False
                        break
            if can_add:
                addable_count += 1
        return addable_count

    # Since full checks are O(K^2 + N*K), and K is small (~10-20), this is fast.
    # N=256, K=20 -> 20^2 + 256*20 = 400 + 5120 = 5520 ops per eval.
    # 10000 evals = 55M ops. Python might struggle in 25s.
    # We need to optimize the "count_addable" or use a heuristic.
    
    # Let's use a greedy construction first to get a good starting point, then SA.
    
    def greedy_maximal():
        """Construct a maximal Sidon set greedily."""
        selected = []
        sums = set()
        
        # Order elements by some heuristic? 
        # Random order is fine for SA start, but let's try to be smart.
        # Pick elements that create "dense" sums?
        
        for x in range(1, N + 1):
            # Check if x can be added
            if 2 * x in sums:
                continue
            conflict = False
            for a in selected:
                if x + a in sums:
                    conflict = True
                    break
            if not conflict:
                selected.append(x)
                sums.add(2 * x)
                for a in selected[:-1]:
                    sums.add(x + a)
        
        # This produces a maximal set because we iterate through all x.
        # If x wasn't added, it's because it conflicted with existing sums.
        # So it's maximal.
        return selected

    # Start with a greedy solution
    current_elems = greedy_maximal()
    current_mask = 0
    for e in current_elems:
        current_mask |= (1 << (e - 1))
        
    best_elems = list(current_elems)
    best_cost = count_addable(current_mask)
    
    # Simulated Annealing
    # Cost:
    # If not Sidon: 10000
    # Else: count_addable (we want 0)
    
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    steps = 0
    max_steps = 20000
    
    while time.time() < deadline and steps < max_steps and T > T_min:
        steps += 1
        
        # Generate neighbor
        r = random.random()
        new_mask = current_mask
        
        if r < 0.5:
            # Remove an element
            if current_mask == 0:
                continue
            lsb = current_mask & -current_mask
            # Pick random bit
            bits = []
            m = current_mask
            while m:
                l = m & -m
                bits.append(l)
                m ^= l
            bit_to_remove = random.choice(bits)
            new_mask = current_mask ^ bit_to_remove
        elif r < 0.8:
            # Add an element
            missing = []
            for x in range(1, N + 1):
                if not (current_mask & (1 << (x - 1))):
                    missing.append(x)
            if not missing:
                continue
            x = random.choice(missing)
            new_mask = current_mask | (1 << (x - 1))
        else:
            # Swap
            if current_mask == 0:
                continue
            bits = []
            m = current_mask
            while m:
                l = m & -m
                bits.append(l)
                m ^= l
            bit_to_remove = random.choice(bits)
            missing = []
            for x in range(1, N + 1):
                if not (current_mask & (1 << (x - 1))):
                    missing.append(x)
            if not missing:
                continue
            x = random.choice(missing)
            new_mask = (current_mask ^ bit_to_remove) | (1 << (x - 1))
            
        # Evaluate new state
        # Fast check for Sidon
        # We can optimize is_sidon by early termination.
        
        # To speed up, we can cache sums? 
        # For SA, we recompute.
        
        # Let's create a faster sidon check
        def fast_sidon_check(mask):
            elems = []
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                elems.append(idx + 1)
                m ^= lsb
                
            if len(elems) < 2:
                return True, set()
                
            sums = set()
            for i in range(len(elems)):
                a = elems[i]
                for j in range(i, len(elems)):
                    b = elems[j]
                    s = a + b
                    if s in sums:
                        return False, sums
                    sums.add(s)
            return True, sums

        is_valid, sums_set = fast_sidon_check(new_mask)
        
        if not is_valid:
            new_cost = 10000
        else:
            # Count addable
            selected_set = set()
            m = new_mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                selected_set.add(idx + 1)
                m ^= lsb
                
            addable = 0
            for x in range(1, N + 1):
                if x in selected_set:
                    continue
                if 2 * x in sums_set:
                    continue
                ok = True
                for a in selected_set:
                    if x + a in sums_set:
                        ok = False
                        break
                if ok:
                    addable += 1
            new_cost = addable
            
        # Current cost
        if not is_sidon(current_mask):
            cur_cost = 10000
        else:
            cur_cost = count_addable(current_mask)
            
        delta = new_cost - cur_cost
        
        if delta < 0 or random.random() < pow(2.718, -delta / T):
            current_mask = new_mask
            if new_cost < best_cost:
                best_cost = new_cost
                # Extract best elems
                best_elems = []
                m = new_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    best_elems.append(idx + 1)
                    m ^= lsb
                    
        T *= cooling
        
    # Final verification and repair if needed
    # The SA might end in an invalid state if we accept invalid moves? 
    # No, we penalize invalid with 10000, so it rarely stays there.
    # But let's ensure the returned set is valid and maximal.
    
    # If best_cost > 0, we need to repair to make it maximal.
    # We can just take the best found, and if it's not maximal, add elements until it is.
    # But adding elements increases cardinality. We want to MINIMIZE cardinality.
    # So we want a set that is ALREADY maximal and small.
    
    # If SA finds a small Sidon set that is NOT maximal, we can't just return it.
    # We must extend it to be maximal.
    # So the objective should really be: Minimize |S| subject to S is Maximal Sidon.
    
    # My cost function was: 10000 if not Sidon, else #addable.
    # This drives #addable to 0.
    # It does NOT explicitly minimize |S| among maximal sets.
    # However, if we start with a greedy maximal set (size ~20-30), and try to remove elements,
    # removals are only accepted if they keep it Sidon.
    # If a removal makes it non-maximal (addable > 0), the cost increases (0 -> positive).
    # So SA will resist removals that break maximality.
    # It will only accept removals that KEEP it maximal (cost stays 0) or reduce cost.
    # So it effectively searches for smaller maximal sets.
    
    # However, if it gets stuck in a local minimum of size K, and there is a size K-1 maximal set,
    # it needs a path of valid maximal sets to get there.
    # If no such path exists, it's stuck.
    
    # To improve, we can add a penalty for size in the cost function?
    # Cost = (addable > 0 ? 1000 : 0) + 0.1 * size
    # This encourages smaller sets, but only if they are maximal (or close).
    
    # Let's restart with a size-aware cost if time permits.
    # For now, return the best found.
    
    # Ensure the returned list is valid
    if not is_sidon(best_elems):
         # Fallback to greedy
         return greedy_maximal()
         
    # Check maximality
    addable_count = count_addable(current_mask) # current_mask might not be best_mask
    # Re-check best_mask
    if count_addable(best_elems_mask) > 0: # Need to convert best_elems to mask
         # Repair: add elements until maximal
         # This increases size, but ensures validity.
         # Since the task requires validity, we must do this.
         pass

    # Optimization: The above logic is a bit loose. 
    # Let's refine the SA loop to track best_mask properly.
    
    # Re-running a cleaner version of the SA logic to be safe.
    
    def run_sa():
        start = time.time()
        limit = 20.0
        
        # Init
        curr = greedy_maximal()
        curr_mask = 0
        for e in curr: curr_mask |= (1 << (e-1))
        
        best = list(curr)
        best_mask = curr_mask
        best_cost = 10000
        
        T = 5.0
        T_min = 0.01
        cool = 0.9995
        
        while time.time() - start < limit and T > T_min:
            # Neighbor
            r = random.random()
            new_mask = curr_mask
            if r < 0.6:
                # Remove
                if curr_mask:
                    bits = []
                    m = curr_mask
                    while m:
                        l = m & -m
                        bits.append(l)
                        m ^= l
                    new_mask = curr_mask ^ random.choice(bits)
            else:
                # Add
                missing = [x for x in range(1, N+1) if not (curr_mask & (1<<(x-1)))]
                if missing:
                    x = random.choice(missing)
                    new_mask = curr_mask | (1 << (x-1))
            
            # Eval
            valid, sums = fast_sidon_check(new_mask)
            if not valid:
                new_cost = 10000
            else:
                sel = set()
                m = new_mask
                while m:
                    l = m & -m
                    sel.add(l.bit_length())
                    m ^= l
                add = 0
                for x in range(1, N+1):
                    if x in sel: continue
                    if 2*x in sums: continue
                    ok = True
                    for a in sel:
                        if x+a in sums:
                            ok = False
                            break
                    if ok: add += 1
                new_cost = add
                
            # Current cost (cache if possible, but recompute for safety)
            cv, cs = fast_sidon_check(curr_mask)
            if not cv:
                c_cost = 10000
            else:
                c_sel = set()
                m = curr_mask
                while m:
                    l = m & -m
                    c_sel.add(l.bit_length())
                    m ^= l
                c_add = 0
                for x in range(1, N+1):
                    if x in c_sel: continue
                    if 2*x in cs: continue
                    ok = True
                    for a in c_sel:
                        if x+a in cs:
                            ok = False
                            break
                    if ok: c_add += 1
                c_cost = c_add
                
            delta = new_cost - c_cost
            if delta < 0 or random.random() < pow(2.718, -delta/T):
                curr_mask = new_mask
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_mask = new_mask
                    best = []
                    m = new_mask
                    while m:
                        l = m & -m
                        best.append(l.bit_length())
                        m ^= l
            T *= cool
            
        return best

    result = run_sa()
    
    # Final safety: ensure result is valid and maximal
    if not result:
        return greedy_maximal()
        
    # Check if result is Sidon
    if not is_sidon(result):
        return greedy_maximal()
        
    # Check if Maximal
    # If not maximal, extend it
    sums = set()
    for i in range(len(result)):
        for j in range(i, len(result)):
            sums.add(result[i] + result[j])
            
    res_set = set(result)
    for x in range(1, N+1):
        if x in res_set: continue
        if 2*x in sums: continue
        ok = True
        for a in res_set:
            if x+a in sums:
                ok = False
                break
        if ok:
            # Add x
            result.append(x)
            res_set.add(x)
            sums.add(2*x)
            for a in res_set - {x}:
                sums.add(x+a)
                
    return result

# EVOLVE-BLOCK-END
