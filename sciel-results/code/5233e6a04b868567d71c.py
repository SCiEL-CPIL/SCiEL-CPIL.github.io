import time
from math import gcd

_start = time.time()
_TIMEOUT = 85

def construct():
    if time.time() - _start > _TIMEOUT:
        return [1]
    
    N = 256
    values = list(range(1, N + 1))
    
    # Precompute coprime masks
    # mask[i] is a bitmask of indices j (0..N) such that gcd(i, j) == 1
    # We use 0-indexed for the mask, so value v is at index v-1
    masks = [0] * (N + 1)
    for i in range(1, N + 1):
        m = 0
        for j in range(1, N + 1):
            if gcd(i, j) == 1:
                m |= (1 << (j - 1))
        masks[i] = m
        
    # Order indices by value descending
    # Index 0 corresponds to value 1
    order = sorted(range(1, N + 1), key=lambda x: -x)
    
    # Initial best solution using a greedy heuristic to prune early
    # Strategy: Greedy merge of prime powers
    def get_initial_solution():
        # Find prime powers
        is_prime = [True] * (N + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(N**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, N + 1, i):
                    is_prime[j] = False
        primes = [p for p in range(2, N + 1) if is_prime[p]]
        
        set1 = [1]
        for p in primes:
            val = p
            while val * p <= N:
                val *= p
            set1.append(val)
        
        # Greedy merge
        current_set = list(set1)
        improved = True
        while improved:
            improved = False
            best_gain = 0
            best_pair = None
            s = sorted(current_set)
            for i in range(len(s)):
                for j in range(i + 1, len(s)):
                    prod = s[i] * s[j]
                    if prod > N:
                        break
                    if prod in current_set:
                        continue
                    if gcd(s[i], s[j]) != 1:
                        continue
                    gain = prod - s[i] - s[j]
                    if gain > best_gain:
                        best_gain = gain
                        best_pair = (s[i], s[j])
            if best_pair:
                a, b = best_pair
                current_set.remove(a)
                current_set.remove(b)
                current_set.append(a * b)
                improved = True
        return current_set

    init_sol = get_initial_solution()
    best_sum = sum(init_sol)
    best_set = list(init_sol)
    
    # B&B DFS
    # To speed up, we can compress the order.
    # Actually, standard B&B on 256 nodes might be slow in Python if the bound is weak.
    # Let's try to improve the bound.
    # Bound: current_sum + sum of values of all available nodes.
    
    # We will use a global variable for best
    global_best_sum = best_sum
    global_best_set = best_set
    
    # Sort order for B&B
    # Using the full range 1..256
    # To make recursion efficient, we pass the index in the sorted order
    
    # Precompute prefix sums of values in the sorted order?
    # No, because availability changes.
    
    # Let's implement DFS with a stack or recursion.
    # Recursion depth 256 is fine.
    
    def dfs(idx, current_sum, available_mask):
        nonlocal global_best_sum, global_best_set
        
        if time.time() - _start > _TIMEOUT - 2:
            return
            
        # Pruning
        # Calculate upper bound
        # Sum of values of all nodes in available_mask with index >= idx?
        # This is expensive to calculate every time.
        # Approximation: Sum of all values in available_mask.
        # Let's precompute value_sum for masks? Impossible.
        
        # Alternative: Just check if current_sum + remaining_potential <= global_best_sum
        # remaining_potential: sum of values of all nodes in available_mask.
        # We can maintain a running sum of available values?
        # When we include a node, we remove it and its neighbors.
        
        # Let's pass `avail_sum` as well.
        # Initial avail_sum = sum(1..256) = 32896
        
    # This state (idx, current_sum, available_mask, avail_sum) is good.
    # But updating avail_sum requires knowing which nodes are removed.
    # removed = available_mask & ~new_mask
    # We need sum of values of bits in `removed`.
    # This is a popcount-like operation but with weights.
    
    # Let's precompute weight_sum for small masks? No, masks are 256 bits.
    # We can iterate through bits of `removed` to subtract from avail_sum.
    # Number of bits set in removed might be large.
    
    # Simpler: Just use the weak bound `current_sum + avail_sum <= global_best_sum`.
    # And hope the greedy initial solution is strong enough.
    
    # Let's try to optimize the order.
    # Nodes with high value and low degree (few conflicts) are good to pick early?
    # Or high value and high degree (block many others)?
    # Usually, sorting by value descending is good for Max Weight Independent Set.
    
    # Let's use a list for the order of nodes to consider.
    # We will iterate through all nodes 256..1.
    
    # To speed up bit iteration:
    # Instead of checking all 256 nodes at each step, we can just iterate.
    
    # Let's implement the DFS.
    
    import sys
    sys.setrecursionlimit(3000)
    
    # Precompute value of each index
    val = [0] * (N + 1)
    for i in range(1, N + 1):
        val[i] = i
        
    # Initial mask: all nodes available
    # Bit 0 is value 1, Bit 255 is value 256
    initial_mask = (1 << N) - 1
    initial_avail_sum = sum(range(1, N + 1))
    
    # Order: 256, 255, ..., 1
    # We need to map index in order to node value
    # order[i] = 256 - i
    # So at step i, we consider node v = 256 - i
    # Check if v is in available_mask
    
    def dfs_step(i, current_sum, available_mask, avail_sum):
        nonlocal global_best_sum, global_best_set
        
        if time.time() - _start > _TIMEOUT - 2:
            return
            
        if i == N:
            if current_sum > global_best_sum:
                global_best_sum = current_sum
                # Reconstruct set? We lost the set.
                # We need to store the set or reconstruct from path.
                # Storing set in recursion is expensive.
                # We can store the choice at each step.
                # But depth is 256.
                # Alternatively, just return the sum and find the set separately?
                # No, we need the set to return.
                
                # Let's modify DFS to pass the current set?
                # List copy at each step is O(N). 256 * 256 * 2^depth? Too slow.
                
                # Better: Store the path of decisions.
                # Or, since we only care about the best, we can store the best set
                # when we update global_best_sum.
                # But we need the set corresponding to current_sum.
                
                # Let's pass `current_set` as a tuple?
                # Tuples are immutable, so we can share prefixes?
                # No, we are adding/removing.
                
                # Let's try passing a list and copying it.
                # If the search space is pruned heavily, this might be okay.
                
                pass
        
        # If node v is not available, skip (exclude)
        v = N - i
        if not (available_mask & (1 << (v - 1))):
            dfs_step(i + 1, current_sum, available_mask, avail_sum)
            return
            
        # Option 1: Exclude v
        # Bound check for exclude branch
        if current_sum + avail_sum > global_best_sum:
             dfs_step(i + 1, current_sum, available_mask, avail_sum)
             
        # Option 2: Include v
        # New mask
        new_mask = available_mask & masks[v]
        # Calculate sum of removed nodes
        removed_mask = available_mask ^ new_mask
        # Sum values of bits in removed_mask
        removed_sum = 0
        rm = removed_mask
        while rm:
            # Get lowest set bit
            lsb = rm & -rm
            # Index of lsb
            # bit_length - 1 gives index
            idx_bit = lsb.bit_length() - 1
            removed_sum += idx_bit + 1
            rm ^= lsb
            
        new_avail_sum = avail_sum - removed_sum
        
        # Bound check for include branch
        if current_sum + v + new_avail_sum > global_best_sum:
             # We need to update the set
             # Let's pass the set
             pass

    # This approach with set passing is risky for performance.
    # Alternative:
    # Run B&B to find the max sum.
    # Then, run a second pass to reconstruct the set?
    # No, the set is determined by the choices.
    
    # Let's try a different tactic:
    # Since N=256 is small, maybe we can use a MILP solver? No, standard lib only.
    
    # Let's stick to the Greedy Merge but with better initial set and more aggressive local search.
    # The B&B might be too slow in Python for 256 nodes without C-level bit operations.
    
    # Let's improve the Greedy Merge.
    # 1. Start with all prime powers.
    # 2. Merge greedily.
    # 3. Local Search:
    #    - 1-for-1 swap: Replace a with b if b > a and b is coprime to all others.
    #    - 1-for-2 swap: Replace a with b, c if b+c > a and b,c coprime to all others and each other.
    #    - 2-for-1 swap: Replace a, b with c if c > a+b and c coprime to all others.
    #    - 2-for-2 swap: Replace a, b with c, d.
    
    # The current program does 2-for-1 (merge).
    # It does NOT do 1-for-1 (replace with a larger coprime number).
    # Example: Set has {2, 3, 5}. Can we replace 2 with 253 (11*23)?
    # 253 is coprime to 3 and 5.
    # Sum change: 253 - 2 = 251. Huge gain.
    # Why didn't the greedy merge do this?
    # Because it started with prime powers. 2 is a prime power. 253 is not a prime power.
    # The greedy merge only combines existing elements. It doesn't introduce new numbers.
    
    # So, we need to allow "introduction" of new numbers.
    
    # Strategy:
    # 1. Identify a "core" set of numbers.
    # 2. For each number in the core, see if it can be replaced by a larger coprime number.
    # 3. Or, see if a pair can be replaced by a single larger number.
    
    # Let's implement a Local Search that considers all numbers 1..256.
    
    def local_search(s):
        s = list(s)
        improved = True
        while improved:
            improved = False
            # Try 1-for-1 replacement
            for i in range(len(s)):
                a = s[i]
                best_replacement = a
                for b in range(256, a, -1):
                    if b == a:
                        continue
                    if b in s:
                        continue
                    # Check if b is coprime to all other elements
                    ok = True
                    for j in range(len(s)):
                        if j == i:
                            continue
                        if gcd(b, s[j]) != 1:
                            ok = False
                            break
                    if ok:
                        best_replacement = b
                        break # Found largest b
                if best_replacement != a:
                    s[i] = best_replacement
                    improved = True
                    break # Restart loop
            
            # Try 2-for-1 replacement (merge)
            if not improved:
                s_sorted = sorted(s)
                for i in range(len(s_sorted)):
                    for j in range(i + 1, len(s_sorted)):
                        a, b = s_sorted[i], s_sorted[j]
                        prod = a * b
                        if prod > 256:
                            continue
                        if prod in s:
                            continue
                        if gcd(a, b) != 1:
                            continue
                        # Check if prod is coprime to all other elements
                        ok = True
                        for k in range(len(s)):
                            if s[k] == a or s[k] == b:
                                continue
                            if gcd(prod, s[k]) != 1:
                                ok = False
                                break
                        if ok:
                            # Remove a, b, add prod
                            s.remove(a)
                            s.remove(b)
                            s.append(prod)
                            improved = True
                            break
                    if improved:
                        break
                        
            # Try 1-for-2 replacement (split)
            if not improved:
                # This is harder. Replace a with b, c.
                # b*c < a? No, b+c > a.
                # b, c coprime to a? No, b, c must be coprime to all other elements in s\{a}.
                # And b, c coprime to each other.
                # And b, c not in s.
                # We want to maximize b+c.
                # This is complex. Let's skip for now and see if 1-for-1 and 2-for-1 are enough.
                pass

        return s

    # Initial set: Prime powers
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    primes = [p for p in range(2, N + 1) if is_prime[p]]
    
    init_set = [1]
    for p in primes:
        val = p
        while val * p <= N:
            val *= p
        init_set.append(val)
        
    current_set = local_search(init_set)
    
    # Run local search multiple times with different starting points?
    # Or just refine the current one.
    
    # Verify
    def is_valid(s):
        if len(s) != len(set(s)):
            return False
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                if gcd(s[i], s[j]) != 1:
                    return False
        return True

    if not is_valid(current_set):
        return [1] + primes

    return [int(x) for x in current_set]

# EVOLVE-BLOCK-END
