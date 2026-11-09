"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import sys
from collections import defaultdict

def construct():
    deadline = time.time() + 55  # safety margin before 90s kill
    n_base = 210
    max_val = 10210
    min_val = n_base + 1  # strictly larger than 210
    
    # Factorize helper
    def factorize(n):
        factors = {}
        d = 2
        while d * d <= n:
            if n % d == 0:
                cnt = 0
                while n % d == 0:
                    n //= d
                    cnt += 1
                factors[d] = cnt
            d += 1 if d == 2 else 2
        if n > 1:
            factors[n] = 1
        return factors
    
    def sf_kernel(n):
        """Square-free kernel of n."""
        k = 1
        for p, e in factorize(n).items():
            if e % 2 == 1:
                k *= p
        return k
    
    # Precompute square-free kernels for all candidates
    candidates = list(range(min_val, max_val + 1))
    kernels = {}
    for x in candidates:
        kernels[x] = sf_kernel(x)
    
    # Collect all primes that appear in any kernel
    all_primes = set()
    for k in kernels.values():
        for p, e in factorize(k).items():
            all_primes.add(p)
    all_primes = sorted(all_primes)
    prime_to_idx = {p: i for i, p in enumerate(all_primes)}
    num_primes = len(all_primes)
    
    # Target: square-free kernel of product must be 210 = 2*3*5*7
    # So we need XOR of selected vectors = vector for 210
    target_vec = 0
    for p in [2, 3, 5, 7]:
        if p in prime_to_idx:
            target_vec |= (1 << prime_to_idx[p])
    
    # Convert each number's kernel to a bitmask
    num_to_mask = {}
    for x in candidates:
        mask = 0
        for p, e in factorize(kernels[x]).items():
            if e % 2 == 1 and p in prime_to_idx:
                mask |= (1 << prime_to_idx[p])
        num_to_mask[x] = mask
    
    # Gaussian elimination over GF(2) to find if target is in span
    # and reconstruct the solution
    def solve_gauss(masks, target):
        """
        Given list of (number, mask) pairs, find if target is in their span.
        Returns (list_of_numbers, max_number) or (None, None).
        Uses row-echelon form and back-substitution.
        """
        if not masks:
            return None, None
        
        # Build augmented matrix: each row is (mask, index)
        # We'll do Gaussian elimination to find a subset that XORs to target
        
        # Create list of (mask, number) sorted by number for determinism
        rows = [(mask, num) for num, mask in masks if mask != 0]
        
        if not rows:
            # All masks are zero, can only achieve target=0
            if target == 0:
                return [], 0
            return None, None
        
        # Gaussian elimination
        pivot_rows = []  # list of (pivot_bit, mask, number)
        
        for mask, num in rows:
            m = mask
            # Try to eliminate using existing pivots
            for pivot_bit, _, _ in pivot_rows:
                if m & (1 << pivot_bit):
                    m ^= (1 << pivot_bit)
                    # We need to track which original rows were used
                    # This simple approach doesn't track that, so let's use a different method
            # Actually, let's use a proper approach with tracking
            
        # Better approach: use a dictionary-based Gaussian elimination
        # that tracks the combination
        
        # Let's use a different strategy: iterative search with memoization
        # For small sets, we can do BFS on the XOR state
        
        # State: current XOR value
        # We want to reach target
        # Each step: XOR with one of the masks
        # We track: (xor_state) -> (set_of_numbers_used, max_number)
        
        # But state space could be large. Let's limit by number of elements.
        
        # Alternative: since we're binary searching on max value, the set size is bounded.
        # Let's try to find a small solution (1-6 elements) using randomized/local search.
        
        return None, None
    
    # Better approach: for each candidate max M, check if target is achievable
    # using numbers in (210, M]
    # Use BFS with state = XOR value, tracking the set of used numbers
    
    # Since the number of distinct primes is limited, the state space is bounded.
    # Let's estimate: primes up to 10210, but in practice only those appearing in (210, M]
    
    # For efficiency, let's try increasing M and check feasibility
    
    best_solution = [840]  # known single-element solution
    best_raw = 630
    
    # Binary search on M
    # Check if solution exists with all elements <= M
    def check_feasible(M, time_limit):
        """Check if target is achievable using numbers in (210, M]."""
        if M < min_val:
            return None
        
        # Get masks for numbers in (210, M]
        masks = [(num, num_to_mask[num]) for num in range(min_val, M + 1) if num_to_mask[num] != 0]
        
        if not masks:
            if target_vec == 0:
                return []
            return None
        
        # BFS on XOR state
        # State: current XOR value
        # We track the minimum max-element to achieve each state
        # and the set of numbers used
        
        # To keep memory manageable, limit the number of states
        start = time.time()
        
        # dist[xor_state] = (max_element_used, frozenset_of_numbers)
        # But frozensets are expensive. Let's use a different approach.
        
        # Actually, let's use a priority queue (Dijkstra-like) where the cost is max element
        # State: xor_value
        # We want to minimize the max element used
        
        import heapq
        
        # (max_element, xor_state, frozenset_of_numbers)
        # But frozensets make this expensive. Let's limit the set size.
        
        # Alternative: just check if target is reachable, don't care about min max
        # Since we're binary searching on M, we just need yes/no
        
        # BFS with visited set
        visited = {0}  # start with XOR = 0 (empty product)
        # For each state, we don't track which numbers were used, just that it's reachable
        
        # But we need to reconstruct the solution. Let's track parent pointers.
        # parent[xor_state] = (prev_xor_state, number_used)
        
        parent = {}  # xor_state -> (prev_xor_state, number)
        
        # Queue: (xor_state)
        from collections import deque
        queue = deque([0])
        
        # Limit the number of states to avoid memory explosion
        max_states = 500000
        state_count = 1
        
        while queue and state_count < max_states:
            if time.time() - start > time_limit:
                break
            state = queue.popleft()
            
            if state == target_vec:
                # Reconstruct path
                path = []
                cur = state
                while cur != 0:
                    prev, num = parent[cur]
                    path.append(num)
                    cur = prev
                path.reverse()
                return path
            
            # Try adding each number
            for num, mask in masks:
                new_state = state ^ mask
                if new_state not in visited:
                    visited.add(new_state)
                    parent[new_state] = (state, num)
                    queue.append(new_state)
                    state_count += 1
                    if state_count >= max_states:
                        break
        
        return None
    
    # Binary search on M
    lo = min_val
    hi = 839  # we know 840 works, try to find better
    
    # First, verify that 840 works
    sol_840 = check_feasible(840, 10)
    if sol_840 is not None:
        # This should find [840] or something else
        pass
    
    # Binary search for the smallest M where a solution exists
    # But we want to minimize M, so we search from low to high
    
    # Actually, let's just iterate M from min_val upward and check
    # This is O(M * check_cost) which could be slow
    
    # Better: binary search
    # But the feasibility is monotonic: if feasible at M, feasible at M+1
    
    # Let's try binary search
    # First check if feasible at hi=839
    # If not, the answer is 840
    
    # Actually, let's just try to find the best solution directly
    
    # Try increasing M from min_val
    # But that's too slow. Let's use binary search.
    
    # Check feasibility at various M values
    # We know M=840 is feasible (single element 840)
    
    # Let's try M values: 211, 212, ..., 839
    # But that's 628 checks, each potentially expensive.
    
    # Let's be smarter: use the structure of the problem.
    
    # For 2-element solutions: a * b * 210 = perfect square
    # a * b = 210 * m^2 for some integer m
    # a, b in (210, M], a != b
    
    # Let's search for 2-element solutions with max < 840
    found_better = False
    best_2elem = None
    best_2elem_max = 840
    
    # For each pair (a, b) with 211 <= a < b <= 839, check if a*b*210 is a perfect square
    # a*b*210 = k^2 => a*b = k^2/210
    # For a*b to be integer and 210 | k^2, we need 210 | k (since 210 is square-free)
    # k = 210 * m, so a*b = 210 * m^2
    
    # m range: a*b in (211*212, 839*838) = (44732, 703082)
    # 210*m^2 in that range => m^2 in (213, 3348) => m in 15..57
    
    for m in range(1, 100):  # be generous
        prod = 210 * m * m
        if prod < 211 * 212:
            continue
        if prod > 839 * 839:
            break
        # Find factor pairs (a, b) with 211 <= a < b <= 839
        # a * b = prod
        a_min = max(211, (prod + 839) // 839)  # a >= prod/839 since b <= 839
        a_max = min(838, prod // 212)  # a <= prod/212 since b >= 212
        a_max = min(a_max, 838)  # a < b <= 839 => a <= 838
        
        for a in range(a_min, a_max + 1):
            if prod % a == 0:
                b = prod // a
                if a < b and 211 <= a and b <= 839:
                    max_val_found = b
                    if max_val_found < best_2elem_max:
                        best_2elem_max = max_val_found
                        best_2elem = [a, b]
                        found_better = True
    
    if found_better:
        best_solution = best_2elem
        best_raw = best_2elem_max - 210
    
    # Try 3-element solutions
    # a * b * c * 210 = perfect square
    # a * b * c = 210 * m^2
    # This is more complex. Let's use the BFS approach for a limited range.
    
    # Try to find any solution with max < best_raw + 210 using BFS
    if best_raw > 0:
        search_max = best_raw + 210 - 1  # we want max < current best
        if search_max >= min_val:
            # Use BFS to find a solution with all elements <= search_max
            # Limit the search
            start_time = time.time()
            time_budget = min(30, deadline - time.time())
            
            masks_in_range = [(num, num_to_mask[num]) for num in range(min_val, search_max + 1) if num_to_mask[num] != 0]
            
            if masks_in_range:
                # BFS
                from collections import deque
                visited = {0}
                parent = {}
                queue = deque([0])
                max_states = 200000
                state_count = 1
                found_sol = None
                
                while queue and state_count < max_states:
                    if time.time() - start_time > time_budget:
                        break
                    state = queue.popleft()
                    
                    if state == target_vec:
                        # Reconstruct
                        path = []
                        cur = state
                        while cur != 0:
                            prev, num = parent[cur]
                            path.append(num)
                            cur = prev
                        path.reverse()
                        found_sol = path
                        break
                    
                    for num, mask in masks_in_range:
                        new_state = state ^ mask
                        if new_state not in visited:
                            visited.add(new_state)
                            parent[new_state] = (state, num)
                            queue.append(new_state)
                            state_count += 1
                            if state_count >= max_states:
                                break
                
                if found_sol is not None:
                    max_in_sol = max(found_sol)
                    if max_in_sol < best_raw + 210:
                        best_solution = found_sol
                        best_raw = max_in_sol - 210
    
    # Verify the solution
    if best_solution:
        prod = 1
        for x in best_solution:
            prod *= x
        prod *= 210
        r = math.isqrt(prod)
        if r * r != prod:
            # Solution is invalid, fall back to known good solution
            best_solution = [840]
            best_raw = 630
    
    # Ensure distinct integers
    if len(set(best_solution)) != len(best_solution):
        best_solution = [840]
        best_raw = 630
    
    # Ensure all > 210 and <= 10210
    if any(x <= 210 or x > 10210 for x in best_solution):
        best_solution = [840]
        best_raw = 630
    
    return [int(x) for x in best_solution]
# EVOLVE-BLOCK-END
