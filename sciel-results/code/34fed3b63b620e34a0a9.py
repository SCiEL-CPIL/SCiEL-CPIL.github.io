import time
import random
from fractions import Fraction

def get_primes(limit):
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = 0
    return [i for i in range(2, limit + 1) if sieve[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    primes = get_primes(1000)
    prime_set = set(primes)

    # Initial set: primes >= 13
    initial_primes = [p for p in primes if p >= 13]
    
    def sum_recip_frac(prime_list):
        return sum(Fraction(1, p) for p in prime_list)

    # Fast greedy check if 1..L can be covered by given primes
    def greedy_can_cover(L, prime_list):
        used_primes = set()
        residues = {}
        covered = [False] * (L + 1)
        
        for x in range(1, L + 1):
            if covered[x]:
                continue
            
            best_p = None
            best_gain = -1
            best_r = None
            
            for p in prime_list:
                if p in used_primes:
                    continue
                
                r = x % p
                gain = 0
                for k in range(0, (L - x) // p + 1):
                    y = x + k * p
                    if not covered[y]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
                    best_r = r
            
            if best_p is None or best_gain <= 0:
                return None
            
            used_primes.add(best_p)
            residues[best_p] = best_r
            
            for k in range(0, (L - x) // best_p + 1):
                y = x + k * best_p
                covered[y] = True
                
        return residues

    # CSP-style backtracking with forward checking for residue assignment
    def csp_solve(L, prime_list, time_budget=5.0):
        """
        Solve the covering problem using backtracking with forward checking.
        Variables: primes in prime_list
        Domains: residues 0..p-1
        Constraints: every x in 1..L is covered by at least one prime
        """
        t0 = time.time()
        n_primes = len(prime_list)
        
        # Precompute for each prime p, the set of x in 1..L that p covers for each residue
        # coverage[p_idx][r] = set of x covered
        # This is expensive for large L, so we use a different approach
        
        # Alternative: work with the uncovered set
        # State: (uncovered_set, assigned_primes, residues)
        # Use bitset for uncovered for efficiency
        
        # For L up to ~1000, we can use Python integers as bitsets
        # bit x-1 is 1 if x is uncovered
        
        # Initial uncovered: all of 1..L
        initial_uncovered = (1 << L) - 1  # bits 0 to L-1 set, representing x=1 to L
        
        # For each prime p, precompute the bitmask for each residue r
        # mask[p_idx][r] has bits set for x in 1..L where x % p == r
        prime_masks = []
        for p in prime_list:
            masks = []
            for r in range(p):
                m = 0
                # x ranges from 1 to L, x % p == r
                # First x >= 1 with x % p == r: 
                # if r == 0, first x is p (if p <= L), else none
                # if r > 0, first x is r (if r <= L), else none
                first_x = r if r > 0 else p
                if first_x > L:
                    masks.append(0)
                    continue
                for x in range(first_x, L + 1, p):
                    m |= (1 << (x - 1))
                masks.append(m)
            prime_masks.append(masks)
        
        # Order primes: smaller primes first (they cover more, so assign them first for better pruning)
        # Actually, for CSP, we should assign variables that constrain the most first
        # Smaller primes cover more numbers, so they constrain more
        order = list(range(n_primes))
        order.sort(key=lambda i: prime_list[i])  # ascending by prime value
        
        # Backtracking state
        best_solution = None
        solution_count = 0
        max_backtracks = 50000  # limit to avoid exponential blowup
        
        def backtrack(idx, uncovered, assigned):
            nonlocal best_solution, solution_count, max_backtracks
            
            if time.time() - t0 > time_budget:
                return False
            if solution_count > max_backtracks:
                return False
            
            # Check if all covered
            if uncovered == 0:
                # Found a solution
                if best_solution is None:
                    best_solution = assigned.copy()
                solution_count += 1
                return True
            
            # If no more primes to assign but still uncovered
            if idx == len(order):
                return False
            
            p_idx = order[idx]
            p = prime_list[p_idx]
            
            # Forward checking: for each possible residue, check if it helps
            # Try residues in order of how much they reduce uncovered
            candidates = []
            for r in range(p):
                mask = prime_masks[p_idx][r]
                new_uncovered = uncovered & ~mask
                # Count newly covered
                newly_covered = bin(uncovered & mask).count('1')
                if newly_covered > 0:
                    candidates.append((newly_covered, r, new_uncovered))
            
            # Sort by most newly covered first (MRV-like: most constraining first)
            candidates.sort(key=lambda x: -x[0])
            
            for _, r, new_uncovered in candidates:
                assigned[p_idx] = r
                if backtrack(idx + 1, new_uncovered, assigned):
                    return True
                # If we found a solution, we can stop (we just need one)
                if best_solution is not None:
                    return True
            
            return False
        
        assigned = [None] * n_primes
        if backtrack(0, initial_uncovered, assigned):
            # Convert back to dict
            result = {}
            for i, p_idx in enumerate(order):
                if assigned[p_idx] is not None:
                    result[prime_list[p_idx]] = assigned[p_idx]
            # Ensure all primes have an assignment (for those not used, assign 0)
            for p in prime_list:
                if p not in result:
                    result[p] = 0
            return result
        return None

    # Start with seed configuration
    current_primes = initial_primes[:]
    current_L = 163
    
    # Verify seed with greedy
    seed_assign = greedy_can_cover(current_L, current_primes)
    if seed_assign is None:
        # Try CSP
        seed_assign = csp_solve(current_L, current_primes, time_budget=10.0)
    if seed_assign is None:
        # Fallback to seed
        return [[p, (i + 1) % p] for i, p in enumerate(initial_primes)]

    # Try to extend L
    candidates_to_add = [p for p in primes if p < 13]
    candidates_to_add.sort(reverse=True) # Try 11, 7, 5, 3, 2
    
    # Track best known solution
    best_primes = current_primes[:]
    best_L = current_L
    best_assignment = seed_assign.copy()
    
    while time.time() < deadline:
        next_L = current_L + 1
        
        # Try current set with greedy first (fast)
        assign = greedy_can_cover(next_L, current_primes)
        if assign is not None:
            current_L = next_L
            best_L = max(best_L, current_L)
            best_assignment = assign.copy()
            best_primes = current_primes[:]
            continue
        
        # Try CSP with current set
        assign = csp_solve(next_L, current_primes, time_budget=5.0)
        if assign is not None:
            current_L = next_L
            best_L = max(best_L, current_L)
            best_assignment = assign.copy()
            best_primes = current_primes[:]
            continue
        
        # If we can't extend L with current primes, try adding a new prime
        added = False
        for p_add in candidates_to_add:
            if p_add in current_primes:
                continue
            
            # Check budget
            new_primes = current_primes + [p_add]
            if sum_recip_frac(new_primes) > 1:
                continue
            
            # Try with new prime
            assign = greedy_can_cover(next_L, new_primes)
            if assign is None:
                assign = csp_solve(next_L, new_primes, time_budget=5.0)
            
            if assign is not None:
                current_primes = new_primes
                candidates_to_add.remove(p_add)
                current_L = next_L
                best_L = max(best_L, current_L)
                best_assignment = assign.copy()
                best_primes = current_primes[:]
                added = True
                break
        
        if not added:
            # Can't extend further with current strategy
            # Try a different strategy: local search on residues for a fixed L
            break
    
    # If we have time, try to improve L via local search on residues
    if time.time() < deadline - 10:
        # Use the best primes we have
        p_set = best_primes
        k = len(p_set)
        
        # Get current residues
        res_list = [best_assignment.get(p, 0) for p in p_set]
        
        # Local search to try to extend coverage
        # We'll try to increase L by adjusting residues
        
        # First, let's find the actual L for the current configuration
        def eval_L(res_list, p_set, L_limit=100000):
            covered = [False] * (L_limit + 1)
            for i, p in enumerate(p_set):
                r = res_list[i]
                for x in range(r if r > 0 else p, L_limit + 1, p):
                    if x >= 1:
                        covered[x] = True
            # Find first False in covered[1:]
            for x in range(1, L_limit + 1):
                if not covered[x]:
                    return x - 1
            return L_limit
        
        current_L_check = eval_L(res_list, p_set)
        
        # Hill climbing on residues
        improvement_found = True
        while improvement_found and time.time() < deadline - 5:
            improvement_found = False
            for i in range(k):
                p = p_set[i]
                old_r = res_list[i]
                # Try all residues? Or random?
                # For small p, try all. For large p, try random.
                if p <= 50:
                    candidates = list(range(p))
                else:
                    candidates = [random.randint(0, p - 1) for _ in range(10)]
                
                for new_r in candidates:
                    if new_r == old_r:
                        continue
                    res_list[i] = new_r
                    new_L = eval_L(res_list, p_set)
                    if new_L > current_L_check:
                        current_L_check = new_L
                        improvement_found = True
                        if current_L_check > best_L:
                            best_L = current_L_check
                            best_assignment = {p_set[j]: res_list[j] for j in range(k)}
                            best_primes = p_set[:]
                        break
                    else:
                        res_list[i] = old_r
        
    # Final verification and return
    if not best_assignment:
        return [[p, (i + 1) % p] for i, p in enumerate(initial_primes)]
    
    # Ensure all primes in best_primes are in the assignment
    for p in best_primes:
        if p not in best_assignment:
            best_assignment[p] = 0
    
    return [[p, best_assignment[p]] for p in best_primes]
