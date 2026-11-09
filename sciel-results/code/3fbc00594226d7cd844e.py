import time
import random
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def get_primes(limit):
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = 0
        return [i for i in range(2, limit + 1) if sieve[i]]

    primes = get_primes(1000)
    prime_set = set(primes)

    # Precompute reciprocal sums as Fractions for exact arithmetic
    def sum_recip(prime_list):
        s = Fraction(0, 1)
        for p in prime_list:
            s += Fraction(1, p)
        return s

    # Fast coverage check using bitsets
    def coverage_bitset(L, assignment):
        """Return bitset where bit i is 1 if (i+1) is covered."""
        bits = 0
        for p, r in assignment.items():
            # x % p == r for x in 1..L
            first = r if r > 0 else p
            if first <= L:
                for x in range(first, L + 1, p):
                    bits |= (1 << (x - 1))
        return bits

    def find_first_gap(bits, L):
        """Find smallest x in 1..L not covered (bit x-1 is 0). Return x or L+1 if all covered."""
        for x in range(1, L + 1):
            if not (bits >> (x - 1)) & 1:
                return x
        return L + 1

    # Initial seed: primes >= 13 with simple assignment
    initial_primes = [p for p in primes if p >= 13]
    
    # Build initial assignment via greedy
    def greedy_assign(L, prime_list):
        covered = [False] * (L + 1)
        assign = {}
        for x in range(1, L + 1):
            if covered[x]:
                continue
            best_p, best_r, best_gain = None, None, -1
            for p in prime_list:
                if p in assign:
                    continue
                r = x % p
                gain = sum(1 for k in range(x, L + 1, p) if not covered[k])
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
                    best_r = r
            if best_p is None:
                return None
            assign[best_p] = best_r
            for k in range(best_r if best_r > 0 else best_p, L + 1, best_p):
                if k <= L:
                    covered[k] = True
        return assign

    # Get initial solution
    current_L = 163
    initial_assign = greedy_assign(current_L, initial_primes)
    if initial_assign is None:
        # Fallback
        return [[p, i % p] for i, p in enumerate(initial_primes)]

    best_primes = initial_primes[:]
    best_L = current_L
    best_assign = initial_assign.copy()

    # Branch and bound over prime-residue selections
    # Nodes: (prime, residue) pairs
    # We want to select at most one residue per prime, sum(1/p) <= 1, maximize coverage
    
    # Strategy: iteratively try to extend coverage by modifying the set
    # Use local search with simulated annealing for efficiency
    
    def evaluate_L(assignment, prime_list):
        """Find largest L such that 1..L all covered."""
        if not assignment:
            return 0
        max_check = 10000
        bits = 0
        for p, r in assignment.items():
            first = r if r > 0 else p
            for x in range(first, max_check + 1, p):
                bits |= (1 << (x - 1))
        L = 0
        for x in range(1, max_check + 1):
            if (bits >> (x - 1)) & 1:
                L = x
            else:
                break
        return L

    # Simulated annealing with gap-focused moves
    def simulated_annealing(L_target, prime_pool, initial_assign, time_budget):
        t0 = time.time()
        current_assign = initial_assign.copy()
        current_L = evaluate_L(current_assign, prime_pool)
        best_assign = current_assign.copy()
        best_L = current_L
        temperature = 10.0
        cooling = 0.995
        
        primes_in_use = set(current_assign.keys())
        
        while time.time() - t0 < time_budget:
            # Generate a move
            move_type = random.random()
            if move_type < 0.4:
                # Change residue of a used prime
                if not primes_in_use:
                    continue
                p = random.choice(list(primes_in_use))
                old_r = current_assign[p]
                new_r = random.randint(0, p - 1)
                if new_r == old_r:
                    continue
                current_assign[p] = new_r
                new_L = evaluate_L(current_assign, prime_pool)
                delta = new_L - current_L
                if delta > 0 or random.random() < pow(2.718, delta / max(temperature, 0.1)):
                    current_L = new_L
                    if new_L > best_L:
                        best_L = new_L
                        best_assign = current_assign.copy()
                else:
                    current_assign[p] = old_r
            elif move_type < 0.7:
                # Add a new prime
                available = [p for p in prime_pool if p not in primes_in_use]
                if not available:
                    continue
                # Check budget
                s = sum_recip(list(primes_in_use))
                candidates = [p for p in available if s + Fraction(1, p) <= 1]
                if not candidates:
                    continue
                p = random.choice(candidates)
                r = random.randint(0, p - 1)
                current_assign[p] = r
                primes_in_use.add(p)
                new_L = evaluate_L(current_assign, prime_pool)
                delta = new_L - current_L
                if delta > 0 or random.random() < pow(2.718, delta / max(temperature, 0.1)):
                    current_L = new_L
                    if new_L > best_L:
                        best_L = new_L
                        best_assign = current_assign.copy()
                else:
                    del current_assign[p]
                    primes_in_use.discard(p)
            elif move_type < 0.9:
                # Remove a prime
                if len(primes_in_use) <= 3:
                    continue
                p = random.choice(list(primes_in_use))
                r = current_assign[p]
                del current_assign[p]
                primes_in_use.discard(p)
                new_L = evaluate_L(current_assign, prime_pool)
                delta = new_L - current_L
                if delta > 0 or random.random() < pow(2.718, delta / max(temperature, 0.1)):
                    current_L = new_L
                    if new_L > best_L:
                        best_L = new_L
                        best_assign = current_assign.copy()
                else:
                    current_assign[p] = r
                    primes_in_use.add(p)
            else:
                # Swap: remove one, add another
                if not primes_in_use:
                    continue
                p_remove = random.choice(list(primes_in_use))
                available = [p for p in prime_pool if p not in primes_in_use]
                if not available:
                    continue
                s_after_remove = sum_recip([p for p in primes_in_use if p != p_remove])
                candidates = [p for p in available if s_after_remove + Fraction(1, p) <= 1]
                if not candidates:
                    continue
                p_add = random.choice(candidates)
                r_add = random.randint(0, p_add - 1)
                r_remove = current_assign[p_remove]
                del current_assign[p_remove]
                primes_in_use.discard(p_remove)
                current_assign[p_add] = r_add
                primes_in_use.add(p_add)
                new_L = evaluate_L(current_assign, prime_pool)
                delta = new_L - current_L
                if delta > 0 or random.random() < pow(2.718, delta / max(temperature, 0.1)):
                    current_L = new_L
                    if new_L > best_L:
                        best_L = new_L
                        best_assign = current_assign.copy()
                else:
                    del current_assign[p_add]
                    primes_in_use.discard(p_add)
                    current_assign[p_remove] = r_remove
                    primes_in_use.add(p_remove)
            
            temperature *= cooling
        
        return best_assign, best_L

    # Main loop: try to extend coverage
    all_primes = primes[:]  # all primes up to 1000
    
    while time.time() < deadline - 5:
        # Try to extend by 1
        next_L = best_L + 1
        
        # First try with current primes
        assign = greedy_assign(next_L, best_primes)
        if assign is not None:
            # Verify
            actual = evaluate_L(assign, best_primes)
            if actual >= next_L:
                best_L = next_L
                best_assign = assign.copy()
                continue
        
        # Try simulated annealing to find better assignment
        remaining_time = deadline - time.time() - 5
        sa_time = min(remaining_time * 0.3, 30.0)
        if sa_time > 2:
            new_assign, new_L = simulated_annealing(next_L, all_primes, best_assign, sa_time)
            if new_L > best_L:
                best_L = new_L
                best_assign = new_assign.copy()
                # Update best_primes to match
                best_primes = list(best_assign.keys())
                continue
        
        # Try adding small primes
        small_primes = [p for p in primes if p < 13]
        found = False
        for p_add in small_primes:
            if p_add in best_primes:
                continue
            temp_primes = best_primes + [p_add]
            s = sum_recip(temp_primes)
            if s > 1:
                # Need to remove some large primes
                temp_primes.sort(reverse=True)
                while s > 1 and len(temp_primes) > 1:
                    p_rm = temp_primes.pop(0)
                    s -= Fraction(1, p_rm)
                if s > 1:
                    continue
            
            assign = greedy_assign(next_L, temp_primes)
            if assign is not None:
                actual = evaluate_L(assign, temp_primes)
                if actual >= next_L:
                    best_L = next_L
                    best_assign = assign.copy()
                    best_primes = [p for p in temp_primes if p in assign]
                    found = True
                    break
        
        if not found:
            # Try random restarts with different prime subsets
            for _ in range(3):
                if time.time() > deadline - 10:
                    break
                # Random subset of primes with budget <= 1
                random.shuffle(all_primes)
                subset = []
                s = Fraction(0, 1)
                for p in all_primes:
                    if s + Fraction(1, p) <= 1:
                        subset.append(p)
                        s += Fraction(1, p)
                    if len(subset) > 50:
                        break
                
                if len(subset) < 5:
                    continue
                
                assign = greedy_assign(next_L, subset)
                if assign is not None:
                    actual = evaluate_L(assign, subset)
                    if actual > best_L:
                        best_L = actual
                        best_assign = assign.copy()
                        best_primes = [p for p in subset if p in assign]
                        found = True
                        break
        
        if not found:
            break

    # Final verification
    final_L = evaluate_L(best_assign, best_primes)
    
    # Format output
    result = []
    for p in sorted(best_primes):
        if p in best_assign:
            result.append([p, best_assign[p]])
    
    return result
