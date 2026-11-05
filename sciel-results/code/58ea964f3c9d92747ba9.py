import time
import random
from math import gcd
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 128
    numbers = list(range(1, n + 1))
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # Precompute LCM table
    lcm_table = {}
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[(i, j)] = lcm(i, j)
    
    def get_lcm(a, b):
        if a <= b:
            return lcm_table[(a, b)]
        else:
            return lcm_table[(b, a)]

    # Find all forbidden triples
    # A triple (a,b,c) is forbidden if lcm(a,b) == lcm(a,c) == lcm(b,c)
    forbidden_triples = []
    # We iterate a < b < c
    for a in range(1, n - 1):
        for b in range(a + 1, n):
            lab = get_lcm(a, b)
            # c must be > b
            for c in range(b + 1, n + 1):
                if get_lcm(a, c) == lab and get_lcm(b, c) == lab:
                    forbidden_triples.append((a, b, c))
                    
    M = len(forbidden_triples)
    
    # Map variables to clauses
    var_clauses = defaultdict(list)
    for i, (a, b, c) in enumerate(forbidden_triples):
        var_clauses[a].append(i)
        var_clauses[b].append(i)
        var_clauses[c].append(i)

    # Initial greedy solution
    def can_add_fast(x, S):
        # Check if adding x to S creates a bad triple
        # A bad triple involving x and two elements in S {u, v}
        # must have lcm(x,u) == lcm(x,v) == lcm(u,v)
        # Group elements in S by lcm(x, elem)
        groups = {}
        for u in S:
            L = get_lcm(x, u)
            if L not in groups:
                groups[L] = []
            groups[L].append(u)
        
        for L, elems in groups.items():
            if len(elems) < 2:
                continue
            # Check pairs in elems
            for i in range(len(elems)):
                for j in range(i + 1, len(elems)):
                    if get_lcm(elems[i], elems[j]) == L:
                        return False
        return True

    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(n)
    prime_powers = set()
    for p in primes:
        power = p
        while power <= n:
            prime_powers.add(power)
            power *= p
    
    core_set = prime_powers | {1}
    
    def num_distinct_primes(x):
        count = 0
        d = 2
        while d * d <= x:
            if x % d == 0:
                count += 1
                while x % d == 0:
                    x //= d
            d += 1
        if x > 1:
            count += 1
        return count

    remaining = [x for x in range(1, n + 1) if x not in core_set]
    
    best_set = set(core_set)
    
    # Greedy passes
    orderings = [
        sorted(remaining, key=lambda x: (num_distinct_primes(x), x)),
        sorted(remaining, key=lambda x: (num_distinct_primes(x), -x)),
        sorted(remaining, key=lambda x: x),
        sorted(remaining, key=lambda x: -x)
    ]
    for _ in range(5):
        rand_order = remaining[:]
        random.shuffle(rand_order)
        orderings.append(rand_order)
        
    for ordering in orderings:
        if time.time() > deadline - 20.0:
            break
        current = set(core_set)
        for x in ordering:
            if time.time() > deadline - 20.0:
                break
            if x in current:
                continue
            if can_add_fast(x, current):
                current.add(x)
        if len(current) > len(best_set):
            best_set = set(current)

    # WalkSAT / Local Search
    # State: boolean array x[1..n]
    # We maintain clause_sat_count and violations
    
    def init_state(S):
        x = [False] * (n + 1)
        for v in S:
            x[v] = True
        clause_sat = [0] * M
        violations = 0
        for i, (a, b, c) in enumerate(forbidden_triples):
            cnt = 0
            if x[a]: cnt += 1
            if x[b]: cnt += 1
            if x[c]: cnt += 1
            clause_sat[i] = cnt
            if cnt == 3:
                violations += 1
        return x, clause_sat, violations

    def flip_var(x, v, clause_sat, violations):
        # Returns new violations
        new_violations = violations
        old_state = x[v]
        for c_idx in var_clauses[v]:
            old_cnt = clause_sat[c_idx]
            if old_state:
                # Turning off
                new_cnt = old_cnt - 1
                if old_cnt == 3:
                    new_violations -= 1
                # new_cnt cannot be 3 if old_cnt was 3 and we subtracted 1
            else:
                # Turning on
                new_cnt = old_cnt + 1
                if new_cnt == 3:
                    new_violations += 1
            clause_sat[c_idx] = new_cnt
        x[v] = not old_state
        return new_violations

    current_x, current_sat, current_viol = init_state(best_set)
    current_size = len(best_set)
    best_size = current_size
    best_x = current_x[:]
    
    # If we started with violations (shouldn't happen with greedy, but safe), fix them first
    # Greedy ensures 0 violations.
    
    max_steps = 50000
    step = 0
    no_improve_count = 0
    
    while time.time() < deadline - 5.0:
        if current_viol == 0:
            # Try to improve size
            # Find a variable v not in set such that flipping it to True doesn't create violations
            # i.e., for all clauses c containing v, sat[c] < 2 (so sat[c]+1 < 3)
            # Actually, if sat[c] == 2, flipping v makes it 3 (violation).
            # So we need sat[c] <= 1 for all c containing v.
            
            improved = False
            # Randomize order of candidates to avoid bias
            candidates = [v for v in range(1, n+1) if not current_x[v]]
            random.shuffle(candidates)
            
            for v in candidates:
                # Check if we can add v
                can_add = True
                for c_idx in var_clauses[v]:
                    if current_sat[c_idx] >= 2:
                        can_add = False
                        break
                if can_add:
                    # Flip v
                    new_viol = flip_var(current_x, v, current_sat, current_viol)
                    # new_viol should be 0
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_x = current_x[:]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    improved = True
                    break
            
            if not improved:
                # Local optimum in terms of size with 0 violations.
                # Try a kick: remove k random elements to break the structure
                # We remove 2 or 3 elements.
                k = random.choice([2, 3])
                in_set = [v for v in range(1, n+1) if current_x[v]]
                if len(in_set) > k:
                    to_remove = random.sample(in_set, k)
                    for v in to_remove:
                        # Flip to False
                        new_viol = flip_var(current_x, v, current_sat, current_viol)
                        current_size -= 1
                    # After kick, we might have 0 violations still (removing never creates violations)
                    # But we lost size. We need to regain it.
                    # The next iterations will try to add elements back, possibly different ones.
                else:
                    # Set too small, restart?
                    break
        else:
            # Violations exist, need to fix
            # Find a random violated clause
            violated_clauses = [i for i in range(M) if current_sat[i] == 3]
            if not violated_clauses:
                current_viol = 0
                continue
                
            c_idx = random.choice(violated_clauses)
            a, b, c = forbidden_triples[c_idx]
            # Pick a random variable from the clause to flip
            # Heuristic: pick the one with lowest "weight" or just random
            v = random.choice([a, b, c])
            # Flip it
            # We want to turn it OFF to reduce violation
            if current_x[v]:
                new_viol = flip_var(current_x, v, current_sat, current_viol)
                current_size -= 1
            else:
                # This shouldn't happen if we picked from a violated clause (all 3 are True)
                pass
        
        step += 1
        if step > max_steps:
            # Restart with perturbation of best solution
            current_x = best_x[:]
            current_size = best_size
            # Re-init sat and viol
            current_x, current_sat, current_viol = init_state(set([v for v in range(1, n+1) if current_x[v]]))
            # Perturb
            in_set = [v for v in range(1, n+1) if current_x[v]]
            if in_set:
                v = random.choice(in_set)
                flip_var(current_x, v, current_sat, current_viol)
                current_size -= 1
            step = 0
            no_improve_count = 0

    # Final result
    final_set = set([v for v in range(1, n+1) if best_x[v]])
    
    # Validate final set just in case
    S = sorted(final_set)
    valid = True
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            lab = get_lcm(S[i], S[j])
            for k in range(j + 1, len(S)):
                if get_lcm(S[i], S[k]) == lab and get_lcm(S[j], S[k]) == lab:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to greedy if walksat broke it (unlikely)
        return list(best_set)
        
    return list(final_set)

