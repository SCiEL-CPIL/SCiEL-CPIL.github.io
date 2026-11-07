import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 256
    all_elements = list(range(2, N + 1))
    
    # Precompute gcd for all pairs? 
    # gcd is fast enough in math module.
    
    # Precompute divisors for updating mult_count
    divisors = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            divisors[j].append(i)
            
    def run_once(seed):
        random.seed(seed)
        S = set(all_elements)
        # mult_count[k] = number of x in S such that k divides x
        mult_count = [0] * (N + 1)
        for x in S:
            for d in divisors[x]:
                mult_count[d] += 1
                
        # conflicts[a] = number of pairs (b,c) in S\{a} such that a | b*c
        conflicts = [0] * (N + 1)
        
        # Initial calculation of conflicts
        # For each a, count pairs (b,c)
        # a | b*c <=> c is multiple of a' = a/gcd(a,b)
        # Sum over b in S\{a} of (count of c in S\{a,b} s.t. c % a' == 0)
        
        for a in S:
            count = 0
            for b in S:
                if b == a:
                    continue
                g = gcd(a, b)
                a_prime = a // g
                # count c in S such that c % a_prime == 0
                # mult_count[a_prime] counts all in S
                # subtract if a is in it
                # subtract if b is in it
                c_count = mult_count[a_prime]
                if a % a_prime == 0:
                    c_count -= 1
                if b % a_prime == 0:
                    c_count -= 1
                count += c_count
            conflicts[a] = count
            
        removed = []
        
        # Greedy removal
        while True:
            if time.time() > deadline:
                break
                
            # Find max conflict
            max_conf = -1
            max_elem = None
            for a in S:
                if conflicts[a] > max_conf:
                    max_conf = conflicts[a]
                    max_elem = a
                    
            if max_conf == 0:
                break
                
            # Remove max_elem
            x = max_elem
            S.remove(x)
            removed.append(x)
            
            # Update conflicts for all a in S
            for a in S:
                # Subtract contribution of x to conflicts[a]
                # Pairs (x, c) and (b, x)
                
                # 1. x=b: a | x*c => c is multiple of a1 = a/gcd(a,x)
                g1 = gcd(a, x)
                a1 = a // g1
                c_count_1 = mult_count[a1]
                if a % a1 == 0:
                    c_count_1 -= 1
                if x % a1 == 0:
                    c_count_1 -= 1
                # Note: we are removing x, so mult_count is still old.
                # The pairs involving x are being removed.
                # The count of valid c's *before* removal is c_count_1.
                # These pairs are no longer in S.
                conflicts[a] -= c_count_1
                
                # 2. x=c: a | b*x => b is multiple of a2 = a/gcd(a,x)
                # a2 is same as a1.
                # b_count_2 is same as c_count_1.
                conflicts[a] -= c_count_1
                
            # Update mult_count
            for d in divisors[x]:
                mult_count[d] -= 1
                
        # Add-back phase
        random.shuffle(removed)
        for y in removed:
            if time.time() > deadline:
                break
            if y in S:
                continue
                
            # Check if y can be added
            can_add = True
            for a in S:
                # Check a | y*c
                g1 = gcd(a, y)
                a1 = a // g1
                c_count = mult_count[a1]
                if a % a1 == 0:
                    c_count -= 1
                # y is not in S yet, so no need to subtract y from mult_count
                # But we need c != a, c != y.
                # mult_count[a1] includes a if a%a1==0. We subtracted it.
                # It does NOT include y.
                # So c_count is number of c in S such that a | y*c.
                # We need c != y. Since y not in S, this is automatic.
                # We need c != a. Subtracted.
                if c_count > 0:
                    can_add = False
                    break
                    
                if not can_add:
                    break
                
                # Check y | a*c
                g2 = gcd(y, a)
                y1 = y // g2
                c_count2 = mult_count[y1]
                if a % y1 == 0:
                    c_count2 -= 1
                # c != a, c != y.
                if c_count2 > 0:
                    can_add = False
                    break
                    
            if can_add:
                S.add(y)
                removed.remove(y)
                for d in divisors[y]:
                    mult_count[d] += 1
                
                # Update conflicts for all a in S (excluding y? no, y is new)
                # Adding y creates new conflicts for existing a's and for y.
                # But we checked that it doesn't create ANY conflicts.
                # So conflicts remain 0 for all.
                # Wait, did we check all?
                # We checked for each a in S:
                # 1. a | y*c for any c in S.
                # 2. y | a*c for any c in S.
                # What about c | a*y?
                # This is covered by symmetry?
                # If we add y, new triples are (a, c, y).
                # Divisor can be a, c, or y.
                # Case 1: a | c*y. Checked (a | y*c).
                # Case 2: c | a*y. 
                #    This is equivalent to checking for divisor c.
                #    In the loop `for a in S`, if we swap roles of a and c, 
                #    we check `c | y*a`?
                #    My check was: for each a in S, check if a divides y*c OR y divides a*c.
                #    This covers:
                #    - Divisor is a: a | y*c (covered)
                #    - Divisor is y: y | a*c (covered)
                #    - Divisor is c: c | a*y.
                #      This is NOT explicitly covered in the loop for a.
                #      However, when the loop variable is `a` (which is some element in S),
                #      and we consider the pair (y, c) where c is the *other* element?
                #      No, the loop iterates over `a` as the *divisor* candidate?
                #      No, `a` is just an element in S.
                #      The check `a | y*c` means "a is a divisor of product of y and some c".
                #      The check `y | a*c` means "y is a divisor of product of a and some c".
                #      What about "c is a divisor of product of a and y"?
                #      This is `c | a*y`.
                #      If we fix `c` and vary `a`, this is covered when the loop variable is `c`?
                #      Let's rename loop variable to `u`.
                #      Check 1: `u | y*v` for some `v in S`.
                #      Check 2: `y | u*v` for some `v in S`.
                #      If we set `u=c`, Check 1 becomes `c | y*a`.
                #      So yes, it is covered when `u` iterates over `c`.
                
                # So conflicts are all 0.
                # But we must update the `conflicts` array?
                # Since we verified no conflicts, `conflicts` for all a in S remains 0.
                # `conflicts[y]` is 0.
                # So no update needed for conflicts values, they stay 0.
                # But wait, if we add y, do we need to update conflicts for future removals?
                # Yes, but since they are 0, and we only remove if > 0, it's fine.
                # However, if we do local search later, we need accurate counts.
                # For now, we just return S.
                
        return list(S)

    best_S = []
    best_size = 0
    
    # Run multiple restarts
    # Estimate time per run: ~0.1s?
    # 85s / 0.1s = 850 runs.
    
    run_id = 0
    while time.time() < deadline:
        S = run_once(run_id)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
        run_id += 1
        
    # Final verification
    # Ensure Python ints
    return [int(x) for x in best_S]

if __name__ == "__main__":
    # Test
    res = construct()
    print(len(res))
