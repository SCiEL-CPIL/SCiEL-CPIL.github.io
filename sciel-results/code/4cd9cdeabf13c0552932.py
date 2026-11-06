import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 90s limit, keep 5s buffer
    
    best_set = []
    best_size = 0
    
    def get_divisors(n):
        """Return divisors of n that are <= 192."""
        divs = set()
        i = 1
        while i * i <= n:
            if n % i == 0:
                if i <= 192:
                    divs.add(i)
                other = n // i
                if other <= 192:
                    divs.add(other)
            i += 1
        return divs

    def get_bad_sets(S):
        """
        Returns:
        bad1: set of d in 1..192 such that d divides sum of 3 distinct elements in S
        bad2_residues: dict mapping a in S to set of residues r such that x % a == r implies invalid
                       r = -(u+v) % a for u,v in S\{a}, u!=v
        """
        bad1 = set()
        n = len(S)
        if n >= 3:
            for i in range(n):
                a = S[i]
                for j in range(i+1, n):
                    b = S[j]
                    for k in range(j+1, n):
                        c = S[k]
                        s = a + b + c
                        bad1.update(get_divisors(s))
        
        bad2_residues = {}
        if n >= 3:
            for i in range(n):
                a = S[i]
                R_a = set()
                # Pairs u, v from S excluding a
                for j in range(n):
                    if j == i: continue
                    u = S[j]
                    for k in range(j+1, n):
                        if k == i: continue
                        v = S[k]
                        # u, v are distinct and != a
                        r = (-(u + v)) % a
                        R_a.add(r)
                bad2_residues[a] = R_a
        
        return bad1, bad2_residues

    def is_valid_candidate(x, S, bad1, bad2_residues):
        if x in bad1:
            return False
        
        for a, R_a in bad2_residues.items():
            if x % a in R_a:
                return False
                
        return True

    # Run multiple randomized greedy attempts
    for run in range(1000):
        if time.time() > deadline:
            break
            
        current_set = []
        remaining = set(range(1, 193))
        
        while remaining:
            if time.time() > deadline:
                break
                
            # Compute bad sets for current S
            bad1, bad2_residues = get_bad_sets(current_set)
            
            valid_candidates = []
            for x in remaining:
                if is_valid_candidate(x, current_set, bad1, bad2_residues):
                    valid_candidates.append(x)
            
            if not valid_candidates:
                break
            
            # Pick a candidate
            x = random.choice(valid_candidates)
            current_set.append(x)
            remaining.remove(x)
            
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
            if best_size == 48:
                break
                
    return best_set
