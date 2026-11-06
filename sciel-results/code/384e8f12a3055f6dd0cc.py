import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    def is_prime(n):
        if n < 2:
            return False
        for q in range(2, int(n**0.5) + 1):
            if n % q == 0:
                return False
        return True
    
    primes = [p for p in range(2, 1001) if is_prime(p)]
    
    # Initial solution: primes from 13 upward with better residue assignment
    initial_primes = [p for p in primes if p >= 13]
    # Assign residues to try to cover small numbers well
    initial_residues = []
    for i, p in enumerate(initial_primes):
        if i < 3:
            initial_residues.append(1)
        elif i < 6:
            initial_residues.append(2)
        else:
            initial_residues.append((i + 1) % p)
    
    classes = list(zip(initial_primes, initial_residues))
    
    max_L = 10000
    
    # Build coverage count array
    count = [0] * (max_L + 1)
    for p, r in classes:
        for n in range(r, max_L + 1, p):
            if n >= 1:
                count[n] += 1
    
    def find_prefix_from(count, start):
        for n in range(start, max_L + 1):
            if count[n] == 0:
                return n - 1
        return max_L
    
    best_L = find_prefix_from(count, 1)
    
    # Local search: optimize residues
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        for i in range(len(classes)):
            p, r = classes[i]
            best_r = r
            best_L_local = best_L
            for new_r in range(p):
                if new_r == r:
                    continue
                # Update counts
                for n in range(r, max_L + 1, p):
                    if n >= 1:
                        count[n] -= 1
                for n in range(new_r, max_L + 1, p):
                    if n >= 1:
                        count[n] += 1
                L_local = best_L
                if best_L < max_L and count[best_L + 1] > 0:
                    L_local = find_prefix_from(count, best_L + 1)
                if L_local > best_L_local:
                    best_L_local = L_local
                    best_r = new_r
                # Revert
                for n in range(new_r, max_L + 1, p):
                    if n >= 1:
                        count[n] -= 1
                for n in range(r, max_L + 1, p):
                    if n >= 1:
                        count[n] += 1
                if time.time() > deadline - 15:
                    break
            if best_r != r:
                for n in range(r, max_L + 1, p):
                    if n >= 1:
                        count[n] -= 1
                for n in range(best_r, max_L + 1, p):
                    if n >= 1:
                        count[n] += 1
                classes[i] = (p, best_r)
                best_L = best_L_local
                improved = True
            if time.time() > deadline - 15:
                break
    
    # Try adding small primes that fit in budget
    current_budget = sum(Fraction(1, p) for p, r in classes)
    for p in sorted(primes):
        if p < 13 and p not in [pp for pp, rr in classes]:
            if current_budget + Fraction(1, p) <= Fraction(1, 1):
                best_new_r = None
                best_L_with = best_L
                for new_r in range(p):
                    for n in range(new_r, max_L + 1, p):
                        if n >= 1:
                            count[n] += 1
                    L_with = best_L
                    if best_L < max_L and count[best_L + 1] > 0:
                        L_with = find_prefix_from(count, best_L + 1)
                    for n in range(new_r, max_L + 1, p):
                        if n >= 1:
                            count[n] -= 1
                    if L_with > best_L_with:
                        best_L_with = L_with
                        best_new_r = new_r
                if best_new_r is not None and best_L_with > best_L:
                    for n in range(best_new_r, max_L + 1, p):
                        if n >= 1:
                            count[n] += 1
                    classes.append((p, best_new_r))
                    current_budget += Fraction(1, p)
                    best_L = best_L_with
                if time.time() > deadline - 10:
                    break
    
    return [[p, r] for p, r in classes]
