# EVOLVE-BLOCK-START
import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin before 60s limit
    
    MAX_M = 128
    moduli = list(range(2, MAX_M + 1))
    
    # Precompute gcds for speed? 
    # gcd is fast, but let's cache it if needed. 
    # 127*127/2 = ~8000 pairs. Negligible.
    
    def is_valid_pair(r1, m1, r2, m2):
        d = math.gcd(m1, m2)
        if d == 1:
            return True
        return (r1 - r2) % d != 0

    def check_set(assignment):
        """Check if the entire assignment is valid."""
        keys = list(assignment.keys())
        for i in range(len(keys)):
            m1 = keys[i]
            r1 = assignment[m1]
            for j in range(i + 1, len(keys)):
                m2 = keys[j]
                r2 = assignment[m2]
                if not is_valid_pair(r1, m1, r2, m2):
                    return False
        return True

    def try_greedy(order):
        """Greedy assignment for a specific order. Returns dict or None if stuck."""
        assignment = {}
        for m in order:
            found = False
            for r in range(m):
                ok = True
                for n, rn in assignment.items():
                    if not is_valid_pair(r, m, rn, n):
                        ok = False
                        break
                if ok:
                    assignment[m] = r
                    found = True
                    break
            if not found:
                return None
        return assignment

    def greedy_skip(order):
        """Greedy skip: add if possible, else skip. Always returns a valid set."""
        assignment = {}
        for m in order:
            for r in range(m):
                ok = True
                for n, rn in assignment.items():
                    if not is_valid_pair(r, m, rn, n):
                        ok = False
                        break
                if ok:
                    assignment[m] = r
                    break
        return assignment

    best_assignment = {}
    best_size = 0

    # 1. Try to get a large initial set using different orderings
    orders = [
        sorted(moduli, reverse=True),
        sorted(moduli),
        sorted(moduli, key=lambda x: x.bit_count()), # Fewer bits might mean simpler?
        sorted(moduli, key=lambda x: -x.bit_count()),
        sorted(moduli, key=lambda x: math.gcd(x, 2**7-1)) # Random-ish
    ]
    
    # Add a few random orders
    for _ in range(5):
        random_order = moduli.copy()
        random.shuffle(random_order)
        orders.append(random_order)

    for order in orders:
        if time.time() > deadline:
            break
        
        # Try full greedy first
        res = try_greedy(order)
        if res is not None:
            if len(res) > best_size:
                best_size = len(res)
                best_assignment = res
            # If we got all 127, we can't do better.
            if best_size == len(moduli):
                break
        
        # Try greedy skip (always valid)
        res_skip = greedy_skip(order)
        if len(res_skip) > best_size:
            best_size = len(res_skip)
            best_assignment = res_skip

    # 2. Local Search to improve
    # We want to maximize the number of moduli.
    # Moves:
    # a) Add a missing modulus with a valid residue.
    # b) Remove a modulus and try to add two others? (Complex)
    # c) Change residue of an existing modulus to allow adding another?
    
    # Simple local search: Try to add missing moduli.
    # If we can't add any, try to change residues to unlock additions.
    
    current = best_assignment.copy()
    current_size = best_size
    
    # List of missing moduli
    missing = [m for m in moduli if m not in current]
    
    # Shuffle missing to try in random order
    random.shuffle(missing)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to add each missing modulus
        for m in missing[:]: # iterate over copy
            if m in current:
                continue
            # Try to find a residue for m
            added = False
            for r in range(m):
                ok = True
                for n, rn in current.items():
                    if not is_valid_pair(r, m, rn, n):
                        ok = False
                        break
                if ok:
                    current[m] = r
                    current_size += 1
                    missing.remove(m)
                    improved = True
                    added = True
                    break
            if added:
                continue
            
            # If couldn't add directly, try to modify existing residues to allow adding m?
            # This is a local search step.
            # Try changing one existing modulus's residue to see if it allows adding m.
            # This is expensive. Let's stick to simple addition first.
            # If simple addition fails for all, we are at a local optimum for this move set.
            
    # If we have time, try a more aggressive local search:
    # Try to swap a modulus? Or just restart with different random seeds for greedy_skip?
    
    # Let's run a few more random greedy_skip rounds if time permits
    rounds = 0
    while time.time() < deadline - 5.0:
        random_order = moduli.copy()
        random.shuffle(random_order)
        res_skip = greedy_skip(random_order)
        if len(res_skip) > best_size:
            best_size = len(res_skip)
            best_assignment = res_skip
            # Reset current to new best and try to add more?
            # For now, just keep the best found.
            # We can re-run the "add missing" logic on the new best.
            current = best_assignment.copy()
            missing = [m for m in moduli if m not in current]
            random.shuffle(missing)
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                for m in missing[:]:
                    if m in current:
                        continue
                    for r in range(m):
                        ok = True
                        for n, rn in current.items():
                            if not is_valid_pair(r, m, rn, n):
                                ok = False
                                break
                        if ok:
                            current[m] = r
                            best_size += 1
                            missing.remove(m)
                            improved = True
                            break
        rounds += 1
        if rounds > 20: # Limit number of restarts
            break

    # Final check: Ensure best_assignment is valid (paranoia)
    if not check_set(best_assignment):
        # Fallback to a known valid small set
        # Try to build a valid set from scratch with a very simple greedy
        fallback = greedy_skip(sorted(moduli, reverse=True))
        if check_set(fallback):
            best_assignment = fallback
        else:
            best_assignment = {2: 1}

    if time.time() > deadline:
        return [[best_assignment[m], m] for m in sorted(best_assignment.keys())]

    return [[best_assignment[m], m] for m in sorted(best_assignment.keys())]
# EVOLVE-BLOCK-END
