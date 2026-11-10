"""Rewritten program with stronger algorithm for maximizing disjoint residue classes."""
# EVOLVE-BLOCK-START
import math
import time
import random

def construct():
    deadline = time.time() + 80.0
    MAX_MOD = 128
    all_moduli = list(range(2, MAX_MOD + 1))
    
    # Precompute gcd table
    gcd_table = {}
    for m1 in all_moduli:
        for m2 in all_moduli:
            if m1 != m2:
                gcd_table[(m1, m2)] = math.gcd(m1, m2)
    
    # Conflict count: number of other moduli with gcd > 1
    conflict_count = {}
    for m in all_moduli:
        conflict_count[m] = sum(1 for m2 in all_moduli if m2 != m and math.gcd(m, m2) > 1)
    
    def find_residue_for(m, selected):
        """Find a valid residue for modulus m given already selected classes."""
        for r in range(m):
            valid = True
            for r_prime, m_prime in selected:
                g = gcd_table.get((m, m_prime), math.gcd(m, m_prime))
                if (r - r_prime) % g == 0:
                    valid = False
                    break
            if valid:
                return r
        return None
    
    def try_add(m, selected):
        r = find_residue_for(m, selected)
        if r is not None:
            selected.append([r, m])
            return True
        return False
    
    # Generate diverse orderings
    orderings = []
    orderings.append(sorted(all_moduli, key=lambda m: conflict_count[m]))
    orderings.append(sorted(all_moduli, key=lambda m: -conflict_count[m]))
    orderings.append(sorted(all_moduli))
    orderings.append(sorted(all_moduli, reverse=True))
    
    def is_prime(n):
        if n < 2:
            return False
        for d in range(2, int(n**0.5)+1):
            if n % d == 0:
                return False
        return True
    
    primes = [m for m in all_moduli if is_prime(m)]
    non_primes = [m for m in all_moduli if not is_prime(m)]
    orderings.append(primes + non_primes)
    orderings.append(non_primes + primes)
    
    for _ in range(20):
        shuffled = all_moduli[:]
        random.shuffle(shuffled)
        orderings.append(shuffled)
    
    for _ in range(10):
        sorted_cf = sorted(all_moduli, key=lambda m: conflict_count[m])
        shuffled = all_moduli[:]
        random.shuffle(shuffled)
        hybrid = [sorted_cf[i] if random.random() < 0.6 else shuffled[i] for i in range(len(all_moduli))]
        orderings.append(hybrid)
    
    best_set = []
    best_len = 0
    
    for order in orderings:
        if time.time() > deadline - 25:
            break
        current = []
        for m in order:
            try_add(m, current)
        if len(current) > best_len:
            best_len = len(current)
            best_set = [list(c) for c in current]
    
    # Local search
    used_moduli = set(m for _, m in best_set)
    unused_moduli = [m for m in all_moduli if m not in used_moduli]
    random.shuffle(unused_moduli)
    
    improved = True
    while improved and time.time() < deadline - 3.0:
        improved = False
        
        # Try adding unused moduli
        for i in range(len(unused_moduli)):
            if time.time() > deadline - 3.0:
                break
            if try_add(unused_moduli[i], best_set):
                unused_moduli.pop(i)
                improved = True
                break
        
        if not improved and unused_moduli:
            # 1-swap: remove one, add one or two
            for idx in range(len(best_set)):
                if time.time() > deadline - 3.0:
                    break
                r_old, m_old = best_set[idx]
                others = [best_set[j] for j in range(len(best_set)) if j != idx]
                
                for m_new in list(unused_moduli):
                    if time.time() > deadline - 3.0:
                        break
                    r_new = find_residue_for(m_new, others)
                    if r_new is not None:
                        new_set = others + [[r_new, m_new]]
                        added_second = False
                        for m_new2 in [m for m in unused_moduli if m != m_new]:
                            r_new2 = find_residue_for(m_new2, new_set)
                            if r_new2 is not None:
                                best_set = new_set + [[r_new2, m_new2]]
                                unused_moduli.remove(m_new)
                                unused_moduli.remove(m_new2)
                                improved = True
                                added_second = True
                                break
                        if not added_second:
                            best_set = new_set
                            unused_moduli.remove(m_new)
                            improved = True
                        break
                if improved:
                    break
        
        if not improved and unused_moduli:
            # Perturbation
            for _ in range(80):
                if time.time() > deadline - 3.0:
                    break
                idx = random.randint(0, len(best_set) - 1)
                r_old, m_old = best_set[idx]
                target_m = unused_moduli[random.randint(0, len(unused_moduli) - 1)]
                
                candidates = [r for r in range(m_old) if r != r_old]
                random.shuffle(candidates)
                
                for r_try in candidates[:40]:
                    valid = True
                    for j in range(len(best_set)):
                        if j == idx:
                            continue
                        r_j, m_j = best_set[j]
                        g = gcd_table.get((m_old, m_j), math.gcd(m_old, m_j))
                        if (r_try - r_j) % g == 0:
                            valid = False
                            break
                    if valid:
                        best_set[idx][0] = r_try
                        r_t = find_residue_for(target_m, best_set)
                        if r_t is not None:
                            best_set.append([r_t, target_m])
                            unused_moduli.remove(target_m)
                            improved = True
                            break
                        else:
                            best_set[idx][0] = r_old
                if improved:
                    break
    
    # Final verification
    n = len(best_set)
    valid = True
    for i in range(n):
        r1, m1 = best_set[i]
        for j in range(i + 1, n):
            r2, m2 = best_set[j]
            g = gcd_table.get((m1, m2), math.gcd(m1, m2))
            if (r1 - r2) % g == 0:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        valid_set = []
        for c in best_set:
            try_add(c[1], valid_set)
        if valid_set:
            best_set = valid_set
        else:
            best_set = [[1, 2]]
    
    return best_set
# EVOLVE-BLOCK-END
