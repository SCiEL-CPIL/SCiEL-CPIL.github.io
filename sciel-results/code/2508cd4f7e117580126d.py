import math
import random
import time

def _is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def _num_divisors(n):
    count = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            count += 1
            if i != n // i:
                count += 1
        i += 1
    return count

def _is_disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r1 - r2) % g != 0

def _can_add(r, m, current_set):
    for r_i, m_i in current_set:
        if not _is_disjoint(r, m, r_i, m_i):
            return False
    return True

def _greedy_fill(moduli_order, current_set):
    for m in moduli_order:
        if any(m == m_i for _, m_i in current_set):
            continue
        for r in range(m):
            if _can_add(r, m, current_set):
                current_set.append([r, m])
                break

def _local_search_improve(current_set, all_moduli, time_limit):
    # Try to add missing moduli by reassigning residues of existing ones
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        in_set = {m for _, m in current_set}
        missing = [m for m in all_moduli if m not in in_set]
        
        for m_new in missing:
            # Try to find a residue for m_new
            found = False
            for r_new in range(m_new):
                if _can_add(r_new, m_new, current_set):
                    current_set.append([r_new, m_new])
                    improved = True
                    found = True
                    break
            
            if found:
                break
                
            # If not found, try to change residues of existing moduli to make room
            # Identify conflicts: for a random r_new, which moduli conflict?
            # Let's pick a random r_new and see conflicts
            r_try = random.randint(0, m_new - 1)
            conflicts = []
            for i, (r_i, m_i) in enumerate(current_set):
                if not _is_disjoint(r_try, m_new, r_i, m_i):
                    conflicts.append(i)
            
            if conflicts:
                # Try to reassign one of the conflicting moduli
                idx = random.choice(conflicts)
                m_conf = current_set[idx][1]
                r_conf_old = current_set[idx][0]
                
                # Try to find a new residue for m_conf that is valid against all others (except m_new)
                # and allows m_new to be added with r_try
                new_r_conf = None
                for r_cand in range(m_conf):
                    if r_cand == r_conf_old:
                        continue
                    # Check if r_cand is valid for m_conf against all other existing moduli
                    ok = True
                    for j, (r_j, m_j) in enumerate(current_set):
                        if j == idx:
                            continue
                        if not _is_disjoint(r_cand, m_conf, r_j, m_j):
                            ok = False
                            break
                    if ok:
                        # Check if this allows m_new with r_try
                        if _is_disjoint(r_try, m_new, r_cand, m_conf):
                            new_r_conf = r_cand
                            break
                
                if new_r_conf is not None:
                    # Apply change
                    current_set[idx][0] = new_r_conf
                    # Now try to add m_new with r_try
                    if _can_add(r_try, m_new, current_set):
                        current_set.append([r_try, m_new])
                        improved = True
                        break # Restart loop with new set
            
            if improved:
                break
                
    return current_set

def construct():
    deadline = time.time() + 80
    max_mod = 128
    all_moduli = list(range(2, max_mod + 1))
    
    primes = [m for m in all_moduli if _is_prime(m)]
    composites = [m for m in all_moduli if not _is_prime(m)]
    composites.sort(key=lambda m: _num_divisors(m), reverse=True) # Harder first
    
    best_set = []
    
    # Strategy 1: Primes first, then composites
    order1 = primes + composites
    set1 = []
    _greedy_fill(order1, set1)
    set1 = _local_search_improve(set1, all_moduli, deadline)
    if len(set1) > len(best_set):
        best_set = set1[:]
        
    # Strategy 2: Composites first (harder first), then primes
    order2 = composites + primes
    set2 = []
    _greedy_fill(order2, set2)
    set2 = _local_search_improve(set2, all_moduli, deadline)
    if len(set2) > len(best_set):
        best_set = set2[:]
        
    # Strategy 3: Randomized restarts with local search
    for _ in range(100):
        if time.time() > deadline:
            break
        random.shuffle(all_moduli)
        current_set = []
        _greedy_fill(all_moduli, current_set)
        current_set = _local_search_improve(current_set, all_moduli, deadline)
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            
    if not best_set:
        return [[0, 2]]
        
    return [[r, m] for r, m in best_set]
