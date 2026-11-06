"""Find maximum pairwise-disjoint residue classes with distinct moduli 2..128."""
import math
import time
import random

MAX_MOD = 128

def gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def are_disjoint(r1, m1, r2, m2):
    g = gcd(m1, m2)
    return (r1 - r2) % g != 0

def check_solution(pairs):
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            r1, m1 = pairs[i]
            r2, m2 = pairs[j]
            if not are_disjoint(r1, m1, r2, m2):
                return False
    return True

def greedy_construct(seed=0):
    rng = random.Random(seed)
    moduli = list(range(2, MAX_MOD + 1))
    rng.shuffle(moduli)
    
    selected = []
    used_moduli = set()
    
    for m in moduli:
        if m in used_moduli:
            continue
        valid_residues = []
        for r in range(m):
            ok = True
            for (r2, m2) in selected:
                if not are_disjoint(r, m, r2, m2):
                    ok = False
                    break
            if ok:
                valid_residues.append(r)
        
        if valid_residues:
            best_r = valid_residues[0]
            selected.append((best_r, m))
            used_moduli.add(m)
    
    return sorted(selected, key=lambda x: x[1])

def local_search(pairs, seed=0, time_limit=5.0):
    rng = random.Random(seed)
    start = time.time()
    current = sorted(pairs, key=lambda x: x[1])
    best = list(current)
    best_len = len(best)
    
    all_moduli = set(range(2, MAX_MOD + 1))
    
    def try_add_one(sol):
        sol_used = set(m for _, m in sol)
        available = [m for m in all_moduli if m not in sol_used]
        rng.shuffle(available)
        for m in available:
            for r in range(m):
                ok = True
                for (r2, m2) in sol:
                    if not are_disjoint(r, m, r2, m2):
                        ok = False
                        break
                if ok:
                    return sol + [(r, m)]
        return None
    
    def try_swap_and_add(sol):
        sol_list = list(sol)
        n = len(sol_list)
        if n == 0:
            return None
        for _ in range(10):
            idx = rng.randrange(n)
            r_old, m_old = sol_list[idx]
            for r_new in range(m_old):
                if r_new == r_old:
                    continue
                candidate = list(sol_list)
                candidate[idx] = (r_new, m_old)
                valid = True
                for i in range(n):
                    for j in range(i + 1, n):
                        r1, m1 = candidate[i]
                        r2, m2 = candidate[j]
                        if not are_disjoint(r1, m1, r2, m2):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                extended = try_add_one(candidate)
                if extended:
                    return extended
        return None
    
    while time.time() - start < time_limit:
        new_sol = try_add_one(current)
        if new_sol:
            current = sorted(new_sol, key=lambda x: x[1])
            if len(current) > best_len:
                best = list(current)
                best_len = len(best)
            continue
        
        new_sol = try_swap_and_add(current)
        if new_sol:
            current = sorted(new_sol, key=lambda x: x[1])
            if len(current) > best_len:
                best = list(current)
                best_len = len(best)
        
        if len(current) > 3 and random.random() < 0.1:
            idx = rng.randrange(len(current))
            perturbed = [p for i, p in enumerate(current) if i != idx]
            new_sol = try_add_one(perturbed)
            if new_sol and len(new_sol) >= len(current) - 1:
                current = sorted(new_sol, key=lambda x: x[1])
    
    return best

def construct():
    deadline = time.time() + 80
    
    seed_sol = [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]
    best_solution = seed_sol
    
    for seed in range(30):
        if time.time() > deadline - 15:
            break
        sol = greedy_construct(seed)
        if check_solution(sol) and len(sol) > len(best_solution):
            best_solution = sol
    
    if time.time() < deadline - 10:
        improved = local_search(best_solution, seed=0, time_limit=min(10, deadline - time.time() - 5))
        if check_solution(improved) and len(improved) > len(best_solution):
            best_solution = improved
    
    for seed in range(1, 8):
        if time.time() > deadline - 5:
            break
        sol = greedy_construct(seed + 50)
        if not check_solution(sol):
            continue
        improved = local_search(sol, seed=seed, time_limit=min(3, max(1, deadline - time.time() - 5)))
        if check_solution(improved) and len(improved) > len(best_solution):
            best_solution = improved
    
    if time.time() < deadline - 3:
        final = local_search(best_solution, seed=99, time_limit=max(1, deadline - time.time() - 3))
        if check_solution(final) and len(final) > len(best_solution):
            best_solution = final
    
    if not best_solution:
        best_solution = [[1, 2], [3, 4]]
    
    return best_solution
