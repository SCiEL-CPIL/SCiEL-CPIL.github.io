import time
from math import gcd
import random

MAXIMUM = 128
DEADLINE = 90.0
SAFETY_MARGIN = 5.0


def _check_pair(r1, m1, r2, m2):
    """Two residue classes are disjoint iff r1 mod gcd(m1,m2) != r2 mod gcd(m1,m2)."""
    g = gcd(m1, m2)
    return r1 % g != r2 % g


def _is_valid_solution(solution):
    if not solution:
        return False
    seen_moduli = set()
    for r, m in solution:
        if m < 2 or m > MAXIMUM:
            return False
        if m in seen_moduli:
            return False
        seen_moduli.add(m)
        if r < 0 or r >= m:
            return False
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            if not _check_pair(solution[i][0], solution[i][1], solution[j][0], solution[j][1]):
                return False
    return True


def _greedy_construct(random_order=False, seed_val=None):
    """
    Dynamic greedy: repeatedly pick the modulus with the fewest valid residues (most constrained).
    If random_order is True, add randomness to tie-breaking to explore different local optima.
    """
    if seed_val is not None:
        random.seed(seed_val)
        
    solution = []
    remaining = list(range(2, MAXIMUM + 1))
    if random_order:
        random.shuffle(remaining)
        
    while remaining and time.time() < DEADLINE - SAFETY_MARGIN:
        best_m = None
        best_r = None
        best_count = None
        best_tie_breaker = None
        
        for m in remaining:
            valid_count = 0
            first_valid = None
            # We can speed this up by precomputing conflicts, but for m<=128 it's fast enough
            for r in range(m):
                ok = True
                for (rp, mp) in solution:
                    if not _check_pair(r, m, rp, mp):
                        ok = False
                        break
                if ok:
                    valid_count += 1
                    if first_valid is None:
                        first_valid = r
            
            if valid_count > 0:
                # Tie-breaking logic
                if best_count is None or valid_count < best_count:
                    best_count = valid_count
                    best_m = m
                    best_r = first_valid
                    if random_order:
                        best_tie_breaker = random.random()
                elif valid_count == best_count:
                    if random_order:
                        if best_tie_breaker is None or random.random() < best_tie_breaker:
                            best_m = m
                            best_r = first_valid
                            best_tie_breaker = random.random()
                    else:
                        # Deterministic tie-break: prefer smaller modulus? Or just first found?
                        # Let's stick to first found in deterministic mode for stability, 
                        # but random mode shuffles 'remaining' so order changes.
                        pass

        if best_m is None:
            break
        
        solution.append([best_r, best_m])
        remaining.remove(best_m)
        
    return solution


def _local_search_improve(solution, deadline):
    """
    Hill climbing local search.
    1. Try to add any unused modulus.
    2. If no add possible, try to remove one existing modulus to allow adding two others? 
       (Simpler: try removing one, then adding one. If net gain > 0, keep).
       Actually, a better move is: 
       - Try adding.
       - If fail, try swapping: remove m1, add m2. If m2 wasn't there, and removing m1 allows m2, 
         and m2 doesn't conflict with others, we gain 0 count but change composition.
         This doesn't help maximize count directly unless it enables future adds.
       - Let's stick to:
         a) Add if possible.
         b) If no add, try "2-swap": remove m1, m2, add m3, m4? Too complex.
         c) Let's do: Remove one m, then try to add as many as possible. If new size > old size, accept.
    """
    best = [list(x) for x in solution]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        used = set(m for _, m in best)
        
        # Phase 1: Try to add any unused modulus
        for m in range(2, MAXIMUM + 1):
            if m in used:
                continue
            for r in range(m):
                ok = True
                for (rp, mp) in best:
                    if not _check_pair(r, m, rp, mp):
                        ok = False
                        break
                if ok:
                    best.append([r, m])
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
            
        # Phase 2: Try to swap one modulus to enable growth
        # For each existing modulus m1, try removing it.
        # Then try to add any unused modulus m2.
        # If we can add m2, and then maybe more, check if total size increased.
        # To be efficient, we just check if removing m1 allows adding *at least one* new modulus 
        # that wasn't addable before? 
        # Actually, if we remove m1, we lose 1. If we can add 2 new ones, we gain 1.
        # Let's try: Remove m1. Run a greedy add loop. If final size > current size, accept.
        
        best_size = len(best)
        
        for idx, (r1, m1) in enumerate(best):
            # Temporarily remove
            temp_solution = [x for i, x in enumerate(best) if i != idx]
            used_temp = set(m for _, m in temp_solution)
            
            # Try to add greedily to temp_solution
            improved_temp = True
            while improved_temp:
                improved_temp = False
                for m in range(2, MAXIMUM + 1):
                    if m in used_temp:
                        continue
                    for r in range(m):
                        ok = True
                        for (rp, mp) in temp_solution:
                            if not _check_pair(r, m, rp, mp):
                                ok = False
                                break
                        if ok:
                            temp_solution.append([r, m])
                            used_temp.add(m)
                            improved_temp = True
                            break
                    if improved_temp:
                        break
            
            if len(temp_solution) > best_size:
                best = temp_solution
                improved = True
                break
                
    return best


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_solution = []
    
    # 1. Deterministic Greedy
    sol = _greedy_construct(random_order=False)
    sol = _local_search_improve(sol, deadline)
    if len(sol) > len(best_solution):
        best_solution = sol
    
    # 2. Multiple Randomized Greedy Restarts
    # We have ~60s. Greedy is very fast (milliseconds). Local search is fast.
    # Let's do as many restarts as we can.
    
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        sol = _greedy_construct(random_order=True, seed_val=attempt)
        sol = _local_search_improve(sol, deadline)
        if len(sol) > len(best_solution):
            best_solution = sol
            # If we hit a very high score, we might want to spend more time polishing?
            # But for now, just keep trying to beat it.
            
        # If we found a solution with size >= 10 (just a guess, adjust if needed), 
        # maybe we can slow down restarts to allow deeper local search?
        # For now, fixed loop.
        
    return best_solution if best_solution else [[0, 2]]
