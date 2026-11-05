"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
from fractions import Fraction

def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def _coverage_L(covered, limit):
    """Find largest L s.t. all 1..L are covered."""
    L = 0
    for i in range(1, limit + 1):
        if covered[i]:
            L = i
        else:
            break
    return L

def construct():
    deadline = _time.time() + 115.0
    MAX_L = 100000
    primes = _sieve(1000)
    BUDGET = Fraction(1)
    
    # Track coverage up to TRACK
    TRACK = 50000  # track up to 50000 for good coverage estimation
    
    # Start with seed solution
    seed_primes = [p for p in primes if p >= 13]
    selection = [(p, (i + 1) % p) for i, p in enumerate(seed_primes)]
    
    # Build coverage array
    covered = bytearray(TRACK + 1)
    for p, r in selection:
        start = r if r >= 1 else r + p
        if start < 1:
            start += p
        for n in range(start, TRACK + 1, p):
            covered[n] = 1
    
    best_L = _coverage_L(covered, TRACK)
    best_sel = list(selection)
    
    # Track residues per prime
    res_by_prime = {}
    for p, r in selection:
        res_by_prime.setdefault(p, set()).add(r)
    
    cur_sum = Fraction(0)
    for p in res_by_prime:
        cur_sum += Fraction(1, p)
    
    import random
    random.seed(123)
    
    no_improve = 0
    max_stall = 20000
    
    while _time.time() < deadline - 3 and no_improve < max_stall:
        op = random.random()
        
        if op < 0.4:
            # Add residue to existing prime
            existing = [p for p, rs in res_by_prime.items() if rs]
            if not existing:
                no_improve += 1
                continue
            p = random.choice(existing)
            avail = [r for r in range(p) if r not in res_by_prime[p]]
            if not avail:
                no_improve += 1
                continue
            r = random.choice(avail)
            
            # Compute new coverage
            new_cov = bytearray(covered)
            start = r if r >= 1 else r + p
            if start < 1:
                start += p
            for n in range(start, TRACK + 1, p):
                new_cov[n] = 1
            new_L = _coverage_L(new_cov, TRACK)
            
            if new_L > best_L:
                covered = new_cov
                selection.append((p, r))
                res_by_prime[p].add(r)
                best_L = new_L
                best_sel = list(selection)
                no_improve = 0
            else:
                no_improve += 1
                
        elif op < 0.6:
            # Add new prime
            used = set(res_by_prime.keys())
            cands = [p for p in primes if p not in used and cur_sum + Fraction(1, p) <= BUDGET]
            if not cands:
                no_improve += 1
                continue
            p = random.choice(cands)
            r = random.randint(0, p - 1)
            
            new_cov = bytearray(covered)
            start = r if r >= 1 else r + p
            if start < 1:
                start += p
            for n in range(start, TRACK + 1, p):
                new_cov[n] = 1
            new_L = _coverage_L(new_cov, TRACK)
            
            if new_L > best_L:
                covered = new_cov
                selection.append((p, r))
                res_by_prime[p] = {r}
                cur_sum += Fraction(1, p)
                best_L = new_L
                best_sel = list(selection)
                no_improve = 0
            else:
                no_improve += 1
                
        elif op < 0.8:
            # Swap residue within same prime
            if not selection:
                no_improve += 1
                continue
            idx = random.randint(0, len(selection) - 1)
            p, r_old = selection[idx]
            avail = [r for r in range(p) if r != r_old]
            if not avail:
                no_improve += 1
                continue
            r_new = random.choice(avail)
            
            new_cov = bytearray(TRACK + 1)
            for pp, rr in selection:
                if pp == p and rr == r_old:
                    rr = r_new
                start = rr if rr >= 1 else rr + pp
                if start < 1:
                    start += pp
                for n in range(start, TRACK + 1, pp):
                    new_cov[n] = 1
            new_L = _coverage_L(new_cov, TRACK)
            
            if new_L > best_L:
                covered = new_cov
                selection[idx] = (p, r_new)
                res_by_prime[p].discard(r_old)
                res_by_prime[p].add(r_new)
                best_L = new_L
                best_sel = list(selection)
                no_improve = 0
            else:
                no_improve += 1
                
        else:
            # Remove a residue (to free budget)
            if not selection:
                no_improve += 1
                continue
            idx = random.randint(0, len(selection) - 1)
            p, r = selection[idx]
            
            new_sel = selection[:idx] + selection[idx+1:]
            new_cov = bytearray(TRACK + 1)
            for pp, rr in new_sel:
                start = rr if rr >= 1 else rr + pp
                if start < 1:
                    start += pp
                for n in range(start, TRACK + 1, pp):
                    new_cov[n] = 1
            new_L = _coverage_L(new_cov, TRACK)
            
            # Accept if we can then add something better, or just to free budget
            # For now, only accept if L stays same (unlikely) or we're stuck
            if new_L >= best_L:
                covered = new_cov
                selection = new_sel
                res_by_prime[p].discard(r)
                if not res_by_prime[p]:
                    del res_by_prime[p]
                    cur_sum -= Fraction(1, p)
                best_L = new_L
                best_sel = list(selection)
                no_improve = 0
            else:
                no_improve += 1
    
    # Final validation
    final = best_sel
    final_sum = Fraction(0)
    fp = set(p for p, r in final)
    for p in fp:
        final_sum += Fraction(1, p)
    if final_sum > BUDGET:
        # Trim
        while final_sum > BUDGET and final:
            p, r = final.pop()
            fp.discard(p)
            if p not in fp:
                final_sum -= Fraction(1, p)
    
    result = [[p, r] for p, r in final]
    return result

# EVOLVE-BLOCK-END
