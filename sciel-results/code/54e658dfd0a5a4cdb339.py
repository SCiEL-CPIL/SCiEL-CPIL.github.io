import math
import time
import random

def construct():
    deadline = time.time() + 85  # 5s safety margin
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def is_disjoint(r1, m1, r2, m2):
        g = gcd(m1, m2)
        return (r1 - r2) % g != 0
    
    # Start with powers of 2
    # For modulus 2^k, use residue 2^(k-1)
    # This ensures for 2^a, 2^b (a>b): gcd=2^b, diff = 2^(a-1)-2^(b-1) = 2^(b-1)(2^(a-b)-1)
    # 2^b divides diff iff 2 divides (2^(a-b)-1) which is false (odd)
    
    base = []
    for k in range(1, 8):  # 2^1 to 2^7 = 2 to 128
        m = 1 << k
        r = 1 << (k-1)
        base.append([r, m])
    
    # Verify base is valid
    for i in range(len(base)):
        for j in range(i+1, len(base)):
            if not is_disjoint(base[i][0], base[i][1], base[j][0], base[j][1]):
                # Fallback to seed
                return [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
    
    best = base[:]
    used_moduli = set(m for _, m in base)
    
    # Try to add more moduli greedily
    candidates = list(range(2, 129))
    candidates = [m for m in candidates if m not in used_moduli]
    
    # Sort candidates by number of constraints (fewer = easier)
    # Actually, try to add in order of increasing difficulty
    random.seed(42)
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        current = best[:]
        cur_used = set(m for _, m in current)
        
        # Shuffle candidates
        cands = [m for m in range(2, 129) if m not in cur_used]
        random.shuffle(cands)
        
        for m in cands:
            if time.time() > deadline:
                break
            
            # Check if m can be added: for all existing, gcd(m, m_i) > 1 (otherwise they'd intersect for any residues)
            # Actually, if gcd(m, m_i) = 1, they always intersect, so m can only be added if gcd(m, m_i) > 1 for all i
            feasible = True
            for r_i, m_i in current:
                if gcd(m, m_i) == 1:
                    feasible = False
                    break
            
            if not feasible:
                continue
            
            # Try to find a residue r in [0, m) such that for all (r_i, m_i) in current:
            # gcd(m, m_i) ∤ (r - r_i)
            found = False
            for r in range(m):
                ok = True
                for r_i, m_i in current:
                    g = gcd(m, m_i)
                    if (r - r_i) % g == 0:
                        ok = False
                        break
                if ok:
                    current.append([r, m])
                    cur_used.add(m)
                    found = True
                    break
            
            if found:
                continue
        
        if len(current) > len(best):
            best = current
    
    return best
