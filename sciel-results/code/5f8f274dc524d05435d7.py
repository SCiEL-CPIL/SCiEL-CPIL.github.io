# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 135
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(n+1) if is_prime[i]]
    
    primes = sieve(1000)
    n_primes = len(primes)
    CAP = 2000  # We'll track coverage up to CAP, and verify beyond if needed
    
    def build_coverage(residues, primes, cap):
        """Build coverage array for 1..cap."""
        cov = [0] * (cap + 1)
        for j, p in enumerate(primes):
            r = residues[j]
            first = r if r > 0 else p
            x = first
            while x <= cap:
                cov[x] += 1
                x += p
        return cov
    
    def get_L_from_cov(cov, cap):
        """Find largest L with all cov[1..L] >= 2."""
        L = 0
        for i in range(1, cap + 1):
            if cov[i] >= 2:
                L = i
            else:
                break
        return L
    
    def verify_L(residues, primes, L, cap=100000):
        """Verify that L is correct by checking beyond cap if needed."""
        if L >= cap:
            # Need to check beyond
            for x in range(cap + 1, L + 1):
                cnt = 0
                for j, p in enumerate(primes):
                    if x % p == residues[j] % p:
                        cnt += 1
                        if cnt >= 2:
                            break
                if cnt < 2:
                    return x - 1
        return L
    
    # Initial assignment
    residues = [0] * n_primes
    targets = list(range(1, n_primes + 1))
    random.shuffle(targets)
    for i, p in enumerate(primes):
        residues[i] = targets[i] % p
    
    cov = build_coverage(residues, primes, CAP)
    current_L = get_L_from_cov(cov, CAP)
    current_L = verify_L(residues, primes, current_L, CAP)
    
    best_residues = residues[:]
    best_L = current_L
    
    stagnation = 0
    while time.time() < deadline and current_L < 100000:
        # Find first gap in 1..CAP
        gap = None
        for x in range(1, CAP + 1):
            if cov[x] < 2:
                gap = x
                break
        
        if gap is None:
            # All of 1..CAP covered, check beyond
            for x in range(CAP + 1, min(CAP + 1000, 100001)):
                cnt = 0
                for j, p in enumerate(primes):
                    if x % p == residues[j] % p:
                        cnt += 1
                        if cnt >= 2:
                            break
                if cnt < 2:
                    gap = x
                    break
            if gap is None:
                break
        
        # Find primes covering the gap
        covering = set()
        for j, p in enumerate(primes):
            if gap % p == residues[j] % p:
                covering.add(j)
        
        needed = 2 - len(covering)
        if needed <= 0:
            current_L = gap - 1
            continue
        
        # Try to fix
        candidates = [j for j in range(n_primes) if j not in covering]
        random.shuffle(candidates)
        
        improved = False
        for j in candidates[:20]:
            p = primes[j]
            new_r = gap % p
            old_r = residues[j]
            
            # Update coverage
            old_first = old_r if old_r > 0 else p
            x = old_first
            while x <= CAP:
                cov[x] -= 1
                x += p
            
            new_first = new_r if new_r > 0 else p
            x = new_first
            while x <= CAP:
                cov[x] += 1
                x += p
            
            residues[j] = new_r
            new_L = get_L_from_cov(cov, CAP)
            new_L = verify_L(residues, primes, new_L, CAP)
            
            if new_L > current_L:
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues[:]
                improved = True
                break
            else:
                # Revert
                residues[j] = old_r
                x = new_first
                while x <= CAP:
                    cov[x] -= 1
                    x += p
                x = old_first
                while x <= CAP:
                    cov[x] += 1
                    x += p
        
        if improved:
            stagnation = 0
        else:
            stagnation += 1
            if stagnation > 10:
                # Perturbation
                for _ in range(5):
                    j = random.randint(0, n_primes - 1)
                    residues[j] = random.randint(0, primes[j] - 1)
                # Rebuild coverage
                cov = build_coverage(residues, primes, CAP)
                new_L = get_L_from_cov(cov, CAP)
                new_L = verify_L(residues, primes, new_L, CAP)
                if new_L > current_L:
                    current_L = new_L
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues[:]
                    stagnation = 0
                else:
                    residues = best_residues[:]
                    cov = build_coverage(residues, primes, CAP)
                    current_L = best_L
                    break
    
    return best_residues
# EVOLVE-BLOCK-END
