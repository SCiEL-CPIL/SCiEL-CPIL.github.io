"""Covering 1..L by prime residue classes with reciprocal budget 1."""
import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
import sys

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

def _covered_prefix_len(classes, cap=100000):
    if not classes:
        return 0
    covered = bytearray(cap + 1)
    for p, r in classes:
        start = r if r >= 1 else p
        for x in range(start, cap + 1, p):
            covered[x] = 1
    L = 0
    for i in range(1, cap + 1):
        if covered[i]:
            L = i
        else:
            break
    return L

def _recip_sum(classes):
    s = Fraction(0, 1)
    for p, _ in classes:
        s += Fraction(1, p)
    return s

def construct():
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    MAXL = 100000
    primes = _primes_upto(1000)
    recip = {p: Fraction(1, p) for p in primes}
    
    class FastCoverage:
        """Fast incremental coverage tracker with count array."""
        def __init__(self, maxl):
            self.maxl = maxl
            self.count = [0] * (maxl + 1)
            self.L = 0
            self.state = {}  # p -> r
        
        def _start(self, p, r):
            return r if r >= 1 else p
        
        def add(self, p, r):
            if p in self.state:
                return False
            start = self._start(p, r)
            for x in range(start, self.maxl + 1, p):
                self.count[x] += 1
            self.state[p] = r
            self._extend_L()
            return True
        
        def remove(self, p):
            if p not in self.state:
                return False
            r = self.state[p]
            start = self._start(p, r)
            for x in range(start, self.maxl + 1, p):
                self.count[x] -= 1
            del self.state[p]
            self._recompute_L()
            return True
        
        def change_residue(self, p, new_r):
            if p not in self.state:
                return False
            old_r = self.state[p]
            if old_r == new_r:
                return False
            old_start = self._start(p, old_r)
            for x in range(old_start, self.maxl + 1, p):
                self.count[x] -= 1
            new_start = self._start(p, new_r)
            for x in range(new_start, self.maxl + 1, p):
                self.count[x] += 1
            self.state[p] = new_r
            self._recompute_L()
            return True
        
        def _extend_L(self):
            """Extend L forward from current position."""
            newL = self.L
            for i in range(self.L + 1, self.maxl + 1):
                if self.count[i] > 0:
                    newL = i
                else:
                    break
            self.L = newL
        
        def _recompute_L(self):
            """Full recompute of L from scratch."""
            newL = 0
            for i in range(1, self.maxl + 1):
                if self.count[i] > 0:
                    newL = i
                else:
                    break
            self.L = newL
        
        def first_gap(self):
            """Return the first uncovered position after L, or None."""
            for i in range(self.L + 1, self.maxl + 1):
                if self.count[i] == 0:
                    return i
            return None
        
        def snapshot(self):
            return dict(self.state)
        
        def restore(self, snap):
            self.state = dict(snap)
            for i in range(self.maxl + 1):
                self.count[i] = 0
            for p, r in self.state.items():
                start = self._start(p, r)
                for x in range(start, self.maxl + 1, p):
                    self.count[x] += 1
            self._recompute_L()
    
    # Phase 1: Greedy initialization with multiple seeds
    def greedy_init(seed):
        random.seed(seed)
        cov = FastCoverage(MAXL)
        budget = Fraction(1, 1)
        order = primes[:]
        random.shuffle(order)
        
        for _ in range(len(primes)):
            if time.time() > DEADLINE:
                break
            best_gain = 0
            best_p = None
            best_r = None
            
            for p in order:
                if p in cov.state:
                    continue
                if budget < recip[p]:
                    continue
                
                # Choose residue to cover L+1 (the frontier)
                target = cov.L + 1
                r = target % p
                
                # Estimate gain: count newly covered positions in [L+1, L+K]
                K = 1000
                gain = 0
                for x in range(cov.L + 1, min(cov.L + K, MAXL) + 1):
                    if cov.count[x] == 0 and (x - r) % p == 0:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
                    best_r = r
            
            if best_p is None or best_gain == 0:
                break
            
            cov.add(best_p, best_r)
            budget -= recip[best_p]
        
        return cov, budget
    
    # Try several seeds
    best_cov = None
    best_L = 0
    best_budget = Fraction(1, 1)
    
    for seed in range(20):
        if time.time() > DEADLINE:
            break
        cov, budget = greedy_init(seed)
        if cov.L > best_L:
            best_L = cov.L
            best_cov = cov
            best_budget = budget
    
    if best_cov is None:
        # Fallback
        primes13 = [p for p in primes if p >= 13]
        return [[p, (i + 1) % p] for i, p in enumerate(primes13)]
    
    # Phase 2: Targeted local search
    cov = best_cov
    budget = best_budget
    best_L_ls = cov.L
    best_state = cov.snapshot()
    
    rng = random.Random(42)
    
    # Helper: try to fill the first gap by adjusting residues or adding primes
    def try_fill_gap(cov, budget, rng, max_attempts=50):
        """Try to fill the first gap after L. Returns (success, new_budget)."""
        gap = cov.first_gap()
        if gap is None:
            return True, budget
        
        # Strategy 1: Change residue of an existing prime to cover the gap
        used_primes = list(cov.state.keys())
        rng.shuffle(used_primes)
        
        for p in used_primes:
            new_r = gap % p
            if new_r == cov.state[p]:
                continue
            
            old_L = cov.L
            old_r = cov.state[p]
            cov.change_residue(p, new_r)
            
            if cov.L > old_L:
                return True, budget
            else:
                # Revert if it didn't help
                cov.change_residue(p, old_r)
        
        # Strategy 2: Add a new prime to cover the gap
        unused = [p for p in primes if p not in cov.state]
        rng.shuffle(unused)
        
        for p in unused:
            if budget < recip[p]:
                continue
            
            r = gap % p
            old_L = cov.L
            if cov.add(p, r):
                budget -= recip[p]
                if cov.L > old_L:
                    return True, budget
                else:
                    cov.remove(p)
                    budget += recip[p]
        
        # Strategy 3: Remove a prime that's not helping, then re-add with better residue
        if len(used_primes) > 10:
            for p in used_primes:
                old_L = cov.L
                old_r = cov.state[p]
                cov.remove(p)
                budget += recip[p]
                
                # Try to re-add with a residue that covers the gap
                if budget >= recip[p]:
                    r = gap % p
                    cov.add(p, r)
                    budget -= recip[p]
                    
                    if cov.L > old_L:
                        return True, budget
                    else:
                        cov.remove(p)
                        budget += recip[p]
                        cov.add(p, old_r)
                        budget -= recip[p]
        
        return False, budget
    
    # Main local search loop
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < DEADLINE:
        gap = cov.first_gap()
        if gap is None:
            break
        
        # Try to fill the gap
        success, budget = try_fill_gap(cov, budget, rng)
        
        if cov.L > best_L_ls:
            best_L_ls = cov.L
            best_state = cov.snapshot()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            # Reset to best and try different randomization
            cov.restore(best_state)
            budget = best_budget
            no_improve_count = 0
            max_no_improve = int(max_no_improve * 1.5)
        
        # Occasionally do a random perturbation to escape local optima
        if rng.random() < 0.01:
            p = rng.choice(list(cov.state.keys()))
            old_r = cov.state[p]
            new_r = rng.randrange(p)
            cov.change_residue(p, new_r)
            
            if cov.L < best_L_ls - 100:
                cov.change_residue(p, old_r)
    
    # Return best found
    if best_L_ls > best_L:
        result = [[p, r] for p, r in best_state.items()]
        return result
    else:
        result = [[p, r] for p, r in best_cov.state.items()]
        return result
# EVOLVE-BLOCK-END
