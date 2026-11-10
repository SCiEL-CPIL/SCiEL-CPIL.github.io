"""
Strategy: Constraint-Satisfaction via Backtracking with Forward Checking
Instead of greedy repair, we treat the problem as finding residues such that
for all x in [1, L], the count of primes p where x ≡ r_p (mod p) is >= 2.

We use a backtracking solver that:
1. Processes integers from 1 upwards
2. For each integer x, if its current coverage is < 2, we "fix" it by changing
   residues of primes that cover x
3. Uses forward checking: after changing a residue, update coverage counts
4. Uses conflict-directed backjumping: if a change breaks a previously satisfied
   integer, backtrack to the last conflicting decision
5. Employs heuristic variable ordering: prioritize primes with small p (more
   frequent coverage) and residues that maximize local density

Key insight: The problem is highly constrained but has many degrees of freedom
(168 residues). A well-guided backtracking solver can explore the solution
space more systematically than greedy methods, especially for finding
doubly-covered prefixes.
"""

import time
import random
from collections import defaultdict

def _primes_upto(n):
    """Generate primes up to n using sieve."""
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n+1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    primes = _primes_upto(1000)
    n_primes = len(primes)
    
    # Initialize residues: use the seed strategy as initial guess
    # The i-th prime covers ceil(i/2)
    residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    
    # Precompute: for each prime index, which integers in [1, MAX_L] it covers
    # MAX_L: we'll search up to 1000 initially, but can extend
    MAX_L = 2000  # Search up to 2000 to see if we can exceed 1000
    
    # Coverage counts: cov[x] = number of primes covering x
    def compute_coverage(res):
        cov = [0] * (MAX_L + 1)
        for i, p in enumerate(primes):
            r = res[i]
            # Find all x in [1, MAX_L] where x ≡ r (mod p)
            # x = r + k*p for k >= 0
            # First x >= 1: if r == 0, x starts at p; else x starts at r
            if r == 0:
                first_x = p
            else:
                first_x = r
            if first_x > MAX_L:
                continue
            for x in range(first_x, MAX_L + 1, p):
                cov[x] += 1
        return cov
    
    def get_score(res):
        cov = compute_coverage(res)
        L = 0
        for x in range(1, MAX_L + 1):
            if cov[x] >= 2:
                L = x
            else:
                break
        return L
    
    # Initial score
    best_score = get_score(residues)
    best_residues = residues[:]
    
    # Backtracking solver with forward checking
    # State: residues, coverage array, current frontier (smallest x with cov < 2)
    
    class Solver:
        def __init__(self):
            self.res = residues[:]
            self.cov = compute_coverage(self.res)
            self.frontier = 1  # Smallest x to check
            self.best_L = best_score
            self.best_res = self.res[:]
            self.history = []  # (x, prime_idx, old_res, new_res) for backtracking
            self.conflicts = defaultdict(list)  # x -> list of prime indices that affect it
            
            # Precompute which primes can cover which x (for fast lookup)
            # For a given x, the primes that can cover x are those p where x mod p is a valid residue
            # Actually, any prime p can cover x by setting r_p = x mod p
            # So for each x, all 168 primes can potentially cover it
            
            # Heuristic: for each x, rank primes by how "useful" they are
            # A prime p is more useful for x if changing its residue to x mod p
            # doesn't break too many other constraints
            
        def update_cov_for_prime(self, prime_idx, old_r, new_r):
            """Update coverage when prime prime_idx changes from old_r to new_r."""
            p = primes[prime_idx]
            # Remove old coverage
            if old_r == 0:
                first_old = p
            else:
                first_old = old_r
            if first_old <= MAX_L:
                for x in range(first_old, MAX_L + 1, p):
                    self.cov[x] -= 1
            
            # Add new coverage
            if new_r == 0:
                first_new = p
            else:
                first_new = new_r
            if first_new <= MAX_L:
                for x in range(first_new, MAX_L + 1, p):
                    self.cov[x] += 1
            
            # Update frontier
            while self.frontier <= MAX_L and self.cov[self.frontier] >= 2:
                self.frontier += 1
            
            # Update best
            if self.frontier - 1 > self.best_L:
                self.best_L = self.frontier - 1
                self.best_res = self.res[:]
        
        def try_fix_x(self, x):
            """Try to fix integer x by changing residues of primes that cover it."""
            if self.cov[x] >= 2:
                return True
            
            need = 2 - self.cov[x]
            
            # Find primes that can cover x: all primes, with residue x mod p
            # Heuristic: prefer primes where changing residue causes minimal disruption
            # Score each prime by how many other integers in [frontier, frontier+10] it would break
            
            candidates = []
            for i, p in enumerate(primes):
                new_r = x % p
                if new_r == self.res[i]:
                    continue  # Already covers x
                
                # Estimate disruption: how many integers in [x, x+20] would lose coverage
                disruption = 0
                old_r = self.res[i]
                for dx in range(0, 21):
                    xx = x + dx
                    if xx > MAX_L:
                        break
                    # Check if xx was covered by old_r
                    if old_r == 0:
                        old_covers = (xx % p == 0 and xx >= p)
                    else:
                        old_covers = (xx % p == old_r and xx >= old_r)
                    
                    # Check if xx is covered by new_r
                    if new_r == 0:
                        new_covers = (xx % p == 0 and xx >= p)
                    else:
                        new_covers = (xx % p == new_r and xx >= new_r)
                    
                    if old_covers and not new_covers:
                        disruption += 1
                    elif not old_covers and new_covers:
                        disruption -= 1
                
                # Prefer primes with low disruption, and small p (more frequent)
                candidates.append((disruption, p, i, new_r))
            
            candidates.sort()
            
            # Try top candidates
            for disruption, p, i, new_r in candidates[:5]:
                if time.time() > deadline:
                    return False
                
                old_r = self.res[i]
                self.history.append((x, i, old_r, new_r))
                self.res[i] = new_r
                self.update_cov_for_prime(i, old_r, new_r)
                
                if self.cov[x] >= 2:
                    # Check if we broke any previously satisfied integers
                    broke = False
                    for hx in range(1, min(x, self.frontier + 10)):
                        if self.cov[hx] < 2:
                            broke = True
                            break
                    
                    if not broke:
                        return True
                    else:
                        # Backtrack
                        self.res[i] = old_r
                        self.update_cov_for_prime(i, new_r, old_r)
                        self.history.pop()
                        # Record conflict
                        self.conflicts[x].append(i)
            
            return False
        
        def solve(self, max_steps=10000):
            steps = 0
            while self.frontier <= MAX_L and steps < max_steps:
                if time.time() > deadline:
                    break
                
                x = self.frontier
                if not self.try_fix_x(x):
                    # Backtrack to last conflict
                    if self.history:
                        # Find the last decision that caused a conflict
                        # Simple: undo last few decisions
                        for _ in range(min(3, len(self.history))):
                            hx, hi, old_r, new_r = self.history.pop()
                            self.res[hi] = old_r
                            self.update_cov_for_prime(hi, new_r, old_r)
                    else:
                        break
                steps += 1
            
            return self.best_L, self.best_res
    
    # Run the solver
    solver = Solver()
    score, res = solver.solve(max_steps=50000)
    
    if score > best_score:
        best_score = score
        best_residues = res
    
    # Local search improvement
    if time.time() < deadline - 20:
        # Simple local search: try changing one residue at a time
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for i in range(n_primes):
                if time.time() > deadline - 10:
                    break
                old_r = best_residues[i]
                p = primes[i]
                # Try a few random residues
                for _ in range(10):
                    if time.time() > deadline - 10:
                        break
                    new_r = random.randint(0, p - 1)
                    if new_r == old_r:
                        continue
                    best_residues[i] = new_r
                    new_score = get_score(best_residues)
                    if new_score > best_score:
                        best_score = new_score
                        improved = True
                    else:
                        best_residues[i] = old_r
                        break
    
    return best_residues
