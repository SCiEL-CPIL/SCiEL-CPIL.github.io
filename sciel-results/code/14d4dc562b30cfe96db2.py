"""
Strategy: Multi-start iterative improvement with targeted gap-filling and pairwise swaps.
Instead of pure simulated annealing, we use a deterministic heuristic to greedily place primes 
to cover the earliest possible gaps, then refine with local search that allows temporary 
deterioration via swaps, restarting if stuck.
"""
import time
import random
from fractions import Fraction
import numpy as np

# EVOLVE-BLOCK-START

def _sieve(n):
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    primes = _sieve(1000)
    N = 100000
    
    # Precompute the indices for each (p, r) to allow fast delta updates
    indices = {}
    for p in primes:
        indices[p] = {}
        for r in range(p):
            start = r if r > 0 else p
            if start <= N:
                indices[p][r] = np.arange(start, N + 1, p)
            else:
                indices[p][r] = np.array([], dtype=int)
                
    # Helper to calculate L given a coverage count array
    def get_L(cc):
        # cc is a numpy array of uint8, length N+1
        # We want the largest L such that cc[1..L] are all > 0
        # Find the first index i >= 1 where cc[i] == 0
        zeros = np.flatnonzero(cc[1:] == 0)
        if len(zeros) == 0:
            return N
        return zeros[0]

    def select_primes_greedy(min_p=2):
        sel = []
        s = Fraction(0, 1)
        for p in primes:
            if p < min_p:
                continue
            cost = Fraction(1, p)
            if s + cost <= Fraction(1, 1):
                sel.append(p)
                s += cost
        return sel, s

    # Generate a few candidate prime sets
    prime_sets = []
    
    # Set A: Greedy from 2
    sel, _ = select_primes_greedy(2)
    prime_sets.append(sel)
    
    # Set B: Greedy from 5 (skip 2, 3)
    sel, _ = select_primes_greedy(5)
    if sel: prime_sets.append(sel)
    
    # Set C: Greedy skipping 3
    sel_b = []
    s = Fraction(0,1)
    for p in primes:
        if p == 3: continue
        if s + Fraction(1,p) <= 1:
            sel_b.append(p)
            s += Fraction(1,p)
    if sel_b: prime_sets.append(sel_b)
    
    best_L = 0
    best_pairs = []
    
    # We will run multiple independent "runs" (restarts) with different prime sets or random initializations
    # Since the prime set is fixed by the budget constraint mostly, we focus on residue initialization.
    
    num_restarts = 10
    
    for run_idx in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Choose a prime set. Cycle through them or pick randomly.
        ps = prime_sets[run_idx % len(prime_sets)]
        if not ps: continue
        
        # Initialize residues
        # Strategy: Greedy placement targeting the first gap
        cover_count = np.zeros(N + 1, dtype=np.uint8)
        assigned_primes = []
        
        # Sort primes. Smaller primes first? Or random?
        # Let's try random order for diversity in restarts, but greedy placement.
        # Actually, placing small primes first is generally better for base coverage.
        # Let's shuffle the order of addition for diversity in some restarts.
        if run_idx % 2 == 0:
            current_primes = ps[:]
        else:
            current_primes = ps[:]
            random.shuffle(current_primes)
            
        for p in current_primes:
            best_r = 0
            best_L_for_p = -1
            
            # Try all residues for this prime
            # Optimization: We only care about extending the current contiguous block.
            # Let current_L be the current max contiguous coverage.
            # We want to find r such that it covers the first gap (current_L + 1) if possible,
            # or extends the block further.
            
            # To be safe and simple, we evaluate all residues but break early if we hit N.
            # However, N=100000 and p is small, so checking all residues is O(N) per prime.
            # Total O(|ps| * N). This is fine.
            
            for r in range(p):
                idx = indices[p][r]
                if len(idx) == 0:
                    continue
                
                # Tentative update
                cover_count[idx] += 1
                new_L = get_L(cover_count)
                cover_count[idx] -= 1 # Revert
                
                if new_L > best_L_for_p:
                    best_L_for_p = new_L
                    best_r = r
            
            # Commit
            idx = indices[p][best_r]
            if len(idx) > 0:
                cover_count[idx] += 1
            assigned_primes.append((p, best_r))
            
        current_L = get_L(cover_count)
        
        if current_L > best_L:
            best_L = current_L
            best_pairs = assigned_primes.copy()
            
        # Local Search Phase: Iterative Improvement with Swaps
        # We look for moves that increase L.
        # If stuck, we look for swaps that might lead to a better configuration later.
        # Given time constraints, we do a limited number of iterations.
        
        max_iters = 2000
        no_improve = 0
        max_no_improve = 200
        
        # We need to track which primes are assigned to which residues
        # assigned_primes is a list of (p, r)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Pick a random prime to move
            idx_prime = random.randint(0, len(assigned_primes) - 1)
            p, old_r = assigned_primes[idx_prime]
            
            # Try to find a better residue for this prime
            # We prioritize residues that cover the current first gap.
            # Let gap = current_L + 1
            # If gap <= N, we want a residue r such that gap % p == r.
            # If such r exists, it's a strong candidate.
            
            gap = current_L + 1
            if gap <= N:
                target_r = gap % p
                # Try the target residue first
                if target_r != old_r:
                    idx_old = indices[p][old_r]
                    idx_new = indices[p][target_r]
                    
                    if len(idx_old) > 0:
                        cover_count[idx_old] -= 1
                    if len(idx_new) > 0:
                        cover_count[idx_new] += 1
                        
                    new_L = get_L(cover_count)
                    
                    if new_L > current_L:
                        assigned_primes[idx_prime] = (p, target_r)
                        current_L = new_L
                        no_improve = 0
                        if current_L > best_L:
                            best_L = current_L
                            best_pairs = assigned_primes.copy()
                        continue
                    else:
                        # Revert
                        if len(idx_new) > 0:
                            cover_count[idx_new] -= 1
                        if len(idx_old) > 0:
                            cover_count[idx_old] += 1
                    
            # If target didn't work or wasn't available, try random residues
            # Limit random tries to keep it fast
            for _ in range(5):
                new_r = random.randint(0, p - 1)
                if new_r == old_r:
                    continue
                
                idx_old = indices[p][old_r]
                idx_new = indices[p][new_r]
                
                if len(idx_old) > 0:
                    cover_count[idx_old] -= 1
                if len(idx_new) > 0:
                    cover_count[idx_new] += 1
                    
                new_L = get_L(cover_count)
                
                if new_L > current_L:
                    assigned_primes[idx_prime] = (p, new_r)
                    current_L = new_L
                    no_improve = 0
                    if current_L > best_L:
                        best_L = current_L
                        best_pairs = assigned_primes.copy()
                    break
                else:
                    # Revert
                    if len(idx_new) > 0:
                        cover_count[idx_new] -= 1
                    if len(idx_old) > 0:
                        cover_count[idx_old] += 1
            
            no_improve += 1
            if no_improve > max_no_improve:
                break
                
    return best_pairs

# EVOLVE-BLOCK-END
