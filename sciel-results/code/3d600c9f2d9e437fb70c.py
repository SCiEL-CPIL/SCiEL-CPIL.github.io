import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    # Generate primes up to 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit+1, i):
                    sieve[j] = False
        return [i for i in range(2, limit+1) if sieve[i]]

    primes = get_primes(1000)
    n = len(primes)  # 168
    L_max = 100000

    def _score(residues, primes, L_max):
        counts = np.zeros(L_max + 1, dtype=np.int32)
        for i in range(len(primes)):
            r = residues[i]
            p = primes[i]
            r_eff = p if r == 0 else r
            if r_eff <= L_max:
                counts[r_eff::p] += 1
        deficient = np.nonzero(counts[1:L_max+1] < 2)[0]
        if len(deficient) == 0:
            return L_max
        return int(deficient[0])

    def _greedy_init(primes, L_max, seed=42):
        random.seed(seed)
        residues = [0] * n
        counts = np.zeros(L_max + 1, dtype=np.int32)
        
        for idx in range(n):
            p = primes[idx]
            best_r = 0
            best_gain = -1
            
            if p <= 100:
                candidates = list(range(p))
            else:
                candidates = random.sample(range(p), min(p, 100))
            
            for r in candidates:
                r_eff = p if r == 0 else r
                if r_eff > 1000:
                    gain = 0
                else:
                    positions = np.arange(r_eff, 1001, p)
                    gain = np.sum(counts[positions] < 2)
                
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            residues[idx] = best_r
            r = best_r
            r_eff = p if r == 0 else r
            if r_eff <= L_max:
                counts[r_eff::p] += 1
                
        return residues

    def _simulated_annealing(residues, primes, L_max, deadline, seed=0):
        random.seed(seed)
        current_L = _score(residues, primes, L_max)
        best_L = current_L
        best_res = residues[:]
        
        counts = np.zeros(L_max + 1, dtype=np.int32)
        for i in range(n):
            r = residues[i]
            p = primes[i]
            r_eff = p if r == 0 else r
            if r_eff <= L_max:
                counts[r_eff::p] += 1
                
        T = 1.0
        T_min = 1e-6
        alpha = 0.9999
        
        while time.time() < deadline:
            i = random.randint(0, n - 1)
            p = primes[i]
            old_r = residues[i]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
            
            old_r_eff = p if old_r == 0 else old_r
            new_r_eff = p if new_r == 0 else new_r
            
            # Apply change
            if old_r_eff <= L_max:
                counts[old_r_eff::p] -= 1
            if new_r_eff <= L_max:
                counts[new_r_eff::p] += 1
                
            # Evaluate
            deficient = np.nonzero(counts[1:L_max+1] < 2)[0]
            if len(deficient) == 0:
                new_L = L_max
            else:
                new_L = int(deficient[0])
            
            # Acceptance
            accepted = False
            if new_L >= current_L:
                accepted = True
            else:
                if T > T_min:
                    prob = np.exp(-(- (new_L - current_L)) / T)
                    if random.random() < prob:
                        accepted = True
            
            if accepted:
                current_L = new_L
                residues[i] = new_r
                if current_L > best_L:
                    best_L = current_L
                    best_res = residues[:]
            else:
                # Revert
                if new_r_eff <= L_max:
                    counts[new_r_eff::p] -= 1
                if old_r_eff <= L_max:
                    counts[old_r_eff::p] += 1
                # residues[i] is already old_r
            
            T *= alpha
            
        return best_res, best_L

    # Phase 1: Greedy Init + SA
    residues = _greedy_init(primes, L_max, seed=42)
    mid_deadline = start_time + 60
    residues, L1 = _simulated_annealing(residues, primes, L_max, mid_deadline, seed=1)
    
    # Phase 2: Perturb and SA again
    if time.time() < deadline - 5:
        residues2 = residues[:]
        # Perturb
        random.seed(99)
        for _ in range(20):
            idx = random.randint(0, n-1)
            residues2[idx] = random.randint(0, primes[idx]-1)
        
        residues2, L2 = _simulated_annealing(residues2, primes, L_max, deadline, seed=2)
        
        if L2 > L1:
            residues = residues2

    return [int(r) for r in residues]
