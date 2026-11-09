"""CSP-inspired approach with fast numpy evaluation and multi-residue local search."""
import time
import random
import math
import numpy as np

# Precompute primes <= 1000
def _get_primes(limit=1000):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

_PRIMES = _get_primes(1000)
_N = len(_PRIMES)  # 168
_NP = np.array(_PRIMES, dtype=np.int64)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def evaluate(residues, max_check=100000):
        """Find largest L such that every x in 1..L has >= 2 covering primes."""
        r = np.array(residues, dtype=np.int64)
        
        # Binary search on L for efficiency
        # First, do a quick check at small L to establish a baseline
        # Then binary search for the maximum L
        
        def check(L):
            """Check if all x in 1..L have >= 2 coverings."""
            if L <= 0:
                return True
            counts = np.zeros(L + 1, dtype=np.int32)
            for i in range(_N):
                p = _NP[i]
                ri = r[i]
                # x = ri + k*p for k >= 0, x in [1, L]
                # k_min: smallest k such that ri + k*p >= 1
                # k_max: largest k such that ri + k*p <= L
                if ri == 0:
                    k_min = 1  # x = p, 2p, ...
                else:
                    k_min = 0  # x = ri, ri+p, ...
                k_max = (L - ri) // p
                if k_min > k_max:
                    continue
                ks = np.arange(k_min, k_max + 1, dtype=np.int64)
                xs = ri + ks * p
                xs = xs[(xs >= 1) & (xs <= L)]
                if len(xs) > 0:
                    np.add.at(counts, xs, 1)
            return np.all(counts[1:] >= 2)
        
        # Quick check at L=100
        if not check(100):
            # Find exact failure point
            lo, hi = 1, 100
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if check(mid):
                    lo = mid
                else:
                    hi = mid - 1
            return lo
        
        # Binary search for maximum L
        lo, hi = 100, max_check
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if check(mid):
                lo = mid
            else:
                hi = mid - 1
        return lo

    def evaluate_fast(residues, current_L):
        """Evaluate starting from current_L, expanding if needed. Returns new L."""
        r = np.array(residues, dtype=np.int64)
        
        def check(L):
            if L <= 0:
                return True
            counts = np.zeros(L + 1, dtype=np.int32)
            for i in range(_N):
                p = _NP[i]
                ri = r[i]
                if ri == 0:
                    k_min = 1
                else:
                    k_min = 0
                k_max = (L - ri) // p
                if k_min > k_max:
                    continue
                ks = np.arange(k_min, k_max + 1, dtype=np.int64)
                xs = ri + ks * p
                xs = xs[(xs >= 1) & (xs <= L)]
                if len(xs) > 0:
                    np.add.at(counts, xs, 1)
            return np.all(counts[1:] >= 2)
        
        # Check if current_L still holds
        if not check(current_L):
            # Find exact failure point
            lo, hi = 1, current_L
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if check(mid):
                    lo = mid
                else:
                    hi = mid - 1
            return lo
        
        # Try to expand
        new_L = current_L
        while new_L < 100000:
            next_L = min(new_L * 2, 100000)
            if check(next_L):
                new_L = next_L
            else:
                # Binary search between new_L and next_L
                lo, hi = new_L, next_L
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if check(mid):
                        lo = mid
                    else:
                        hi = mid - 1
                return lo
        return new_L

    # Initial solution: greedy construction
    def greedy_init():
        """Greedily assign residues to maximize double coverage."""
        residues = [0] * _N
        # Start with a reasonable initial assignment
        for i, p in enumerate(_PRIMES):
            residues[i] = 1 % p  # x=1 is covered by all
        return residues

    # Multi-residue local search
    def multi_local_search(residues, max_time, batch_size=3):
        """Local search that changes multiple residues at once."""
        current = list(residues)
        current_score = evaluate_fast(current, 100)
        
        improved = True
        no_improve_count = 0
        while improved and time.time() < max_time:
            improved = False
            no_improve_count += 1
            
            # Try random batches of residues
            for _ in range(50):
                if time.time() >= max_time:
                    break
                    
                # Pick a batch of indices to change
                batch = random.sample(range(_N), min(batch_size, _N))
                old_vals = [current[i] for i in batch]
                
                # Try random new values for the batch
                for _trial in range(10):
                    new_vals = []
                    for i in batch:
                        p = _PRIMES[i]
                        new_vals.append(random.randint(0, p - 1))
                    
                    for i, v in zip(batch, new_vals):
                        current[i] = v
                    
                    new_score = evaluate_fast(current, current_score)
                    
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                        no_improve_count = 0
                        break
                    else:
                        # Revert
                        for i, v in zip(batch, old_vals):
                            current[i] = v
                
                if improved:
                    break
            
            if no_improve_count > 20:
                # Reset batch size and try larger batches
                batch_size = min(batch_size + 2, 10)
                no_improve_count = 0
        
        return current, current_score

    # Simulated annealing style
    def simulated_annealing(residues, max_time):
        """SA with multi-residue moves."""
        current = list(residues)
        current_score = evaluate_fast(current, 100)
        best = list(current)
        best_score = current_score
        
        T = 50.0
        T_min = 0.1
        cooling = 0.995
        
        while time.time() < max_time:
            # Pick a random batch
            batch_size = random.choice([1, 2, 3, 5])
            batch = random.sample(range(_N), min(batch_size, _N))
            old_vals = [current[i] for i in batch]
            
            new_vals = []
            for i in batch:
                p = _PRIMES[i]
                new_vals.append(random.randint(0, p - 1))
            
            for i, v in zip(batch, new_vals):
                current[i] = v
            
            new_score = evaluate_fast(current, current_score)
            
            if new_score >= current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best = list(current)
            else:
                # Accept with probability exp((new_score - current_score) / T)
                delta = new_score - current_score
                if delta < 0:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        current_score = new_score
                    else:
                        # Revert
                        for i, v in zip(batch, old_vals):
                            current[i] = v
            
            T = max(T_min, T * cooling)
        
        return best, best_score

    # Main search
    random.seed(42)
    
    # Start with greedy initialization
    best_residues = greedy_init()
    best_score = evaluate_fast(best_residues, 100)
    
    # Phase 1: Local search from greedy
    if time.time() < deadline - 20.0:
        best_residues, best_score = multi_local_search(best_residues, deadline - 20.0, batch_size=3)
    
    # Phase 2: Simulated annealing from best
    if time.time() < deadline - 10.0:
        sa_residues, sa_score = simulated_annealing(best_residues, deadline - 10.0)
        if sa_score > best_score:
            best_score = sa_score
            best_residues = sa_residues
    
    # Phase 3: More local search from best
    if time.time() < deadline - 5.0:
        ls_residues, ls_score = multi_local_search(best_residues, deadline - 5.0, batch_size=5)
        if ls_score > best_score:
            best_score = ls_score
            best_residues = ls_residues
    
    return best_residues
