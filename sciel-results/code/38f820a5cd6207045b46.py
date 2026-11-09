"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import numpy as np

N = 5000
_DEADLINE = time.time() + 320.0  # 330s wall-clock, 10s safety margin

# EVOLVE-BLOCK-START

def _primes(limit):
    """Return list of primes <= limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, ok in enumerate(sieve) if ok]

_ALL_PRIMES = _primes(N)
_MAX_K = 1000

# Precompute indices for each prime and residue class
_RESIDUE_INDICES = {}
for p in _ALL_PRIMES:
    res = []
    for r in range(p):
        res.append(np.arange(r, N + 1, p))
    _RESIDUE_INDICES[p] = res

class AdmissibleTupleSolver:
    def __init__(self, k, primes_to_use):
        self.k = k
        self.primes = primes_to_use
        self.P = len(primes_to_use)
        self.cover_count = np.zeros(N + 1, dtype=np.int32)
        self.residues = [0] * self.P
        
        for i, p in enumerate(self.primes):
            r = random.randint(0, p - 1)
            self.residues[i] = r
            indices = _RESIDUE_INDICES[p][r]
            if len(indices) > 0:
                self.cover_count[indices] += 1
        
        self.alive = (self.cover_count == 0)
        self.survivor_count = int(np.sum(self.alive))
        
        # Store for fast access
        self._res_indices = [_RESIDUE_INDICES[p] for p in self.primes]

    def _update(self, i, new_r):
        """
        Update residue of prime i to new_r.
        Returns delta in survivor_count.
        """
        old_r = self.residues[i]
        if old_r == new_r:
            return 0
            
        delta = 0
        p = self.primes[i]
        res_arrs = self._res_indices[i]
        
        # Remove old
        old_indices = res_arrs[old_r]
        if len(old_indices) > 0:
            # Decrement cover count
            self.cover_count[old_indices] -= 1
            # Count how many became alive (cover_count == 0)
            delta += int(np.sum(self.cover_count[old_indices] == 0))
            
        # Add new
        new_indices = res_arrs[new_r]
        if len(new_indices) > 0:
            # Count how many become dead (cover_count == 0 before increment)
            delta -= int(np.sum(self.cover_count[new_indices] == 0))
            self.cover_count[new_indices] += 1
            
        self.residues[i] = new_r
        self.survivor_count += delta
        return delta

    def local_search(self, time_limit, max_iters=2000000):
        """
        Local search to maximize survivor_count.
        """
        best_survivors = self.survivor_count
        best_residues = self.residues.copy()
        
        # Simple Hill Climbing with random restarts within time limit?
        # Or just pure hill climbing? Pure hill climbing gets stuck.
        # Let's use a "First Ascent" or "Best Ascent" with some noise?
        # Given the time, let's try a mix:
        # 1. Try to find any improving move.
        # 2. If none found, do a random perturbation (like SA but low temp).
        
        # Actually, let's just do standard SA with a very fast loop.
        temp = 5.0
        cooling = 0.99999
        
        start_time = time.time()
        iters = 0
        no_improve_count = 0
        best_in_window = self.survivor_count
        
        while iters < max_iters and (time.time() - start_time) < time_limit:
            iters += 1
            
            if iters % 1000 == 0 and time.time() > _DEADLINE - 20:
                break
            
            # Choose a prime to update
            # Bias towards smaller primes? Smaller primes affect more numbers.
            # Or just uniform.
            i = random.randint(0, self.P - 1)
            p = self.primes[i]
            new_r = random.randint(0, p - 1)
            old_r = self.residues[i]
            
            if new_r == old_r:
                continue
                
            delta = self._update(i, new_r)
            new_surv = self.survivor_count
            
            if new_surv >= best_survivors:
                best_survivors = new_surv
                best_residues = self.residues.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                # Accept with probability exp(-delta/temp) if delta < 0
                if delta < 0 and temp > 1e-6:
                    if random.random() < math.exp(delta / temp):
                        pass # Accept (already applied)
                    else:
                        self._update(i, old_r) # Revert
                else:
                    # If delta < 0 and not accepted, revert.
                    # If delta >= 0, it was accepted (since we check >= best, but here we are in else, so delta < 0? No.
                    # Wait. If new_surv >= best_survivors, we accept.
                    # If new_surv < best_survivors:
                    #   It might still be an improvement over CURRENT state?
                    #   SA accepts if new_state > old_state OR with prob.
                    #   My current logic:
                    #   1. Apply change.
                    #   2. Check if global best improved.
                    #   3. If not global best, decide to revert or not based on SA acceptance.
                    #   
                    #   But SA acceptance depends on CURRENT state vs NEW state.
                    #   old_surv = self.survivor_count - delta
                    #   if new_surv > old_surv: always accept.
                    #   else: accept with prob.
                    
                    # I need to know old_surv.
                    # old_surv = self.survivor_count - delta
                    # If new_surv > old_surv, we should accept.
                    # My code above:
                    # if new_surv >= best_survivors: accept (update best)
                    # else: 
                    #    if delta < 0 (worse than current? No, delta is change. If delta < 0, new < old)
                    #    Wait. delta = new - old.
                    #    If delta < 0, new < old. This is a worsening move.
                    #    If delta > 0, new > old. This is an improving move.
                    
                    # So:
                    # if delta > 0:
                    #    Accept. (And check global best)
                    # else (delta <= 0):
                    #    Accept with prob exp(delta/temp).
                    #    If rejected, revert.
                    
                    # My current code structure is a bit mixed. Let's clean it up.
                    # I'll just revert if delta < 0 and random check fails.
                    # If delta >= 0, we keep it (improvement).
                    # But wait, if delta >= 0, we already handled it in `if new_surv >= best_survivors`?
                    # No, `best_survivors` is global max. `self.survivor_count` is current.
                    # If delta > 0, new > old. It's an improvement. We should accept.
                    # If delta < 0, new < old. Worsening.
                    
                    # So:
                    # if delta > 0:
                    #    # Always accept
                    #    # Update global best if needed
                    # else:
                    #    # Maybe accept
                    #    if random...
                    #       # Accept (do nothing, already applied)
                    #    else:
                    #       # Revert
                    #       self._update(i, old_r)
                    
                    # The block above was:
                    # if new_surv >= best_survivors:
                    #    ...
                    # else:
                    #    no_improve_count += 1
                    #    if delta < 0 and temp > 1e-6:
                    #       if random < exp(delta/temp): pass
                    #       else: revert
                    #    else:
                    #       # This branch is reached if delta >= 0 but new_surv < best_survivors?
                    #       # Impossible if delta > 0, because new > old. If old was not best, new might not be best.
                    #       # But delta > 0 means improvement. We should accept it.
                    #       # So this `else` block is WRONG. It doesn't revert, but it also doesn't explicitly accept?
                    #       # It just falls through. Since we already applied the update, falling through means accepting.
                    #       # So it accepts.
                    #       # But wait, if delta < 0 and we DIDN'T revert, it accepts.
                    #       # So this logic is:
                    #       # If delta < 0:
                    #       #   If prob > exp: Revert.
                    #       #   Else: Accept.
                    #       # If delta >= 0:
                    #       #   Accept.
                    #       # This is correct SA.
                    pass
                    
            temp *= cooling
            if temp < 0.1:
                temp = 0.1 # Floor to prevent underflow, or just let it be 0?
                # If temp is 0, we only accept improvements.
                # This is fine.
                
        return best_survivors, best_residues

def construct():
    """
    Constructs the largest admissible tuple in [0, N].
    """
    # Binary search for max K
    # Range: 1 to 500
    # Theoretical max is roughly N / log(N) ? No, N * prod(1-1/p).
    # For K=500, primes up to 500.
    # Density ~ e^-gamma / ln(500) ~ 0.56 / 6.2 ~ 0.09.
    # 5000 * 0.09 = 450.
    # So K=500 is likely infeasible.
    # K=400? Primes up to 400. ln(400)=6.0. 0.56/6.0 = 0.093. 5000*0.093=465.
    # K=300? Primes up to 300. ln(300)=5.7. 0.56/5.7=0.098. 5000*0.098=490.
    # So 300-400 is the sweet spot.
    
    low = 1
    high = 500
    best_k = 1
    best_tuple = None
    
    # Precompute masks for all primes up to high
    # Already done in _RESIDUE_INDICES
    
    while low <= high:
        if time.time() > _DEADLINE - 30:
            break
            
        mid = (low + high) // 2
        primes_to_use = [p for p in _ALL_PRIMES if p <= mid]
        
        if not primes_to_use:
            best_k = mid
            best_tuple = list(range(mid))
            low = mid + 1
            continue
            
        solver = AdmissibleTupleSolver(mid, primes_to_use)
        
        # Allocate time based on remaining budget and difficulty
        # Harder (larger K) needs more time? Or easier needs less?
        # Let's give it a fixed chunk of time, e.g., 20s, but scale down if deadline is close.
        time_left = _DEADLINE - 10 - time.time()
        if time_left <= 0:
            break
            
        # Spend at most 20s or 50% of remaining time?
        # Let's say 20s per check is safe.
        check_time = min(20.0, time_left * 0.5)
        
        best_survivors, best_res = solver.local_search(check_time)
        
        if best_survivors >= mid:
            # Feasible
            # Reconstruct the tuple
            alive = np.ones(N + 1, dtype=bool)
            for p, r in zip(primes_to_use, best_res):
                indices = _RESIDUE_INDICES[p][r]
                if len(indices) > 0:
                    alive[indices] = False
                    
            result_indices = np.where(alive)[0]
            # Take the first 'mid' survivors
            best_tuple = result_indices[:mid].tolist()
            best_k = mid
            low = mid + 1
        else:
            high = mid - 1

    if best_tuple is None:
        return []
        
    return best_tuple

# EVOLVE-BLOCK-END
