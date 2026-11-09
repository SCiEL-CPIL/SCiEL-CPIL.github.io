"""
Strategy: Greedy Local Search with Incremental Updates and Multi-Start Restarts.

Instead of Simulated Annealing with full O(P*N) re-evaluations, we:
1. Use a greedy initialization that minimizes removals.
2. Perform local search by changing one residue at a time.
3. Crucially, update the survivor count incrementally in O(N/p) time per move,
   rather than O(P*N). This allows ~1000x more iterations.
4. Use multiple random restarts to explore different local optima.
5. Binary search for the maximum feasible size k.
"""

import time
import random
import math
import numpy as np

N = 5000
_DEADLINE = time.time() + 320.0  # Safety margin

def _primes(limit):
    """Return list of primes <= limit."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, ok in enumerate(sieve) if ok]

# Precompute primes up to N
ALL_PRIMES = _primes(N)

def construct():
    """
    Constructs the largest admissible tuple in [0, N].
    """
    
    # Precompute modulo indices for speed
    # mod_indices[p] = np.array([x % p for x in range(N+1)])
    # We also precompute the indices for each residue class for each prime?
    # That might use too much memory. 
    # Let's stick to mod_indices[p] and use np.where or boolean masks.
    # Actually, for incremental updates, we need to know which indices are in residue r.
    # We can precompute a list of lists: residue_indices[p][r] = list of x in [0,N] where x%p==r.
    # Memory: Sum over p<=N of (N/p) * (p) ~ N * sum(1) ~ N * pi(N) ~ 5000 * 600 = 3M integers.
    # 3M integers * 4 bytes = 12MB. This is very manageable.
    
    residue_indices = {}
    for p in ALL_PRIMES:
        res_list = [[] for _ in range(p)]
        # Efficient way to fill:
        for r in range(p):
            # x = r, r+p, r+2p ... <= N
            res_list[r] = list(range(r, N + 1, p))
        residue_indices[p] = res_list

    def get_survivors_full(primes_list, residues):
        """
        Compute number of survivors given a list of primes and corresponding residues.
        Used for initial evaluation and verification.
        """
        if not primes_list:
            return N + 1
        
        valid = np.ones(N + 1, dtype=bool)
        
        for p, r in zip(primes_list, residues):
            # Mark all x in residue r mod p as invalid
            # Using the precomputed lists is faster than boolean mask on full array?
            # Actually, boolean mask on full array is O(N).
            # Marking individual indices is O(N/p).
            # For small p, O(N/p) is much smaller than N.
            # For large p, N/p is small.
            # Let's use the precomputed lists for speed.
            for idx in residue_indices[p][r]:
                valid[idx] = False
            
        return np.sum(valid)

    def greedy_init(primes_list):
        """
        Greedy initialization: process primes in increasing order.
        Choose residue that minimizes new removals (i.e., removes the fewest currently alive numbers).
        """
        residues = []
        valid = np.ones(N + 1, dtype=bool)
        
        for p in primes_list:
            # Count alive numbers in each residue class
            # We need to check valid[idx] for idx in residue_indices[p][r]
            counts = np.zeros(p, dtype=int)
            for r in range(p):
                # Sum of valid[idx] for idx in residue_indices[p][r]
                # This is O(N/p) per r, so O(N) per p.
                # We can vectorize this?
                # valid is a numpy array.
                # Let's just loop over r and use sum(valid[idx] for idx in ...)
                # Or better: create a numpy array of indices for all residues? No, they are different lengths.
                # Let's use a loop over residues, it's fast enough for greedy.
                res_idx = residue_indices[p][r]
                # Convert to numpy for fast sum?
                # valid[res_idx] returns a numpy array of bools.
                counts[r] = np.sum(valid[res_idx])
            
            min_count = np.min(counts)
            candidates = np.where(counts == min_count)[0]
            # Tie-breaker: random or first? Random might help diversity.
            r_min = candidates[random.randint(0, len(candidates)-1)]
            
            residues.append(r_min)
            
            # Update valid
            for idx in residue_indices[p][r_min]:
                valid[idx] = False
                
        return residues

    def local_search(primes_list, init_residues, time_budget):
        """
        Perform local search to maximize survivors.
        Uses incremental updates.
        """
        k = len(primes_list)
        if k == 0:
            return get_survivors_full(primes_list, init_residues), init_residues
        
        current_residues = init_residues[:]
        
        # Initial evaluation
        # We maintain a 'valid' array to allow O(N/p) updates.
        valid = np.ones(N + 1, dtype=bool)
        for p, r in zip(primes_list, current_residues):
            for idx in residue_indices[p][r]:
                valid[idx] = False
        current_score = np.sum(valid)
        
        best_score = current_score
        best_residues = current_residues[:]
        
        start_time = time.time()
        no_improvement_count = 0
        max_no_improvement = 5000 # Restart or stop if stuck
        
        while time.time() - start_time < time_budget:
            # Pick a random prime index
            idx = random.randint(0, k - 1)
            p = primes_list[idx]
            old_r = current_residues[idx]
            
            # Pick a new random residue
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
            
            # Calculate delta in score
            # Score = sum(valid)
            # Changing residue for p from old_r to new_r:
            # 1. Un-remove indices in old_r that are NOT removed by other primes.
            # 2. Remove indices in new_r that are NOT removed by other primes.
            
            # This incremental logic is tricky because "removed by other primes" depends on the current state.
            # However, we can compute the change by looking at the current `valid` array.
            
            # Let A = set of indices in residue old_r mod p.
            # Let B = set of indices in residue new_r mod p.
            
            # Currently, all indices in A are marked False in `valid` (if they weren't already False by others).
            # Actually, `valid[idx]` is False if idx is in ANY forbidden residue.
            
            # If we change old_r to new_r:
            # Indices in A that were ONLY removed by this prime (i.e., valid[idx] was False solely due to this prime? No, we don't track counts).
            # We can't easily know if an index was removed by *other* primes.
            
            # Alternative approach for incremental update:
            # We can't just look at `valid` because it's a boolean mask.
            # We need to know how many indices in A are "solely" removed by this prime to know how many we gain.
            # We need to know how many indices in B are "not" removed by other primes to know how many we lose.
            
            # This requires tracking "hit counts" or similar, which is expensive to update.
            
            # Given the complexity of correct incremental updates with boolean masks,
            # and N=5000 is small, let's revert to a slightly faster full re-evaluation
            # but use a very tight loop and early termination if possible.
            # OR, use the "hit count" approach.
            
            # Let's try the Hit Count approach.
            # hit_count[x] = number of primes p such that x % p == current_residue[p].
            # If hit_count[x] > 0, x is removed.
            # Score = count of x where hit_count[x] == 0.
            
            # This is much better for incremental updates!
            
            # Let's re-initialize with hit counts.
            # We need to restart the local search logic with hit counts.
            
            # Actually, let's just restart the whole function logic with hit counts from the beginning.
            # It's cleaner.
            
            break # Break out to use the hit-count version below

        # --- Hit Count Implementation ---
        
        # Re-initialize state with hit counts
        hit_counts = np.zeros(N + 1, dtype=int)
        for p, r in zip(primes_list, current_residues):
            for idx in residue_indices[p][r]:
                hit_counts[idx] += 1
        
        current_score = np.sum(hit_counts == 0)
        best_score = current_score
        best_residues = current_residues[:]
        
        start_time = time.time()
        no_improvement_count = 0
        
        while time.time() - start_time < time_budget:
            if no_improvement_count > max_no_improvement:
                break
                
            idx = random.randint(0, k - 1)
            p = primes_list[idx]
            old_r = current_residues[idx]
            
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
            
            # Calculate delta
            # Indices in old_r: A
            # Indices in new_r: B
            
            # For x in A:
            # hit_counts[x] decreases by 1.
            # If hit_counts[x] was 1, it becomes 0 -> Score +1.
            # If hit_counts[x] was > 1, it becomes > 0 -> Score unchanged.
            
            # For x in B:
            # hit_counts[x] increases by 1.
            # If hit_counts[x] was 0, it becomes 1 -> Score -1.
            # If hit_counts[x] was > 0, it becomes > 1 -> Score unchanged.
            
            # Note: A and B are disjoint for a given prime p (since old_r != new_r).
            
            gain = 0
            loss = 0
            
            for x in residue_indices[p][old_r]:
                if hit_counts[x] == 1:
                    gain += 1
            
            for x in residue_indices[p][new_r]:
                if hit_counts[x] == 0:
                    loss += 1
            
            delta = gain - loss
            
            if delta > 0:
                # Accept
                for x in residue_indices[p][old_r]:
                    hit_counts[x] -= 1
                for x in residue_indices[p][new_r]:
                    hit_counts[x] += 1
                current_residues[idx] = new_r
                current_score += delta
                no_improvement_count = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_residues = current_residues[:]
            elif delta == 0:
                # Neutral move, accept with some probability to escape plateaus?
                # Or just reject to stay focused?
                # Let's accept 50% of the time to allow wandering.
                if random.random() < 0.5:
                    for x in residue_indices[p][old_r]:
                        hit_counts[x] -= 1
                    for x in residue_indices[p][new_r]:
                        hit_counts[x] += 1
                    current_residues[idx] = new_r
                    no_improvement_count += 1
            else:
                # Delta < 0
                no_improvement_count += 1
                
        return best_score, best_residues

    # Binary search for the maximum admissible size m
    low = 0
    high = N + 1
    best_m = 0
    best_residues_final = []
    
    # Time budget allocation
    # We have ~300s.
    # Binary search takes ~12-13 steps (log2(5000)).
    # Each step runs local search.
    # We should allocate time proportionally or uniformly?
    # Uniform is safer.
    # Total time ~ 300s.
    # Steps ~ 13.
    # Time per step ~ 20s.
    # But we also have initialization and final construction.
    # Let's use 15s per local search run.
    
    TIME_PER_RUN = 15.0
    
    while low <= high:
        if time.time() > _DEADLINE - 10:
            break
            
        mid = (low + high) // 2
        
        # Get primes <= mid
        # Primes are sorted. We can find the index.
        # Or just filter.
        primes_to_use = [p for p in ALL_PRIMES if p <= mid]
        
        if not primes_to_use:
            survivors = N + 1
            final_res = []
        else:
            # Greedy init
            init_res = greedy_init(primes_to_use)
            
            # Local search
            survivors, final_res = local_search(primes_to_use, init_res, TIME_PER_RUN)
            
        if survivors >= mid:
            best_m = mid
            best_residues_final = final_res
            low = mid + 1
        else:
            high = mid - 1

    # Construct the final set
    if best_m == 0:
        return []
        
    primes_to_use = [p for p in ALL_PRIMES if p <= best_m]
    
    if not primes_to_use:
        return list(range(best_m))
        
    if not best_residues_final:
        best_residues_final = greedy_init(primes_to_use)
        
    # Generate valid indices
    valid = np.ones(N + 1, dtype=bool)
    for p, r in zip(primes_to_use, best_residues_final):
        for idx in residue_indices[p][r]:
            valid[idx] = False
            
    result_indices = np.where(valid)[0]
    
    if len(result_indices) < best_m:
        # Fallback: return what we have
        return result_indices.tolist()
        
    final_set = result_indices[:best_m].tolist()
    
    return final_set

# EVOLVE-BLOCK-END
