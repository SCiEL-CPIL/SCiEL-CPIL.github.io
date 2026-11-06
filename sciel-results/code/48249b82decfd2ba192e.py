"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import math

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = False
    return np.where(sieve)[0]

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # 1. Precompute primes up to N
    primes = _primes(N)
    primes_list = primes.tolist()
    
    # We only care about primes p <= N+1 potentially, but effectively p <= current_size.
    # In SA, we will track the size.
    
    # 2. Simulated Annealing Setup
    # State: boolean mask of size N+1
    # Objective: Maximize size, subject to admissibility.
    # Penalty formulation:
    # For each prime p <= |S|, let C_p be the number of residues covered.
    # If C_p == p, we have a violation.
    # Penalty = sum over p <= |S| of max(0, C_p - p + 1) * weight
    # Actually, simpler: If C_p == p, penalty += 1.
    # To make it smooth for SA (though we use discrete moves), we can just count violations.
    # Score = Size - Lambda * Violations
    
    # Initialize with a random valid-ish set or full set?
    # Full set has many violations.
    # Let's start with a set that is already admissible but maybe small, 
    # or start with full set and let SA remove/swap to optimize.
    # Starting with full set is risky as penalty is huge.
    # Let's start with the result of a fast greedy removal (like the seed) to have a valid start.
    
    # Fast Greedy Init (Deterministic)
    mask_init = np.ones(N + 1, dtype=bool)
    current_size_init = N + 1
    for p in primes_list:
        if p > current_size_init:
            break
        active_points = np.where(mask_init)[0]
        residues = active_points % p
        counts = np.bincount(residues, minlength=p)
        if np.all(counts > 0):
            min_residue = np.argmin(counts)
            to_remove = (np.arange(N + 1) % p == min_residue) & mask_init
            mask_init[to_remove] = False
            current_size_init -= np.sum(to_remove)
            
    best_mask = mask_init.copy()
    best_size = current_size_init
    best_score = best_size  # Valid, so penalty 0
    
    # Precompute prime data for fast evaluation
    # We need to evaluate the state quickly.
    # State: mask (bool array)
    # To evaluate penalty:
    # 1. Get active points
    # 2. For each p <= len(active), check if all residues covered.
    
    # Optimization:
    # Instead of iterating all primes every time, we can maintain counts.
    # But SA moves are single bit flips (add/remove one point).
    # If we remove point x:
    #   For each p, if x was the only point in residue r, count[r] drops to 0.
    #   If count[r] was 1 and becomes 0, and all other counts were >0, then C_p decreases by 1.
    #   If C_p becomes p-1, violation might be resolved.
    # If we add point x:
    #   For each p, if residue r was empty, count[r] becomes 1.
    #   C_p increases by 1.
    #   If C_p becomes p, violation is created.
    
    # This incremental update is O(pi(N)) per move.
    # pi(5000) ~ 670.
    # 670 ops per move is fast.
    
    # Let's implement the state with precomputed counts.
    
    def get_initial_counts(mask):
        counts = {}
        active = np.where(mask)[0]
        size = len(active)
        for p in primes_list:
            if p > size:
                break
            if len(active) == 0:
                counts[p] = np.zeros(p, dtype=int)
            else:
                residues = active % p
                counts[p] = np.bincount(residues, minlength=p)
        return counts, size

    # Initial state for SA
    mask = best_mask.copy()
    counts, current_size = get_initial_counts(mask)
    
    # Precompute list of primes relevant to potential max size (N+1)
    # We need counts for p up to N+1? 
    # If size grows, we need to check larger primes.
    # Let's precompute counts for all primes up to N+1 initially? 
    # No, only those <= current_size matter for penalty.
    # But if we add points, size grows.
    # Let's maintain counts for all primes in primes_list.
    # Memory: 670 primes * avg p size. Sum of p for p<=5000 is approx N^2/(2logN) ~ 5000^2/20 ~ 1.2M ints. 
    # This is fine.
    
    # Re-initialize counts for ALL primes in primes_list
    counts_all = {}
    active = np.where(mask)[0]
    for p in primes_list:
        if len(active) == 0:
            counts_all[p] = np.zeros(p, dtype=int)
        else:
            residues = active % p
            counts_all[p] = np.bincount(residues, minlength=p)
    
    # Helper to calculate penalty
    def calc_penalty(size, counts):
        pen = 0
        for p in primes_list:
            if p > size:
                break
            c = counts[p]
            covered = np.sum(c > 0)
            if covered == p:
                pen += 1
        return pen

    # Helper to calculate score
    def calc_score(size, counts):
        pen = calc_penalty(size, counts)
        return size - pen * 100  # Heavy penalty to prioritize validity
    
    # SA Parameters
    T_start = 10.0
    T_end = 0.01
    num_iterations = 50000  # Adjust based on time
    
    # Precompute points for fast access
    points = np.arange(N + 1)
    
    current_score = calc_score(current_size, counts_all)
    
    # List of primes to check in penalty loop
    # We can optimize the penalty calc by only checking primes where covered == p.
    # But checking all is O(670) which is fine.
    
    for i in range(num_iterations):
        if time.time() > deadline:
            break
            
        # Cooling schedule
        # Linear or Exponential? Exponential is standard.
        # T = T_start * (T_end/T_start)^(i/num_iterations)
        T = T_start * (T_end / T_start) ** (i / num_iterations)
        
        # Choose a random move: flip a bit
        # To make moves more effective, we can bias towards adding/removing points
        # that are "borderline". But random is simple.
        
        # Pick random index
        idx = np.random.randint(0, N + 1)
        
        if mask[idx]:
            # Remove point idx
            new_size = current_size - 1
            # Update counts
            old_counts = counts_all
            # We need to update counts for all primes
            # For each p, count[idx % p] -= 1
            
            # Check if move is beneficial or accept with prob exp(-delta/T)
            # Calculate new penalty
            # Instead of recalculating full penalty, we can track which primes change status.
            # But full recalc is O(670). 50000 * 670 = 33M ops. 
            # Python loop over 670 primes is slow. 
            # We need vectorized penalty or faster update.
            
            # Let's try to optimize the penalty calculation.
            # We can maintain a count of "violated primes".
            # violated_primes = sum(1 for p if covered_p == p)
            
            # Let's maintain `covered_counts` array for each prime?
            # No, just maintain `is_violated` boolean for each prime.
            
            # Let's restructure to maintain state incrementally.
            
            # State:
            # mask
            # current_size
            # counts_all: dict p -> array of counts
            # covered_counts: dict p -> int (number of non-zero residues)
            # violated: int (number of primes p <= size where covered_counts[p] == p)
            
            # Let's initialize this structure properly.
            
            pass # Placeholder, will implement in the loop below properly
            
    # The above loop structure is getting complex to implement inline without testing.
    # Let's write a cleaner, optimized SA loop.
    
    # Reset for actual SA run
    mask = best_mask.copy()
    current_size = best_size
    
    # Initialize incremental state
    counts_all = {}
    covered_counts = {}
    violated = 0
    
    active = np.where(mask)[0]
    for p in primes_list:
        if len(active) == 0:
            counts_all[p] = np.zeros(p, dtype=int)
            covered_counts[p] = 0
        else:
            residues = active % p
            c = np.bincount(residues, minlength=p)
            counts_all[p] = c
            covered_counts[p] = np.sum(c > 0)
            
    # Calculate initial violated
    for p in primes_list:
        if p > current_size:
            break
        if covered_counts[p] == p:
            violated += 1
            
    current_penalty = violated
    current_score = current_size - current_penalty * 100
    best_score_sa = current_score
    best_mask_sa = mask.copy()
    
    T_start = 5.0
    T_end = 0.001
    # Estimate max iterations possible
    # Each iter: 
    # - Pick idx (fast)
    # - Update counts for ~670 primes (Python loop, ~1-2us per prime? 1ms total?)
    # - Check acceptance
    # 1ms * 100000 = 100s. Too slow?
    # Let's try to vectorize the update? Hard because each prime has different modulus.
    # We can precompute residues for all points for all primes?
    # Residues[p, x] = x % p.
    # Memory: 670 * 5000 = 3.3M ints. 13MB. OK.
    
    residues_table = np.zeros((len(primes_list), N + 1), dtype=np.uint16)
    for i, p in enumerate(primes_list):
        residues_table[i] = points % p
        
    # Now update is:
    # For each prime i (where p_i <= max_size_relevant):
    #   r = residues_table[i, idx]
    #   old_count = counts_all[p_i][r]
    #   new_count = old_count +/- 1
    #   counts_all[p_i][r] = new_count
    #   if old_count == 0 and new_count == 1: covered_counts[p_i] += 1
    #   if old_count == 1 and new_count == 0: covered_counts[p_i] -= 1
    #   if p_i <= new_size:
    #      if covered_counts[p_i] == p_i: violated += 1 (if was not violated)
    #      if covered_counts[p_i] == p_i - 1: violated -= 1 (if was violated)
    #   # Also need to handle size change affecting which primes are "active"
    #   # If size increases, a new prime p_new might become active.
    #   # If size decreases, a prime p_old might become inactive.
    
    # This is still a Python loop over 670 primes.
    # 670 iterations in Python is ~0.1ms - 0.5ms.
    # 10,000 iterations -> 1-5 seconds.
    # We can do ~100,000 iterations in 300s? Maybe.
    
    num_iters = 100000
    for i in range(num_iters):
        if time.time() > deadline:
            break
            
        T = T_start * (T_end / T_start) ** (i / num_iters)
        
        idx = np.random.randint(0, N + 1)
        
        if mask[idx]:
            # Remove
            new_size = current_size - 1
            delta_penalty = 0
            delta_size = -1
            
            # Update state
            # We need to track changes to 'violated'
            # And handle the activation/deactivation of primes due to size change.
            
            # 1. Deactivate primes p where current_size < p <= new_size? No, p <= current_size are active.
            # Active primes are p <= current_size.
            # If we remove, new_size = current_size - 1.
            # Primes p such that new_size < p <= current_size become inactive.
            # Only p = current_size can be a prime in this range (since primes are sparse).
            # If current_size is prime, it was active, now inactive.
            
            p_deact = current_size
            if p_deact in covered_counts: # Check if prime
                # It was active. If it was violated, it no longer contributes.
                if covered_counts[p_deact] == p_deact:
                    delta_penalty -= 1 # Penalty decreases by 1 (violation removed from active set)
                # Note: We don't change counts, just ignore it.
            
            # 2. Update counts for active primes (p <= new_size)
            # Also primes that might become violated/unviolated.
            # We iterate over all primes p <= new_size.
            # Actually, we should iterate over all primes in primes_list that are <= max(current_size, new_size)
            # to handle the status change.
            
            # Let's just iterate over all primes in primes_list.
            # If p > current_size and p > new_size: ignore.
            
            for p in primes_list:
                if p > current_size and p > new_size:
                    continue # Not active in either state
                
                r = residues_table[primes_list.index(p), idx] # This index lookup is slow!
                
            # Optimization: Precompute index mapping or store residues in a list aligned with primes_list.
            # residues_list[i] corresponds to primes_list[i]
            
            # Let's restructure the loop to use indices.
            
            # Re-do the inner loop efficiently
            # We'll just accept the overhead of the loop for now, but optimize the index access.
            
            # Actually, let's just break out if this is too slow and fall back to a simpler heuristic.
            # Given the time constraint and complexity, a simpler "Local Search" with restarts might be safer.
            # The SA implementation above is prone to bugs in the incremental penalty tracking.
            
            # Let's switch strategy to **Greedy Local Search with Multiple Restarts and Randomized Swaps**.
            # 1. Start with a valid set (from greedy).
            # 2. Try to add a point. If invalid, try to remove a different point to make it valid?
            #    Or: Try to swap (add x, remove y) if it improves size or maintains size with better structure?
            #    Actually, size is the only metric.
            #    So we want to ADD points.
            #    If adding x makes it invalid, we must remove some y to fix the violation.
            #    If we can find y such that adding x and removing y results in a valid set, size stays same.
            #    This doesn't help size.
            #    We need to find a sequence of additions that are valid.
            
            # The "Expansion" in the seed was trying to add points that were "safe".
            # The issue is that "safe" points might not exist if we removed too much.
            
            # New Idea: **Iterative Deepening A* like search for the "best" residue to remove**.
            # Instead of removing min count, we look ahead.
            # But that's expensive.
            
            # Let's stick to the **Simulated Annealing** but implement it correctly and efficiently.
            # The key is to make the inner loop fast.
            # Python loop over 670 primes is the bottleneck.
            # Can we vectorize the penalty update?
            # No, because the condition `covered_counts[p] == p` depends on the specific count.
            
            # Alternative: Use a "Penalty Score" that is smooth.
            # Score = Size - Lambda * Sum( max(0, Covered_p - p + 1) )
            # This is still hard to update incrementally without loop.
            
            # Let's try a different approach: **Greedy with Backtracking**.
            # 1. Process primes in increasing order.
            # 2. At each prime p, if all residues covered, we must remove one.
            # 3. Instead of removing min count, we try removing each residue (or a sample) and see which leads to a larger final set.
            # 4. This is a tree search. Branching factor is p (up to 5000). Depth is ~670.
            # 5. Too big.
            
            # However, we only need to make a choice for small primes.
            # For large primes p, the set size is likely < p, so no removal needed.
            # Wait, if p <= |S|, we MUST remove.
            # |S| starts at 5000.
            # Primes up to 5000.
            # Most primes are large.
            # For p > 2500, |S| will likely be < p after removing for small primes?
            # If we remove ~1/2 for p=2, ~1/3 for p=3...
            # Size drops fast.
            # After p=2, size ~2500.
            # After p=3, size ~1600.
            # After p=5, size ~1000.
            # After p=7, size ~700.
            # After p=11, size ~500.
            # After p=13, size ~380.
            # ...
            # Size drops below p quickly.
            # So only small primes require removal!
            # Let's check:
            # Start 5000.
            # p=2: remove 2500. Size 2500.
            # p=3: remove 833. Size 1667.
            # p=5: remove 333. Size 1334.
            # p=7: remove 190. Size 1144.
            # p=11: remove 104. Size 1040.
            # p=13: remove 80. Size 960.
            # p=17: remove 56. Size 904.
            # p=19: remove 47. Size 857.
            # p=23: remove 37. Size 820.
            # p=29: remove 28. Size 792.
            # p=31: remove 25. Size 767.
            # p=37: remove 20. Size 747.
            # p=41: remove 18. Size 729.
            # p=43: remove 17. Size 712.
            # p=47: remove 15. Size 697.
            # p=53: remove 13. Size 684.
            # p=59: remove 11. Size 673.
            # p=61: remove 11. Size 662.
            # p=67: remove 9. Size 653.
            # p=71: remove 9. Size 644.
            # p=73: remove 8. Size 636.
            # p=79: remove 8. Size 628.
            # p=83: remove 7. Size 621.
            # p=89: remove 7. Size 614.
            # p=97: remove 6. Size 608.
            # p=101: remove 6. Size 602.
            # ...
            # Size is still > p for a long time.
            # When does Size < p?
            # If we remove 1/p of current size, size multiplies by (1-1/p).
            # Product over primes p <= P of (1-1/p) ~ 1/(e^gamma ln P).
            # Size(P) ~ 5000 / (ln P).
            # We need 5000 / ln P < P.
            # P ln P > 5000.
            # If P=100, 100*4.6 = 460. No.
            # If P=1000, 1000*6.9 = 6900. Yes.
            # So for P > ~800, Size < P.
            # So we only need to make removal decisions for primes p < ~800.
            # Number of primes < 800 is ~139.
            # Branching factor is p.
            # Tree search is still huge.
            
            # BUT, for large p (e.g. p=700), the removal cost is small (1/700 of size).
            # The choice of which residue to remove matters less?
            # No, it affects the distribution for subsequent primes.
            
            # However, the number of "critical" primes is small.
            # For p > 100, the impact of the choice is small.
            # So we can use a **Greedy** approach for p > 50, and **Beam Search** for p <= 50.
            
            # Let's implement Beam Search for the first few primes.
            
            pass # End of analysis
            
    # Given the complexity and time, I will implement the **Beam Search** strategy.
    # It is deterministic, fast, and explores multiple paths.
    
    def beam_search_init():
        # State: (mask, size)
        # We will keep a list of states (beam).
        # For each state, for each prime p, if violation, generate new states by removing each residue.
        # Keep top K states by size.
        
        beam = [(np.ones(N + 1, dtype=bool), N + 1)]
        beam_width = 100
        
        # We only need to process primes up to a limit where size < p.
        # Let's process primes in order.
        
        for p in primes_list:
            if time.time() > deadline:
                break
                
            new_beam = []
            
            for mask, size in beam:
                if p > size:
                    # No violation possible, keep state
                    new_beam.append((mask, size))
                    continue
                
                active_points = np.where(mask)[0]
                residues = active_points % p
                counts = np.bincount(residues, minlength=p)
                
                if not np.all(counts > 0):
                    # Already valid for this p
                    new_beam.append((mask, size))
                    continue
                
                # Violation: must remove one residue.
                # Generate candidates for each residue r
                # To limit explosion, we can restrict to top few residues by count?
                # No, removing min count is best locally.
                # But maybe removing a max count is better globally? Unlikely.
                # Let's generate states for ALL residues?
                # p can be up to 5000. 100 states * 5000 = 500,000 states. Too many.
                
                # Restrict to residues with lowest counts?
                # Or just pick the best one (Greedy) and add some randomness?
                # Beam search is best if we keep top K.
                # If we generate p children, we keep top K.
                # For small p (2, 3, 5), p is small.
                # For large p, we can just do Greedy (1 child) or top 5 children.
                
                if p <= 10:
                    candidates = range(p)
                else:
                    # Pick top 5 residues with lowest counts
                    # Or just the min?
                    # Let's pick top 3 to balance.
                    sorted_residues = np.argsort(counts)
                    candidates = sorted_residues[:3]
                    
                for r in candidates:
                    new_mask = mask.copy()
                    to_remove = (np.arange(N + 1) % p == r) & new_mask
                    new_mask[to_remove] = False
                    new_size = size - np.sum(to_remove)
                    new_beam.append((new_mask, new_size))
                    
            # Sort new_beam by size (descending) and keep top beam_width
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:beam_width]
            
            # Optional: Stop if all states in beam have size < next_prime
            # Or if beam converges?
            
        # Return the best state from the final beam
        if beam:
            best_mask, best_size = beam[0]
            return best_mask, best_size
        else:
            return np.ones(N + 1, dtype=bool), N + 1

    best_mask, best_size = beam_search_init()
    
    # Expansion phase on the best result
    # Similar to seed, try to add back points.
    
    mask = best_mask
    current_size = best_size
    
    # Precompute empty residues for relevant primes
    relevant_primes = [p for p in primes_list if p <= current_size + 1]
    
    empty_flags = {}
    empty_counts = {}
    
    active_points = np.where(mask)[0]
    for p in relevant_primes:
        if len(active_points) == 0:
            empty_flags[p] = np.ones(p, dtype=bool)
            empty_counts[p] = p
        else:
            residues = active_points % p
            present = np.zeros(p, dtype=bool)
            present[residues] = True
            empty_flags[p] = ~present
            empty_counts[p] = np.sum(empty_flags[p])
            
    candidates = np.where(~mask)[0]
    
    # Score candidates by safety (number of primes where residue is present)
    scores = np.zeros(len(candidates))
    for i, x in enumerate(candidates):
        score = 0
        for p in relevant_primes:
            if p > current_size + 1:
                break
            r = x % p
            if not empty_flags[p][r]:
                score += 1
        scores[i] = score
        
    order = np.argsort(-scores)
    
    for idx in order:
        if time.time() > deadline:
            break
            
        x = candidates[idx]
        new_size = current_size + 1
        
        safe = True
        for p in relevant_primes:
            if p > new_size:
                break
            r = x % p
            if empty_flags[p][r]:
                if empty_counts[p] == 1:
                    safe = False
                    break
                    
        if safe:
            mask[x] = True
            current_size += 1
            
            for p in relevant_primes:
                if p > current_size:
                    break
                r = x % p
                if empty_flags[p][r]:
                    empty_flags[p][r] = False
                    empty_counts[p] -= 1
                    
    result = np.where(mask)[0].tolist()
    return result
# EVOLVE-BLOCK-END
