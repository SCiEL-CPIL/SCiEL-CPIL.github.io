import numpy as np
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Generate Primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    n_primes = len(primes)
    primes_np = np.array(primes)
    
    # We check up to 2000 to give some room, but the score is capped by the first failure.
    # The task says "largest L such that every integer 1..L satisfies at least two".
    # If we reach 1000, that's the target. Let's check up to 2000 to be safe, 
    # but the score calculation will stop at the first failure.
    MAX_CHECK = 2000
    x_values = np.arange(1, MAX_CHECK + 1)
    
    # Precompute x mod p for all x and p.
    # Shape: (MAX_CHECK, n_primes)
    x_mod_p = x_values[:, None] % primes_np[None, :]
    
    # Helper: Compute the current score (L)
    # Returns the largest L such that counts[x] >= 2 for all x in 1..L
    def compute_score(residues):
        match = (x_mod_p == residues[None, :])
        counts = match.sum(axis=1)
        invalid = counts < 2
        if not np.any(invalid):
            return MAX_CHECK
        else:
            first_invalid_idx = np.argmax(invalid)
            return int(first_invalid_idx)
    
    # Helper: Batch compute scores for a list of residues (list of arrays)
    # This is slower but allows parallel evaluation of multiple candidates if needed,
    # but for SA, we usually evaluate one move at a time. 
    # However, we can vectorize the "try move" logic.
    
    # Strategy: Multi-start Simulated Annealing
    # We will run multiple independent SA chains in parallel (conceptually, 
    # but sequentially in Python unless we use threads, which is hard with GIL 
    # and numpy overhead. Instead, we'll do a fast sequential SA with aggressive 
    # restarts or a "island" model where we keep a few best states and mutate them).
    
    # Actually, a better approach for speed: 
    # 1. Initialize with random residues.
    # 2. Use a "Best Move" heuristic within SA: instead of random single moves, 
    #    evaluate all possible single-residue changes and pick the best one 
    #    that improves score or is accepted by SA.
    #    This is expensive: 168 primes * (p-1) residues. 
    #    Sum(p-1) for p<=1000 is approx 1000 * 168 / 2? No, sum of primes ~ 30000.
    #    So ~30000 possible moves. 
    #    Evaluating each move naively is O(MAX_CHECK * 1) if we precompute deltas?
    
    # Let's optimize the move evaluation.
    # Current counts: C[x]
    # If we change residue of prime i from r_old to r_new:
    # - For all x where x % p_i == r_old, C[x] decreases by 1.
    # - For all x where x % p_i == r_new, C[x] increases by 1.
    # We can precompute for each prime i and residue r, the boolean mask of x's.
    # masks[i, r] is a boolean array of shape (MAX_CHECK,)
    # This requires 168 * 1000 * 2000 booleans = 336M booleans = 336 MB. 
    # This is too much memory.
    
    # Alternative:
    # We stick to the "Frontier" idea but make it stochastic and parallelized 
    # via "Islands" (multiple independent searches).
    
    # Let's try a different tactic: 
    # The current program gets 3.36. This is likely L ~ 336? 
    # Wait, score is value/seed. If seed is 1000, and score is 3.36, L is ~336?
    # Or is seed 1? "1.0 equals the seed program". 
    # The task says "Maximize; L >= 1000 would realize...".
    # "The evaluator measures the raw value (higher is better). The search score is value/seed".
    # If the seed program produces L=1000, score is 1.0. 
    # If current program score is 3.36, does it produce L=3360? 
    # Or is the "seed" a specific baseline L?
    # "1.0 equals the seed program". 
    # Let's assume the seed program produces some L_seed.
    # If score is 3.36, and it's L/L_seed, then L is 3.36 * L_seed.
    # If L_seed is small, say 100, then L is 336.
    # If L_seed is 1000, then L is 3360.
    # Given the difficulty, L=3360 is very high. L=336 is more plausible for a simple greedy.
    # Let's assume the goal is to maximize L.
    
    # New Strategy: 
    # 1. Start with a good initial guess (e.g., r_p = 1 for all p, or random).
    # 2. Use a "Hill Climbing" approach that evaluates the impact of changing 
    #    ONE residue at a time, but does so efficiently.
    # 3. To avoid getting stuck, use "Simulated Annealing" logic: 
    #    Accept worse moves with probability exp(-delta/T).
    # 4. To speed up, we don't evaluate all 30000 moves. We evaluate a random 
    #    subset of moves (e.g., 100 random prime-residue pairs) per step.
    
    # Let's implement a fast SA.
    
    def get_initial_residues():
        # Try a few heuristics for initialization
        # 1. All 1s
        # 2. Random
        # 3. r_p = p // 2
        return np.random.randint(0, primes_np)
    
    best_residues = None
    best_score = -1
    
    num_islands = 5
    island_residues = [get_initial_residues() for _ in range(num_islands)]
    island_scores = [compute_score(r) for r in island_residues]
    
    for i in range(num_islands):
        if island_scores[i] > best_score:
            best_score = island_scores[i]
            best_residues = island_residues[i].copy()
    
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    
    while time.time() < deadline - 1.0:
        # Pick a random island to mutate
        island_idx = np.random.randint(0, num_islands)
        curr_res = island_residues[island_idx]
        curr_score = island_scores[island_idx]
        
        # Generate a random move: pick a random prime index and a new residue
        p_idx = np.random.randint(0, n_primes)
        old_r = curr_res[p_idx]
        new_r = np.random.randint(0, primes[p_idx])
        
        if new_r == old_r:
            continue
            
        curr_res[p_idx] = new_r
        new_score = compute_score(curr_res)
        
        # Acceptance criterion
        if new_score > curr_score:
            island_scores[island_idx] = new_score
            if new_score > best_score:
                best_score = new_score
                best_residues = curr_res.copy()
        else:
            # Metropolis criterion
            # Delta is negative. We want to accept if exp(-delta/T) > random
            # delta = new_score - curr_score
            delta = new_score - curr_score
            if delta < 0:
                # Calculate acceptance probability
                # To avoid underflow, if delta is very large negative, prob is 0
                if T > 0:
                    # exp(delta / T)
                    # Note: delta is negative, so delta/T is negative.
                    # We want exp(-|delta|/T)
                    prob = np.exp(delta / T)
                    if np.random.random() < prob:
                        island_scores[island_idx] = new_score
                    else:
                        # Reject move
                        curr_res[p_idx] = old_r
            # If delta == 0, always accept (or reject, doesn't matter much, but accept to explore)
            else:
                island_scores[island_idx] = new_score
        
        # Cool down
        T = max(T * cooling, T_min)
        
        # Occasionally, restart an island if it's stuck (low score) and time is available
        # This helps escape local minima.
        if curr_score < best_score * 0.8 and np.random.random() < 0.05:
            island_residues[island_idx] = get_initial_residues()
            island_scores[island_idx] = compute_score(island_residues[island_idx])
            if island_scores[island_idx] > best_score:
                best_score = island_scores[island_idx]
                best_residues = island_residues[island_idx].copy()
                
    return [int(r) for r in best_residues]

# Ensure numpy random seed is set for reproducibility if needed
np.random.seed(42)

if __name__ == "__main__":
    res = construct()
    print(len(res))
