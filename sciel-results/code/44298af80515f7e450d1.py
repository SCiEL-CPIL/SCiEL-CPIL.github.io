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
    
    # The range of integers to check.
    # We start with a generous range to allow the greedy algorithm to push the frontier.
    # If the greedy algorithm gets stuck early, we might not reach 1000.
    # We check up to 2000 to give some room.
    MAX_CHECK = 2000
    x_values = np.arange(1, MAX_CHECK + 1)
    
    # Precompute x mod p for all x and p.
    # Shape: (MAX_CHECK, n_primes)
    x_mod_p = x_values[:, None] % primes_np[None, :]
    
    # Initial Solution
    # A common strong start for covering systems is to align residues to cover small numbers.
    # Let's start with r_p = 1 for all primes, which covers x=1, p+1, etc.
    # Or better, initialize to 0? 
    # If r_p = 0, x is covered if x is multiple of p.
    # If r_p = 1, x is covered if x = 1 mod p.
    # Let's try r_p = 1.
    current_residues = np.ones(n_primes, dtype=np.int64)
    
    # Helper: Compute the current score (L)
    def compute_score(residues):
        match = (x_mod_p == residues[None, :])
        counts = match.sum(axis=1)
        invalid = counts < 2
        if not np.any(invalid):
            return MAX_CHECK
        else:
            first_invalid_idx = np.argmax(invalid)
            return int(first_invalid_idx)
    
    current_score = compute_score(current_residues)
    best_residues = current_residues.copy()
    best_score = current_score
    
    # If we already have a high score, we might just refine.
    # But usually, greedy starts low and climbs.
    
    # Greedy Frontier Extension Strategy
    # We look at the first gap (current_score + 1).
    # We want to assign residues to cover this gap AND future gaps.
    # Instead of fixing one prime at a time globally, we focus on the "active" frontier.
    
    # A more effective greedy approach for this specific "doubly covered prefix" problem:
    # Iterate x from 1 to MAX_CHECK.
    # For each x, check if it is doubly covered by the current residues.
    # If not, we need to "fix" it.
    # However, changing a residue affects future x's.
    
    # Let's use a "Repair" strategy:
    # 1. Start with random or heuristic residues.
    # 2. Find the first x that is not doubly covered.
    # 3. Identify the "deficit" at x. It needs at least 2 matches.
    # 4. Choose a prime p that currently does NOT match x (so x % p != r_p) and change r_p to x % p.
    #    To choose which prime, we can pick one that maximizes the "potential" coverage for future numbers.
    #    A simple heuristic: pick the prime p that covers the most other numbers in the vicinity [x, x+K] that are currently under-covered.
    
    # To make this efficient, we can precompute for each prime p, the list of x's it covers for a given residue.
    # But residues change.
    
    # Let's try a "Local Search" focused on the gap.
    # This is different from SA which explores globally. This is a directed hill-climbing on the gap.
    
    # Even better: **Greedy Construction from Scratch**
    # We will assign residues one by one? No, the order matters.
    
    # Let's try a **Constraint Satisfaction** style greedy:
    # We maintain the current best prefix L.
    # While L < MAX_CHECK and time < deadline:
    #   Let x = L + 1 be the first uncovered point.
    #   We need to increase the count of matches at x to >= 2.
    #   Current count at x is C_x.
    #   We need to change residues of primes that do NOT currently match x.
    #   Changing r_p to x % p will increase the match count at x by 1.
    #   But it might break matches at other points.
    #   
    #   Strategy: 
    #   1. Identify all primes p where x % p != r_p.
    #   2. For each such p, simulate changing r_p to x % p.
    #   3. Calculate the new score.
    #   4. Pick the p that yields the highest new score.
    #   5. If the best new score > current score, accept.
    #   6. If no single change improves the score, try a pair of changes? (Expensive)
    #   7. If stuck, maybe randomize some residues to escape local optimum?
    
    # This "Best Single Move to Fix Gap" is a strong heuristic.
    
    # Optimization: 
    # Instead of recalculating the full score for every candidate p, we can estimate.
    # But MAX_CHECK is small (2000), so full recalculation is fast (2000 * 168 ops ~ 300k ops).
    # 1000 such evaluations per step is 300M ops. In Python/Numpy this might be slow if done in a loop.
    # Vectorization: We can evaluate all candidate moves at once?
    # 
    # Let's implement the "Best Single Move" loop with Numpy vectorization for the candidate evaluation.
    
    # Precompute x_mod_p again? We have it.
    
    # State: current_residues
    # Current Score: L
    
    # Loop
    while time.time() < deadline - 1.0:
        if current_score >= MAX_CHECK:
            break
            
        gap_x = current_score + 1
        gap_idx = current_score  # index in x_values (0-based, x_values[0]=1)
        
        # Find primes that do NOT currently match gap_x
        # match_at_gap = (x_mod_p[gap_idx] == current_residues)
        # candidates = np.where(~match_at_gap)[0]
        
        match_at_gap = (x_mod_p[gap_idx] == current_residues)
        candidates = np.where(~match_at_gap)[0]
        
        if len(candidates) == 0:
            # All primes already match gap_x. But it's not doubly covered?
            # This implies count < 2, but all primes match? Impossible if n_primes >= 2.
            # Unless n_primes < 2. But we have 168.
            # So this case shouldn't happen if count < 2.
            # If count >= 2, gap_x would be covered, contradiction.
            # So if we are here, something is wrong or we hit MAX_CHECK logic issue.
            break
        
        # For each candidate prime index p_idx in candidates:
        #   new_residues = current_residues.copy()
        #   new_residues[p_idx] = gap_x % primes[p_idx]
        #   score = compute_score(new_residues)
        
        # Vectorized evaluation of candidates:
        # We want to compute the score for each candidate move.
        # Score is the first index where count < 2.
        
        # Let's construct a matrix of new residues for all candidates.
        # Shape: (num_candidates, n_primes)
        # This might be memory heavy if num_candidates is large (up to 168).
        # 168 * 168 = ~28k ints. Small.
        
        num_cands = len(candidates)
        
        # Create a copy of residues for each candidate
        # cand_residues[i, j] = current_residues[j]
        # cand_residues[i, candidates[i]] = gap_x % primes[candidates[i]]
        
        # Efficiently:
        # Start with a broadcast of current_residues
        # cand_residues = np.tile(current_residues, (num_cands, 1))
        # Then update the specific column for each row.
        
        # Actually, we can compute the new match counts directly.
        # For a candidate move at prime p_idx:
        # The match matrix changes only in column p_idx.
        # old_matches = (x_mod_p == current_residues[None, :])
        # new_matches = old_matches.copy()
        # new_matches[:, p_idx] = (x_mod_p[:, p_idx] == new_r)
        # new_counts = new_matches.sum(axis=1)
        
        # We can vectorize the "new_counts" calculation for all candidates at once.
        # 
        # Let M be the old match matrix (MAX_CHECK, n_primes).
        # Let C be the candidate indices (num_cands,).
        # Let R be the new residues for these candidates (num_cands,).
        # 
        # For each candidate i:
        #   col = C[i]
        #   r = R[i]
        #   new_col_match = (x_mod_p[:, col] == r)
        #   delta_count = new_col_match - M[:, col]  # (MAX_CHECK,)
        #   old_counts = M.sum(axis=1)  # (MAX_CHECK,)
        #   new_counts = old_counts + delta_count
        #   score_i = first index where new_counts < 2
        
        # We can compute `delta_count` for all candidates in a batch.
        # delta_counts[i, x_idx] = (x_mod_p[x_idx, C[i]] == R[i]) - M[x_idx, C[i]]
        
        # x_mod_p[x_idx, C[i]] is a bit tricky to vectorize directly without gather.
        # x_mod_p is (MAX_CHECK, n_primes).
        # We want x_mod_p[:, C] which is (MAX_CHECK, num_cands).
        # x_mod_p_cands = x_mod_p[:, C]
        
        # R is (num_cands,). We need to broadcast R to (1, num_cands) to compare with x_mod_p_cands.
        # new_matches_cands = (x_mod_p_cands == R[None, :])  # (MAX_CHECK, num_cands)
        
        # M_cands = M[:, C]  # (MAX_CHECK, num_cands)
        
        # delta_counts = new_matches_cands.astype(int) - M_cands.astype(int)  # (MAX_CHECK, num_cands)
        
        # old_counts = M.sum(axis=1)  # (MAX_CHECK,)
        # new_counts = old_counts[:, None] + delta_counts  # (MAX_CHECK, num_cands)
        
        # Now find the first index where new_counts < 2 for each column.
        # invalid = new_counts < 2  # (MAX_CHECK, num_cands)
        # first_invalid = np.argmax(invalid, axis=0)  # (num_cands,)
        # If no invalid, argmax returns 0. We need to handle the "all valid" case.
        # all_valid = ~invalid.any(axis=0)
        # scores = np.where(all_valid, MAX_CHECK, first_invalid)
        
        # This is a very efficient vectorized evaluation of all single-move candidates!
        
        # Let's implement this.
        
        # 1. Get current match matrix M
        M = (x_mod_p == current_residues[None, :])
        old_counts = M.sum(axis=1)
        
        # 2. Prepare candidate data
        C = candidates
        R = x_mod_p[gap_idx, C]  # The new residue for each candidate prime is gap_x % p
        
        # 3. Compute delta counts
        # x_mod_p_cands = x_mod_p[:, C]
        # This is a fancy indexing operation. (MAX_CHECK, num_cands)
        x_mod_p_cands = x_mod_p[:, C]
        
        # new_matches_cands: True if x_mod_p[x, p_candidate] == R_candidate
        # R is (num_cands,), broadcast to (1, num_cands)
        new_matches_cands = (x_mod_p_cands == R[None, :])
        
        # M_cands: The old match status for these columns
        M_cands = M[:, C]
        
        # delta_counts: (MAX_CHECK, num_cands)
        # Note: new_matches_cands and M_cands are bool. Cast to int for subtraction.
        delta_counts = new_matches_cands.astype(np.int8) - M_cands.astype(np.int8)
        
        # new_counts: (MAX_CHECK, num_cands)
        new_counts = old_counts[:, None].astype(np.int8) + delta_counts
        
        # 4. Find scores
        invalid = new_counts < 2
        # If any row is True in a column, that column has a gap.
        # We want the FIRST row index where it is True.
        # np.argmax returns 0 if no True. 
        # We need to distinguish "no gap" (score = MAX_CHECK) from "gap at 0" (score = 0).
        
        # has_gap = invalid.any(axis=0)  # (num_cands,)
        # first_gap_idx = np.argmax(invalid, axis=0)  # (num_cands,)
        # scores = np.where(has_gap, first_gap_idx, MAX_CHECK)
        
        # Optimization: 
        # If the gap is at gap_idx (which it is, by definition, since we are trying to fix it),
        # the score will be at least gap_idx.
        # Actually, the score is the first x that fails.
        
        has_gap = invalid.any(axis=0)
        first_gap_idx = np.argmax(invalid, axis=0)
        scores = np.where(has_gap, first_gap_idx, MAX_CHECK)
        
        # 5. Select best candidate
        best_idx_in_scores = np.argmax(scores)
        best_candidate_idx = C[best_idx_in_scores]
        best_new_score = scores[best_idx_in_scores]
        
        # 6. Apply move if it improves
        if best_new_score > current_score:
            current_residues[best_candidate_idx] = R[best_idx_in_scores]
            current_score = best_new_score
            
            if current_score > best_score:
                best_score = current_score
                best_residues = current_residues.copy()
        else:
            # Stuck. No single move improves the score.
            # This can happen if fixing the gap breaks a previous gap? 
            # No, previous gaps are already covered. 
            # It can happen if the move fixes the current gap but creates a new gap earlier? 
            # No, changing a residue can only break coverage at some x. 
            # If it breaks coverage at x < gap_x, then the score would decrease.
            # If it breaks coverage at x > gap_x, the score might stay the same or increase.
            # If best_new_score == current_score, it means the gap at gap_x is fixed, 
            # but a new gap appears at gap_x (or later)? 
            # If the gap at gap_x is fixed, the first gap must be > gap_x.
            # So score should be > current_score.
            # Why would best_new_score == current_score?
            # Maybe the move doesn't actually increase the count at gap_x to >= 2?
            # We picked candidates where match_at_gap was False. 
            # So changing r_p to gap_x % p WILL make it match gap_x.
            # So count at gap_x increases by 1.
            # If old count was 1, new count is 2. Gap fixed.
            # If old count was 0, new count is 1. Gap NOT fixed.
            # 
            # Ah! We need to ensure that the move actually fixes the gap.
            # If old count at gap_x is 0, we need TWO moves to fix it.
            # If old count at gap_x is 1, one move is enough.
            
            old_count_at_gap = old_counts[gap_idx]
            
            if old_count_at_gap < 2:
                # If old count is 0, a single move makes it 1. Still < 2.
                # So score doesn't improve.
                # We need to accept moves that don't immediately improve the score 
                # but set up for a future improvement? 
                # Or, we can look for a PAIR of moves.
                
                # For simplicity, let's try to accept a move that increases the count at gap_x 
                # even if it doesn't immediately fix it, provided it doesn't break anything else?
                # This is getting complex.
                
                # Alternative: If stuck, perform a random restart or random perturbation.
                # Let's try a random perturbation: change 2 random residues to random values.
                # This might escape local minima.
                
                # But first, let's check if we can just accept a move that keeps the score same 
                # but increases coverage at the gap?
                # The "score" is just the prefix length. 
                # If we are stuck at L, it means L+1 is not doubly covered.
                # If count(L+1) == 1, we need 1 more match.
                # If count(L+1) == 0, we need 2 more matches.
                
                # If count(L+1) == 1:
                # Any single move that adds a match at L+1 should fix it, UNLESS it breaks a previous match.
                # Can it break a previous match? 
                # Yes, if the prime p was matching some x < L+1, and we change r_p, it no longer matches x.
                # If x was only covered by p and one other, and we break p, x becomes singly covered.
                # Then the new first gap is x, which is < L+1. So score decreases.
                # Our vectorized check handles this: it finds the NEW first gap.
                # If best_new_score == current_score, it means the new first gap is at the same index?
                # That's impossible if we fixed the gap at current_score+1.
                # Unless the move broke a gap at current_score? 
                # But current_score is the first gap. So x <= current_score are covered.
                # If we break a match at x < current_score, the new gap is < current_score.
                # So score would decrease.
                # If best_new_score == current_score, it implies the first gap is still at current_score?
                # That means the move did NOT fix the gap at current_score+1?
                # But we ensured the new residue matches gap_x.
                # So the count at gap_x increased.
                # If it was 1, it's now 2. Gap fixed.
                # If it was 0, it's now 1. Gap NOT fixed.
                
                # So if old_count_at_gap == 0, we are stuck with single moves.
                # We need to make two moves simultaneously?
                # Or, we can accept a move that increases the count from 0 to 1, 
                # even if the score doesn't improve, as long as it doesn't decrease.
                # Then, on the next iteration, we try to fix it again.
                
                # Let's modify the acceptance criteria:
                # Accept if best_new_score >= current_score.
                # If equal, we are making progress on the gap count (0->1 or 1->2 but broken elsewhere? No, if broken elsewhere score drops).
                # If score is equal, it means the first gap is still at the same place?
                # If old_count_at_gap == 1, and we fix it, score MUST increase (unless we break an earlier one, which decreases score).
                # If old_count_at_gap == 0, and we make it 1, score stays same (gap still there).
                # So accepting equal score is safe if it increases the count at the gap.
                
                # Let's check if the best move increased the count at the gap.
                # We can compute the new count at the gap for the best candidate.
                # new_count_at_gap = old_count_at_gap + 1 (since we picked a non-matching prime)
                # So if old_count_at_gap < 2, the count increases.
                
                if best_new_score >= current_score:
                    current_residues[best_candidate_idx] = R[best_idx_in_scores]
                    current_score = best_new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_residues = current_residues.copy()
                else:
                    # Best move decreases score.
                    # Try a random move?
                    # Pick a random prime and random residue.
                    # This is a fallback.
                    rand_idx = np.random.randint(0, n_primes)
                    old_r = current_residues[rand_idx]
                    new_r = np.random.randint(0, primes[rand_idx])
                    if new_r != old_r:
                        current_residues[rand_idx] = new_r
                        new_score = compute_score(current_residues)
                        if new_score >= current_score:
                            current_score = new_score
                            if new_score > best_score:
                                best_score = new_score
                                best_residues = current_residues.copy()
                        else:
                            # Revert if it gets worse? 
                            # Or keep it to escape?
                            # Let's keep it if it's not much worse?
                            # For now, revert to stay safe.
                            current_residues[rand_idx] = old_r

    return [int(r) for r in best_residues]

# Ensure numpy random seed is set for reproducibility if needed
np.random.seed(42)

if __name__ == "__main__":
    res = construct()
    print(len(res))
