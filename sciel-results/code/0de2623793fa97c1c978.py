"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def _compute_sizes(A):
    """Compute |A+A| and |A*A| for set A."""
    A_list = list(A)
    n = len(A_list)
    sums = set()
    prods = set()
    for i in range(n):
        for j in range(i, n):
            sums.add(A_list[i] + A_list[j])
            prods.add(A_list[i] * A_list[j])
    return len(sums), len(prods)

def _score(A):
    s, p = _compute_sizes(A)
    return max(s, p)

def _get_candidates():
    """Generate a diverse candidate pool focused on collision-prone numbers."""
    candidates = set()
    # Small integers are crucial for sumset minimization
    candidates.update(range(1, 101))
    
    # Highly composite numbers for product set minimization
    # Numbers with many divisors are more likely to produce repeated products
    def count_divisors(n):
        count = 0
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                count += 2
        if int(math.isqrt(n)) ** 2 == n:
            count -= 1
        return count

    # Add numbers with high divisor counts up to 10000
    # We prioritize smaller highly composite numbers as they also help with sums
    for x in range(1, 10001):
        if count_divisors(x) >= 8: # Threshold for "structured"
            candidates.add(x)
            
    # Ensure we have enough candidates
    if len(candidates) < 500:
        candidates.update(range(1, 500))
        
    return list(candidates)

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    candidates = _get_candidates()
    
    # Greedy construction: Select 20 numbers one by one
    # minimizing the increase in max(|A+A|, |A*A|)
    A = set()
    
    # Pre-compute initial empty state
    current_sums = set()
    current_prods = set()
    current_score = 0
    
    # To speed up greedy selection, we can just compute the full score for each candidate
    # Since n=20 is small, and candidate pool is ~500, this is feasible.
    # O(20 * 500 * 20^2) = 4M operations approx.
    
    for _ in range(20):
        best_cand = None
        best_score = float('inf')
        best_sums = None
        best_prods = None
        
        # Shuffle candidates to break ties randomly
        cand_list = candidates[:]
        random.shuffle(cand_list)
        
        for c in cand_list:
            if c in A:
                continue
            if c > 10000:
                continue
            
            # Calculate new sums and prods
            # New sums: {a + c for a in A} U {2c}
            # New prods: {a * c for a in A} U {c*c}
            
            # Optimization: We can compute the size of the union without creating full sets if we are careful,
            # but for n=20, creating sets is fast enough.
            
            new_sums = current_sums.copy()
            new_prods = current_prods.copy()
            
            for a in A:
                new_sums.add(a + c)
                new_prods.add(a * c)
            
            new_sums.add(2 * c)
            new_prods.add(c * c)
            
            s_size = len(new_sums)
            p_size = len(new_prods)
            score = max(s_size, p_size)
            
            if score < best_score:
                best_score = score
                best_cand = c
                best_sums = new_sums
                best_prods = new_prods
        
        if best_cand is not None:
            A.add(best_cand)
            current_sums = best_sums
            current_prods = best_prods
            current_score = best_score
        else:
            # Fallback if no valid candidate found (should not happen)
            for c in range(1, 10001):
                if c not in A:
                    A.add(c)
                    break
            
            if (time.time() - start_time) > time_limit:
                return sorted(A)
            
            # Recalculate score
            s, p = _compute_sizes(A)
            current_score = max(s, p)
            current_sums, current_prods = set(), set() # Reset for next iteration if we didn't track properly
            # Actually, let's just restart the tracking if we hit fallback
            A_list = list(A)
            current_sums = set()
            current_prods = set()
            for i in range(len(A_list)):
                for j in range(i, len(A_list)):
                    current_sums.add(A_list[i] + A_list[j])
                    current_prods.add(A_list[i] * A_list[j])
            
            if (time.time() - start_time) > time_limit:
                return sorted(A)

    best_A = set(A)
    best_score = current_score
    
    # Local Search: Swap optimization
    # Try to improve by swapping one element with another from the pool
    # We restrict the pool for swaps to top candidates to save time
    swap_pool = candidates[:]
    random.shuffle(swap_pool)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        A_list = list(A)
        random.shuffle(A_list)
        
        for old_val in A_list:
            if (time.time() - start_time) > time_limit:
                break
            
            # Try swapping old_val with new_val
            # We limit the number of swaps to try per iteration to avoid timeout
            for new_val in swap_pool[:100]: # Check top 100 random candidates
                if new_val in A or new_val == old_val:
                    continue
                if new_val > 10000:
                    continue
                
                # Calculate new score
                # Remove old_val contributions, add new_val contributions
                # This is complex to do incrementally with sets if we don't store pairs.
                # Given n=20, full recalculation is O(400) which is fast.
                
                new_A = A.copy()
                new_A.discard(old_val)
                new_A.add(new_val)
                
                new_score = _score(new_A)
                
                if new_score < current_score:
                    A = new_A
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_A = set(A)
                    improved = True
                    break # Restart sweep with new A
    
    return sorted(best_A)
# EVOLVE-BLOCK-END
