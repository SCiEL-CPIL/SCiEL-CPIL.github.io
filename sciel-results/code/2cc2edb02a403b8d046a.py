"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def _score_set(U):
    """Calculate the score for a given set U."""
    if not U:
        return 0.0
    U = list(set(U))
    if 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    # Compute sumset and difference set sizes
    # For small sets, direct computation is fine. 
    # For larger sets, we might need to be careful, but mixed radix sets 
    # have structured sum/diff sets.
    # However, the score formula uses |U+U| and |U-U|.
    # Let's compute them directly for correctness, using numpy for speed.
    U_arr = np.array(U, dtype=np.int64)
    
    # Sumset
    # To avoid O(N^2) memory blowup for very large N, we can use a set 
    # but for mixed radix sets with small k, N is small.
    # The seed was [0,1,3] (size 3).
    # Mixed radix sets can grow. Let's assume N < 1000 for now.
    # If N is large, we might need a smarter way, but let's start with direct.
    
    sums = set()
    diffs = set()
    
    # Optimization: if the set is a product set with no carries, 
    # |U+U| and |U-U| can be computed from the radices.
    # But for general local search steps, we might break the structure.
    # So we compute directly.
    
    # Vectorized sum/diff for small N
    if len(U) < 200:
        u_sums = U_arr[:, None] + U_arr[None, :]
        u_diffs = U_arr[:, None] - U_arr[None, :]
        sum_size = len(np.unique(u_sums))
        diff_size = len(np.unique(u_diffs))
    else:
        # Fallback for larger sets (shouldn't happen often in early search)
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(a+b)
                diffs.add(a-b)
        sum_size = len(sums)
        diff_size = len(diffs)
        
    ratio = diff_size / sum_size
    if ratio <= 0:
        return 0.0
        
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 0.0
        
    score = 1 + log_ratio / log_denom
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Initial best
    best_score = 0.0
    best_radices = [2, 2, 2]  # Example: {0,1} x {0,2} x {0,4} -> {0,1,2,3,4,5,6,7}
    # Actually, let's start with a simple mixed radix set
    # U = { sum a_i m_i } where 0 <= a_i < r_i
    # Let's define radices r = [r0, r1, ..., rk-1]
    # m_0 = 1
    # m_{i+1} = m_i * r_i
    
    def build_set_from_radices(radices):
        m = 1
        set_elements = [0]
        for r in radices:
            new_elements = []
            for a in range(r):
                for x in set_elements:
                    new_elements.append(x + a * m)
            set_elements = new_elements
            m *= r
        return set_elements

    # Initial candidate
    current_radices = [2, 2, 2, 2] # Size 16, max 15
    current_set = build_set_from_radices(current_radices)
    current_score = _score_set(current_set)
    
    if current_score > best_score:
        best_score = current_score
        best_radices = current_radices[:]

    # Local search on radices
    # We can change a radix r_i to r_i + delta or r_i - delta
    # Or swap radices? Order matters for max element.
    # Smaller radices should generally be earlier to keep max element smaller?
    # Actually, m_i grows. If we put a large radix early, m grows fast.
    # To minimize max element for a fixed product size, we should keep radices balanced?
    # No, max element is sum (r_i - 1) m_i.
    # If r = [2, 10], m = [1, 2]. Elements: 0..1, 2..11. Max 11.
    # If r = [10, 2], m = [1, 10]. Elements: 0..9, 10..19. Max 19.
    # So smaller radices first is better for minimizing max element.
    
    num_radices = len(current_radices)
    
    # We can also add or remove radices
    # Let's stick to fixed length first, then try changing length.
    
    while time.time() < deadline:
        # Randomly pick a position to modify
        i = random.randint(0, num_radices - 1)
        
        # Try changing radix at i
        old_r = current_radices[i]
        # Propose new radix
        # Keep it small to control max element
        # Range 2 to 10
        new_r = random.randint(2, 10)
        
        # Ensure it's different
        if new_r == old_r:
            continue
            
        candidate_radices = current_radices[:]
        candidate_radices[i] = new_r
        
        # Sort radices in non-decreasing order to minimize max element
        # Wait, does sorting preserve the set structure properties?
        # Yes, the set is just the product. The "radices" are just the sizes of each dimension.
        # The mixed radix representation is unique regardless of order, but the max element depends on order.
        # To minimize max element, sort radices ascending.
        candidate_radices.sort()
        
        # Rebuild set
        candidate_set = build_set_from_radices(candidate_radices)
        
        # Check time
        if time.time() > deadline - 2.0:
            break
            
        candidate_score = _score_set(candidate_set)
        
        if candidate_score > current_score:
            current_score = candidate_score
            current_radices = candidate_radices
            if current_score > best_score:
                best_score = current_score
                best_radices = current_radices[:]
        else:
            # Sometimes allow worse moves? Or just greedy?
            # Greedy is safer for time budget.
            pass

        # Occasionally try changing the number of radices
        if random.random() < 0.05:
            if num_radices < 8:
                # Add a radix
                candidate_radices = current_radices + [2]
                candidate_radices.sort()
                candidate_set = build_set_from_radices(candidate_radices)
                candidate_score = _score_set(candidate_set)
                if candidate_score > current_score:
                    current_score = candidate_score
                    current_radices = candidate_radices
                    num_radices = len(current_radices)
                    if current_score > best_score:
                        best_score = current_score
                        best_radices = current_radices[:]
            elif num_radices > 2:
                # Remove a radix (the largest one to reduce size/max)
                candidate_radices = current_radices[:-1]
                candidate_set = build_set_from_radices(candidate_radices)
                candidate_score = _score_set(candidate_set)
                if candidate_score > current_score:
                    current_score = candidate_score
                    current_radices = candidate_radices
                    num_radices = len(current_radices)
                    if current_score > best_score:
                        best_score = current_score
                        best_radices = current_radices[:]

    # Return the best set found
    final_set = build_set_from_radices(best_radices)
    return final_set

# EVOLVE-BLOCK-END
