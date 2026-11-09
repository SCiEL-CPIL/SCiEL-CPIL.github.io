import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    N = 512
    # 1. Generate candidates and weights
    candidates = list(range(1, N))
    weights = [1.0 / (N - v) for v in candidates]
    
    # Sort by weight descending
    order = sorted(range(N-1), key=lambda i: -weights[i])
    # order[0] is index of 511, order[1] is 510, etc.
    
    # Split into Top K and Bottom
    K = 40  # Top 40 candidates
    top_indices = order[:K]
    bottom_indices = order[K:]
    
    top_values = [candidates[i] for i in top_indices]
    top_weights = [weights[i] for i in top_indices]
    
    bottom_values = [candidates[i] for i in bottom_indices]
    bottom_weights = [weights[i] for i in bottom_indices]
    
    # Precompute prime factors for Bottom numbers
    # We need to know which primes block which bottom numbers.
    # Let's get all primes < 512
    def sieve(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(N)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Mask for each bottom number: bits set for its prime factors
    bottom_masks = [0] * len(bottom_values)
    for i, v in enumerate(bottom_values):
        mask = 0
        temp = v
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                mask |= (1 << prime_to_idx[p])
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            mask |= (1 << prime_to_idx[temp])
        bottom_masks[i] = mask
        
    total_bottom_weight = sum(bottom_weights)
    
    # Conflict graph for Top K
    # top_conflict[i] is bitmask of indices in top_indices that conflict with top_indices[i]
    top_conflict = [0] * K
    top_prime_masks = [0] * K
    for i, v in enumerate(top_values):
        temp = v
        mask = 0
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                mask |= (1 << prime_to_idx[p])
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            mask |= (1 << prime_to_idx[temp])
        top_prime_masks[i] = mask
        
    for i in range(K):
        for j in range(i + 1, K):
            if top_prime_masks[i] & top_prime_masks[j]:
                top_conflict[i] |= (1 << j)
                top_conflict[j] |= (1 << i)

    best_score = 0.0
    best_selection = []  # List of indices in top_indices
    
    def calculate_bottom_score(blocked_mask):
        """Sum of weights of bottom numbers coprime to blocked_mask."""
        s = 0.0
        for i in range(len(bottom_values)):
            if not (bottom_masks[i] & blocked_mask):
                s += bottom_weights[i]
        return s

    def branch_and_bound(idx, selected_mask, current_top_score, blocked_prime_mask):
        nonlocal best_score, best_selection
        
        if time.time() > deadline:
            return
            
        # Upper bound
        # 1. Current top score
        # 2. Sum of weights of all remaining top nodes (optimistic: assume we can take all)
        # 3. Sum of weights of valid bottom nodes
        remaining_top_sum = sum(top_weights[j] for j in range(idx, K))
        
        # Optimistic bottom score: assume no new primes are blocked by future top nodes
        # This is a valid upper bound because blocking more primes only reduces bottom score
        optimistic_bottom_score = calculate_bottom_score(blocked_prime_mask)
        
        ub = current_top_score + remaining_top_sum + optimistic_bottom_score
        
        if ub <= best_score + 1e-12:
            return

        if idx == K:
            # Calculate actual bottom score
            actual_bottom_score = calculate_bottom_score(blocked_prime_mask)
            total_score = current_top_score + actual_bottom_score
            if total_score > best_score:
                best_score = total_score
                # Reconstruct selection
                sel = []
                for i in range(K):
                    if (selected_mask >> i) & 1:
                        sel.append(i)
                best_selection = sel
            return

        # Branch on top_indices[idx]
        
        # Option 1: Exclude
        branch_and_bound(idx + 1, selected_mask, current_top_score, blocked_prime_mask)
        
        # Option 2: Include
        # Check if compatible with previously selected
        compatible = True
        for j in range(idx):
            if (selected_mask >> j) & 1:
                if (top_conflict[idx] >> j) & 1:
                    compatible = False
                    break
                    
        if compatible:
            new_selected_mask = selected_mask | (1 << idx)
            new_blocked_mask = blocked_prime_mask | top_prime_masks[idx]
            new_top_score = current_top_score + top_weights[idx]
            
            branch_and_bound(idx + 1, new_selected_mask, new_top_score, new_blocked_mask)

    # Run B&B
    branch_and_bound(0, 0, 0.0, 0)
    
    # Construct result
    result = []
    for i in best_selection:
        result.append(top_values[i])
        
    # Add valid bottom numbers
    # Determine blocked primes from best_selection
    final_blocked_mask = 0
    for i in best_selection:
        final_blocked_mask |= top_prime_masks[i]
        
    for i in range(len(bottom_values)):
        if not (bottom_masks[i] & final_blocked_mask):
            result.append(bottom_values[i])
            
    return result
