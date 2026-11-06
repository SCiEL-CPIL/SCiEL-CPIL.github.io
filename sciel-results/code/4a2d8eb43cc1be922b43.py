import random
import time

def construct():
    """
    Returns a list of 32 distinct integers in 1..512 maximizing the number
    of distinct d > 0 such that a, a+d, a+2d are in the set.
    """
    
    def score_from_bitset(bitset):
        """
        Calculate the number of distinct differences d > 0 
        such that a, a+d, a+2d are in S for some a.
        Takes a precomputed bitset.
        """
        count = 0
        for d in range(1, 256):
            if bitset & (bitset >> d) & (bitset >> (2 * d)):
                count += 1
        return count

    def build_bitset(S):
        bitset = 0
        for v in S:
            bitset |= (1 << (v - 1))
        return bitset

    # Initialization
    random.seed(42)
    candidates = [
        list(range(1, 33)),
        list(range(1, 17)) + list(range(257, 273)),
        list(range(1, 17)) + list(range(129, 145)),
        random.sample(range(1, 513), 32),
        random.sample(range(1, 513), 32),
        random.sample(range(1, 513), 32)
    ]
    
    best_set = max(candidates, key=lambda S: score_from_bitset(build_bitset(S)))
    best_score = score_from_bitset(build_bitset(best_set))
    
    start_time = time.time()
    time_limit = 110.0
    
    current_set = best_set[:]
    current_bitset = build_bitset(current_set)
    current_score = best_score
    
    # Local Search with Swaps
    while time.time() - start_time < time_limit:
        improved = False
        n = len(current_set)
        
        # Try all swaps
        for i in range(n):
            for j in range(i + 1, n):
                # Swap values in set
                v_i = current_set[i]
                v_j = current_set[j]
                
                # Update bitset
                # Clear bits for v_i and v_j
                current_bitset &= ~(1 << (v_i - 1))
                current_bitset &= ~(1 << (v_j - 1))
                # Set bits for swapped values (which are just v_j and v_i)
                current_bitset |= (1 << (v_j - 1))
                current_bitset |= (1 << (v_i - 1))
                
                new_score = score_from_bitset(current_bitset)
                
                if new_score > current_score:
                    # Accept swap
                    current_set[i] = v_j
                    current_set[j] = v_i
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_set = current_set[:]
                    improved = True
                    break
                else:
                    # Revert bitset
                    current_bitset &= ~(1 << (v_j - 1))
                    current_bitset &= ~(1 << (v_i - 1))
                    current_bitset |= (1 << (v_i - 1))
                    current_bitset |= (1 << (v_j - 1))
            
            if improved:
                break
        
        if not improved:
            # Local optimum reached. Try random perturbation.
            # Change 2-4 elements randomly to jump to a new basin.
            num_changes = random.randint(2, 4)
            for _ in range(num_changes):
                idx = random.randint(0, 31)
                old_val = current_set[idx]
                new_val = random.randint(1, 512)
                # Ensure distinctness
                while new_val in current_set:
                    new_val = random.randint(1, 512)
                
                # Update bitset
                current_bitset &= ~(1 << (old_val - 1))
                current_bitset |= (1 << (new_val - 1))
                current_set[idx] = new_val
            
            current_score = score_from_bitset(current_bitset)
            
            # If perturbation was too bad, revert to best
            if current_score < best_score - 10:
                current_set = best_set[:]
                current_bitset = build_bitset(current_set)
                current_score = best_score

    return best_set
