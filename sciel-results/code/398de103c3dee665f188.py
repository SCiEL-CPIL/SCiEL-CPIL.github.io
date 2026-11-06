import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    # Primes up to 127
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    num_primes = len(primes)
    
    # Precompute factor vectors as tuples for hashing
    def get_vec(n):
        v = [0] * num_primes
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
        return tuple(v)
        
    vecs = {i: get_vec(i) for i in range(2, 129)}
    zero_vec = (0,) * num_primes
    
    # Build the conflict graph for Maximum Independent Set
    # Vertices: integers 2-128
    # Two vertices i, j conflict if they cannot both be in a valid sequence
    # 
    # The key insight: in a strictly increasing sequence, the order is fixed.
    # So we need to find the longest subsequence of [2,3,...,128] where all
    # consecutive block products are distinct.
    #
    # This is NOT a simple independent set problem because the constraint is
    # global (depends on the entire sequence structure), not pairwise.
    #
    # Instead, let's use a smarter beam search with better state representation.
    
    # Use a dictionary-based approach with tuple vectors for speed
    # State: (sequence_tuple, prefix_sums_tuple, products_set)
    
    # Let's use a different approach: dynamic programming with pruning
    # dp[i] = best sequence ending at i
    # But the state space is too large for pure DP.
    
    # Improved beam search with better pruning and wider exploration
    
    # First, let's identify which pairs of numbers are "compatible" in adjacent positions
    # Two numbers a < b can be adjacent if no product collision occurs with just these two
    # But this is not sufficient - we need global constraints.
    
    # Let's use a more aggressive beam search with multiple strategies
    
    def check_validity(seq):
        """Check if a sequence has all distinct consecutive block products."""
        prefix_sums = [zero_vec]
        prod_set = set()
        
        for x in seq:
            v_x = vecs[x]
            v_new = tuple(a + b for a, b in zip(prefix_sums[-1], v_x))
            prefix_sums.append(v_new)
            
            for i in range(len(prefix_sums) - 1):
                diff = tuple(a - b for a, b in zip(v_new, prefix_sums[i]))
                if diff in prod_set:
                    return False
                prod_set.add(diff)
        
        return True
    
    # Greedy initialization: start with all numbers, remove conflicts
    # This gives us a good starting point
    
    # Build pairwise conflict: can i and j be in the same sequence?
    # Actually, the constraint is more complex. Let's use a different approach.
    
    # Strategy: Use iterative improvement starting from a greedy solution
    # 1. Start with all numbers [2, 3, ..., 128]
    # 2. While there are product collisions, remove the "most conflicting" element
    # 3. Try to add back removed elements
    
    # Better strategy: Use a simulated annealing / local search approach
    
    # Let's try a beam search with tuple-based state for speed
    
    beam = [([], (zero_vec,), frozenset())]
    beam_width = 500
    
    for n in range(2, 129):
        if time.time() > deadline:
            break
            
        new_beam = []
        v_n = vecs[n]
        
        for seq, psums, pset in beam:
            # Option 1: Exclude n
            new_beam.append((seq, psums, pset))
            
            # Option 2: Include n
            v_new = tuple(a + b for a, b in zip(psums[-1], v_n))
            
            # Check all suffix products ending at n
            valid = True
            new_prods = []
            
            for i in range(len(psums)):
                diff = tuple(a - b for a, b in zip(v_new, psums[i]))
                if diff in pset:
                    valid = False
                    break
                new_prods.append(diff)
            
            if valid:
                new_seq = seq + (n,)
                new_psums = psums + (v_new,)
                new_pset = pset.union(new_prods)
                new_beam.append((new_seq, new_psums, new_pset))
        
        # Select top beam_width states by sequence length, then by lexicographic order
        new_beam.sort(key=lambda x: (len(x[0]), x[0]), reverse=True)
        beam = new_beam[:beam_width]
        
        # Early termination check
        if beam and len(beam[0][0]) >= 110:
            # Try to tighten the beam
            beam_width = min(beam_width, 100)
    
    # Get best sequence from beam
    if beam:
        best_seq = list(beam[0][0])
    else:
        best_seq = []
    
    # Local improvement: try to insert missing numbers
    if time.time() < deadline - 10.0:
        best_seq = sorted(best_seq)
        seq_set = set(best_seq)
        candidates = [x for x in range(2, 129) if x not in seq_set]
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for c in candidates:
                if c in seq_set:
                    continue
                
                # Find insertion position
                ins_idx = 0
                for i, x in enumerate(best_seq):
                    if x < c:
                        ins_idx = i + 1
                    else:
                        break
                
                # Try inserting c at ins_idx
                test_seq = best_seq[:ins_idx] + [c] + best_seq[ins_idx:]
                
                # Verify validity
                if check_validity(test_seq):
                    best_seq = test_seq
                    seq_set.add(c)
                    improved = True
        
        best_seq = sorted(best_seq)
    
    # Final check and return
    if check_validity(best_seq):
        return best_seq
    else:
        # Fallback: return a valid shorter sequence
        return [i for i in range(2, 129) if i in best_seq][:len(best_seq)//2] if best_seq else []

# Run a quick test
if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)
