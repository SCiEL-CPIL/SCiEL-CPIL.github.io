"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

_START_TIME = time.time()
_DEADLINE = _START_TIME + 55.0  # Safety margin for 60s limit

def construct():
    """
    Returns a list of distinct integers in 2..40 such that all 2^k subset products are distinct.
    Maximizes the number of selected integers.
    
    Strategy:
    1. Model the problem using prime exponent vectors.
    2. The condition is that all subset sums of the selected exponent vectors are distinct.
    3. This is equivalent to finding the largest subset of vectors that is "sum-free" in the sense that no two distinct subsets have the same sum.
    4. We use a Maximum Independent Set (MIS) formulation on a conflict graph.
       - Vertices: candidate numbers (2..40).
       - Edge between u and v if {u, v} cannot be part of any valid set? No, that's too strong.
       - Actually, the property is global. If we select a set S, all 2^|S| sums must be distinct.
       - A simpler constraint: If v_a + v_b = v_c + v_d, then we cannot have {a,b,c,d} in a way that creates ambiguity?
       - Wait, the condition is: For any two distinct subsets A, B of S, sum(A) != sum(B).
       - This implies that for any distinct a, b in S, v_a != v_b (which is true for distinct integers).
       - It also implies that v_a + v_b != v_c for any distinct a,b,c in S.
       - It also implies that v_a + v_b != v_c + v_d for any distinct a,b,c,d in S.
       
       This is a "Sidon set" like property in the group (Z^+)^12, but with subset sums.
       
       Let's use a Backtracking Search with strong pruning based on the number of subset sums.
       Key Insight: The number of subset sums is 2^k. The range of possible sums is bounded by the sum of all vectors.
       If the "density" of the sums in the bounding box is too high, we might prune?
       
       Actually, a more direct approach:
       We want to maximize k.
       Let's try to find the maximum k by checking if a set of size k exists.
       We can use a recursive search that adds elements one by one.
       
       To make it fast:
       1. Precompute all vectors.
       2. Sort candidates by "complexity" (e.g., number of prime factors) descending? 
          Primes are "simple" (one non-zero entry). Composites are "complex".
          Primes are easier to pack?
          Let's try sorting by the L1 norm of the vector ascending. Primes have norm 1.
       3. Use a set of tuples for current subset sums.
       4. Pruning: If current_size + remaining_candidates <= best_size, prune.
       5. Additional Pruning: 
          The maximum possible number of subset sums is 2^k.
          The "volume" of the space occupied by the sums must be at least 2^k.
          This is hard to compute dynamically.
          
       Let's stick to a highly optimized Backtrack with:
       - Candidate ordering: Sort by "difficulty" or "constraint impact".
       - Symmetry breaking: If two candidates have identical vectors (impossible here as integers are distinct), skip.
       - Early termination if deadline is reached.
       
       To improve on the current 12 (score 1.3333 implies 12 elements? 12/9=1.333. Seed likely 9. Current is 12).
       Can we get 13?
       
       Let's try a different heuristic for ordering:
       Numbers with fewer prime factors are "sparser".
       Let's try to prioritize primes first? Or composites?
       
       Let's try a randomized restart approach with different orderings to escape local optima.
    """
    
    candidates = list(range(2, 41))
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    def get_vector(n):
        v = [0] * num_primes
        for i, p in enumerate(primes):
            while n % p == 0:
                v[i] += 1
                n //= p
        return tuple(v)
        
    vectors = {x: get_vector(x) for x in candidates}
    
    # Heuristic: Sort by number of prime factors (support size) then by value
    # Primes have support size 1.
    def support_size(v):
        return sum(1 for x in v if x > 0)
        
    # Let's try multiple orderings in parallel (sequentially in time)
    orderings = [
        sorted(candidates, key=lambda x: (support_size(vectors[x]), x)),
        sorted(candidates, key=lambda x: (-support_size(vectors[x]), x)),
        sorted(candidates, key=lambda x: x),
        sorted(candidates, key=lambda x: -x)
    ]
    
    best_size = 0
    best_set = []
    
    initial_sums = {tuple([0]*num_primes)}
    
    def backtrack(ordering, start_idx, current_set, current_sums, current_size):
        nonlocal best_size, best_set
        
        if time.time() > _DEADLINE:
            return
            
        if current_size > best_size:
            best_size = current_size
            best_set = list(current_set)
            
        # Pruning
        if current_size + (len(ordering) - start_idx) <= best_size:
            return
            
        for i in range(start_idx, len(ordering)):
            if time.time() > _DEADLINE:
                return
                
            x = ordering[i]
            v = vectors[x]
            
            # Check collision
            can_add_flag = True
            # Optimization: Check if v is in sums? No, v is a vector, sums are sums.
            # We need to check if (s + v) is in sums for any s in sums.
            # This is O(2^k * D).
            
            # Fast check: 
            # If we have many sums, this is slow.
            # Let's just do it.
            
            for s in current_sums:
                nv = tuple(s[j] + v[j] for j in range(num_primes))
                if nv in current_sums:
                    can_add_flag = False
                    break
                    
            if can_add_flag:
                # Add
                new_sums = set(current_sums)
                for s in current_sums:
                    nv = tuple(s[j] + v[j] for j in range(num_primes))
                    new_sums.add(nv)
                    
                current_set.append(x)
                backtrack(ordering, i + 1, current_set, new_sums, current_size + 1)
                current_set.pop()
                
    # Try each ordering with a time slice
    time_per_ordering = (_DEADLINE - time.time()) / len(orderings)
    
    for ordering in orderings:
        if time.time() > _DEADLINE:
            break
        backtrack(ordering, 0, [], initial_sums, 0)
        
    # Ensure types are correct
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
