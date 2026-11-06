"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def _is_cap_set(vectors, n):
    """Check if a list of tuples is a cap set."""
    if len(vectors) <= 2:
        return True
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            for k in range(j + 1, len(vectors)):
                a, b, c = vectors[i], vectors[j], vectors[k]
                # Check if a, b, c form an arithmetic progression
                # In Z_3^n, a, b, c form AP iff a + b + c = 0 mod 3
                # Equivalently, for each coordinate i: a_i + b_i + c_i is 0 or 3
                # Since a_i, b_i, c_i in {0,1,2}, sum is 0,1,2,3,4,5,6.
                # Sum is 0 mod 3 iff sum is 0 or 3.
                # Sum is 0 iff all are 0.
                # Sum is 3 iff they are {0,1,2} in some order.
                # So we just need to check if any coordinate has all distinct values.
                # Wait, if a_i=0, b_i=1, c_i=2, sum=3=0 mod 3.
                # If a_i=0, b_i=0, c_i=0, sum=0=0 mod 3.
                # If a_i=0, b_i=0, c_i=1, sum=1 != 0.
                # So the condition "a+b+c=0 mod 3" is equivalent to "for all i, a_i, b_i, c_i are NOT all distinct".
                # Actually, it's simpler: a, b, c are in AP if and only if a+b+c == 0 in the group.
                # In Z_3^n, a+b+c = (a_1+b_1+c_1, ..., a_n+b_n+c_n).
                # This is 0 iff each coordinate sum is 0 mod 3.
                # Let's just compute the sum.
                valid = True
                for coord in range(n):
                    s = a[coord] + b[coord] + c[coord]
                    if s % 3 != 0:
                        valid = False
                        break
                if valid:
                    return False
    return True

def _greedy_cap_set(n, time_limit=10.0):
    """Generate a cap set using greedy insertion with randomization."""
    start_time = time.time()
    # Generate all vectors
    all_vectors = []
    for i in range(3**n):
        vec = []
        x = i
        for _ in range(n):
            vec.append(x % 3)
            x //= 3
        all_vectors.append(tuple(vec))
    
    random.shuffle(all_vectors)
    
    cap_set = []
    for v in all_vectors:
        if time.time() - start_time > time_limit:
            break
        # Check if adding v keeps it a cap set
        # Optimization: check against existing elements
        is_valid = True
        for i in range(len(cap_set)):
            for j in range(i + 1, len(cap_set)):
                a, b = cap_set[i], cap_set[j]
                # Check if v, a, b form AP
                # v + a + b = 0 mod 3?
                valid_triplet = True
                for coord in range(n):
                    if (v[coord] + a[coord] + b[coord]) % 3 != 0:
                        valid_triplet = False
                        break
                if valid_triplet:
                    is_valid = False
                    break
            if not is_valid:
                break
        if is_valid:
            cap_set.append(v)
    
    return cap_set

def _local_search(cap_set, n, time_limit=10.0):
    """Improve cap set via local search swaps."""
    start_time = time.time()
    current_set = list(cap_set)
    current_size = len(current_set)
    
    # Precompute all vectors for potential swaps
    # This might be too large for large n, but for small n it's fine.
    # If n is large, we can just try random swaps or neighbor swaps.
    # Given the time limit, let's assume n is small enough or we use a restricted search.
    # For safety, let's only try swapping with vectors that are "close" or just random.
    # Actually, generating all vectors is 3^n. If n=10, that's 59049. Manageable.
    # If n=20, that's 3.4e9. Not manageable.
    # The task doesn't specify n, but "cap set" problems usually have small n for exact solutions.
    # Let's assume n <= 15 for full enumeration, else use random sampling for swaps.
    
    if 3**n < 100000:
        all_vectors = []
        for i in range(3**n):
            vec = []
            x = i
            for _ in range(n):
                vec.append(x % 3)
                x //= 3
            all_vectors.append(tuple(vec))
        
        # Create a set of current vectors for O(1) lookup
        current_set_tuples = set(current_set)
        
        while time.time() - start_time < time_limit:
            improved = False
            # Try swapping each element in current_set with each element in all_vectors
            # This is O(N^2 * 3^n * n) which might be too slow.
            # Let's be smarter: for each element in current_set, try to replace it.
            
            for i in range(len(current_set)):
                if time.time() - start_time > time_limit:
                    break
                v_to_remove = current_set[i]
                # Try replacing v_to_remove with a random vector not in current_set
                # Or iterate through candidates?
                # Let's just try a few random candidates per iteration to keep it fast.
                for _ in range(10):
                    v_new = random.choice(all_vectors)
                    if v_new in current_set_tuples:
                        continue
                    
                    # Check if (current_set - {v_to_remove}) U {v_new} is a cap set
                    # This requires checking all triplets in the new set.
                    # Expensive.
                    # Approximation: Check if v_new conflicts with any pair in (current_set - {v_to_remove})
                    # And check if v_to_remove was "critical" (i.e., its removal allows v_new).
                    
                    # Let's just do the full check for the new set if it's small.
                    # If n is small, the set size is small.
                    new_set = current_set[:i] + current_set[i+1:] + [v_new]
                    
                    # Check cap set property
                    is_valid = True
                    for a_idx in range(len(new_set)):
                        for b_idx in range(a_idx + 1, len(new_set)):
                            for c_idx in range(b_idx + 1, len(new_set)):
                                a, b, c = new_set[a_idx], new_set[b_idx], new_set[c_idx]
                                valid_triplet = True
                                for coord in range(n):
                                    if (a[coord] + b[coord] + c[coord]) % 3 != 0:
                                        valid_triplet = False
                                        break
                                if valid_triplet:
                                    is_valid = False
                                    break
                            if not is_valid:
                                break
                        if not is_valid:
                            break
                    
                    if is_valid:
                        # Swap successful
                        current_set[i] = v_new
                        current_set_tuples.remove(v_to_remove)
                        current_set_tuples.add(v_new)
                        improved = True
                        break # Move to next element to remove
            
            if not improved:
                # No improvement found in this pass, maybe break or continue with different randoms
                # Let's break to avoid infinite loops of no-op
                break
                
    return current_set

def priority(vector, n):
    # The task contract says: "vectors of {0,1,2}^n are added greedily in descending priority"
    # "Score: cap set size"
    # "The evaluator measures the raw value"
    
    # Wait, the interface is `priority(vector: tuple, n: int) -> float`.
    # And the description says: "vectors ... are added greedily in descending priority when they keep the set a cap set".
    # This implies the EVALUATOR runs the greedy algorithm using my `priority` function.
    # I do NOT control the greedy algorithm itself in the evaluator.
    # I only control the priority assigned to each vector.
    
    # My previous thought process assumed I could run local search. But if the evaluator just calls `priority` for each vector and adds it if valid, I can't do local search.
    # I can only influence the ORDER in which vectors are considered.
    
    # So the strategy is: assign priorities such that the greedy algorithm picks a "good" set.
    # "Good" means large size.
    # To get a large set, we should pick vectors that are "independent" or "spread out".
    # Vectors with many 0s might be better? Or balanced?
    # Inspiration 1 (1.13) used: "Static heuristic favoring balanced, low-autocorrelation vectors".
    
    # Let's try a heuristic that favors vectors with a specific structure.
    # For cap sets in Z_3^n, a common construction is using vectors with weight 1 (one 1, rest 0) or weight 2?
    # Actually, the set of all vectors with exactly one non-zero coordinate is a cap set?
    # Let e_i be the vector with 1 at i and 0 elsewhere.
    # e_i + e_j + e_k = 0?
    # If i, j, k distinct: sum has 1s at i, j, k. Not 0.
    # If i=j: 2e_i + e_k. Not 0.
    # So the standard basis is a cap set. Size n.
    
    # Can we do better?
    # The maximum size of a cap set in Z_3^n is roughly 3^n / (n * 2^n) or something?
    # Actually, the upper bound is (3/2)^n? No.
    # For n=3, max size is 9.
    # For n=4, max size is 20.
    
    # A simple heuristic: prioritize vectors with low "conflict potential".
    # Conflicts arise when a+b+c=0.
    # If we pick vectors with many 0s, they are less likely to sum to 0 with others?
    # Let's try prioritizing by Hamming weight (number of non-zero elements).
    # Or by the sum of coordinates?
    
    # Let's try: prioritize vectors with fewer non-zero elements.
    # This creates a set of "sparse" vectors.
    
    # vector is a tuple of {0,1,2}
    # Count non-zeros
    non_zeros = sum(1 for x in vector if x != 0)
    
    # Return a priority such that lower non_zeros have HIGHER priority.
    # Since the evaluator sorts in descending priority, we want high value for good vectors.
    # So return -non_zeros or 10 - non_zeros.
    
    # Let's add a bit of randomness or tie-breaking to avoid lexicographic bias if weights are equal.
    # But priority must be deterministic? "Static heuristic".
    # If I return just -non_zeros, then all vectors with same weight have same priority.
    # The evaluator might break ties lexicographically or randomly.
    # To be safe, I can add a small deterministic tie-breaker based on the vector content.
    
    # Let's try: priority = 100 - non_zeros
    # This favors sparse vectors.
    
    # Another idea: Inspiration 1 mentioned "low-autocorrelation".
    # Autocorrelation of a vector?
    # Maybe sum of (v_i - v_j)^2?
    
    # Let's stick to the simplest effective heuristic: Sparsity.
    
    # Also, consider the value 2. In Z_3, 2 is -1.
    # Maybe vectors with only 0 and 1 are better than those with 2?
    # Let's penalize 2s.
    
    score = 0
    for x in vector:
        if x == 1:
            score += 1
        elif x == 2:
            score += 2
    
    # We want to minimize score?
    # If we minimize score, we prefer 0s, then 1s, then 2s.
    # This is equivalent to minimizing the sum of coordinates.
    
    # Let's return -score.
    return -score

# EVOLVE-BLOCK-END
