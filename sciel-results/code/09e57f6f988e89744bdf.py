import math
import time

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255,
    with gcd(a, b) > 1 for all distinct pairs.
    Maximizes cardinality.
    """
    N = 255
    deadline = time.time() + 55.0

    # 255 = 3 * 5 * 17
    # Every element in the set must share a factor with 255, so must be divisible by 3, 5, or 17.
    # But additionally, all pairs must have gcd > 1.
    #
    # The set of all multiples of 3 in [1, 255] is a valid clique of size 85.
    # Can we do better by mixing in multiples of 5 or 17?
    #
    # Key observation: if we include any multiple of 3 that is NOT a multiple of 5 or 17
    # (e.g., 3, 6, 9, 12, 15 is multiple of 5... wait 15 is multiple of 3 and 5),
    # then any other element must share a factor with it.
    #
    # Let's think about this more carefully using a graph/clique approach.
    # Actually, let's try a greedy/optimization approach:
    #
    # The candidate pool is all numbers in [1,255] divisible by 3, 5, or 17.
    # We need a maximum clique in the graph where edges = gcd>1, containing 255.
    #
    # Since 255 = 3*5*17, 255 is connected to all candidates.
    #
    # Strategy: Try all "prime factor strategies":
    # 1. All multiples of 3: size 85
    # 2. All multiples of 5: size 51 (but 255 is a multiple of 5, so valid)
    # 3. All multiples of 17: size 15
    #
    # Can we mix? If we take all multiples of 3, can we add a multiple of 5 not divisible by 3?
    # Such a number, say x, must have gcd(x, y) > 1 for all y in our set.
    # If y=3 is in the set, gcd(x,3) must be >1, meaning x must be divisible by 3.
    # But x is not divisible by 3 by assumption. Contradiction.
    # So if 3 is in the set, we can't add any number not divisible by 3.
    #
    # What if we remove 3? Then we need to check all remaining multiples of 3.
    # This becomes complex. Let's try a different approach:
    #
    # Use the fact that for a pairwise noncoprime set, there's a theorem that says
    # the maximum size is achieved by taking all multiples of the smallest prime
    # dividing the forced element. But let's verify computationally.
    #
    # Let's try a branch-and-bound or greedy approach on the candidate set.
    #
    # Candidate set: all numbers in [1,255] divisible by 3, 5, or 17.
    candidates = [x for x in range(1, 256) if x % 3 == 0 or x % 5 == 0 or x % 17 == 0]
    
    # Build adjacency: for each candidate, which other candidates have gcd>1?
    # Use bitsets for efficiency.
    n_cand = len(candidates)
    cand_index = {x: i for i, x in enumerate(candidates)}
    
    # 255 must be in the set
    idx_255 = cand_index[255]
    
    # Build adjacency matrix using lists of sets
    adj = [set() for _ in range(n_cand)]
    for i in range(n_cand):
        for j in range(i+1, n_cand):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Now find maximum clique containing idx_255
    # Use a simple branch and bound
    best_clique = {idx_255}
    best_size = 1
    
    # Greedy: start with 255, add candidates that connect to all current members
    current = {idx_255}
    # Candidates that connect to all in current
    candidates_for_greedy = [i for i in range(n_cand) if i != idx_255 and all(j in adj[i] for j in current)]
    
    # Sort by degree (number of connections) descending
    candidates_for_greedy.sort(key=lambda i: len(adj[i]), reverse=True)
    
    for i in candidates_for_greedy:
        # Check if i connects to all in current
        if all(j in adj[i] for j in current):
            current.add(i)
    
    if len(current) > best_size:
        best_size = len(current)
        best_clique = set(current)
    
    # Now try to improve with local search / branch and bound
    # Try removing elements and adding others
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_set = set(best_clique)
        # Try adding each non-member that connects to all members
        for i in range(n_cand):
            if i in current_set:
                continue
            if all(j in adj[i] for j in current_set):
                new_set = current_set | {i}
                if len(new_set) > best_size:
                    best_size = len(new_set)
                    best_clique = new_set
                    improved = True
                    break
        
        # Try removing one element and adding two
        if not improved:
            for remove_i in list(current_set):
                if remove_i == idx_255:
                    continue
                reduced = current_set - {remove_i}
                # Try adding elements
                for add_i in range(n_cand):
                    if add_i in reduced:
                        continue
                    if all(j in adj[add_i] for j in reduced):
                        new_set = reduced | {add_i}
                        if len(new_set) > best_size:
                            best_size = len(new_set)
                            best_clique = new_set
                            improved = True
                            break
                    if time.time() > deadline:
                        break
                if improved:
                    break
    
    # Convert back to values
    result = [candidates[i] for i in best_clique]
    result.sort()
    
    # Verify 255 is in the set
    if 255 not in result:
        result.append(255)
    
    return result
