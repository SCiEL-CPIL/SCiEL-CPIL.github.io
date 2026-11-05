import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50 seconds to be safe within 60s limit
    
    # Candidate sets of primes to try, ordered by size (smaller is better)
    # We want to minimize the number of distinct primes.
    # Let's try sets of size 4, 5, 6, etc.
    # Note: If we use primes {2,3}, sums must be 2^a 3^b. 
    # 2^a 3^b + 2^c 3^d = 2^e 3^f.
    # This is a strong constraint.
    
    prime_sets = [
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13],
        [2, 3, 5, 7, 11, 13, 17],
        [2, 3, 5, 7, 11, 13, 17, 19]
    ]
    
    # Fallback to seed solution if no better found
    best_result = list(range(1, 13))
    best_score = 9 # Score for 1..12 is 9 primes: 2,3,5,7,11,13,17,19,23
    
    def is_smooth(num, primes):
        if num <= 1:
            return True
        for p in primes:
            while num % p == 0:
                num //= p
        return num == 1

    def generate_smooth_numbers(max_val, primes):
        smooths = []
        # Simple generation by iterating and checking
        # 10000 is small enough for this
        for i in range(1, max_val + 1):
            if is_smooth(i, primes):
                smooths.append(i)
        return smooths

    def find_clique(nodes, edges, target_size, deadline):
        # nodes: list of ints
        # edges: set of frozensets or dict of sets for adjacency
        # Using recursive backtracking
        if len(nodes) < target_size:
            return None
        
        # Sort nodes by degree (descending) to find cliques faster
        def degree(n):
            return len(edges[n])
            
        nodes = sorted(nodes, key=degree, reverse=True)
        
        # Precompute adjacency sets for faster intersection
        adj = {n: set() for n in nodes}
        for n1 in nodes:
            for n2 in edges[n1]:
                if n2 in adj:
                    adj[n1].add(n2)
        
        def backtrack(current_clique, candidates):
            if time.time() > deadline:
                return None
            if len(current_clique) == target_size:
                return current_clique
            if len(candidates) < target_size - len(current_clique):
                return None
            
            for i in range(len(candidates)):
                node = candidates[i]
                new_clique = current_clique + [node]
                # New candidates must be adjacent to node and from the remaining candidates
                # To maintain order and avoid duplicates, we only look at candidates after i
                new_candidates = [c for c in candidates[i+1:] if c in adj[node]]
                res = backtrack(new_clique, new_candidates)
                if res:
                    return res
            return None

        return backtrack([], nodes)

    for primes in prime_sets:
        if time.time() > deadline:
            break
            
        smooths = generate_smooth_numbers(10000, primes)
        if len(smooths) < 12:
            continue
            
        # Build graph
        # We only care about edges where sum is smooth
        # To save time, we can limit the number of candidates if the set is huge
        # But smooths for {2,3,5,7} up to 10000 is small (~100-200)
        # For larger sets, it grows.
        
        # Optimization: Only keep top N candidates?
        # Actually, for small prime sets, the graph is dense enough.
        # For larger sets, we might need to be smarter.
        # Let's just try all pairs if count is manageable (< 500)
        
        if len(smooths) > 1000:
            # Too many candidates, skip or sample?
            # For now, skip very large sets to avoid O(N^2) explosion if needed
            # But {2,3,5,7,11,13} should be fine.
            pass

        edges = {n: set() for n in smooths}
        
        # Build edges
        # O(N^2)
        # If N is ~500, N^2 = 250k, which is fast.
        for i in range(len(smooths)):
            u = smooths[i]
            for j in range(i + 1, len(smooths)):
                v = smooths[j]
                if is_smooth(u + v, primes):
                    edges[u].add(v)
                    edges[v].add(u)
        
        # Find clique of size 12
        # We can restrict search to nodes with degree >= 11
        candidates = [n for n in smooths if len(edges[n]) >= 11]
        
        if len(candidates) >= 12:
            clique = find_clique(candidates, edges, 12, deadline)
            if clique:
                # Verify
                valid = True
                prime_factors = set()
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        s = clique[i] + clique[j]
                        # Factor s to get primes
                        temp = s
                        for p in primes:
                            while temp % p == 0:
                                prime_factors.add(p)
                                temp //= p
                        if temp != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    # This is a valid solution with |primes| distinct primes
                    # Since we iterate by increasing size of primes, the first one found is optimal for this family
                    # But we should check if |primes| < best_score
                    if len(primes) < best_score:
                        best_result = clique
                        best_score = len(primes)
                        # We can break if we found a very good one, e.g., 4 or 5
                        # But let's keep searching for smaller sets in next iterations?
                        # No, we iterate prime_sets in increasing size.
                        # So if we found one with size 4, it's the best possible (min 3? maybe 3 is impossible).
                        # Let's return immediately if we find a small one?
                        # The task is to minimize. 4 is better than 9.
                        # Is 3 possible? {2,3,5}. Sums 2^a 3^b 5^c.
                        # Let's let the loop continue to check smaller sets if we started with larger?
                        # No, we started with [2,3,5,7].
                        # So if we find a clique for [2,3,5,7], score is 4.
                        # We should check if [2,3,5] is possible?
                        # Let's add [2,3,5] to the list.
                        
                        # If we find a solution with 4 primes, it's very good.
                        # We can break if best_score == 4? 
                        # Is 3 possible?
                        # 12 numbers from {2,3,5}-smooth.
                        # Sums must be {2,3,5}-smooth.
                        # This is a known hard problem. Likely 4 is the min.
                        # Let's add [2,3,5] to the list to be sure.
                        
                        # Since we iterate from smallest set, if we found a solution, it's the best for this family.
                        # But we might have skipped [2,3,5] if it wasn't in the list.
                        # Let's restart with a refined list if we find a solution?
                        # Or just add [2,3,5] to the initial list.
                        
                        pass # Continue to next prime set? No, we want the smallest.
                        # Since prime_sets is ordered by size, the first success is the best.
                        # So we can break.
                        break

    return best_result

# Add [2,3,5] to the list in the function? 
# I defined prime_sets inside construct. Let's modify the list.
# I will rewrite the function to include [2,3,5].

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50 seconds to be safe within 60s limit
    
    prime_sets = [
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13],
        [2, 3, 5, 7, 11, 13, 17],
        [2, 3, 5, 7, 11, 13, 17, 19]
    ]
    
    best_result = list(range(1, 13))
    best_score = 9
    
    def is_smooth(num, primes):
        if num <= 1:
            return True
        for p in primes:
            while num % p == 0:
                num //= p
        return num == 1

    def generate_smooth_numbers(max_val, primes):
        smooths = []
        for i in range(1, max_val + 1):
            if is_smooth(i, primes):
                smooths.append(i)
        return smooths

    def find_clique(nodes, edges, target_size, deadline):
        if len(nodes) < target_size:
            return None
        
        def degree(n):
            return len(edges[n])
            
        nodes = sorted(nodes, key=degree, reverse=True)
        
        adj = {n: set() for n in nodes}
        for n1 in nodes:
            for n2 in edges[n1]:
                if n2 in adj:
                    adj[n1].add(n2)
        
        def backtrack(current_clique, candidates):
            if time.time() > deadline:
                return None
            if len(current_clique) == target_size:
                return current_clique
            if len(candidates) < target_size - len(current_clique):
                return None
            
            for i in range(len(candidates)):
                node = candidates[i]
                new_clique = current_clique + [node]
                new_candidates = [c for c in candidates[i+1:] if c in adj[node]]
                res = backtrack(new_clique, new_candidates)
                if res:
                    return res
            return None

        return backtrack([], nodes)

    for primes in prime_sets:
        if time.time() > deadline:
            break
            
        smooths = generate_smooth_numbers(10000, primes)
        if len(smooths) < 12:
            continue
            
        if len(smooths) > 2000:
            # Skip if too many to build graph quickly?
            # 2000^2 = 4M ops. Might be tight in Python.
            # But for [2,3,5], count is small.
            # For [2,3,5,7], count is ~300.
            # For [2,3,5,7,11], count is ~1000.
            # Let's allow up to 1000 for safety.
            if len(smooths) > 1000:
                continue

        edges = {n: set() for n in smooths}
        
        for i in range(len(smooths)):
            u = smooths[i]
            for j in range(i + 1, len(smooths)):
                v = smooths[j]
                if is_smooth(u + v, primes):
                    edges[u].add(v)
                    edges[v].add(u)
        
        candidates = [n for n in smooths if len(edges[n]) >= 11]
        
        if len(candidates) >= 12:
            clique = find_clique(candidates, edges, 12, deadline)
            if clique:
                valid = True
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        s = clique[i] + clique[j]
                        if not is_smooth(s, primes):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    if len(primes) < best_score:
                        best_result = clique
                        best_score = len(primes)
                    break # Found best for this family size, move to next (smaller? no, we iterate smallest first)
                    # Since we iterate smallest first, first success is global best for these sets.
                    # But what if [2,3,5] fails and [2,3,5,7] succeeds?
                    # We break only if we found a solution.
                    # If [2,3,5] fails, loop continues to [2,3,5,7].
                    # So breaking is correct.

    return best_result
