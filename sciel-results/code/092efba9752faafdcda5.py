import itertools
import time
import sys

def construct(n):
    start_time = time.time()
    deadline = 120  # 120 seconds
    
    # For small n, use classic construction or brute force if very small
    if n <= 5:
        # Classic construction is optimal for n <= 4, and likely close for 5
        # But TSP works for all
        pass
    
    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    num_perms = len(perms)
    
    # Map permutation to index for fast lookup
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Precompute overlap costs
    # Cost[i][j] = n - overlap(perms[i], perms[j])
    # overlap is max k such that perms[i][n-k:] == perms[j][:k]
    
    # To speed up, we can use a dictionary for suffixes
    # But for n=7, 5040^2 is ~25M. 
    # Vectorized approach or optimized loop.
    
    # Let's try to compute the cost matrix efficiently.
    # We only need the cost for the TSP.
    
    # Greedy TSP Heuristic
    # 1. Start with a random or fixed permutation
    # 2. Repeatedly pick the nearest unvisited neighbor
    # 3. Apply 2-opt improvements
    
    # To make it faster, we can precompute the "distance" on the fly using LUTs for suffixes.
    # For each permutation p, and each k in 1..n, suffix = p[n-k:].
    # We can create a dictionary: suffix -> list of indices starting with that suffix? 
    # No, we need prefix of j to match suffix of i.
    # So: dict_suffix_to_indices[suffix] = [indices of perms starting with suffix]?
    # Wait, overlap(i, j) depends on suffix of i and prefix of j.
    
    # Let's just compute the full cost matrix for n=7. It's 25 million entries.
    # In Python, this might take ~5-10 seconds.
    
    if n > 6:
        # For larger n, we might need a more efficient heuristic or approximation.
        # But the task specifies n=7.
        pass

    # Precompute costs
    # cost[i][j]
    # We can store as a list of lists of small ints.
    
    # Optimization: 
    # For each i, we want to find j that maximizes overlap.
    # We can sort j's by their prefix.
    
    # Let's use a simple loop for now, but break if time is up.
    
    # If n is small enough, compute full matrix.
    if n <= 7:
        costs = [[0] * num_perms for _ in range(num_perms)]
        # Symmetry? No, overlap(i,j) != overlap(j,i) generally.
        # But cost(i,j) = n - overlap(i,j).
        
        # To speed up:
        # Group permutations by their first element, second, etc.
        # Actually, just loop.
        
        for i in range(num_perms):
            pi = perms[i]
            for j in range(num_perms):
                pj = perms[j]
                # Calculate overlap
                # Max possible overlap is n-1 (since they are distinct permutations, overlap n implies i==j)
                # We check k from n-1 down to 1
                # If pi[n-k:] == pj[:k]
                
                # Fast check:
                # Check k=n-1
                if pi[1:] == pj[:n-1]:
                    costs[i][j] = 1
                elif n >= 2 and pi[2:] == pj[:n-2]:
                    costs[i][j] = 2
                elif n >= 3 and pi[3:] == pj[:n-3]:
                    costs[i][j] = 3
                elif n >= 4 and pi[4:] == pj[:n-4]:
                    costs[i][j] = 4
                elif n >= 5 and pi[5:] == pj[:n-5]:
                    costs[i][j] = 5
                elif n >= 6 and pi[6:] == pj[:n-6]:
                    costs[i][j] = 6
                else:
                    costs[i][j] = n  # overlap 0
                
        # Greedy Tour
        unvisited = set(range(num_perms))
        tour = [0]
        unvisited.remove(0)
        current = 0
        
        while unvisited:
            best_next = -1
            best_cost = float('inf')
            for j in unvisited:
                c = costs[current][j]
                if c < best_cost:
                    best_cost = c
                    best_next = j
            tour.append(best_next)
            unvisited.remove(best_next)
            current = best_next
            
        # 2-Opt Improvement
        # Repeat until no improvement or time limit
        improved = True
        while improved and (time.time() - start_time) < deadline - 10:
            improved = False
            for i in range(len(tour) - 2):
                if (time.time() - start_time) > deadline - 10:
                    break
                # Try reversing tour[i+1..j]
                # Cost change:
                # Old: cost(tour[i], tour[i+1]) + cost(tour[j], tour[j+1])
                # New: cost(tour[i], tour[j]) + cost(tour[i+1], tour[j+1])
                # Note: tour is a path, not a cycle.
                # So we only check boundaries of the reversed segment.
                
                # We can optimize by checking only local segments first?
                # For now, full 2-opt.
                for j in range(i + 1, len(tour) - 1):
                    c_old = costs[tour[i]][tour[i+1]] + costs[tour[j]][tour[j+1]]
                    c_new = costs[tour[i]][tour[j]] + costs[tour[i+1]][tour[j+1]]
                    if c_new < c_old:
                        tour[i+1:j+1] = reversed(tour[i+1:j+1])
                        improved = True
                        # Break early? Or continue?
                        # Usually continue to find more improvements.
        
        # Construct word
        word = list(perms[tour[0]])
        for k in range(1, len(tour)):
            prev = perms[tour[k-1]]
            curr = perms[tour[k]]
            # Find max overlap
            # We already know the cost, so overlap = n - cost
            overlap = n - costs[tour[k-1]][tour[k]]
            word.extend(curr[overlap:])
        
        return "".join(map(str, word))
    else:
        # Fallback to classic construction for larger n or if TSP is too slow
        word = [1]
        for m in range(2, n + 1):
            perms_prev, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms_prev.append(window)
            new = []
            for p in perms_prev:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
