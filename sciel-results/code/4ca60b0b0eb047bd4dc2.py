import random
import time
import sys

def construct(n):
    if n <= 4:
        # Standard construction is optimal for n <= 4
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    # New Approach: Greedy Construction with Backtracking (Warnsdorff-like)
    
    import itertools
    
    # Precompute all valid permutations
    all_perms = set(itertools.permutations(range(1, n + 1)))
    total_perms = len(all_perms)
    
    # Helper to check if a window is a permutation
    def is_perm(window):
        return window in all_perms

    # We will build the string as a list of integers
    # State: current string, visited set
    # We use DFS with backtracking
    
    best_word = None
    start_time = time.time()
    time_limit = 115.0
    
    # To speed up, we can precompute transitions?
    # For a given suffix (n-1 chars), what are the valid next chars?
    # There are n! possible suffixes. n=7 -> 5040 suffixes.
    # For each suffix, there are at most 2 valid next chars?
    # No, a suffix of length n-1 is a partial permutation.
    # E.g., n=3, suffix (1,2) -> next must be 3.
    # Suffix (1,3) -> next must be 2.
    # In general, for a suffix of length n-1 consisting of distinct digits,
    # the missing digit is the only valid next digit to form a permutation.
    # Wait. If the suffix has distinct digits, there is exactly 1 missing digit.
    # So the next digit is DETERMINED by the suffix.
    # This means the graph of permutations is a set of disjoint cycles/paths?
    # No, the suffix is the *last* n-1 digits of the *previous* permutation.
    # Let P = p1...pn. The next permutation Q must overlap with P.
    # The overlap is the suffix of P and prefix of Q.
    # If we fix the last n-1 digits of P, the first n-1 digits of Q are fixed.
    # The last digit of Q is determined by the requirement that Q is a permutation.
    # So, from any permutation P, there is exactly ONE next permutation Q that overlaps
    # with P by n-1 positions?
    # Let's check.
    # P = 1234. Suffix 234. Q must start with 234. Q = 234X. X must be 1. Q=2341.
    # So yes, the transition is deterministic if we require overlap n-1.
    # But we don't *have* to overlap by n-1. We can overlap by less.
    # If we overlap by k < n-1, we have more freedom.
    
    # The standard construction uses overlaps of n-1 mostly, but sometimes less?
    # No, the standard construction (factoradic) uses a specific structure.
    
    # If we restrict to overlaps of n-1, we traverse a single cycle of length n!.
    # This cycle contains all n! permutations?
    # Yes, the graph where edges connect P to Q if P's suffix (n-1) == Q's prefix (n-1)
    # is a single cycle containing all n! permutations.
    # This is the "standard" superpermutation of length n! + n - 1.
    # But we want the SHORTEST.
    # Shorter superpermutations use overlaps of n-2, n-3, etc.
    
    # So, at each step, we can choose to overlap by k = n-1, n-2, ..., 1.
    # If we overlap by k, we append n-k characters.
    # The new window is determined by the last k chars of current + n-k new chars.
    # The new window must be a permutation.
    
    # Strategy:
    # 1. Start with a random permutation.
    # 2. At each step, choose an overlap length k from {n-1, n-2, ..., 1}.
    # 3. For a given k, the prefix of the next perm is fixed (last k of current).
    # 4. We need to complete the rest of the permutation (n-k digits) such that
    #    the whole thing is a valid permutation.
    # 5. This leaves (n-k)! possibilities? No, the remaining digits must be the
    #    complement of the prefix. So there are (n-k)! ways to complete it.
    # 6. We want to choose the completion that allows future progress.
    
    # Heuristic:
    # Prefer smaller k (more new characters) if it leads to a "dead end" less likely?
    # Or prefer larger k (less new characters) to save length?
    # Larger k saves length immediately, but might trap us.
    # Smaller k adds length, but gives more flexibility.
    
    # Let's try a Greedy approach:
    # At each step, try k = n-1 down to 1.
    # For each k, generate all valid next permutations.
    # Score them by "flexibility" (number of valid next-next steps).
    # Pick the best.
    # If we get stuck, backtrack.
    
    # This is complex. Let's try a simpler randomized greedy with backtracking.
    
    def solve():
        # State: current_perm, visited_set, path (list of perms)
        # We use a stack for backtracking
        
        # To make it efficient, we represent the state by the current permutation
        # and the set of visited permutations.
        
        # Random start
        start_perm = tuple(random.sample(range(1, n + 1), n))
        
        # We will use a recursive DFS with a depth limit (number of steps)
        # Since n! is 5040, the path length is roughly 5040 + n - 1.
        # Recursion depth 5040 is fine.
        
        def dfs(current_perm, visited, path):
            nonlocal best_word, time_limit, start_time
            
            if time.time() - start_time > time_limit:
                return False
            
            if len(visited) == total_perms:
                # Found a solution!
                # Construct the word
                word = list(path[0])
                for i in range(1, len(path)):
                    p_prev = path[i-1]
                    p_curr = path[i]
                    # Find max overlap
                    max_k = 0
                    for k in range(n, 0, -1):
                        if p_prev[-k:] == p_curr[:k]:
                            max_k = k
                            break
                    word.extend(p_curr[max_k:])
                best_word = word
                return True
            
            # Determine candidates
            # Candidates are permutations that can follow current_perm.
            # They must share a prefix with current_perm's suffix.
            # We can choose overlap k from n-1 down to 1.
            
            # To avoid huge branching, we limit the overlap choices.
            # Usually, optimal superpermutations use overlaps of n-1 and n-2.
            # Let's try k = n-1 and k = n-2 first.
            
            candidates = []
            
            for k in [n-1, n-2]: # Try high overlaps first to minimize length
                prefix = current_perm[-k:]
                # Find all perms starting with prefix
                # We can precompute a dict: prefix -> list of perms
                # But n=7, k=6, prefix is 6 digits.
                # There is only 1 perm for k=n-1.
                # For k=n-2, there are 2 perms.
                
                # Actually, let's just generate them on the fly or precompute.
                # Precomputing all perms by prefix of length n-1 and n-2 is efficient.
                
            # Let's precompute adjacency lists for k=n-1 and k=n-2.
            
        # Precomputation
        # adj[n-1][suffix] = [perm]
        # adj[n-2][suffix] = [perm1, perm2]
        
        adj_n1 = {}
        adj_n2 = {}
        
        for p in all_perms:
            # k = n-1
            suffix = tuple(p[1:]) # No, suffix of PREVIOUS is prefix of CURRENT.
            # If current is P, next Q has Q[:k] == P[-k:]
            # So for Q, the relevant part is Q[:k].
            # We want to find Q given P.
            # So we build a map: P[-k:] -> Q.
            
            # For k=n-1:
            key_n1 = tuple(p[:-1]) # This is the prefix of P, not suffix.
            # Wait. P = p1...pn. Suffix is p2...pn.
            # Q must start with p2...pn.
            # So key is p[1:].
            key_n1 = tuple(p[1:])
            if key_n1 not in adj_n1:
                adj_n1[key_n1] = []
            adj_n1[key_n1].append(p)
            
            # For k=n-2:
            key_n2 = tuple(p[2:])
            if key_n2 not in adj_n2:
                adj_n2[key_n2] = []
            adj_n2[key_n2].append(p)
            
        # Note: adj_n1 maps a suffix of length n-1 to the UNIQUE permutation starting with it.
        # adj_n2 maps a suffix of length n-2 to the 2 permutations starting with it.
        
        # Let's run the DFS
        
        # We need to track visited permutations.
        # Using a set is O(1) lookup.
        
        # To handle backtracking efficiently, we pass `visited` as a set and copy it?
        # Copying a set of 5040 items 5040 times is 25M ops. Might be slow in Python.
        # Alternative: use a global set and add/remove on recursion.
        
        global_visited = set()
        
        def dfs_opt(current_perm, depth):
            nonlocal best_word
            
            if time.time() - start_time > time_limit:
                return False
            
            if depth == total_perms:
                # Reconstruct
                # We need to store the path.
                # Storing the path of 5040 tuples is fine.
                pass
            
            # Generate candidates
            # 1. k = n-1
            key_n1 = tuple(current_perm[1:])
            cand_n1 = adj_n1.get(key_n1, [])
            
            # 2. k = n-2
            key_n2 = tuple(current_perm[2:])
            cand_n2 = adj_n2.get(key_n2, [])
            
            # Combine and filter visited
            candidates = []
            for q in cand_n1:
                if q not in global_visited:
                    candidates.append((q, n-1))
            for q in cand_n2:
                if q not in global_visited:
                    candidates.append((q, n-2))
            
            # If no candidates, dead end
            if not candidates:
                return False
            
            # Heuristic: Sort candidates
            # Prefer those that lead to more options?
            # Or just random?
            random.shuffle(candidates)
            
            # Try each candidate
            for q, k in candidates:
                global_visited.add(q)
                path.append(q)
                
                if dfs_opt(q, depth + 1):
                    return True
                
                # Backtrack
                path.pop()
                global_visited.remove(q)
                
            return False
        
        # Initialize
        global_visited = set()
        start_perm = tuple(random.sample(range(1, n + 1), n))
        global_visited.add(start_perm)
        path = [start_perm]
        
        # Run DFS
        # This might be too slow if it explores too much.
        # We add a random restart mechanism.
        
        found = False
        for restart in range(100):
            if time.time() - start_time > time_limit:
                break
                
            # Reset state
            global_visited = set()
            start_perm = tuple(random.sample(range(1, n + 1), n))
            global_visited.add(start_perm)
            path = [start_perm]
            
            # Limit recursion depth or use iterative?
            # Python recursion limit is ~1000. We need 5040.
            # We must increase recursion limit or use iterative DFS.
            
        # The recursive approach will hit recursion limit.
        # We need to implement iterative DFS or increase limit.
        # sys.setrecursionlimit(10000)
        
        sys.setrecursionlimit(10000)
        
        # Let's try the recursive one with the limit increased.
        
        found = False
        for restart in range(50):
            if time.time() - start_time > time_limit:
                break
            
            global_visited = set()
            start_perm = tuple(random.sample(range(1, n + 1), n))
            global_visited.add(start_perm)
            path = [start_perm]
            
            # We need to modify dfs_opt to use the non-local path and visited
            # and return True/False.
            
            # The previous dfs_opt definition was nested and used nonlocal.
            # Let's restructure to be cleaner.
            
            def dfs_iter(current, depth):
                # This is a placeholder, we will use the recursive one defined below
                pass
            
            # Actually, let's just use the recursive one defined in the closure.
            # But it needs to be defined inside the restart loop or use nonlocal properly.
            
            # Let's define a helper that takes state and returns result.
            
            def search():
                nonlocal path, global_visited
                
                def dfs(current, depth):
                    if time.time() - start_time > time_limit:
                        return False
                    
                    if depth == total_perms:
                        return True
                    
                    key_n1 = tuple(current[1:])
                    cand_n1 = adj_n1.get(key_n1, [])
                    
                    key_n2 = tuple(current[2:])
                    cand_n2 = adj_n2.get(key_n2, [])
                    
                    candidates = []
                    for q in cand_n1:
                        if q not in global_visited:
                            candidates.append(q)
                    for q in cand_n2:
                        if q not in global_visited:
                            candidates.append(q)
                            
                    if not candidates:
                        return False
                        
                    random.shuffle(candidates)
                    
                    for q in candidates:
                        global_visited.add(q)
                        path.append(q)
                        
                        if dfs(q, depth + 1):
                            return True
                        
                        path.pop()
                        global_visited.remove(q)
                        
                    return False
                
                return dfs(start_perm, 1)
            
            if search():
                # Reconstruct word
                word = list(path[0])
                for i in range(1, len(path)):
                    p_prev = path[i-1]
                    p_curr = path[i]
                    max_k = 0
                    for k in range(n, 0, -1):
                        if p_prev[-k:] == p_curr[:k]:
                            max_k = k
                            break
                    word.extend(p_curr[max_k:])
                
                best_word = word
                found = True
                break
                
        if found and best_word:
            return "".join(map(str, best_word))
        
        # Fallback to standard construction if search fails
        def get_standard_superperm(n):
            word = [1]
            for m in range(2, n + 1):
                perms, seen = [], set()
                for i in range(len(word) - m + 2):
                    window = tuple(word[i:i + m - 1])
                    if len(set(window)) == m - 1 and window not in seen:
                        seen.add(window)
                        perms.append(window)
                new = []
                for p in perms:
                    block = list(p) + [m] + list(p)
                    overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                    new += block[overlap:]
                word = new
            return word
            
        std_word = get_standard_superperm(n)
        return "".join(map(str, std_word))

