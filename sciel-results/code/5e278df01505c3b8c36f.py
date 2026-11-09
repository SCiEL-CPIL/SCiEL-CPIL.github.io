import time
import random
import math

def construct():
    t0 = time.time()
    DEADLINE = 88.0
    n = 180
    
    # Precompute triples (a,b,c) such that 1/a = 1/b + 1/c
    # a < b < c
    # (b-a)(c-a) = a^2
    triples = []
    for a in range(1, n + 1):
        a2 = a * a
        d = 1
        while d * d <= a2:
            if a2 % d == 0:
                e = a2 // d
                if d < e:
                    b = a + d
                    c = a + e
                    if c <= n:
                        triples.append((a, b, c))
            d += 1
    
    # Build conflict lists: for each vertex i, list of (j,k) pairs that form a triple with i
    conflicts = [[] for _ in range(n)]
    for (a, b, c) in triples:
        conflicts[a - 1].append((b - 1, c - 1))
        conflicts[b - 1].append((a - 1, c - 1))
        conflicts[c - 1].append((a - 1, b - 1))
    
    best_set = None
    best_size = 0
    
    def try_add(S, a):
        """Try to add element a (0-indexed) to set S. Returns True if successful."""
        for (u, v) in conflicts[a]:
            if S[u] and S[v]:
                return False
        S[a] = True
        return True
    
    def try_remove(S, a):
        """Remove element a from set S."""
        S[a] = False
    
    def greedy_complete(S):
        """Try to add all possible elements to S in order of least conflicts."""
        cur_size = sum(S)
        # Compute conflict count for each missing element
        missing = [i for i in range(n) if not S[i]]
        # Sort by number of conflicts (fewer first)
        missing.sort(key=lambda x: len(conflicts[x]))
        for a in missing:
            if try_add(S, a):
                cur_size += 1
        return cur_size
    
    def local_search_improved(S, max_passes=20):
        """Improved local search: try swaps (remove one, add another) and additions."""
        cur_size = sum(S)
        
        for _ in range(max_passes):
            if time.time() - t0 > DEADLINE - 3:
                break
            improved = False
            
            # Phase 1: Try to add any missing element
            missing = [i for i in range(n) if not S[i]]
            random.shuffle(missing)
            for a in missing:
                if try_add(S, a):
                    cur_size += 1
                    improved = True
            
            # Phase 2: Try 1-for-1 swaps (remove present, add missing)
            present = [i for i in range(n) if S[i]]
            missing = [i for i in range(n) if not S[i]]
            random.shuffle(present)
            random.shuffle(missing)
            
            for a in present:
                # Temporarily remove a
                S[a] = False
                cur_size -= 1
                added_count = 0
                added_elements = []
                for b in missing:
                    if try_add(S, b):
                        added_count += 1
                        added_elements.append(b)
                # Keep the swap if it's beneficial (we added at least 1)
                if added_count >= 1:
                    cur_size += added_count
                    improved = True
                    missing = [x for x in missing if x not in set(added_elements)]
                else:
                    # Restore a
                    S[a] = True
                    cur_size += 1
            
            if not improved:
                break
        
        return cur_size
    
    def simulated_annealing(S, t_start=1.0, t_end=0.01, max_iters=5000):
        """Simulated annealing to escape local optima."""
        cur_size = sum(S)
        T = t_start
        cooling = (t_end / t_start) ** (1.0 / max_iters)
        
        for it in range(max_iters):
            if time.time() - t0 > DEADLINE - 3:
                break
            T *= cooling
            
            # Choose a random move: flip a bit
            a = random.randint(0, n - 1)
            old_state = S[a]
            
            # Compute delta in size
            if old_state:
                # Removing a: size decreases by 1
                S[a] = False
                delta = -1
            else:
                # Adding a: check if valid
                can_add = True
                for (u, v) in conflicts[a]:
                    if S[u] and S[v]:
                        can_add = False
                        break
                if can_add:
                    S[a] = True
                    delta = 1
                else:
                    # Can't add, so reject this move
                    continue
            
            # Accept with probability exp(delta/T) if delta < 0
            if delta >= 0:
                pass  # Always accept improvements
            else:
                if random.random() > math.exp(delta / max(T, 0.001)):
                    # Reject: undo the move
                    S[a] = old_state
                    cur_size -= delta  # Restore size
                    continue
            
            cur_size += delta
        
        # Final greedy completion
        cur_size = greedy_complete(S)
        return cur_size
    
    # Strategy 1: Suffix + Greedy Add (keep existing approach)
    for K in range(1, n + 1):
        if time.time() - t0 > DEADLINE - 15:
            break
            
        S = [False] * n
        for i in range(K - 1, n):
            S[i] = True
            
        # Check if this suffix is valid
        valid_suffix = True
        for (a, b, c) in triples:
            if a >= K and b >= K and c >= K:
                valid_suffix = False
                break
        
        if not valid_suffix:
            continue
            
        cur_size = n - K + 1
        if cur_size > best_size:
            best_size = cur_size
            best_set = S[:]
            
        # Greedy add smaller numbers
        for a in range(K - 1, 0, -1):
            if S[a - 1]:
                continue
            can_add = True
            for (u, v) in conflicts[a - 1]:
                if S[u] and S[v]:
                    can_add = False
                    break
                    
            if can_add:
                S[a - 1] = True
                cur_size += 1
                if cur_size > best_size:
                    best_size = cur_size
                    best_set = S[:]
                    
    # Strategy 2: Improved local search from best set
    if best_set is not None and time.time() - t0 < DEADLINE - 5:
        S2 = best_set[:]
        cur_size = local_search_improved(S2)
        if cur_size > best_size:
            best_size = cur_size
            best_set = S2[:]

    # Strategy 3: Multiple random restarts with improved local search
    if time.time() - t0 < DEADLINE - 10:
        for restart in range(40):
            if time.time() - t0 > DEADLINE - 5:
                break
            
            K = random.randint(20, 140)
            S = [False] * n
            for i in range(K - 1, n):
                S[i] = True
                
            valid = True
            for (a, b, c) in triples:
                if a >= K and b >= K and c >= K:
                    valid = False
                    break
            
            if not valid:
                continue
            
            cur_size = n - K + 1
            
            # Greedy add
            for a in range(K - 1, 0, -1):
                if S[a - 1]:
                    continue
                can_add = True
                for (u, v) in conflicts[a - 1]:
                    if S[u] and S[v]:
                        can_add = False
                        break
                if can_add:
                    S[a - 1] = True
                    cur_size += 1
            
            # Improved local search
            cur_size = local_search_improved(S)
            
            if cur_size > best_size:
                best_size = cur_size
                best_set = S[:]

    # Strategy 4: Simulated annealing from best set
    if best_set is not None and time.time() - t0 < DEADLINE - 8:
        S3 = best_set[:]
        cur_size = simulated_annealing(S3, t_start=1.0, t_end=0.01, max_iters=3000)
        if cur_size > best_size:
            best_size = cur_size
            best_set = S3[:]

    if best_set is None:
        best_set = [False] * n
        best_set[n-1] = True
        best_size = 1

    # Final polish
    if time.time() - t0 < DEADLINE - 2:
        S_final = best_set[:]
        cur_size = local_search_improved(S_final)
        if cur_size > best_size:
            best_size = cur_size
            best_set = S_final[:]

    out = [i + 1 for i in range(n) if best_set[i]]
    
    return out
