"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute LCM for all pairs
    lcm = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            lcm[i][j] = i * j // gcd(i, j)
            lcm[j][i] = lcm[i][j]
            
    # Find all bad triples
    bad_triples = set()
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            lab = lcm[a][b]
            for c in range(b+1, N+1):
                if lcm[a][c] == lab and lcm[b][c] == lab:
                    bad_triples.add((a, b, c))
    
    # Precompute conflict pairs for each element
    # conflict_pairs[x] = list of (y, z) such that {x, y, z} is a bad triple
    conflict_pairs = [[] for _ in range(N+1)]
    degrees = [0] * (N + 1)
    for a, b, c in bad_triples:
        conflict_pairs[a].append((b, c))
        conflict_pairs[b].append((a, c))
        conflict_pairs[c].append((a, b))
        degrees[a] += 1
        degrees[b] += 1
        degrees[c] += 1
        
    # Convert to sets for faster lookup if needed, but lists are fine for iteration
    # Actually, for can_add, we iterate over conflict_pairs[x] and check membership in S.
    
    def can_add(x, S):
        """Check if x can be added to set S without creating a bad triple."""
        for y, z in conflict_pairs[x]:
            if y in S and z in S:
                return False
        return True

    def is_valid(S):
        """Verify that S contains no bad triple."""
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    a, b, c = S_list[i], S_list[j], S_list[k]
                    # Sort to match bad_triples format
                    if a > b: a, b = b, a
                    if b > c: b, c = c, b
                    if a > b: a, b = b, a
                    if (a, b, c) in bad_triples:
                        return False
        return True

    def greedy_fill(order):
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        return S

    def polish(S):
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for x in elements:
                if x not in S and can_add(x, S):
                    S.add(x)
                    changed = True
        return S

    def simulated_annealing(S, max_iter=50000):
        """
        SA with 1-remove, 1-add moves.
        """
        S = set(S)
        current_size = len(S)
        best_local = set(S)
        best_local_size = current_size
        
        start_temp = 1.0
        end_temp = 0.01
        # Adjust cooling rate based on max_iter
        cooling = (end_temp / start_temp) ** (1.0 / max(max_iter, 1))
        temp = start_temp
        
        complement = [x for x in elements if x not in S]
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            if current_size == 0:
                break
                
            # 1. Pick random element to remove
            to_remove = random.choice(list(S))
            S.remove(to_remove)
            complement.append(to_remove)
            
            # 2. Try to add a random element from complement
            # To speed up, we can try a few random candidates
            added = None
            # Try up to 5 random candidates
            for _ in range(min(5, len(complement))):
                idx = random.randint(0, len(complement) - 1)
                cand = complement[idx]
                if can_add(cand, S):
                    S.add(cand)
                    complement.pop(idx)
                    added = cand
                    break
            
            if added is None:
                # Revert
                S.add(to_remove)
                complement.remove(to_remove)
            else:
                new_size = len(S)
                delta = new_size - current_size
                # Accept if better, or with probability exp(-delta/T) if worse
                if delta > 0:
                    current_size = new_size
                    if current_size > best_local_size:
                        best_local_size = current_size
                        best_local = set(S)
                else:
                    # delta is 0 or -1
                    if random.random() < pow(temp, 1.0):
                        current_size = new_size
                        if current_size > best_local_size:
                            best_local_size = current_size
                            best_local = set(S)
                    else:
                        # Revert
                        S.discard(added)
                        S.add(to_remove)
                        complement.remove(to_remove)
                        complement.append(added)
            
            temp *= cooling
            
        return best_local

    best_set = set()
    best_size = 0
    
    # 1. Greedy strategies
    strategies = []
    # Natural
    strategies.append(elements[:])
    # Reverse
    strategies.append(elements[::-1])
    # By degree asc
    strategies.append(sorted(elements, key=lambda x: degrees[x]))
    # By degree desc
    strategies.append(sorted(elements, key=lambda x: -degrees[x]))
    
    # Random shuffles
    for _ in range(10):
        rand_order = elements[:]
        random.shuffle(rand_order)
        strategies.append(rand_order)
        
    for order in strategies:
        if time.time() > deadline - 30:
            break
        S = greedy_fill(order)
        S = polish(S)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
    # 2. Simulated Annealing from best greedy
    if time.time() < deadline - 20:
        S_sa = simulated_annealing(set(best_set), max_iter=100000)
        S_sa = polish(S_sa)
        if len(S_sa) > best_size:
            best_size = len(S_sa)
            best_set = set(S_sa)
            
    # 3. More SA restarts if time permits
    remaining_time = deadline - 15 - time.time()
    if remaining_time > 5:
        # Run a few more SA instances from different starting points
        for i in range(5):
            if time.time() > deadline - 15:
                break
            # Start from a random valid set
            rand_order = elements[:]
            random.shuffle(rand_order)
            S_start = greedy_fill(rand_order)
            S_start = polish(S_start)
            
            # Run SA
            S_sa = simulated_annealing(set(S_start), max_iter=50000)
            S_sa = polish(S_sa)
            
            if len(S_sa) > best_size:
                best_size = len(S_sa)
                best_set = set(S_sa)
                
    # Final polish
    if time.time() < deadline - 2:
        best_set = polish(best_set)
        
    # Final verification
    if not is_valid(best_set):
        best_set = {1}
        
    return list(best_set)
# EVOLVE-BLOCK-END
