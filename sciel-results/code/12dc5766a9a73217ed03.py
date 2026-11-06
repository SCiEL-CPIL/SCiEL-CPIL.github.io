import time
import random
import math
from itertools import combinations

def construct():
    deadline = time.time() + 280
    
    # 1. Generate Paley Graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    current_n = q
    current_adj = adj

    def get_k4s_i4s(adj, n):
        k4s = []
        i4s = []
        for subset in combinations(range(n), 4):
            is_clique = True
            is_indep = True
            for i in range(4):
                u = subset[i]
                for j in range(i + 1, 4):
                    v = subset[j]
                    if (adj[u] >> v) & 1:
                        is_indep = False
                    else:
                        is_clique = False
                    if not is_clique and not is_indep:
                        break
                if not is_clique and not is_indep:
                    break
            if is_clique:
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                k4s.append(mask)
            elif is_indep:
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                i4s.append(mask)
        return k4s, i4s

    def find_valid_neighbor(adj, n, k4s, i4s, deadline):
        """
        Use Simulated Annealing to find a valid neighbor set S for a new vertex n.
        A set S is valid if:
        1. No K4 in G is a subset of S.
        2. No I4 in G is a subset of V \ S.
        """
        full_mask = (1 << n) - 1
        
        # Precompute for faster delta calculation?
        # For now, just calculate Energy from scratch if lists are small.
        # Number of K4s and I4s in 37-vertex graph is roughly < 1000.
        # So iterating through them is fast.
        
        def energy(S):
            bad_k = 0
            bad_i = 0
            V_minus_S = full_mask ^ S
            for km in k4s:
                if (km & S) == km:
                    bad_k += 1
            for im in i4s:
                if (im & V_minus_S) == im:
                    bad_i += 1
            return bad_k + bad_i

        # Initialize S randomly
        S = 0
        for v in range(n):
            if random.random() < 0.5:
                S |= (1 << v)
        
        current_energy = energy(S)
        
        T = 10.0
        T_min = 0.01
        cooling = 0.999
        
        steps = 0
        max_steps = 50000
        
        best_S = S
        best_energy = current_energy
        
        while steps < max_steps and time.time() < deadline - 5:
            # Pick a random vertex to flip
            v = random.randint(0, n - 1)
            new_S = S ^ (1 << v)
            
            new_energy = energy(new_S)
            
            delta = new_energy - current_energy
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                S = new_S
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_S = S
                    if best_energy == 0:
                        break
            
            T *= cooling
            steps += 1
            
            # If we are stuck, restart with higher T?
            if steps % 5000 == 0 and current_energy > 0:
                # Reheat slightly
                T = max(T, 5.0)
                
        if best_energy == 0:
            # Convert mask to set
            S_set = set()
            temp = best_S
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                S_set.add(v)
                temp ^= lsb
            return S_set
        return None

    # Verify start
    def verify(adj, n):
        for subset in combinations(range(n), 5):
            is_clique = True
            is_indep = True
            for i in range(5):
                u = subset[i]
                for j in range(i + 1, 5):
                    v = subset[j]
                    if (adj[u] >> v) & 1:
                        is_indep = False
                    else:
                        is_clique = False
                    if not is_clique and not is_indep:
                        break
                if not is_clique and not is_indep:
                    break
            if is_clique or is_indep:
                return False
        return True

    if not verify(current_adj, current_n):
        edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
        return q, edges

    k4s, i4s = get_k4s_i4s(current_adj, current_n)

    # Try to grow
    for target_n in range(current_n + 1, 43):
        if time.time() > deadline - 10:
            break
            
        success = False
        for attempt in range(5): # Fewer attempts, SA is longer
            if time.time() > deadline - 10:
                break
            S = find_valid_neighbor(current_adj, current_n, k4s, i4s, deadline)
            if S is not None:
                # Add vertex
                new_idx = current_n
                current_adj.append(0)
                for v in S:
                    current_adj[v] |= (1 << new_idx)
                    current_adj[new_idx] |= (1 << v)
                current_n += 1
                
                # Recompute K4s and I4s
                k4s, i4s = get_k4s_i4s(current_adj, current_n)
                success = True
                break
        
        if not success:
            break

    # Final check
    if verify(current_adj, current_n):
        edges = []
        for i in range(current_n):
            for j in range(i + 1, current_n):
                if (current_adj[i] >> j) & 1:
                    edges.append((i, j))
        return current_n, edges

    # Fallback
    edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    return q, edges
