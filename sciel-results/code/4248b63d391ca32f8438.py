import random
import time
import math

def construct():
    d = 7
    n_vertices = 1 << d
    all_edges = []
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y))

    edge_index = {e: i for i, e in enumerate(all_edges)}
    num_edges = len(all_edges)  # 448

    # Precompute for each edge, the list of C4s it belongs to.
    edge_c4s = [[] for _ in range(num_edges)]
    for (u, v) in all_edges:
        diff = u ^ v
        i = diff.bit_length() - 1
        ei = edge_index[(u, v)]
        for j in range(d):
            if j == i:
                continue
            u_prime = u ^ (1 << j)
            v_prime = v ^ (1 << j)
            e_opp = (min(u_prime, v_prime), max(u_prime, v_prime))
            e_side1 = (min(u, u_prime), max(u, u_prime))
            e_side2 = (min(v, v_prime), max(v, v_prime))
            a = edge_index[e_opp]
            b = edge_index[e_side1]
            c = edge_index[e_side2]
            edge_c4s[ei].append((a, b, c))

    def is_valid_add(ei, S):
        for (a, b, c) in edge_c4s[ei]:
            if a in S and b in S and c in S:
                return False
        return True

    def greedy_construct(order):
        S = set()
        for ei in order:
            if is_valid_add(ei, S):
                S.add(ei)
        return S

    def local_search_sa(S, max_time=5.0):
        S = set(S)
        current_size = len(S)
        best_S = set(S)
        best_size = current_size
        
        T = 1.0
        T_min = 0.01
        cooling = 0.9995
        start = time.time()
        
        while time.time() - start < max_time:
            # Move generation
            # 1. Add random valid edge
            # 2. Swap: remove one from S, add one not in S
            # 3. Remove random edge (to escape)
            
            r = random.random()
            
            if r < 0.4:
                # Try to add
                candidates = [ei for ei in range(num_edges) if ei not in S]
                if not candidates:
                    continue
                ei = random.choice(candidates)
                if is_valid_add(ei, S):
                    S.add(ei)
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_S = set(S)
            elif r < 0.8:
                # Try swap
                candidates_out = [ei for ei in range(num_edges) if ei not in S]
                if not candidates_out:
                    continue
                ei = random.choice(candidates_out)
                
                # Find blocking C4s
                blocking = []
                for (a, b, c) in edge_c4s[ei]:
                    if a in S and b in S and c in S:
                        blocking.append((a, b, c))
                
                if not blocking:
                    S.add(ei)
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_S = set(S)
                else:
                    # Pick random edge from a random blocking C4 to remove
                    c4 = random.choice(blocking)
                    rem = random.choice(c4)
                    
                    # Check validity of swap
                    # We remove rem, add ei
                    # Is ei valid in S \ {rem}?
                    # We need to check if any OTHER C4 containing ei is fully in S \ {rem}.
                    # Since rem was in *one* blocking C4, other C4s might still be blocked.
                    # So we must call is_valid_add(ei, S \ {rem})
                    new_S = set(S)
                    new_S.remove(rem)
                    if is_valid_add(ei, new_S):
                        new_S.add(ei)
                        # Size is same
                        # Accept neutral move?
                        # In SA, we accept neutral moves with prob 1 usually, or to allow wandering.
                        # Let's accept it.
                        S = new_S
                        # current_size unchanged
            else:
                # Remove random edge (bad move)
                if S:
                    rem = random.choice(list(S))
                    S.remove(rem)
                    current_size -= 1
                    # Delta E = +1 (cost increases by 1)
                    # Accept with prob exp(-1/T)
                    if random.random() < math.exp(-1.0 / T):
                        # Accept
                        pass
                    else:
                        # Reject
                        S.add(rem)
                        current_size += 1
            
            T *= cooling
            if T < T_min:
                T = T_min
                # If we hit min T, we can restart T or just continue
                # Continuing at min T is a random walk.
        
        return best_S

    best_S = set()
    best_size = 0
    
    time_limit = 110.0
    start_time = time.time()
    
    # ILS Loop
    while time.time() - start_time < time_limit:
        # 1. Greedy
        order = list(range(num_edges))
        random.shuffle(order)
        S = greedy_construct(order)
        
        # 2. Local Search
        S = local_search_sa(S, max_time=2.0)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
            if best_size >= 308:
                break
        
        # 3. Perturb (for next iteration's start? No, we start fresh with Greedy)
        # ILS usually perturbs the *current* best or the *current* solution.
        # Starting fresh with Greedy is a form of restart.
        # To make it ILS, we could perturb best_S and run SA on that.
        # Let's try perturbing best_S.
        if best_S and time.time() - start_time < time_limit - 5.0:
            # Perturb best_S
            S_pert = set(best_S)
            # Remove 5 random edges
            to_remove = random.sample(list(S_pert), min(5, len(S_pert)))
            for r in to_remove:
                S_pert.remove(r)
            # Re-add greedily
            # We can just run SA on S_pert
            S_pert = local_search_sa(S_pert, max_time=2.0)
            if len(S_pert) > best_size:
                best_size = len(S_pert)
                best_S = set(S_pert)
                if best_size >= 308:
                    break

    return [[all_edges[i][0], all_edges[i][1]] for i in best_S]
