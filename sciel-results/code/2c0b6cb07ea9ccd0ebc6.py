"""Kissing number in dimension 11 - Energy-based local search."""
import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin

    # Generate W4 vectors
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    N4 = len(vectors_v4)

    # Conflict matrix
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    # Neighbor lists
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]

    rng = np.random.default_rng(42)

    def randomized_greedy():
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = np.array([len(n) for n in neighbors], dtype=np.int32)

        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]

            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False

            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)

        return in_set

    def energy_local_search(in_set, max_iter=2000):
        """
        Density-based local search:
        - Energy = sum of squared violations max(0, dot-2)^2
        - Allow temporary conflicts (energy > 0) to escape local optima
        - Moves: add, remove, swap, 2-for-1
        - After each move, if energy decreases or is 0, accept
        - Periodic greedy removal pass to restore validity
        """
        in_set = in_set.copy()
        
        # Track which vertices are in the set
        in_indices = np.where(in_set)[0]
        
        # Conflict count for each vertex: how many set members conflict with it
        cc = np.zeros(N4, dtype=np.int32)
        for i in in_indices:
            cc[neighbors[i]] += 1
        
        # Energy: sum of squared violations
        # For vertices in set, violations = cc[i] (number of conflicts with other set members)
        # But we need to count each pair once. Let's use: energy = sum over in_set of cc[i] / 2
        # Actually, let's just track the number of conflicting pairs
        # energy = number of pairs (i,j) both in set with conflicts[i,j]
        
        def compute_energy(indices):
            if len(indices) == 0:
                return 0
            sub_conf = conflicts[np.ix_(indices, indices)]
            return int(np.triu(sub_conf, 1).sum())
        
        current_energy = compute_energy(in_indices)
        current_size = len(in_indices)
        
        best_in_set = in_set.copy()
        best_size = current_size
        best_energy = current_energy
        
        for iteration in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            if len(in_indices) == 0:
                break
            
            # Try all possible moves and pick the best energy improvement
            best_move = None
            best_new_energy = current_energy
            
            # 1. Add moves: add vertex v not in set
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) > 0:
                # For each candidate, new energy = current_energy + cc[v] (conflicts with set)
                # But we also need to check if adding v creates new conflicts among set members? No,
                # adding v only adds conflicts between v and existing set members.
                # new_energy = current_energy + cc[v]
                # We want to minimize energy, so pick v with smallest cc[v]
                # But we also want to increase size, so among those with same energy, pick any
                min_cc = np.min(cc[not_in_set])
                # Only add if it doesn't increase energy too much, or if energy is 0
                # Actually, let's allow adds that don't increase energy
                add_cands = not_in_set[cc[not_in_set] <= min_cc]
                if len(add_cands) > 0:
                    v = add_cands[rng.integers(len(add_cands))]
                    new_energy = current_energy + cc[v]
                    if new_energy <= best_new_energy:
                        best_new_energy = new_energy
                        best_move = ('add', v)
            
            # 2. Remove moves: remove vertex v in set
            if len(in_indices) > 0:
                # Removing v reduces energy by cc[v] (conflicts between v and other set members)
                # But cc[v] counts conflicts with ALL set members, including v itself? No,
                # cc[v] = number of set members that conflict with v (excluding v since no self-conflict)
                # So removing v reduces energy by cc[v]
                # new_energy = current_energy - cc[v]
                # We want to maximize reduction, so pick v with largest cc[v]
                max_cc = np.max(cc[in_indices])
                if max_cc > 0:
                    rem_cands = in_indices[cc[in_indices] == max_cc]
                    v = rem_cands[rng.integers(len(rem_cands))]
                    new_energy = current_energy - cc[v]
                    if new_energy < best_new_energy:
                        best_new_energy = new_energy
                        best_move = ('remove', v)
            
            # 3. Swap moves: remove u in set, add v not in set
            # new_energy = current_energy - cc[u] + (cc[v] - conflicts[u,v])
            # where conflicts[u,v] is 1 if u and v conflict, else 0
            # We need cc[v] to be the conflict count of v with set members EXCLUDING u
            # cc[v] currently includes conflicts with u if u is in set and conflicts with v
            # So new_cc_v_after_removal = cc[v] - (1 if conflicts[u,v] else 0)
            # new_energy = current_energy - cc[u] + (cc[v] - (1 if conflicts[u,v] else 0))
            
            # This is expensive to do for all pairs. Let's sample.
            if len(in_indices) > 0 and len(not_in_set) > 0:
                n_swaps = min(50, len(in_indices) * len(not_in_set))
                for _ in range(n_swaps):
                    u = in_indices[rng.integers(len(in_indices))]
                    v = not_in_set[rng.integers(len(not_in_set))]
                    if conflicts[u, v]:
                        new_energy = current_energy - cc[u] + cc[v] - 1
                    else:
                        new_energy = current_energy - cc[u] + cc[v]
                    if new_energy < best_new_energy:
                        best_new_energy = new_energy
                        best_move = ('swap', u, v)
            
            # 4. 2-for-1 moves: remove u1, u2 in set, add v not in set
            # This can help escape local optima
            if len(in_indices) >= 2 and len(not_in_set) > 0:
                n_moves = min(20, len(in_indices) * len(not_in_set))
                for _ in range(n_moves):
                    u1_idx = rng.integers(len(in_indices))
                    u2_idx = rng.integers(len(in_indices))
                    if u1_idx == u2_idx:
                        continue
                    u1 = in_indices[u1_idx]
                    u2 = in_indices[u2_idx]
                    v = not_in_set[rng.integers(len(not_in_set))]
                    
                    # Compute new energy
                    # Remove u1, u2: energy decreases by cc[u1] + cc[u2] - (1 if conflicts[u1,u2] else 0)
                    # Add v: energy increases by (cc[v] - (1 if conflicts[u1,v] else 0) - (1 if conflicts[u2,v] else 0))
                    # But we need to be careful: cc[v] includes conflicts with u1 and u2 if they conflict with v
                    reduction = cc[u1] + cc[u2]
                    if conflicts[u1, u2]:
                        reduction -= 1
                    addition = cc[v]
                    if conflicts[u1, v]:
                        addition -= 1
                    if conflicts[u2, v]:
                        addition -= 1
                    new_energy = current_energy - reduction + addition
                    if new_energy < best_new_energy:
                        best_new_energy = new_energy
                        best_move = ('2for1', u1, u2, v)
            
            # Apply best move if it improves energy
            if best_move is not None and best_new_energy < current_energy:
                move_type = best_move[0]
                if move_type == 'add':
                    v = best_move[1]
                    in_set[v] = True
                    cc[neighbors[v]] += 1
                    in_indices = np.where(in_set)[0]
                elif move_type == 'remove':
                    v = best_move[1]
                    in_set[v] = False
                    cc[neighbors[v]] -= 1
                    in_indices = np.where(in_set)[0]
                elif move_type == 'swap':
                    u, v = best_move[1], best_move[2]
                    in_set[u] = False
                    cc[neighbors[u]] -= 1
                    in_set[v] = True
                    cc[neighbors[v]] += 1
                    in_indices = np.where(in_set)[0]
                elif move_type == '2for1':
                    u1, u2, v = best_move[1], best_move[2], best_move[3]
                    in_set[u1] = False
                    in_set[u2] = False
                    cc[neighbors[u1]] -= 1
                    cc[neighbors[u2]] -= 1
                    in_set[v] = True
                    cc[neighbors[v]] += 1
                    in_indices = np.where(in_set)[0]
                
                current_energy = best_new_energy
                current_size = len(in_indices)
                
                # Track best valid solution (energy == 0)
                if current_energy == 0 and current_size > best_size:
                    best_size = current_size
                    best_in_set = in_set.copy()
                    best_energy = 0
                
            else:
                # No improving move found. Try a perturbation: random swap to escape
                if len(in_indices) > 0 and len(not_in_set) > 0:
                    # Random swap
                    u = in_indices[rng.integers(len(in_indices))]
                    v = not_in_set[rng.integers(len(not_in_set))]
                    in_set[u] = False
                    cc[neighbors[u]] -= 1
                    in_set[v] = True
                    cc[neighbors[v]] += 1
                    in_indices = np.where(in_set)[0]
                    current_energy = compute_energy(in_indices)
                    current_size = len(in_indices)
                
                # Periodic greedy removal to restore validity
                if iteration % 100 == 50 and current_energy > 0:
                    # Greedily remove vertices to reduce energy to 0
                    while current_energy > 0 and time.time() < deadline - 10:
                        # Find vertex with highest conflict count
                        max_cc_idx = np.argmax(cc[in_indices])
                        v = in_indices[max_cc_idx]
                        if cc[v] == 0:
                            break
                        in_set[v] = False
                        cc[neighbors[v]] -= 1
                        in_indices = np.where(in_set)[0]
                        current_energy = compute_energy(in_indices)
                        current_size = len(in_indices)
                    
                    if current_energy == 0 and current_size > best_size:
                        best_size = current_size
                        best_in_set = in_set.copy()
        
        return best_in_set, best_size

    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)

    # Run multiple restarts
    n_restarts = 0
    max_restarts = 80

    while time.time() < deadline - 15 and n_restarts < max_restarts:
        n_restarts += 1

        in_set = randomized_greedy()
        in_set, cur_size = energy_local_search(in_set)

        if cur_size > best_size:
            best_size = cur_size
            best_in_set = in_set.copy()

        # Perturbation: Remove a few random elements from best set and re-optimize
        if n_restarts > 1 and time.time() < deadline - 20:
            perturbed = best_in_set.copy()
            indices = np.where(perturbed)[0]
            if len(indices) > 5:
                to_remove = rng.choice(indices, size=min(5, len(indices)//2), replace=False)
                perturbed[to_remove] = False
                perturbed, psize = energy_local_search(perturbed)
                if psize > best_size:
                    best_size = psize
                    best_in_set = perturbed.copy()

    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]

    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)

    final_vectors = np.vstack([result_v4, vectors_v1])
    return final_vectors
