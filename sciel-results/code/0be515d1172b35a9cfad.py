import numpy as np
import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11

    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)
    
    # Precompute masks for intersection checks
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)
    
    # Build Conflict Matrix
    # conflict[i, j] is True if supports[i] and supports[j] intersect in >= 3 elements
    conflict = np.zeros((N_S, N_S), dtype=bool)
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = support_masks[i] & support_masks[j]
            if inter.bit_count() >= 3:
                conflict[i, j] = True
                conflict[j, i] = True

    # Adjacency list for faster iteration in some contexts, but we will use matrix ops
    # adj[i] is a boolean mask of neighbors
    adj = conflict # conflict[i] is already a boolean array of length N_S

    rng = np.random.RandomState(42)

    def greedy_fill(x, blocked):
        """
        Greedily add any possible node.
        Returns updated x and blocked.
        """
        changed = True
        while changed:
            changed = False
            # Find candidates: not in x, not blocked
            candidates = np.where((~x) & (~blocked))[0]
            if len(candidates) == 0:
                break
            
            # Pick one (randomly to break symmetry, or first one for determinism in polish)
            # For greedy fill, we can pick all compatible ones? No, adding one blocks others.
            # We pick one at a time to maximize packing in a greedy sense?
            # Actually, if we pick all candidates at once, they might conflict with each other.
            # So we must pick one, update blocked, then repeat.
            
            # Optimization: If we pick a random candidate, we update blocked.
            v = candidates[rng.randint(len(candidates))]
            x[v] = True
            # Update blocked: any neighbor of v is now blocked
            blocked = np.logical_or(blocked, adj[v])
            changed = True
        return x, blocked

    def local_search(x_init, max_iter, time_limit):
        """
        Simulated Annealing / Random Walk on Independent Set.
        """
        x = x_init.copy()
        # Initialize blocked
        blocked = np.zeros(N_S, dtype=bool)
        # Any node conflicting with any selected node is blocked
        # blocked = np.any(conflict[np.where(x)[0]], axis=0)
        # But we also need to consider that selected nodes are "occupied"
        # Let's just recompute blocked from scratch for safety at start of LS
        in_set = np.where(x)[0]
        if len(in_set) > 0:
            blocked = np.any(conflict[in_set], axis=0)
        # Nodes in the set are effectively blocked from being added again, 
        # but our "add" logic checks ~x.
        
        current_score = int(np.sum(x))
        best_x = x.copy()
        best_score = current_score
        
        # We don't strictly need T if we accept all valid moves that don't decrease score,
        # but sometimes we need to decrease score to escape local minima.
        # However, for MIS, "valid move" usually means maintaining independence.
        # We can use:
        # 1. Add a node (Score +1) - Always accept.
        # 2. Remove a node (Score -1) - Accept with prob exp(-1/T) or if it enables a better swap.
        # 3. Swap (Remove 1, Add 1) (Score 0) - Accept if it leads to better future? 
        #    Actually, a swap that keeps score same might open up space for additions.
        
        # Strategy:
        # - Try to Add. If success, accept.
        # - If cannot Add, try to Swap (Remove u, Add v). If success, accept.
        # - If cannot Swap, try to Remove (to unblock).
        
        T = 5.0
        cooling = 0.999
        
        for it in range(max_iter):
            if time.time() > time_limit:
                break
                
            r = rng.random()
            
            # 1. Try to Add
            candidates = np.where((~x) & (~blocked))[0]
            if len(candidates) > 0:
                v = candidates[rng.randint(len(candidates))]
                x[v] = True
                blocked = np.logical_or(blocked, adj[v])
                current_score += 1
                if current_score > best_score:
                    best_score = current_score
                    best_x = x.copy()
                continue # Loop again, try to add more
            
            # Cannot add. Try Swap or Remove.
            
            if r < 0.7:
                # Swap: Remove u, Add v
                in_set = np.where(x)[0]
                if len(in_set) == 0:
                    continue
                u = in_set[rng.randint(len(in_set))]
                
                # Temporarily remove u
                x_temp = x.copy()
                x_temp[u] = False
                # Recompute blocked for x_temp
                in_set_temp = np.where(x_temp)[0]
                if len(in_set_temp) > 0:
                    blocked_temp = np.any(conflict[in_set_temp], axis=0)
                else:
                    blocked_temp = np.zeros(N_S, dtype=bool)
                
                # Find candidates that were blocked by u but are now free
                # Candidates must not be in x_temp and not in blocked_temp
                candidates_swap = np.where((~x_temp) & (~blocked_temp))[0]
                
                if len(candidates_swap) > 0:
                    v = candidates_swap[rng.randint(len(candidates_swap))]
                    # Apply swap
                    x = x_temp
                    x[v] = True
                    # Update blocked
                    blocked = np.logical_or(blocked_temp, adj[v])
                    # Score remains same (removed 1, added 1)
                    # But we might have unlocked potential for future adds
                    # We don't update best_score here as it's not an improvement yet
                    # But we keep the state.
                    continue
            
            else:
                # Remove: Just remove a node to unblock others
                in_set = np.where(x)[0]
                if len(in_set) == 0:
                    continue
                u = in_set[rng.randint(len(in_set))]
                x[u] = False
                # Recompute blocked
                in_set_new = np.where(x)[0]
                if len(in_set_new) > 0:
                    blocked = np.any(conflict[in_set_new], axis=0)
                else:
                    blocked = np.zeros(N_S, dtype=bool)
                current_score -= 1
                # Accept this move (it's a descent, but allows escape)
                # We don't update best_x here.
                continue
                
            T *= cooling

        return best_x

    best_support_set = set()
    best_size = 0
    
    num_restarts = 80
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Initialize x0
        x0 = np.zeros(N_S, dtype=bool)
        
        if r % 2 == 0:
            # Random Greedy Init
            order = rng.permutation(N_S)
            blocked = np.zeros(N_S, dtype=bool)
            for v in order:
                if not x0[v] and not blocked[v]:
                    x0[v] = True
                    blocked = np.logical_or(blocked, adj[v])
        else:
            # Greedy Init with different heuristic (e.g. highest degree first? or random)
            # Let's try "Least Conflicts" or just random permutation with different seed
            rng_local = np.random.RandomState(42 + r * 100)
            order = rng_local.permutation(N_S)
            blocked = np.zeros(N_S, dtype=bool)
            for v in order:
                if not x0[v] and not blocked[v]:
                    x0[v] = True
                    blocked = np.logical_or(blocked, adj[v])
                    
        # Greedy Fill to ensure maximality
        blocked_init = np.zeros(N_S, dtype=bool)
        in_set_init = np.where(x0)[0]
        if len(in_set_init) > 0:
            blocked_init = np.any(conflict[in_set_init], axis=0)
        x0, blocked_init = greedy_fill(x0, blocked_init)
        
        remaining_time = deadline - time.time()
        # Allocate time based on remaining budget
        # 10s margin for polish and construction
        max_iter = int(max(1000, (remaining_time - 10.0) * 100)) # ~100 iterations per second? 
        # Actually, let's just run until time limit
        time_limit = deadline - 5.0
        
        x_best = local_search(x0, max_iter, time_limit)
        size = int(np.sum(x_best))
        
        if size > best_size:
            best_size = size
            best_support_set = set(np.where(x_best)[0].tolist())
            
    # Final Polish: Ensure no node can be added
    if best_support_set:
        polish_x = set(best_support_set)
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            for v in range(N_S):
                if v in polish_x:
                    continue
                # Check if v conflicts with any in polish_x
                # conflict[v] is a boolean array. 
                # We need to check if any index in polish_x has conflict[v][u] == True
                # This is slow in Python loop. 
                # Better: maintain a bitset or use numpy.
                # Since N_S is 330, and polish_x is ~15-20, this loop is 330*20 = 6600 ops per pass.
                # Passes are few.
                compatible = True
                for u in polish_x:
                    if conflict[v, u]:
                        compatible = False
                        break
                if compatible:
                    polish_x.add(v)
                    changed = True
        if len(polish_x) > best_size:
            best_size = len(polish_x)
            best_support_set = polish_x

    # Construct Vectors
    vectors = []
    
    # Type A: 22 vectors
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors.append(v2)
        
    # Type D: 16 vectors per support
    for s_idx in best_support_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors.append(v)
            
    if not vectors:
        return np.array([], dtype=np.int8).reshape(0, D)
        
    return np.array(vectors, dtype=np.int8)
