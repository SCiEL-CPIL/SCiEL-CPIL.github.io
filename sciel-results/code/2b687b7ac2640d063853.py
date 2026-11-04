"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11

    # Type A: Vectors with one non-zero entry +/- 2
    # Norm = 2.
    type_a_vectors = []
    for i in range(D):
        type_a_vectors.append(np.array([2 if j == i else 0 for j in range(D)], dtype=np.int64))
        type_a_vectors.append(np.array([-2 if j == i else 0 for j in range(D)], dtype=np.int64))
        
    # Type D: Vectors with four non-zero entries +/- 1.
    # Norm = 2.
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)
    
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)
    
    # Precompute conflict lists: conflict if intersection >= 3
    conflict_lists = []
    for i in range(N_S):
        conf = []
        for j in range(N_S):
            if (support_masks[i] & support_masks[j]).bit_count() >= 3:
                conf.append(j)
        conflict_lists.append(conf)
    
    # Precompute compatibility as bitsets for fast operations
    # compat[i] is a bitset of nodes compatible with i
    compat_bits = []
    for i in range(N_S):
        bits = 0
        for j in range(N_S):
            if (support_masks[i] & support_masks[j]).bit_count() <= 2:
                bits |= (1 << j)
        compat_bits.append(bits)

    def greedy_init(seed_order=None):
        x = np.zeros(N_S, dtype=bool)
        blocked = np.zeros(N_S, dtype=bool)
        if seed_order is None:
            order = list(range(N_S))
            random.shuffle(order)
        else:
            order = seed_order
        for v in order:
            if not x[v] and not blocked[v]:
                x[v] = True
                for c in conflict_lists[v]:
                    blocked[c] = True
        return x

    def local_search_sa(x, max_iters=2000, T_start=10.0, T_end=0.1):
        """Simulated annealing on the support selection problem."""
        current_x = x.copy()
        current_size = int(np.sum(current_x))
        best_x = current_x.copy()
        best_size = current_size
        
        # Compute blocked status
        blocked = np.zeros(N_S, dtype=bool)
        for i in range(N_S):
            if current_x[i]:
                for c in conflict_lists[i]:
                    blocked[c] = True
        
        T = T_start
        cooling = (T_end / T_start) ** (1.0 / max(1, max_iters))
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            T *= cooling
            
            # Choose a random move: add, remove, or swap
            move_type = random.random()
            selected_indices = np.where(current_x)[0]
            unselected_indices = np.where(~current_x)[0]
            
            if len(selected_indices) == 0 or len(unselected_indices) == 0:
                break
                
            new_x = current_x.copy()
            new_size = current_size
            
            if move_type < 0.3 and len(unselected_indices) > 0:
                # Add a random unselected node
                j = unselected_indices[random.randint(0, len(unselected_indices)-1)]
                if not blocked[j]:
                    new_x[j] = True
                    new_size += 1
                else:
                    # Can't add, skip
                    continue
            elif move_type < 0.6 and len(selected_indices) > 0:
                # Remove a random selected node
                i = selected_indices[random.randint(0, len(selected_indices)-1)]
                new_x[i] = False
                new_size -= 1
            else:
                # Swap: remove i, add j
                i = selected_indices[random.randint(0, len(selected_indices)-1)]
                j = unselected_indices[random.randint(0, len(unselected_indices)-1)]
                new_x[i] = False
                new_x[j] = True
                new_size = current_size  # size unchanged
            
            # Validate the new solution
            valid = True
            for idx in range(N_S):
                if new_x[idx]:
                    for c in conflict_lists[idx]:
                        if new_x[c]:
                            valid = False
                            break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Accept or reject
            delta = new_size - current_size
            if delta > 0 or random.random() < np.exp(delta / max(T, 1e-9)):
                current_x = new_x
                current_size = new_size
                
                # Update blocked
                blocked = np.zeros(N_S, dtype=bool)
                for i in range(N_S):
                    if current_x[i]:
                        for c in conflict_lists[i]:
                            blocked[c] = True
                
                if current_size > best_size:
                    best_size = current_size
                    best_x = current_x.copy()
                    
        return best_x

    best_x = None
    best_size = 0
    
    # Run multiple restarts with different strategies
    n_restarts = 80
    for r in range(n_restarts):
        if time.time() > deadline - 15:
            break
            
        # Different initialization strategies
        if r % 3 == 0:
            x = greedy_init()
        elif r % 3 == 1:
            # Greedy with degree ordering (prefer low-degree nodes first)
            degrees = [len(conflict_lists[i]) for i in range(N_S)]
            order = sorted(range(N_S), key=lambda i: degrees[i])
            random.shuffle(order)  # slight randomization
            x = greedy_init(order)
        else:
            # Random subset start
            x = np.zeros(N_S, dtype=bool)
            k = random.randint(50, 150)
            candidates = random.sample(range(N_S), k)
            for v in candidates:
                ok = True
                for c in conflict_lists[v]:
                    if x[c]:
                        ok = False
                        break
                if ok:
                    x[v] = True
        
        size = int(np.sum(x))
        
        # Simulated annealing
        x_sa = local_search_sa(x, max_iters=500)
        size_sa = int(np.sum(x_sa))
        
        if size_sa > best_size:
            best_size = size_sa
            best_x = x_sa.copy()
            
        if size > best_size:
            best_size = size
            best_x = x.copy()

    if best_x is None:
        best_x = greedy_init()
        best_size = int(np.sum(best_x))

    # Construct the vectors
    type_d_vectors = []
    for i in range(N_S):
        if best_x[i]:
            support = supports[i]
            for signs in itertools.product([1, -1], repeat=4):
                v = np.zeros(D, dtype=np.int64)
                for idx, s in zip(support, signs):
                    v[idx] = s
                type_d_vectors.append(v)
                
    all_vectors = type_a_vectors + type_d_vectors
    result = np.array(all_vectors, dtype=np.int64)
    
    return result

# EVOLVE-BLOCK-END
