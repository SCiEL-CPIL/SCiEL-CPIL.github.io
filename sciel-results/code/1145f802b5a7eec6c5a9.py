import numpy as np
import itertools
import time
import random
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    D = 11

    # Type A: Vectors with one non-zero entry +/- 2
    # Norm squared = 4.
    vectors_A = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors_A.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors_A.append(v2)

    vectors_A = np.array(vectors_A, dtype=np.int64)

    # Type D: Vectors with four non-zero entries +/- 1
    # Norm squared = 4.
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    N_S = len(supports)

    # Precompute intersection sizes for all pairs of supports
    inter = np.zeros((N_S, N_S), dtype=np.int8)
    masks = []
    for s in supports:
        m = 0
        for i in s:
            m |= (1 << i)
        masks.append(m)

    for i in range(N_S):
        m_i = masks[i]
        for j in range(i, N_S):
            m_j = masks[j]
            inter_len = bin(m_i & m_j).count('1')
            inter[i, j] = inter_len
            inter[j, i] = inter_len

    # Conflict graph for independent set (intersection >= 3)
    adj = [set() for _ in range(N_S)]
    for i in range(N_S):
        for j in range(N_S):
            if i != j and inter[i, j] >= 3:
                adj[i].add(j)
                adj[j].add(i)

    def greedy_mis_min_degree(adj, N_S, rng):
        """Greedy MIS using minimum-degree heuristic with randomization."""
        selected = set()
        available = set(range(N_S))
        # Compute initial degrees
        deg = np.array([len(adj[i]) for i in range(N_S)], dtype=np.int32)
        
        while available:
            if time.time() > deadline - 30:
                break
            # Find available nodes with minimum degree (among available neighbors)
            # For efficiency, use a heap
            # Actually, let's just pick from available nodes
            # Use a small random subset to speed up if available is large
            if len(available) > 50:
                candidates = list(rng.sample(available, 50))
            else:
                candidates = list(available)
            
            best_node = None
            best_key = (10**9, 10**9)
            for v in candidates:
                # Degree among available neighbors
                deg_av = sum(1 for u in adj[v] if u in available)
                # Tie-break randomly
                key = (deg_av, rng.random())
                if key < best_key:
                    best_key = key
                    best_node = v
            
            if best_node is None:
                break
                
            selected.add(best_node)
            available.discard(best_node)
            for u in adj[best_node]:
                available.discard(u)
            
        return selected

    def local_search_mis(in_set, adj):
        """Try to add more nodes to the independent set."""
        changed = True
        in_set_list = list(in_set)
        while changed:
            changed = False
            for v in range(N_S):
                if v not in in_set_list:
                    conflict = any(u in in_set_list for u in adj[v])
                    if not conflict:
                        in_set_list.append(v)
                        changed = True
        return set(in_set_list)

    def local_search_remove_and_readd(in_set, adj):
        """Try swapping: remove one node, add two if possible."""
        in_set_list = list(in_set)
        improved = True
        while improved:
            improved = False
            for i, v in enumerate(in_set_list):
                # Try removing v
                in_set_list.remove(v)
                new_set = set(in_set_list)
                old_size = len(new_set)
                
                # Try adding any available node
                for u in range(N_S):
                    if u not in new_set and not any(x in new_set for x in adj[u]):
                        new_set.add(u)
                        if len(new_set) > old_size + 1:  # Net gain
                            in_set_list = list(new_set)
                            improved = True
                            break
                # Restore v if no improvement
                if not improved or len(in_set_list) <= old_size:
                    in_set_list.append(v)
        return set(in_set_list)

    best_size = 0
    best_set = set()
    rng = random.Random(42)

    num_restarts = 2000
    for r in range(num_restarts):
        if time.time() > deadline - 40:
            break

        # Use min-degree greedy
        in_set = greedy_mis_min_degree(adj, N_S, rng)
        
        # Local search to add more
        in_set = local_search_mis(in_set, adj)
        
        # Occasionally try swap-based local search
        if r % 10 == 0:
            in_set = local_search_remove_and_readd(in_set, adj)

        size = len(in_set)
        if size > best_size:
            best_size = size
            best_set = set(in_set)

    # Generate vectors for best_set
    vectors_D = []
    for s_idx in best_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors_D.append(v)

    vectors_D = np.array(vectors_D, dtype=np.int64)

    # Try to add more vectors greedily from all possible Type D vectors
    all_vectors_D = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(s, signs):
                v[idx] = sign
            all_vectors_D.append(v)

    all_vectors_D = np.array(all_vectors_D, dtype=np.int64)

    current_vectors = list(vectors_D)
    current_set = set(map(tuple, current_vectors))

    indices_all = list(range(len(all_vectors_D)))
    rng.shuffle(indices_all)

    for i in indices_all:
        if time.time() > deadline - 5:
            break

        v = all_vectors_D[i]
        if tuple(v) in current_set:
            continue

        can_add = True
        for u in current_vectors:
            if np.dot(v, u) > 2:
                can_add = False
                break

        if can_add:
            current_vectors.append(v)
            current_set.add(tuple(v))

    final_vectors_D = np.array(current_vectors, dtype=np.int64)

    final_vectors = np.vstack([vectors_A, final_vectors_D])

    return final_vectors
