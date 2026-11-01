"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1

    # Group W4 vectors by their support (tuple of nonzero positions)
    support_to_indices = {}
    for i in range(N4):
        support = tuple(np.nonzero(vectors_v4[i])[0])
        if support not in support_to_indices:
            support_to_indices[support] = []
        support_to_indices[support].append(i)
    
    supports = list(support_to_indices.keys())
    n_supports = len(supports)
    
    # Build support-level conflict matrix:
    # Two supports conflict if they share >= 3 positions
    support_conflicts = np.zeros((n_supports, n_supports), dtype=bool)
    for i in range(n_supports):
        si = set(supports[i])
        for j in range(i + 1, n_supports):
            sj = set(supports[j])
            if len(si & sj) >= 3:
                support_conflicts[i, j] = True
                support_conflicts[j, i] = True
    
    # Each support contributes 16 vectors if selected
    vectors_per_support = 16
    
    # Spectral Initialization Strategy
    # Compute the principal eigenvector of the conflict graph adjacency matrix
    # The components of the eigenvector approximate the "importance" or "centrality"
    # of each node in the context of the graph's structure.
    # For MIS, we want to pick nodes that are "well-connected" to others in a way that 
    # allows many others to be excluded, or conversely, we can use the eigenvector 
    # of the Laplacian or just the adjacency to rank nodes.
    # Actually, for Independent Set, we want to pick a set of non-adjacent nodes.
    # A common heuristic is to pick the node with the smallest degree first (greedy min-degree).
    # Spectral approach: The principal eigenvector of the adjacency matrix A 
    # corresponds to the largest eigenvalue. 
    # Let's compute the eigenvector of the adjacency matrix.
    
    adjacency = support_conflicts.astype(float)
    
    try:
        # Use sparse or dense eigenvector calculation
        # n_supports is 330, so dense is fine and fast
        eigvals, eigvecs = np.linalg.eig(adjacency)
        # Find the index of the largest eigenvalue
        max_idx = np.argmax(eigvals)
        principal_eigvec = eigvecs[:, max_idx]
        # Take absolute values as we care about magnitude of participation
        spectral_scores = np.abs(principal_eigvec)
    except Exception:
        # Fallback to degree-based scores if spectral fails
        spectral_scores = support_conflicts.sum(axis=1).astype(float)
        
    # Normalize scores
    spectral_scores /= (np.max(spectral_scores) + 1e-9)
    
    # Combine with degree for a robust ranking
    degrees = support_conflicts.sum(axis=1)
    # We want to pick nodes that are "central" but allow many others to be picked?
    # Actually, for Independent Set, a good greedy is often min-degree.
    # Let's try a hybrid: rank by (degree - spectral_score) or similar?
    # Or simply use spectral score to guide the SA initialization.
    
    # Let's create a few different initializations:
    # 1. Greedy Min-Degree (proven good)
    # 2. Spectral-Guided Greedy: Pick nodes with high spectral score but low degree?
    #    Or pick nodes with high spectral score, then repair conflicts?
    #    Let's try: Sort by spectral score descending, pick if compatible.
    #    This might pick a "core" of the graph which is bad for IS.
    #    Let's try: Sort by spectral score ascending? No.
    #    
    # Better Spectral Idea: 
    # The eigenvector of the Laplacian L = D - A.
    # The smallest non-zero eigenvalue (algebraic connectivity) relates to connectivity.
    # Let's stick to a simpler spectral heuristic:
    # Use the principal eigenvector to define a "soft" constraint or weight.
    # In SA, we can bias the selection of nodes to add based on their spectral score.
    
    neighbors_sup = [np.where(support_conflicts[i])[0] for i in range(n_supports)]
    neighbor_counts_sup = np.array([len(n) for n in neighbors_sup], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_support_set = np.zeros(n_supports, dtype=bool)
    
    def greedy_min_degree_supports():
        """Greedy independent set with min-degree heuristic on support graph."""
        in_set = np.zeros(n_supports, dtype=bool)
        alive = np.ones(n_supports, dtype=bool)
        degrees = neighbor_counts_sup.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors_sup[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= support_conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set

    def spectral_greedy_supports():
        """Greedy independent set guided by spectral scores (lowest score first)."""
        in_set = np.zeros(n_supports, dtype=bool)
        alive = np.ones(n_supports, dtype=bool)
        # Sort indices by spectral score ascending (pick 'peripheral' nodes first?)
        # Or descending? 
        # Let's try ascending: nodes with low centrality are less likely to conflict with many.
        order = np.argsort(spectral_scores)
        
        for s in order:
            if not alive[s]:
                continue
            if not np.any(support_conflicts[s] & in_set):
                in_set[s] = True
                # Remove neighbors from alive
                nbrs = neighbors_sup[s]
                alive[nbrs] = False
                alive[s] = False
                
        return in_set

    def sa_local_search_supports(in_set_init, max_time=10.0):
        """Improved SA with add/remove/swap moves on support graph."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        
        T = 2.0
        cooling = 0.999
        min_T = 0.03
        
        # count_conflicts[s] = number of neighbors of s that are in in_set
        count_conflicts = np.zeros(n_supports, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors_sup[i]] += 1
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        
        while time.time() < sa_deadline:
            move_type = rng.random()
            
            if move_type < 0.7:
                # Add move: pick a support not in set
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    continue
                    
                # Weighted selection: prefer supports with fewer conflicts
                # Incorporate spectral score as a tie-breaker or soft weight
                counts = count_conflicts[not_in_set].astype(float)
                # Lower counts are better. 
                # We can also bias towards higher spectral scores if counts are equal?
                # Let's stick to counts for stability, but use spectral for init.
                weights = np.exp(-counts / max(T, 0.1))
                weights /= weights.sum()
                s = not_in_set[rng.choice(len(not_in_set), p=weights)]
                
                k = count_conflicts[s]
                delta = 1 - k
                
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.5
                else:
                    accept = rng.random() < np.exp(delta / T)
                    
                if accept:
                    to_remove = np.where(support_conflicts[s] & in_set)[0]
                    
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors_sup[u]] -= 1
                    
                    in_set[s] = True
                    current_size += delta
                    count_conflicts[neighbors_sup[s]] += 1
            elif move_type < 0.9 and current_size > 0:
                # Remove move: pick a support in set to remove
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) == 0:
                    continue
                    
                s = in_set_indices[rng.integers(len(in_set_indices))]
                
                # Removing always decreases size by 1
                # Only accept if it helps overall (rarely, for escape)
                accept = rng.random() < np.exp(-1.0 / T)
                
                if accept:
                    in_set[s] = False
                    current_size -= 1
                    count_conflicts[neighbors_sup[s]] -= 1
            else:
                # Swap move: remove one in-set support, add one not-in-set
                in_set_indices = np.where(in_set)[0]
                not_in_set = np.where(~in_set)[0]
                if len(in_set_indices) == 0 or len(not_in_set) == 0:
                    continue
                
                s_remove = in_set_indices[rng.integers(len(in_set_indices))]
                s_add = not_in_set[rng.integers(len(not_in_set))]
                
                # Check if s_add conflicts with (in_set - s_remove)
                in_set_minus = in_set.copy()
                in_set_minus[s_remove] = False
                conflicts_after = np.sum(support_conflicts[s_add] & in_set_minus)
                
                # Delta: +1 for add, -1 for remove, -conflicts_after for forced removals?
                # No, swap just replaces. If s_add conflicts with others in in_set_minus, 
                # we can't do a simple swap unless we also remove those.
                # Let's only accept if s_add has NO conflicts with in_set_minus.
                if conflicts_after == 0:
                    # Valid swap
                    in_set[s_remove] = False
                    count_conflicts[neighbors_sup[s_remove]] -= 1
                    
                    in_set[s_add] = True
                    count_conflicts[neighbors_sup[s_add]] += 1
                    
                    # Size remains same. Accept with probability based on T? 
                    # Or just accept to explore.
                    if rng.random() < 0.5:
                        pass # Keep
                    else:
                        # Revert
                        in_set[s_add] = False
                        count_conflicts[neighbors_sup[s_add]] -= 1
                        in_set[s_remove] = True
                        count_conflicts[neighbors_sup[s_remove]] += 1
                else:
                    # Complex swap: remove s_remove and all conflicts of s_add
                    # This is effectively an "Add with forced removals"
                    to_remove = np.where(support_conflicts[s_add] & in_set)[0]
                    # s_remove might be in to_remove or not
                    # Let's treat it as an Add move if it improves size
                    # Net change: +1 (add) - len(to_remove) (remove)
                    # If len(to_remove) == 0, it's an Add.
                    # If len(to_remove) > 0, it's a multi-removal.
                    # We'll skip complex swaps to keep SA fast and stable.
                    pass
            
            if current_size > local_best_size:
                local_best_size = current_size
                local_best_in_set = in_set.copy()
                stuck_counter = 0
            else:
                stuck_counter += 1
            
            if stuck_counter > 2000:
                T = max(T, 1.5)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size
    
    # Multiple restarts with varied strategies
    n_restarts = 0
    max_restarts = 300
    
    def greedy_max_degree_supports():
        """Greedy independent set with max-degree heuristic (opposite of min)."""
        in_set = np.zeros(n_supports, dtype=bool)
        alive = np.ones(n_supports, dtype=bool)
        degrees = neighbor_counts_sup.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            max_deg = np.max(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == max_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors_sup[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= support_conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set
    
    def random_init_supports(density=0.15):
        """Random initialization with given density, then repair conflicts."""
        in_set = np.zeros(n_supports, dtype=bool)
        candidates = rng.integers(0, n_supports, size=int(n_supports * density * 3))
        
        for s in candidates:
            if not in_set[s] and not np.any(support_conflicts[s] & in_set):
                in_set[s] = True
        
        return in_set
    
    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1
        
        # Choose initialization strategy
        # Use spectral greedy in some restarts
        strategy = n_restarts % 4
        if strategy == 0:
            in_set = greedy_min_degree_supports()
        elif strategy == 1:
            in_set = greedy_max_degree_supports()
        elif strategy == 2:
            in_set = spectral_greedy_supports()
        else:
            in_set = random_init_supports()
            
        current_size = in_set.sum()
        
        if current_size > best_size:
            best_size = current_size
            best_support_set = in_set.copy()
        
        # SA Local Search
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break
        
        sa_time = min(remaining_time * 0.5, 12.0)
        if sa_time < 2:
            sa_time = 2.0
            
        sa_result, sa_size = sa_local_search_supports(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_support_set = sa_result.copy()
        
        # Final greedy polish: try to add any remaining non-conflicting supports
        if time.time() < deadline - 3:
            polish_set = best_support_set.copy()
            polished = True
            while polished and time.time() < deadline - 3:
                polished = False
                not_in = np.where(~polish_set)[0]
                if len(not_in) == 0:
                    break
                # Find supports with zero conflicts with current set
                for s in not_in:
                    if not np.any(support_conflicts[s] & polish_set):
                        polish_set[s] = True
                        polished = True
                        if polish_set.sum() > best_size:
                            best_size = polish_set.sum()
                            best_support_set = polish_set.copy()
    
    # Build final vector list from selected supports
    selected_support_indices = np.where(best_support_set)[0]
    final_vectors = []
    
    for si in selected_support_indices:
        support = supports[si]
        for idx in support_to_indices[support]:
            final_vectors.append(vectors_v4[idx])
    
    # Add all W1 vectors (22 of them)
    final_vectors.extend(v1_candidates)
    
    result = np.array(final_vectors, dtype=np.int64)
    
    # Fallback: if something went wrong, use deterministic construction
    if len(result) == 0:
        all_supports = list(itertools.combinations(range(D), 4))
        selected_supports = []
        for s in all_supports:
            valid = True
            for t in selected_supports:
                intersection = set(s).intersection(t)
                if len(intersection) > 2:
                    valid = False
                    break
            if valid:
                selected_supports.append(s)
                if len(selected_supports) >= 32:
                    break
        
        fallback_vectors = []
        for s in selected_supports:
            for signs in itertools.product([1, -1], repeat=4):
                v = [0] * D
                for pos, sign in zip(s, signs):
                    v[pos] = sign
                fallback_vectors.append(v)
        
        # Add W1 vectors
        for pos in range(D):
            for sign in [1, -1]:
                v = [0] * D
                v[pos] = 2 * sign
                fallback_vectors.append(v)
        
        result = np.array(fallback_vectors, dtype=np.int64)
    
    return result

# EVOLVE-BLOCK-END
