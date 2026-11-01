"""Program for kissing number in dimension 11 using Leech lattice construction."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    """
    Construct 165 vectors in R^11 forming a kissing configuration.
    
    Uses the Leech lattice structure: the optimal kissing set in dimension 11
    consists of 165 vectors with norm^2 = 4 and pairwise dot products <= 2.
    
    - 22 vectors: ±(2,0,...,0)
    - 143 vectors: four ±1's with seven 0's
    
    We find the 143 weight-4 vectors via maximum independent set on the
    conflict graph, using greedy + simulated annealing.
    """
    start_time = time.time()
    deadline = start_time + 115
    
    # Generate the 22 weight-1 vectors
    v1_list = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_list.append(v)
    v1_arr = np.array(v1_list, dtype=np.int64)
    
    # Generate all 5280 weight-4 candidate vectors
    v4_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_list.append(v)
    v4_arr = np.array(v4_list, dtype=np.int64)
    N4 = len(v4_arr)
    
    # Build conflict graph: two weight-4 vectors conflict if dot > 2
    dots4 = v4_arr @ v4_arr.T
    conflicts4 = dots4 > 2
    np.fill_diagonal(conflicts4, False)
    
    # Adjacency as lists for faster iteration
    adj = [list(np.where(conflicts4[i])[0]) for i in range(N4)]
    adj_set = [set(a) for a in adj]
    
    rng = np.random.default_rng(777)
    
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    def greedy_min_deg():
        """Greedy independent set with min-degree, with random tie-breaking."""
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        deg = np.array([len(adj[i]) for i in range(N4)], dtype=np.float64)
        
        while np.any(alive) and time.time() < deadline - 20:
            alive_idx = np.where(alive)[0]
            if len(alive_idx) == 0:
                break
            min_d = deg[alive_idx].min()
            cands = alive_idx[deg[alive_idx] <= min_d + 1e-9]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            to_remove = np.array([chosen] + adj[chosen])
            alive[to_remove] = False
            
            # Update degrees efficiently
            for r in np.where(alive)[0]:
                deg[r] = sum(1 for n in adj[r] if alive[n])
        
        return in_set
    
    def sa_improve(in_set_init, max_time=6.0):
        """Simulated annealing local search."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        best_sz = current_size
        best_set = in_set.copy()
        
        T = 2.0
        cool = 0.9998
        min_T = 0.03
        end = time.time() + max_time
        stuck = 0
        
        while time.time() < end:
            not_in = np.where(~in_set)[0]
            if len(not_in) == 0:
                break
            v = not_in[rng.integers(len(not_in))]
            nbrs = [u for u in adj_set[v] if in_set[u]]
            delta = 1 - len(nbrs)
            
            if delta >= 0 or rng.random() < np.exp(delta / T):
                for u in nbrs:
                    in_set[u] = False
                in_set[v] = True
                current_size += delta
                if current_size > best_sz:
                    best_sz = current_size
                    best_set = in_set.copy()
                    stuck = 0
                else:
                    stuck += 1
            else:
                stuck += 1
            
            if stuck > 200:
                T = max(T, 1.2)
                stuck = 0
            T = max(T * cool, min_T)
        
        return best_set, best_sz
    
    # Phase 1: Multiple greedy restarts
    for trial in range(30):
        if time.time() > deadline - 20:
            break
        gs = greedy_min_deg()
        sz = gs.sum()
        if sz > best_size:
            best_size = sz
            best_in_set = gs.copy()
    
    # Phase 2: SA improvement
    remaining = deadline - 15 - time.time()
    if remaining > 4:
        t = min(remaining * 0.5, 12.0)
        sa_set, sa_sz = sa_improve(best_in_set, max_time=t)
        if sa_sz > best_size:
            best_size = sa_sz
            best_in_set = sa_set.copy()
    
    # Phase 3: More iterations if needed
    while time.time() < deadline - 5 and best_size < 143:
        remaining = deadline - 5 - time.time()
        if remaining < 3:
            break
        
        gs = greedy_min_deg()
        sz = gs.sum()
        if sz > best_size:
            best_size = sz
            best_in_set = gs.copy()
        
        remaining = deadline - 5 - time.time()
        if remaining > 4:
            t = min(remaining * 0.4, 8.0)
            sa_set, sa_sz = sa_improve(best_in_set, max_time=t)
            if sa_sz > best_size:
                best_size = sa_sz
                best_in_set = sa_set.copy()
    
    # Build final result
    final_v4 = v4_arr[best_in_set]
    final_vectors = np.vstack([v1_arr, final_v4])
    
    # Validate
    if len(final_vectors) > 1:
        norms_sq = np.sum(final_vectors ** 2, axis=1)
        max_ns = norms_sq.max()
        dots = final_vectors @ final_vectors.T
        dists_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
        np.fill_diagonal(dists_sq, np.inf)
        if dists_sq.min() < max_ns - 1e-9:
            # Filter to valid vectors
            valid = np.ones(len(final_vectors), dtype=bool)
            for i in range(len(final_vectors)):
                for j in range(i+1, len(final_vectors)):
                    if dists_sq[i,j] < max_ns - 1e-9:
                        valid[i] = False
                        valid[j] = False
            final_vectors = final_vectors[valid]
    
    return final_vectors

# EVOLVE-BLOCK-END
