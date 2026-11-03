"""Kissing number in dimension 11: vector-level MIS with sign-pattern selection."""
import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Generate all C(11,4) = 330 supports
    supports = list(itertools.combinations(range(D), 4))
    n_sup = len(supports)  # 330

    # All sign patterns for 4 coordinates
    sign_patterns = list(itertools.product([1, -1], repeat=4))
    n_sp = len(sign_patterns)  # 16

    # Total candidates: 330 * 16 = 5280
    n_cand = n_sup * n_sp

    # Precompute: for each support, which sign patterns conflict with which
    # Within same support: two patterns conflict if they agree on >= 3 of 4 coords
    # i.e., differ in <= 1 coord
    within_conflict = np.zeros((n_sp, n_sp), dtype=np.uint8)
    for i in range(n_sp):
        for j in range(i + 1, n_sp):
            # Count differences
            diffs = sum(1 for a, b in zip(sign_patterns[i], sign_patterns[j]) if a != b)
            if diffs <= 1:  # agree on >= 3
                within_conflict[i, j] = 1
                within_conflict[j, i] = 1

    # For supports intersecting in 3: precompute the 3 shared positions
    # and which sign patterns conflict (agree on all 3 shared coords)
    # Map support index to its set
    support_sets = [set(s) for s in supports]

    # Build conflict matrix for all 5280 candidates
    # This is large: 5280 x 5280. Use uint8.
    adj = np.zeros((n_cand, n_cand), dtype=np.uint8)

    # Fill within-support conflicts
    for si in range(n_sup):
        base = si * n_sp
        for i in range(n_sp):
            for j in range(n_sp):
                if within_conflict[i, j]:
                    adj[base + i, base + j] = 1

    # Fill between-support conflicts (only for supports intersecting in 3)
    for si in range(n_sup):
        for sj in range(si + 1, n_sup):
            inter = support_sets[si] & support_sets[sj]
            if len(inter) != 3:
                continue
            # Find positions in each support corresponding to shared coords
            pos_si = [supports[si].index(c) for c in inter]
            pos_sj = [supports[sj].index(c) for c in inter]
            base_i = si * n_sp
            base_j = sj * n_sp
            for i in range(n_sp):
                for j in range(n_sp):
                    # Check if signs agree on all 3 shared coords
                    agree = all(sign_patterns[i][pos_si[k]] == sign_patterns[j][pos_sj[k]] for k in range(3))
                    if agree:
                        adj[base_i + i, base_j + j] = 1
                        adj[base_j + j, base_i + i] = 1

    # Add Type A vectors: 22 vectors (±2e_i for i in 0..10)
    # They are compatible with all Type B vectors, so they're always included.
    # We don't need to include them in the conflict graph.
    n_type_a = 2 * D  # 22

    # Now solve MIS on the 5280-node graph
    n = n_cand

    def greedy_is(seed=0):
        rng = np.random.default_rng(seed)
        remaining = np.ones(n, dtype=np.uint8)
        selected = np.zeros(n, dtype=np.uint8)
        
        while True:
            if time.time() > deadline - 20.0:
                break
            rem_idx = np.where(remaining > 0)[0]
            if len(rem_idx) == 0:
                break
            
            deg = (adj * remaining[np.newaxis, :]).sum(axis=1)
            deg[remaining == 0] = n + 1
            
            min_deg = deg.min()
            candidates = np.where((deg == min_deg) & (remaining > 0))[0]
            if len(candidates) == 0:
                break
            
            pick = rng.choice(candidates)
            selected[pick] = 1
            remaining[pick] = 0
            remaining[adj[pick] == 1] = 0
        
        return selected

    def local_search_sa(selected, n_steps=100000, seed=0):
        rng = np.random.default_rng(seed)
        rand = random.Random(seed)
        
        S = selected.copy()
        current_size = int(S.sum())
        best_size = current_size
        best_S = S.copy()
        
        T = 3.0
        cooling = 0.9998
        min_T = 0.1
        
        step = 0
        while step < n_steps and time.time() < deadline - 15.0:
            step += 1
            v = rng.integers(0, n)
            
            if S[v]:
                if rand.random() < 0.02:
                    S[v] = 0
                    current_size -= 1
                continue
            
            conflicts = np.where(adj[v] & S)[0]
            n_conf = len(conflicts)
            
            if n_conf == 0:
                S[v] = 1
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_S = S.copy()
            else:
                delta = 1 - n_conf
                if delta >= 0:
                    S[conflicts] = 0
                    S[v] = 1
                    current_size += delta
                    if current_size > best_size:
                        best_size = current_size
                        best_S = S.copy()
                else:
                    if T > min_T:
                        p = np.exp(delta / T)
                        if rand.random() < p:
                            S[conflicts] = 0
                            S[v] = 1
                            current_size += delta
                            if current_size > best_size:
                                best_size = current_size
                                best_S = S.copy()
            
            T = max(min_T, T * cooling)
        
        return best_S, best_size

    # Multiple restarts
    best_S = np.zeros(n, dtype=np.uint8)
    best_size = 0
    
    n_restarts = 20
    for i in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        S_init = greedy_is(seed=1000 + i)
        init_size = int(S_init.sum())
        
        S_sa, sa_size = local_search_sa(S_init, n_steps=50000, seed=2000 + i)
        
        if sa_size > best_size:
            best_size = sa_size
            best_S = S_sa.copy()
        
        if i % 4 == 0 and time.time() < deadline - 40.0:
            S_sa2, sa2_size = local_search_sa(S_init, n_steps=100000, seed=5000 + i)
            if sa2_size > best_size:
                best_size = sa2_size
                best_S = S_sa2.copy()

    # Construct vectors
    vectors = []
    for idx in range(n):
        if best_S[idx]:
            si = idx // n_sp
            sp = idx % n_sp
            sup = supports[si]
            signs = sign_patterns[sp]
            v = [0] * D
            for pos, s in zip(sup, signs):
                v[pos] = s
            vectors.append(v)
    
    # Add Type A vectors
    for i in range(D):
        for sign in (1, -1):
            v = [0] * D
            v[i] = sign * 2
            vectors.append(v)
    
    result = np.array(vectors, dtype=np.int64)
    return result
