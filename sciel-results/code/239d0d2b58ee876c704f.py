import numpy as np
import random
import itertools

def construct():
    D = 11
    supports = list(itertools.combinations(range(D), 4))
    n_sup = len(supports)
    
    # Build adjacency list
    adj = [set() for _ in range(n_sup)]
    for i in range(n_sup):
        s_i = set(supports[i])
        for j in range(i+1, n_sup):
            s_j = set(supports[j])
            if len(s_i.intersection(s_j)) == 3:
                adj[i].add(j)
                adj[j].add(i)
                
    # 1. Greedy Initialization
    def greedy_is():
        remaining = set(range(n_sup))
        current_set = set()
        while remaining:
            # Pick vertex with min degree in remaining
            min_deg = float('inf')
            best_v = None
            for v in remaining:
                deg = len(adj[v].intersection(remaining))
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                    if deg == 0: break
            current_set.add(best_v)
            to_remove = {best_v}.union(adj[best_v].intersection(remaining))
            remaining.difference_update(to_remove)
        return current_set

    best_set = greedy_is()
    best_size = len(best_set)
    
    # 2. Simulated Annealing
    # We will try to improve the set
    # State: S (set of indices)
    # Move: Pick v not in S. Let N be neighbors of v in S.
    #       New S = S - N + {v}.
    #       Delta = 1 - len(N).
    
    # We run SA for a fixed number of steps or until time limit
    # Time limit 120s. We should be careful not to exceed.
    # Let's do 50,000 steps.
    
    S = best_set.copy()
    current_size = len(S)
    T = 1.0
    cooling = 0.999
    
    # Precompute neighbors list for faster access? adj is already list of sets.
    
    # To speed up, we can maintain a list of "candidates" or just pick random v from range(n_sup)
    # and check if v in S.
    
    for step in range(50000):
        # Pick random v
        v = random.randint(0, n_sup - 1)
        
        if v in S:
            # We can try to remove v?
            # Or just skip. Removing v reduces size by 1.
            # It might allow adding many others later.
            # Let's allow removal with low probability or specific logic.
            # For now, just skip if v in S to focus on "swap/add" moves.
            # Actually, if we only do swaps where we replace N with v, and v is not in S,
            # we never remove a vertex without replacing it.
            # This preserves the property that we are always in the set of independent sets.
            # But we might get stuck.
            # Let's allow a pure removal move with small probability?
            # Or just rely on the swap.
            continue
            
        # v is not in S
        # Find N = neighbors of v in S
        N = adj[v].intersection(S)
        
        if len(N) == 0:
            # We can add v
            S.add(v)
            current_size += 1
            if current_size > best_size:
                best_size = current_size
                best_set = S.copy()
        else:
            # Move: S -> S - N + {v}
            delta = 1 - len(N)
            
            # Accept if delta > 0 or random < exp(delta/T)
            if delta > 0:
                # This is an improvement (should not happen if S is maximal, 
                # but S might not be maximal if we did weird moves? 
                # If S is maximal, every v not in S has at least one neighbor in S.
                # So len(N) >= 1.
                # If len(N) == 1, delta = 0.
                # If len(N) > 1, delta < 0.
                # So delta > 0 is impossible for maximal S.
                pass
            else:
                # delta <= 0
                # Probability of acceptance
                if delta == 0:
                    # Neutral move, accept with 0.5?
                    if random.random() < 0.5:
                        S.difference_update(N)
                        S.add(v)
                        # Size unchanged
                else:
                    # delta < 0
                    # exp(delta / T)
                    # T decreases
                    p = np.exp(delta / T)
                    if random.random() < p:
                        S.difference_update(N)
                        S.add(v)
                        current_size += delta # delta is negative
                
        T *= cooling
        
        # Update best
        if current_size > best_size:
            best_size = current_size
            best_set = S.copy()
            
    # Construct vectors from best_set
    vectors = []
    for idx in best_set:
        sup = supports[idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, s in enumerate(sup):
                v[s] = signs[k]
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
