import numpy as np
import time

def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    adj = np.array(adjacency, dtype=np.int8)
    np.fill_diagonal(adj, 0)
    
    start_time = time.time()
    deadline = 140.0
    
    S_total = np.sum(adj)
    
    def get_cut(x):
        xAx = x @ (adj @ x)
        return 0.25 * (S_total - xAx)
    
    # Spectral initialization
    D = np.diag(adj.sum(axis=1))
    L = D - adj
    
    try:
        vals, vecs = np.linalg.eigh(L)
        fiedler = vecs[:, 1]
        x_spec = np.sign(fiedler)
        x_spec[x_spec == 0] = 1
    except:
        x_spec = np.ones(n, dtype=np.int8)
    
    rng = np.random.default_rng(42)
    
    def local_search_1flip(x, max_iters=10000):
        S = adj @ x
        C = get_cut(x)
        iters = 0
        while iters < max_iters:
            gains = x * S
            i = np.argmax(gains)
            max_gain = gains[i]
            if max_gain <= 0:
                break
            v = x[i]
            x[i] = -v
            C += max_gain
            S -= 2 * v * adj[:, i]
            iters += 1
        return x, C, iters
    
    def local_search_2swap(x, max_iters=5000):
        """2-swap local search: flip two vertices at a time."""
        S = adj @ x
        C = get_cut(x)
        iters = 0
        while iters < max_iters:
            # Compute gain for flipping pair (i, j)
            # Gain = x_i*S_i + x_j*S_j - 2*A_ij*x_i*x_j
            # Vectorized: for each i, compute gains with all j > i
            gains_1 = x * S  # x_i * S_i for each i
            # For pair (i,j): gain = gains_1[i] + gains_1[j] - 2*adj[i,j]*x[i]*x[j]
            # We need to find the best pair.
            # Naive O(n^2) scan each iteration is too slow for many iterations.
            # Instead, do a few rounds of best-pair search.
            
            # Find top candidate i
            i = np.argmax(gains_1)
            if gains_1[i] <= 0:
                # No single flip helps, but maybe a 2-swap does
                # Try a few random pairs
                found = False
                for _ in range(20):
                    j = rng.integers(0, n)
                    if j == i:
                        continue
                    # Compute gain for pair (i, j)
                    gain = gains_1[i] + gains_1[j] - 2 * adj[i, j] * x[i] * x[j]
                    if gain > 0:
                        # Flip both i and j
                        vi = x[i]
                        vj = x[j]
                        x[i] = -vi
                        x[j] = -vj
                        C += gain
                        # Update S
                        # S changes due to both flips
                        # Delta for i: -2*vi * adj[:,i]
                        # Delta for j: -2*vj * adj[:,j]
                        S -= 2 * vi * adj[:, i]
                        S -= 2 * vj * adj[:, j]
                        gains_1 = x * S
                        iters += 1
                        found = True
                        break
                if not found:
                    break
            else:
                # Best single flip helps, but try to see if a 2-swap is even better
                # For the best i, find best j
                # gain(i,j) = gains_1[i] + gains_1[j] - 2*adj[i,j]*x[i]*x[j]
                # = gains_1[i] + (gains_1[j] - 2*adj[i,j]*x[i]*x[j])
                # For fixed i, we want to maximize gains_1[j] - 2*adj[i,j]*x[i]*x[j]
                # This is gains_1[j] - 2*x[i]*adj[i,j]*x[j]
                # = gains_1[j] - 2*x[i]*x[j]*adj[i,j]
                
                # Compute for all j != i
                j_gains = gains_1.copy()
                # Subtract the interaction term
                interaction = 2 * x[i] * x * adj[i, :]
                j_gains[i] = -1e9  # Don't pick i
                j_gains -= interaction
                j = np.argmax(j_gains)
                if j == i or j_gains[j] <= 0:
                    # No good 2-swap with i, do single flip
                    v = x[i]
                    x[i] = -v
                    C += gains_1[i]
                    S -= 2 * v * adj[:, i]
                    gains_1 = x * S
                    iters += 1
                else:
                    # Do 2-swap
                    vi = x[i]
                    vj = x[j]
                    x[i] = -vi
                    x[j] = -vj
                    gain = gains_1[i] + gains_1[j] - 2 * adj[i, j] * vi * vj
                    C += gain
                    S -= 2 * vi * adj[:, i]
                    S -= 2 * vj * adj[:, j]
                    gains_1 = x * S
                    iters += 1
        
        return x, C, iters
    
    best_C = -1
    best_x = None
    
    # Phase 1: Spectral init + 1-flip LS
    x_curr = x_spec.copy()
    x_ls, C_ls, _ = local_search_1flip(x_curr)
    if C_ls > best_C:
        best_C = C_ls
        best_x = x_ls.copy()
    
    if time.time() > deadline - 20:
        return ((best_x + 1) // 2).astype(int)
    
    # Phase 2: 2-swap polish on best
    x_curr = best_x.copy()
    x_ls2, C_ls2, _ = local_search_2swap(x_curr, max_iters=2000)
    if C_ls2 > best_C:
        best_C = C_ls2
        best_x = x_ls2.copy()
    
    if time.time() > deadline - 10:
        return ((best_x + 1) // 2).astype(int)
    
    # Phase 3: Multi-start with 1-flip LS
    num_restarts = 200
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        if i % 3 == 0:
            x_curr = rng.choice([-1, 1], size=n).astype(np.int8)
        elif i % 3 == 1:
            # Perturb best: flip random 10% of nodes
            x_curr = best_x.copy()
            num_flips = max(1, n // 10)
            flip_idx = rng.choice(n, size=num_flips, replace=False)
            x_curr[flip_idx] *= -1
        else:
            # Perturb spectral: flip random 20% of nodes
            x_curr = x_spec.copy()
            num_flips = max(1, n // 5)
            flip_idx = rng.choice(n, size=num_flips, replace=False)
            x_curr[flip_idx] *= -1
        
        x_ls, C_ls, _ = local_search_1flip(x_curr)
        if C_ls > best_C:
            best_C = C_ls
            best_x = x_ls.copy()
    
    # Phase 4: 2-swap polish on final best if time permits
    if time.time() < deadline - 5:
        x_curr = best_x.copy()
        x_ls2, C_ls2, _ = local_search_2swap(x_curr, max_iters=3000)
        if C_ls2 > best_C:
            best_C = C_ls2
            best_x = x_ls2.copy()
    
    if best_x is None:
        best_x = rng.choice([-1, 1], size=n).astype(np.int8)
    
    labels = (best_x + 1) // 2
    return labels.astype(int)
