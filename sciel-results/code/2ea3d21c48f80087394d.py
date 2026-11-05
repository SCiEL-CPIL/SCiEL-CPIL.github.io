import numpy as np
import time

def partition(adjacency):
    adj = np.asarray(adjacency, dtype=np.int8)
    n = adj.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    start_time = time.time()
    DEADLINE = start_time + 140.0

    def time_left():
        return DEADLINE - time.time()

    E = int(np.sum(adj)) // 2
    degrees = adj.sum(axis=1)
    # Precompute neighbor lists for speed
    neighbors = [np.where(adj[i] > 0)[0] for i in range(n)]
    deg_list = [len(neighbors[i]) for i in range(n)]

    def cut_size(x):
        # x is +/- 1
        return E - 0.5 * (x @ (adj @ x))

    def spectral_init():
        # Laplacian
        L = np.diag(degrees.astype(np.float64)) - adj.astype(np.float64)
        # Symmetric eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(L)
        # Second smallest eigenvalue
        idx = np.argsort(eigvals)
        v = eigvecs[:, idx[1]]
        # Threshold at 0
        x = np.sign(v)
        # Handle zeros (rare)
        x[x == 0] = 1
        return x

    def local_search(x, max_iters=200000):
        # x is +/- 1
        Ax = adj @ x
        gains = 2 * x * Ax
        current_cut = cut_size(x)
        
        for _ in range(max_iters):
            if time_left() < 1.0:
                break
            
            # Find vertex with max gain
            v = int(np.argmax(gains))
            max_gain = gains[v]
            
            if max_gain <= 0:
                break
            
            # Flip v
            x_old_v = x[v]
            x[v] = -x_old_v
            
            # Update Ax and gains incrementally
            # Ax[k] changes by -2*x_old_v*adj[k,v]
            # gains[k] = 2*x[k]*Ax[k]
            # For k != v: gains[k] -= 2*x_old_v*adj[k,v]*x[k]
            # For k == v: gains[v] = 2*x[v]*Ax[v] = 2*(-x_old_v)*(Ax[v]-2*x_old_v*adj[v,v])
            # Since adj[v,v]=0: gains[v] = 2*(-x_old_v)*Ax[v]
            # Old gains[v] = 2*x_old_v*Ax[v], so new gains[v] = -old gains[v]
            
            nbrs = neighbors[v]
            if len(nbrs) > 0:
                # Update Ax for neighbors
                Ax[nbrs] -= 2 * x_old_v
                # Update gains for neighbors
                gains[nbrs] -= 4 * x_old_v * x[nbrs]
            
            # Update gain for v itself
            gains[v] = -gains[v]
            
            current_cut += max_gain
            
        return x, current_cut

    def two_flip_local_search(x, max_iters=30000):
        # x is +/- 1
        # Use incremental updates for pair flips
        Ax = adj @ x
        g = x * Ax  # g[k] = x[k] * Ax[k]
        
        # Precompute sign product matrix S[i,j] = x[i]*x[j] for the adjacency pattern
        # We'll update g incrementally
        
        for _ in range(max_iters):
            if time_left() < 1.0:
                break
            
            # Find best pair using vectorized computation but only for edges
            # G[i,j] = g[i] + g[j] - 2*x[i]*x[j]*adj[i,j]
            # Only need to check pairs where adj[i,j]=1 for the correction term
            # For non-adjacent pairs: G[i,j] = g[i] + g[j]
            
            # Efficient approach: compute base G = g[i]+g[j] for all pairs,
            # then subtract 2*x[i]*x[j]*adj[i,j]
            G = g[:, None] + g[None, :]
            # Subtract the edge correction
            G -= 2.0 * x[:, None] * x[None, :] * adj
            # Mask diagonal and lower triangle
            np.fill_diagonal(G, -np.inf)
            G = np.triu(G, k=1)
            
            idx = np.unravel_index(np.argmax(G), G.shape)
            best_gain = G[idx]
            
            if best_gain <= 0:
                break
            
            i, j = idx
            x_old_i = x[i]
            x_old_j = x[j]
            
            # Update Ax
            Ax -= 2.0 * x_old_i * adj[:, i]
            Ax -= 2.0 * x_old_j * adj[:, j]
            
            # Flip x
            x[i] = -x_old_i
            x[j] = -x_old_j
            
            # Recompute g (O(n) - cheap)
            g = x * Ax
            
        return x

    def hybrid_local_search(x, max_iters=50000):
        x, _ = local_search(x, max_iters=max_iters//2)
        x = two_flip_local_search(x, max_iters=max_iters//2)
        x, _ = local_search(x, max_iters=max_iters//2)
        return x

    def ils(x, num_iters=10000):
        best_x = x.copy()
        best_cut = cut_size(x)
        no_improve = 0
        max_no_improve = 80
        
        for i in range(num_iters):
            if time_left() < 5.0:
                break
            
            # Adaptive perturbation
            base_k = 2 + no_improve // 40
            k = min(base_k, max(1, n // 5))
            if n > k:
                flip_idx = np.random.choice(n, k, replace=False)
                x_pert = x.copy()
                x_pert[flip_idx] = -x_pert[flip_idx]
            else:
                x_pert = x.copy()
            
            # Local search
            x_ls = hybrid_local_search(x_pert)
            cut_ls = cut_size(x_ls)
            
            if cut_ls > best_cut:
                best_cut = cut_ls
                best_x = x_ls
                x = x_ls
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= max_no_improve:
                    # Big jump: flip a larger random subset
                    k2 = min(8 + no_improve // 20, n // 3)
                    if n > k2:
                        flip_idx2 = np.random.choice(n, k2, replace=False)
                        x_pert2 = x.copy()
                        x_pert2[flip_idx2] = -x_pert2[flip_idx2]
                        x_ls2 = hybrid_local_search(x_pert2)
                        cut_ls2 = cut_size(x_ls2)
                        if cut_ls2 > best_cut:
                            best_cut = cut_ls2
                            best_x = x_ls2
                            x = x_ls2
                            no_improve = 0
                        else:
                            no_improve = 0
        
        return best_x, best_cut

    # 1. Spectral Init
    x = spectral_init()
    
    # 2. Initial Local Search
    x = hybrid_local_search(x)
    best_x = x.copy()
    best_cut = cut_size(x)
    
    # 3. ILS
    x, cut = ils(x, num_iters=10000)
    if cut > best_cut:
        best_cut = cut
        best_x = x

    # 4. Final high-precision polish
    if time_left() > 3.0:
        for _ in range(5):
            if time_left() < 3.0:
                break
            x = hybrid_local_search(best_x, max_iters=100000)
            cut = cut_size(x)
            if cut > best_cut:
                best_cut = cut
                best_x = x
            else:
                break

    # Convert to 0/1 labels
    labels = (best_x + 1) // 2
    return labels.astype(int)
