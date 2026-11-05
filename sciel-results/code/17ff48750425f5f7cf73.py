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

    def cut_size(x):
        # x is +/- 1
        return E - 0.5 * (x @ (adj @ x))

    def spectral_init():
        # Laplacian
        L = np.diag(degrees) - adj
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

    def local_search(x, max_iters=100000):
        # x is +/- 1
        Ax = adj @ x
        gains = 2 * x * Ax
        current_cut = cut_size(x)
        
        for _ in range(max_iters):
            if time_left() < 1.0:
                break
            
            # Find vertex with max gain
            v = np.argmax(gains)
            max_gain = gains[v]
            
            if max_gain <= 0:
                break
            
            # Flip v
            x_old_v = x[v]
            
            # Update Ax
            # Ax = Ax - 2 * x_old_v * adj[:, v]
            Ax -= 2 * x_old_v * adj[:, v]
            
            # Flip x
            x[v] = -x_old_v
            
            # Update gains
            # For neighbors u of v:
            # gains[u] -= 4 * x[u] * x_old_v * adj[u, v]
            # Note: x[u] is current x[u]
            # adj[u, v] is 1 if connected
            neighbors = np.where(adj[v] > 0)[0]
            if len(neighbors) > 0:
                gains[neighbors] -= 4 * x[neighbors] * x_old_v * adj[neighbors, v]
            
            # For v itself:
            # gains[v] = -gains[v]
            gains[v] = -gains[v]
            
            # Update current cut
            current_cut += max_gain
            
        return x, current_cut

    def two_flip_local_search(x, max_iters=50000):
        # x is +/- 1
        # This explores pairs of flips to escape single-flip local optima
        # We compute the gain for flipping pair (i, j)
        # gain(i, j) = 2 * (x[i]*Ax[i] + x[j]*Ax[j]) - 2 * x[i]*x[j]*adj[i,j]*2
        # Wait, let's derive properly.
        # Cut(x) = sum_{u<v} x[u]x[v] adj[u,v] * (-1) + E? No.
        # Cut(x) = 0.5 * sum_{u,v} (1 - x[u]x[v]) adj[u,v] = E - 0.5 sum x[u]x[v] adj[u,v]
        # Let S = sum x[u]x[v] adj[u,v] = x^T A x
        # If we flip i and j:
        # x'[i] = -x[i], x'[j] = -x[j]
        # S' = S - 2 x[i] sum_{v!=i} x[v] adj[i,v] - 2 x[j] sum_{v!=j} x[v] adj[j,v] + 4 x[i] x[j] adj[i,j]
        # Note: sum_{v!=i} x[v] adj[i,v] = (Ax)_i - x[i] adj[i,i] = (Ax)_i (since adj[i,i]=0)
        # So S' = S - 2 x[i] (Ax)_i - 2 x[j] (Ax)_j + 4 x[i] x[j] adj[i,j]
        # Change in Cut = -0.5 (S' - S) = x[i] (Ax)_i + x[j] (Ax)_j - 2 x[i] x[j] adj[i,j]
        # Let g[i] = x[i] (Ax)_i
        # gain(i, j) = g[i] + g[j] - 2 x[i] x[j] adj[i,j]
        
        Ax = adj @ x
        g = x * Ax
        
        for _ in range(max_iters):
            if time_left() < 1.0:
                break
            
            # We want to find max gain(i, j) for i < j
            # gain(i, j) = g[i] + g[j] - 2 x[i] x[j] adj[i,j]
            # If adj[i,j] == 0, gain = g[i] + g[j]
            # If adj[i,j] == 1, gain = g[i] + g[j] - 2 x[i] x[j]
            
            # Efficiently finding the best pair is O(n^2).
            # n=100, so 10000 operations per check.
            # This is fast enough.
            
            best_gain = 0
            best_i = -1
            best_j = -1
            
            # Vectorized computation of gains for all pairs?
            # G = g[:, None] + g[None, :] - 2 * x[:, None] * x[None, :] * adj
            # We only want upper triangle i < j
            G = g[:, None] + g[None, :] - 2 * x[:, None] * x[None, :] * adj
            # Mask lower triangle and diagonal
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            G[~mask] = -np.inf
            
            # Find max
            idx = np.unravel_index(np.argmax(G), G.shape)
            best_gain = G[idx]
            
            if best_gain <= 0:
                break
            
            best_i, best_j = idx
            
            # Flip i and j
            x_old_i = x[best_i]
            x_old_j = x[best_j]
            
            # Update Ax
            # Ax changes due to flipping i and j
            # Ax'[k] = Ax[k] - 2 x_old_i adj[k, i] - 2 x_old_j adj[k, j]
            # Except for k=i and k=j where x changes too?
            # Let's just recompute Ax for simplicity? No, n=100, recompute is cheap.
            # But let's try to update.
            # For k != i, j:
            # Ax'[k] = Ax[k] - 2 x_old_i adj[k, i] - 2 x_old_j adj[k, j]
            # For k = i:
            # Ax'[i] = sum_{v} x'[v] adj[i, v] = -x_old_i adj[i, i] + sum_{v!=i} x'[v] adj[i, v]
            # x'[i] = -x_old_i
            # x'[j] = -x_old_j
            # x'[v] = x[v] for v != i, j
            # Ax'[i] = -x_old_i * 0 + (-x_old_j) adj[i, j] + sum_{v!=i, j} x[v] adj[i, v]
            # Ax[i] = x_old_i * 0 + x_old_j adj[i, j] + sum_{v!=i, j} x[v] adj[i, v]
            # Ax'[i] = Ax[i] - 2 x_old_j adj[i, j]
            # Similarly Ax'[j] = Ax[j] - 2 x_old_i adj[i, j]
            
            Ax -= 2 * x_old_i * adj[:, best_i]
            Ax -= 2 * x_old_j * adj[:, best_j]
            
            # Fix Ax[i] and Ax[j]
            # The subtraction above was done for all k, including i and j.
            # For k=i:
            # Ax[i] was reduced by 2 x_old_i adj[i, i] (0) + 2 x_old_j adj[i, j]
            # So Ax[i] is now Ax_old[i] - 2 x_old_j adj[i, j]
            # But we need Ax'[i] = Ax_old[i] - 2 x_old_j adj[i, j]
            # Wait, is that correct?
            # Let's check.
            # Ax_old[i] = sum_v x[v] adj[i, v] = x[j] adj[i, j] + sum_{v!=i, j} x[v] adj[i, v]
            # Ax_new[i] = sum_v x'[v] adj[i, v] = x'[j] adj[i, j] + sum_{v!=i, j} x[v] adj[i, v]
            # x'[j] = -x[j]
            # Ax_new[i] = -x[j] adj[i, j] + sum_{v!=i, j} x[v] adj[i, v]
            # Ax_new[i] = Ax_old[i] - 2 x[j] adj[i, j]
            # My update: Ax -= 2 x_old_j adj[:, j]
            # So Ax[i] -= 2 x_old_j adj[i, j]
            # This matches!
            
            # Same for j.
            
            # Update x
            x[best_i] = -x_old_i
            x[best_j] = -x_old_j
            
            # Update g
            # g[k] = x[k] Ax[k]
            # For k != i, j: x[k] same, Ax[k] changed.
            # For k = i: x[i] flipped, Ax[i] changed.
            # g_new[i] = -x_old_i * (Ax_old[i] - 2 x_old_j adj[i, j])
            # g_old[i] = x_old_i * Ax_old[i]
            # This is getting complicated to update g incrementally.
            # Since n=100, let's just recompute g.
            g = x * Ax
            
            # Update current cut
            # We don't track current_cut in this function, we just return x.
            # But we need to know if we improved.
            # We can compute cut_size(x) at the end, or track it.
            # Let's track it.
            # cut_new = cut_old + best_gain
            # We don't have cut_old passed in. Let's just return x and let caller compute cut.
            # Or pass cut in.
            # Let's just return x. The caller can check if cut improved.
            # But local_search usually returns the best found.
            # Let's assume the caller checks.
            
        return x

    def hybrid_local_search(x, max_iters=50000):
        # Alternate between single flip and double flip
        # Or just do single flip until stuck, then double flip until stuck.
        
        x, _ = local_search(x, max_iters=max_iters//2)
        x = two_flip_local_search(x, max_iters=max_iters//2)
        x, _ = local_search(x, max_iters=max_iters//2)
        
        return x

    def ils(x, num_iters=1000):
        best_x = x.copy()
        best_cut = cut_size(x)
        
        for i in range(num_iters):
            if time_left() < 5.0:
                break
            
            # Perturb
            x_pert = x.copy()
            # Flip 5 random vertices
            k = 5
            if n > k:
                flip_idx = np.random.choice(n, k, replace=False)
                x_pert[flip_idx] = -x_pert[flip_idx]
            
            # Local search
            x_ls = hybrid_local_search(x_pert)
            cut_ls = cut_size(x_ls)
            
            if cut_ls > best_cut:
                best_cut = cut_ls
                best_x = x_ls
                x = x_ls
            else:
                # Revert to previous best (standard ILS)
                pass
                
        return best_x, best_cut

    # 1. Spectral Init
    x = spectral_init()
    
    # 2. Initial Local Search
    x = hybrid_local_search(x)
    best_x = x
    best_cut = cut_size(x)
    
    # 3. ILS
    x, cut = ils(x, num_iters=5000)
    if cut > best_cut:
        best_cut = cut
        best_x = x

    # 4. Final Polish
    if time_left() > 1.0:
        x = hybrid_local_search(best_x)
        cut = cut_size(x)
        if cut > best_cut:
            best_cut = cut
            best_x = x

    # Convert to 0/1 labels
    # x is +/- 1. Map 1 -> 1, -1 -> 0
    labels = (best_x + 1) // 2
    return labels.astype(int)
