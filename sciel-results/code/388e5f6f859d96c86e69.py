import time
import random
import numpy as np

def construct():
    n = 40
    deadline = time.time() + 140
    all_pairs = np.triu_indices(n, k=1)
    all_u = all_pairs[0].astype(np.int32)
    all_v = all_pairs[1].astype(np.int32)
    n_pairs = len(all_u)
    I = np.eye(n, dtype=np.uint8)

    # ---- vectorized C6 existence check ----
    # Returns bool array of length n_pairs: True iff adding that edge creates a C6
    def c6_mask(adj):
        A = adj.astype(np.uint8)
        A2 = A @ A
        # For each non-edge (u,v): does a path u-x-y-z-v exist?
        # Equivalent: exists x in N(u), z in N(v), x!=z, (N(x) ∩ N(z)) \ {u,v} != 0
        # Vectorize: for each (u,v), compute sum over x,z of indicator
        # Use: (A2[u,v] > 0) means path length 2. But we need path length 4.
        # Path u-x-y-z-v: count = sum_x sum_z [x in N(u)] [z in N(v)] [x!=z] [N(x)∩N(z) \ {u,v} != 0]
        # Alternative: (A^4)[u,v] > 0 AND no shorter path... but A^4 counts walks.
        # Better: use the direct formula.
        # For each pair (u,v): 
        #   M[u,v] = sum_{x != u,v} sum_{z != u,v,x} [A[u,x]] [A[v,z]] max(0, (A[x,:] & A[z,:]) sum - [A[x,u]] - [A[x,v]] - [A[z,u]] - [A[z,v]] + ...)
        # This is complex. Let's use a simpler but still fast approach:
        # Compute B = A^2 (number of length-2 paths). A C6 through (u,v) exists iff
        # there exist x in N(u), z in N(v), x != z, x != v, z != u, with common neighbor.
        # common_neighbor_count(x,z) = (A[x,:] & A[z,:]).sum()
        # We need: exists x in N(u)\{v}, z in N(v)\{u,x} s.t. common_neighbor_count(x,z) - [u in N(x)∩N(z)] - [v in N(x)∩N(z)] > 0
        # Since u in N(x) (x in N(u)) and v in N(z) (z in N(v)):
        #   common_neighbor_count(x,z) >= 1 + [u==z or v==x or ...]
        # Actually simpler: y must be in N(x) ∩ N(z) and y != u, y != v, y != x, y != z.
        # (A[x,:] & A[z,:]).sum() counts all common neighbors including possibly u and v.
        # u is in N(x) iff A[x,u]=1 (yes, since x in N(u)). u is in N(z) iff A[z,u]=1.
        # v is in N(x) iff A[x,v]=1. v is in N(z) iff A[z,v]=1 (yes, since z in N(v)).
        # So common neighbors of x and z include u (if A[z,u]=1) and v (if A[x,v]=1).
        # valid_y_count = (A[x,:] & A[z,:]).sum() - A[z,u] - A[x,v]
        # We need valid_y_count > 0 for some valid (x,z).
        # 
        # Let's compute this with numpy broadcasting for all pairs at once.
        # A2[u,v] = sum_k A[u,k]*A[k,v] = number of length-2 paths from u to v.
        # A C6 through (u,v) exists iff A4[u,v] >= 1 where A4 counts length-4 paths,
        # BUT we must exclude paths that repeat vertices (not simple).
        # For a simple graph, a walk of length 4 from u to v that is a simple path
        # (all 5 vertices distinct) is exactly what we need.
        # A^4[u,v] counts all walks of length 4. Some may repeat vertices.
        # However, if A^4[u,v] > 0 and A[u,v] == 0 and A^2[u,v] == 0, 
        # does that guarantee a C6? Not necessarily - could be a walk u-x-u-z-v (repeats u).
        # 
        # Let's use a different approach: compute for each pair (u,v) whether there's
        # a valid C6 using explicit but vectorized logic.
        
        # For efficiency, let's compute:
        # For each x (40 values), for each z (40 values), x != z:
        #   cn[x,z] = (A[x,:] & A[z,:]).sum() - A[z, :] @ A[x, :]  # no wait
        #   Actually cn_raw[x,z] = (A[x,:] & A[z,:]).sum()
        #   cn_valid[x,z] = cn_raw[x,z] - A[x,z] - A[z,x]  # subtract x and z themselves? No.
        #   We need to subtract u and v specifically for each (u,v) pair.
        #   cn_valid_for_uv[x,z] = cn_raw[x,z] - A[z,u] - A[x,v]
        #   (u is common neighbor of x,z iff A[x,u] AND A[z,u]; but A[x,u]=1 always since x in N(u).
        #    So u is common neighbor iff A[z,u]=1.)
        #   (v is common neighbor of x,z iff A[x,v] AND A[z,v]; A[z,v]=1 always since z in N(v).
        #    So v is common neighbor iff A[x,v]=1.)
        #   Also need x != z, x != v, z != u, y != x, y != z.
        #   y != x: x is in N(x)? No (simple graph, no self-loops). So x is never a common neighbor of x and z.
        #   y != z: similarly z is never in N(z). So no issue.
        #   So: valid_y_count(x,z;u,v) = cn_raw[x,z] - A[z,u] - A[x,v]
        #   And we need x in N(u) \ {v}, z in N(v) \ {u}, x != z.
        
        # This is still O(n^2) per pair if done naively. Let me think of a fully vectorized way.
        # 
        # Alternative: For the specific task of checking ALL non-edges at once:
        # Create a 4D tensor? Too large (40^4 = 2.5M, actually fine for uint8).
        # 
        # Let's just do it with moderate vectorization:
        # Precompute cn_raw: 40x40 matrix where cn_raw[x,z] = (A[x,:] & A[z,:]).sum()
        # Then for each (u,v) pair, check if exists x,z with:
        #   A[u,x]=1, A[v,z]=1, x!=z, x!=v, z!=u, cn_raw[x,z] - A[z,u] - A[x,v] > 0
        
        # Actually, let me just use a semi-vectorized approach that's still much faster than pure Python:
        # For each u (40 times), vectorize over all v.
        
        result = np.zeros(n_pairs, dtype=bool)
        cn_raw = (A @ A.T).astype(np.int32)  # cn_raw[x,z] = |N(x) ∩ N(z)|... wait no
        # A @ A.T: (A @ A.T)[x,z] = sum_k A[x,k]*A[k,z] = |N(x) ∩ N(z)|
        # Yes! cn_raw[x,z] = (A @ A.T)[x,z]
        
        for idx in range(n_pairs):
            u = all_u[idx]
            v = all_v[idx]
            if A[u, v]:
                continue
            # x must be in N(u) \ {v}
            # z must be in N(v) \ {u, x}
            # Check: exists x in N(u)\{v}, z in N(v)\{u,x} with cn_raw[x,z] - A[z,u] - A[x,v] > 0
            
            # Get neighbors
            nu = np.where(A[u, :])[0]
            nv = np.where(A[v, :])[0]
            
            # Filter
            nu = nu[nu != v]
            nv = nv[nv != u]
            
            if len(nu) == 0 or len(nv) == 0:
                continue
            
            # For each x in nu, z in nv, x != z:
            #   valid = cn_raw[x,z] - A[z,u] - A[x,v] > 0
            # Vectorize over (x,z) pairs
            x_arr = nu[:, np.newaxis]
            z_arr = nv[np.newaxis, :]
            
            # Mask: x != z
            valid_xz = (x_arr != z_arr)
            
            # Get cn_raw values
            cn_vals = cn_raw[x_arr, z_arr]
            # Subtract A[z,u] and A[x,v]
            au_z = A[z_arr, u]  # A[z,u] for each (x,z)
            av_x = A[x_arr, v]  # A[x,v] for each (x,z)
            
            valid_count = cn_vals - au_z - av_x
            found = np.any(valid_count[valid_xz] > 0)
            result[idx] = found
        
        return result

    # ---- greedy add phase (vectorized) ----
    def greedy_add(adj, rng):
        A = adj
        count = int(A.sum() // 2)
        
        # Find all non-edges
        non_edge_mask = ~A.astype(bool)
        non_edge_mask = non_edge_mask | non_edge_mask.T  # symmetric
        np.fill_diagonal(non_edge_mask, True)  # exclude self-loops
        # Actually we want upper triangle non-edges
        non_edge_idx = np.where(non_edge_mask[all_u, all_v])[0]
        
        # Shuffle
        order = rng.permutation(len(non_edge_idx))
        
        for idx in order:
            if time.time() > deadline - 5:
                break
            u = all_u[non_edge_idx[idx]]
            v = all_v[non_edge_idx[idx]]
            if A[u, v]:
                continue
            # Check C6 for this single edge
            nu = np.where(A[u, :])[0]
            nv = np.where(A[v, :])[0]
            nu = nu[nu != v]
            nv = nv[nv != u]
            if len(nu) == 0 or len(nv) == 0:
                A[u, v] = 1
                A[v, u] = 1
                count += 1
                continue
            cn_raw = (A[u, :] & A[v, :]).sum()  # not quite right, let me redo
            # For this specific (u,v): check if C6 exists
            # x in N(u)\{v}, z in N(v)\{u}, x!=z, cn[x,z] - A[z,u] - A[x,v] > 0
            found = False
            for x in nu:
                for z in nv:
                    if x == z:
                        continue
                    cn = int((A[x, :] & A[z, :]).sum())
                    if cn - int(A[z, u]) - int(A[x, v]) > 0:
                        found = True
                        break
                if found:
                    break
            if not found:
                A[u, v] = 1
                A[v, u] = 1
                count += 1
        return A, count

    # ---- 2-opt swap: remove e1, e2; add e3, e4 ----
    def try_2opt_swap(adj, count, rng):
        """Try to find a 2-exchange that increases edge count."""
        A = adj
        # Get current edges
        edges = [(all_u[i], all_v[i]) for i in range(n_pairs) if A[all_u[i], all_v[i]]]
        if len(edges) < 2:
            return A, count, False
        
        # Get non-edges
        non_edges = [(all_u[i], all_v[i]) for i in range(n_pairs) if not A[all_u[i], all_v[i]]]
        if len(non_edges) < 2:
            return A, count, False
        
        # Try a few random 2-swaps
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            e1, e2 = rng.sample(edges, 2)
            f1, f2 = rng.sample(non_edges, 2)
            
            # Remove e1, e2
            A[e1[0], e1[1]] = 0; A[e1[1], e1[0]] = 0
            A[e2[0], e2[1]] = 0; A[e2[1], e2[0]] = 0
            new_count = count - 2
            
            # Try to add f1, f2
            added = 0
            for f in [f1, f2]:
                u, v = f
                if A[u, v]:
                    continue
                # Check C6
                nu = np.where(A[u, :])[0]
                nv = np.where(A[v, :])[0]
                nu = nu[nu != v]
                nv = nv[nv != u]
                found = False
                for x in nu:
                    for z in nv:
                        if x == z:
                            continue
                        cn = int((A[x, :] & A[z, :]).sum())
                        if cn - int(A[z, u]) - int(A[x, v]) > 0:
                            found = True
                            break
                    if found:
                        break
                if not found:
                    A[u, v] = 1
                    A[v, u] = 1
                    added += 1
            
            new_count += added
            if new_count > count:
                return A, new_count, True
            else:
                # Revert
                A[e1[0], e1[1]] = 1; A[e1[1], e1[0]] = 1
                A[e2[0], e2[1]] = 1; A[e2[1], e2[0]] = 1
                # Remove f's that were added
                for f in [f1, f2]:
                    u, v = f
                    if A[u, v]:
                        A[u, v] = 0
                        A[v, u] = 0
        return A, count, False

    # ---- main search loop ----
    rng = random.Random(42)
    best_count = 0
    best_adj = None
    
    for restart in range(200):
        if time.time() > deadline - 10:
            break
        
        # Initialize: random graph with moderate density
        density = 0.3 + 0.3 * (restart / 200)  # increase density over time
        A = np.zeros((n, n), dtype=np.uint8)
        for i in range(n_pairs):
            if rng.random() < density:
                u, v = all_u[i], all_v[i]
                A[u, v] = 1
                A[v, u] = 1
        
        # Check initial graph is C6-free (it might not be for high density)
        # For safety, start with lower density and build up
        # Actually, let's start with a known C6-free structure and perturb
        
        # Better initialization: start with random 2-regular-ish graph and add
        A = np.zeros((n, n), dtype=np.uint8)
        # Create a random 2-regular graph (union of cycles)
        perm = rng.sample(range(n), n)
        for i in range(n):
            A[perm[i], perm[(i+1) % n]] = 1
            A[perm[(i+1) % n], perm[i]] = 1
        # Add some random edges
        for _ in range(30):
            u, v = rng.randint(0, n-1), rng.randint(0, n-1)
            if u != v:
                A[u, v] = 1
                A[v, u] = 1
        
        count = int(A.sum() // 2)
        
        # Greedy add phase
        A, count = greedy_add(A, rng)
        
        # 2-opt swaps
        improved = True
        while improved and time.time() < deadline - 10:
            A, count, improved = try_2opt_swap(A, count, rng)
        
        if count > best_count:
            best_count = count
            best_adj = A.copy()
        
        # Perturb best for next restart
        if best_adj is not None and restart < 195:
            A = best_adj.copy()
            # Remove 3-6 random edges
            edges = [(all_u[i], all_v[i]) for i in range(n_pairs) if A[all_u[i], all_v[i]]]
            if edges:
                k = min(rng.randint(3, 6), len(edges))
                for e in rng.sample(edges, k):
                    A[e[0], e[1]] = 0
                    A[e[1], e[0]] = 0
                count = int(A.sum() // 2)
                # Greedy add from perturbed state
                A, count = greedy_add(A, rng)
                # 2-opt
                improved = True
                while improved and time.time() < deadline - 10:
                    A, count, improved = try_2opt_swap(A, count, rng)
                if count > best_count:
                    best_count = count
                    best_adj = A.copy()
    
    # Convert to edge list
    result = []
    for i in range(n_pairs):
        if best_adj[all_u[i], all_v[i]]:
            result.append([int(all_u[i]), int(all_v[i])])
    
    return result
